"""
PCA + PCA^-1
PCA + PLS + PCA^-1
PLS
Kraken
Kraken + learnable PCA
non-linear Kraken
cov-VAE/CLIP pretraining + MLP + Kraken learnable PCA
GNN
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from scipy import stats
import torch
import torch.nn as nn
from loss import create_loss_fn

def predict_from_loader(model, data_loader, device=None):
    """Generate predictions from a model using a data loader."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"]
            y = batch["y"]
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets


class CrossModalPCA(nn.Module):
    """
    Cross-modal PCA model implemented in PyTorch.
    Maps source modality to target modality using PCA decomposition from training data.
    Given a base data object (which provides population mean and loadings for each modality), 
    it projects from source modality into source PCA space, truncates to num_components, 
    then reconstructs (inverse transforms) via the target modality's PCA.

    Typical intended use is for source and target in {'SC', 'FC'}.

    Args:
        base: Dataset base object (e.g., HCP_Base) providing PCA means/loadings for both modalities on train partition.
        num_components: Number of components to use (int).
        source: 'SC' or 'FC' (modality for input)
        target: 'SC' or 'FC' (modality for output)
    """
    def __init__(self, base, num_components=256,device=None):
        super().__init__()
        self.base = base
        self.num_components = num_components

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Select appropriate means, loadings for source and target
        if base.source == "SC":
            self.source_mean = torch.tensor(base.sc_train_avg, dtype=torch.float32, device=self.device)
            self.source_loadings = torch.tensor(base.sc_train_loadings, dtype=torch.float32, device=self.device)
        elif base.source == "FC":
            self.source_mean = torch.tensor(base.fc_train_avg, dtype=torch.float32, device=self.device)
            self.source_loadings = torch.tensor(base.fc_train_loadings, dtype=torch.float32, device=self.device)

        if base.target == "SC":
            self.target_mean = torch.tensor(base.sc_train_avg, dtype=torch.float32, device=self.device)
            self.target_loadings = torch.tensor(base.sc_train_loadings, dtype=torch.float32, device=self.device)
        elif base.target == "FC":
            self.target_mean = torch.tensor(base.fc_train_avg, dtype=torch.float32, device=self.device)
            self.target_loadings = torch.tensor(base.fc_train_loadings, dtype=torch.float32, device=self.device)

        print(f"Source mean shape: {self.source_mean.shape}")
        print(f"Source loadings shape: {self.source_loadings.shape}")
        print(f"Target mean shape: {self.target_mean.shape}")
        print(f"Target loadings shape: {self.target_loadings.shape}")
        
        # Only keep first k components, and explicitly ensure float32 dtype to avoid matmul dtype mismatch
        self.register_buffer('source_loadings_k', self.source_loadings[:, :self.num_components].to(torch.float32))
        self.register_buffer('target_loadings_k', self.target_loadings[:, :self.num_components].to(torch.float32))
        self.register_buffer('source_mean_', self.source_mean.to(torch.float32))
        self.register_buffer('target_mean_', self.target_mean.to(torch.float32))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor(batch_size, d)): source modality tensor.

        Returns:
            y_hat (torch.Tensor(batch_size, d)): predicted target modality.
        """
        x = x.to(self.device).to(torch.float32)

        # 1. Center input
        x_centered = x - self.source_mean_

        # 2. Project into truncated source PCA space (batch @ (d, k)) => (batch, k)
        z = torch.matmul(x_centered, self.source_loadings_k)

        # 3. Reconstruct with target PCA (inverse transform): (batch, k) @ (k, d) => (batch, d)
        y_hat = torch.matmul(z, self.target_loadings_k.t()) + self.target_mean_

        return y_hat

    def to(self, *args, **kwargs): # for device mismatch issues
        super().to(*args, **kwargs)
        self.source_mean_ = self.source_mean_.to(*args, **kwargs)
        self.target_mean_ = self.target_mean_.to(*args, **kwargs)
        self.source_loadings_k = self.source_loadings_k.to(*args, **kwargs)
        self.target_loadings_k = self.target_loadings_k.to(*args, **kwargs)
        return self


class CrossModal_PLS_SVD(nn.Module):
    """
    Direct PLS-SVD cross-modal model WITHOUT PCA pre-projection.
    
    Uses implicit SVD via scipy.sparse.linalg.svds with a LinearOperator to compute
    the top-k singular vectors of the cross-covariance matrix X.T @ Y WITHOUT
    explicitly forming the full matrix. This is memory-efficient for high-dimensional data.
    
    Memory: O(n * p) instead of O(p * p)
    Time: O(k * n * p) per iteration of the Lanczos algorithm
    
    Architecture: y_hat = (x - μ_x) @ W_x @ B @ W_y.T + μ_y
    
    Where:
    - W_x (p_x, k): Left singular vectors of X.T @ Y (source loadings)
    - W_y (p_y, k): Right singular vectors of X.T @ Y (target loadings)  
    - B (k, k): Latent-space regression matrix
    
    Args:
        base: Dataset base object providing raw connectivity matrices and partition indices.
        n_components: Number of PLS components (latent dimensions).
        device: torch device. Defaults to CUDA if available.
    """
    def __init__(self, base, n_components=10, device=None):
        super().__init__()
        from scipy.sparse.linalg import LinearOperator, svds
        
        self.n_components = n_components
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Get training data indices
        train_indices = base.trainvaltest_partition_indices["train"]
        
        # Get raw data based on source/target modalities
        if base.source == "SC":
            X_train = base.sc_upper_triangles[train_indices].astype(np.float64)
        elif base.source == "FC":
            X_train = base.fc_upper_triangles[train_indices].astype(np.float64)
        else:
            raise ValueError(f"Unknown source modality: {base.source}")
            
        if base.target == "SC":
            Y_train = base.sc_upper_triangles[train_indices].astype(np.float64)
        elif base.target == "FC":
            Y_train = base.fc_upper_triangles[train_indices].astype(np.float64)
        else:
            raise ValueError(f"Unknown target modality: {base.target}")
        
        n_samples, p_x = X_train.shape
        _, p_y = Y_train.shape
        
        # Compute means for centering
        source_mean_np = X_train.mean(axis=0)
        target_mean_np = Y_train.mean(axis=0)
        
        # Center the training data
        X_centered = X_train - source_mean_np
        Y_centered = Y_train - target_mean_np
        
        print(f"Fitting PLS-SVD (implicit): X shape {X_train.shape}, Y shape {Y_train.shape}")
        print(f"n_components: {n_components}")
        print(f"Memory-efficient: NOT forming {p_x}x{p_y} cross-covariance matrix")
        
        # Define implicit cross-covariance operator C = X.T @ Y
        # C has shape (p_x, p_y) but we never form it explicitly
        # Instead we define matvec (C @ v) and rmatvec (C.T @ v)
        def matvec(v):
            """Compute C @ v = X.T @ (Y @ v) without forming C"""
            return X_centered.T @ (Y_centered @ v)
        
        def rmatvec(v):
            """Compute C.T @ v = Y.T @ (X @ v) without forming C"""
            return Y_centered.T @ (X_centered @ v)
        
        C_operator = LinearOperator(
            shape=(p_x, p_y),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64
        )
        
        # Compute top-k singular vectors using iterative Lanczos algorithm
        # This only requires matrix-vector products, not the full matrix
        print(f"Computing top-{n_components} singular vectors via implicit SVD...")
        U, S, Vt = svds(C_operator, k=n_components)
        
        # svds returns in ascending order, reverse to descending
        idx = np.argsort(S)[::-1]
        U = U[:, idx]      # (p_x, k) - source weights
        S = S[idx]         # (k,) - singular values
        Vt = Vt[idx, :]    # (k, p_y) - target weights transposed
        
        print(f"Singular values: {S[:5]}..." if len(S) > 5 else f"Singular values: {S}")
        
        # W_x and W_y are the PLS weights (singular vectors)
        W_x = U                    # (p_x, k)
        W_y = Vt.T                 # (p_y, k)
        
        # Compute latent projections for training data
        X_latent = X_centered @ W_x  # (n, k)
        Y_latent = Y_centered @ W_y  # (n, k)
        
        # Fit regression in latent space: Y_latent ≈ X_latent @ B
        B = np.linalg.lstsq(X_latent, Y_latent, rcond=None)[0]  # (k, k)
        
        # Store as torch tensors
        self.register_buffer('source_mean', torch.tensor(source_mean_np, dtype=torch.float32, device=device))
        self.register_buffer('target_mean', torch.tensor(target_mean_np, dtype=torch.float32, device=device))
        self.register_buffer('W_x', torch.tensor(W_x, dtype=torch.float32, device=device))  # (p_x, k)
        self.register_buffer('W_y', torch.tensor(W_y, dtype=torch.float32, device=device))  # (p_y, k)
        self.register_buffer('B', torch.tensor(B, dtype=torch.float32, device=device))      # (k, k)
        self.register_buffer('singular_values', torch.tensor(S, dtype=torch.float32, device=device))
        
        # Compute train R² for reference
        Y_pred_train = X_latent @ B @ W_y.T + target_mean_np
        train_r2 = 1 - np.sum((Y_train - Y_pred_train)**2) / np.sum((Y_train - Y_train.mean(axis=0))**2)
        print(f"PLS-SVD train R²: {train_r2:.4f}")
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, p_x) source modality tensor
        Returns:
            y_hat: (batch_size, p_y) predicted target modality
        """
        x = x.to(self.device).to(torch.float32)
        
        # 1. Center input
        x_centered = x - self.source_mean
        
        # 2. Project to latent space via source PLS weights
        z_x = torch.matmul(x_centered, self.W_x)  # (batch, k)
        
        # 3. Transform in latent space
        z_y = torch.matmul(z_x, self.B)  # (batch, k)
        
        # 4. Back-project via target PLS weights
        y_hat = torch.matmul(z_y, self.W_y.T) + self.target_mean  # (batch, p_y)
        
        return y_hat
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


class CrossModal_PCA_PLS(nn.Module):
    """
    Cross-modal PCA model implemented in PyTorch.
    Given a base data object (which provides population mean and loadings for each modality), 
    it projects from source modality into source PCA space, truncates to num_components, 
    the PCA is learned between the PCA of the source and target modalities,
    then reconstructs (inverse transforms) via the target modality's PCA.

    # example: X @ B_sc = C ← PLS regression → C_hat @ B_fc^T = X_hat 

    Typical intended use is for source and target in {'SC', 'FC'}.

    Args:
        base: Dataset base object (e.g., HCP_Base) providing PCA means/loadings for both modalities on train partition.
        num_components: Number of components to use (int).
        source: 'SC' or 'FC' (modality for input)
        target: 'SC' or 'FC' (modality for output)
    """
    def __init__(self, base, n_components_pca=32, n_components_pls=4, device=None):
        super().__init__()
        self.base = base
        self.n_components_pca = n_components_pca
        self.n_components_pls = n_components_pls

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Select appropriate means, loadings, and scores for source and target
        if base.source == "SC":
            self.source_mean = torch.tensor(base.sc_train_avg, dtype=torch.float32, device=self.device)
            self.source_loadings = torch.tensor(base.sc_train_loadings, dtype=torch.float32, device=self.device)
            self.source_scores = torch.tensor(base.sc_train_scores, dtype=torch.float32, device=self.device)  # shape: (n_train, n_components_sc)
        elif base.source == "FC":
            self.source_mean = torch.tensor(base.fc_train_avg, dtype=torch.float32, device=self.device)
            self.source_loadings = torch.tensor(base.fc_train_loadings, dtype=torch.float32, device=self.device)
            self.source_scores = torch.tensor(base.fc_train_scores, dtype=torch.float32, device=self.device)  # shape: (n_train, n_components_fc)

        if base.target == "SC":
            self.target_mean = torch.tensor(base.sc_train_avg, dtype=torch.float32, device=self.device)
            self.target_loadings = torch.tensor(base.sc_train_loadings, dtype=torch.float32, device=self.device)
            self.target_scores = torch.tensor(base.sc_train_scores, dtype=torch.float32, device=self.device)  # shape: (n_train, n_components_sc)
        elif base.target == "FC":
            self.target_mean = torch.tensor(base.fc_train_avg, dtype=torch.float32, device=self.device)
            self.target_loadings = torch.tensor(base.fc_train_loadings, dtype=torch.float32, device=self.device)
            self.target_scores = torch.tensor(base.fc_train_scores, dtype=torch.float32, device=self.device)  # shape: (n_train, n_components_fc)

        # --- Fit PLS model on the training data's PCA scores ---
        # Only use first n_components_pca components for PLS (np->cpu for sklearn)
        X_pls = self.source_scores[:, :self.n_components_pca].cpu().numpy()
        Y_pls = self.target_scores[:, :self.n_components_pca].cpu().numpy()

        self.pls_pca_fusion_model = PLSRegression(n_components=self.n_components_pls)
        self.pls_pca_fusion_model.fit(X_pls, Y_pls)
        print("PLS model fit score: ", self.pls_pca_fusion_model.score(X_pls, Y_pls))

        print(f"PCA source scores shape: {self.source_scores.shape}")
        print("PLS source scores shape: ", self.pls_pca_fusion_model.x_scores_.shape)
        print("PLS target scores shape: ", self.pls_pca_fusion_model.y_scores_.shape)
        print(f"PCA target scores shape: {self.target_scores.shape}")
        
        # Only keep first k components, and explicitly ensure float32 dtype to avoid matmul dtype mismatch
        self.register_buffer('source_loadings_k', self.source_loadings[:, :self.n_components_pca].to(torch.float32))
        self.register_buffer('target_loadings_k', self.target_loadings[:, :self.n_components_pca].to(torch.float32))
        self.register_buffer('source_mean_', self.source_mean.to(torch.float32))
        self.register_buffer('target_mean_', self.target_mean.to(torch.float32))

        # https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/cross_decomposition/_pls.py#L564
        self.register_buffer('x_rotations_', torch.tensor(self.pls_pca_fusion_model.x_rotations_, dtype=torch.float32, device=self.device))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor(batch_size, d)): source modality tensor.

        Returns:
            y_hat (torch.Tensor(batch_size, d)): predicted target modality.
        """
        x = x.to(self.device).to(torch.float32)

        # 1. Center input
        x_centered = x - self.source_mean_

        # 2. Project into truncated source PCA space (batch @ (d, k)) => (batch, k)
        z = torch.matmul(x_centered, self.source_loadings_k)

        # 3. Run through learned PLS model in PCA space to get y_hat_pca
        #    - Switch to cpu and numpy for sklearn
        z_np = z.detach().cpu().numpy()
        y_hat_pca_np = self.pls_pca_fusion_model.predict(z_np)

        # 4. Convert PLS output back to torch and move to device
        y_hat_pca = torch.tensor(y_hat_pca_np, dtype=torch.float32, device=self.device)

        # 5. Reconstruct with target PCA (inverse transform): (batch, k) @ (k, d) => (batch, d)
        y_hat = torch.matmul(y_hat_pca, self.target_loadings_k.t()) + self.target_mean_
        return y_hat

    def to(self, *args, **kwargs): # for device mismatch issues
        super().to(*args, **kwargs)
        self.source_mean_ = self.source_mean_.to(*args, **kwargs)
        self.target_mean_ = self.target_mean_.to(*args, **kwargs)
        self.source_loadings_k = self.source_loadings_k.to(*args, **kwargs)
        self.target_loadings_k = self.target_loadings_k.to(*args, **kwargs)
        return self


class CrossModal_PCA_PLS_learnable(nn.Module):
    """
    Learnable cross-modal encoder-decoder model.
    
    Architecture: (x - μ_s) @ W_enc -> dropout -> @ W_mid -> dropout -> @ W_dec + μ_t
    
    Can be initialized from PCA+PLS (default) or randomly (random_init=True).
    Supports dropout and L1/L2 regularization for preventing overfitting.

    Args:
        base: Dataset base object providing PCA means/loadings/scores.
        n_components_pca_source: Encoder output dimension (latent dim from source).
        n_components_pca_target: Decoder input dimension (latent dim for target).
        n_components_pls: PLS components (only used if random_init=False).
        device: torch device.
        learn_means: Make means learnable. Default False.
        learn_encoder: Make W_enc learnable. Default True.
        learn_mid: Make W_mid learnable. Default True.
        learn_decoder: Make W_dec learnable. Default True.
        random_init: If True, use random initialization instead of PCA/PLS. Default False.
        dropout: Dropout probability (0 = no dropout). Default 0.0.
        l1_reg: L1 regularization weight. Default 0.0.
        l2_reg: L2 regularization weight. Default 0.0.
        lr: Learning rate. Default 1e-4.
        epochs: Number of training epochs. Default 100.
        loss_fn: Loss type string or module. Default 'mse'.
        loss_alpha: Weight for weighted_mse. Default 0.5.
    """
    def __init__(self, base, n_components_pca_source=256, n_components_pca_target=256, 
                 n_components_pls=64, device=None, learn_means=False,
                 learn_encoder=False, learn_mid=True, learn_decoder=False,
                 random_init=False, dropout=0.3, l1_reg=0.0, l2_reg=0.001,
                 lr=1e-4, epochs=100, loss_fn='demeaned_mse', loss_alpha=0.5):
        super().__init__()
        self.n_components_pca_source = n_components_pca_source
        self.n_components_pca_target = n_components_pca_target
        self.random_init = random_init
        self.dropout_p = dropout
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Loss function
        if isinstance(loss_fn, str):
            self.loss_fn = create_loss_fn(loss_fn, base=base, alpha=loss_alpha)
        else:
            self.loss_fn = loss_fn
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(device)

        # Get dimensions and means from base
        if base.source == "SC":
            source_mean, source_loadings, source_scores = base.sc_train_avg, base.sc_train_loadings, base.sc_train_scores
        else:
            source_mean, source_loadings, source_scores = base.fc_train_avg, base.fc_train_loadings, base.fc_train_scores
        if base.target == "SC":
            target_mean, target_loadings, target_scores = base.sc_train_avg, base.sc_train_loadings, base.sc_train_scores
        else:
            target_mean, target_loadings, target_scores = base.fc_train_avg, base.fc_train_loadings, base.fc_train_scores

        d_source = source_loadings.shape[0]
        d_target = target_loadings.shape[0]
        k_src, k_tgt = n_components_pca_source, n_components_pca_target

        # Means (always from data, optionally learnable)
        self.source_mean = nn.Parameter(torch.tensor(source_mean, dtype=torch.float32, device=device), requires_grad=learn_means)
        self.target_mean = nn.Parameter(torch.tensor(target_mean, dtype=torch.float32, device=device), requires_grad=learn_means)

        # Fit PLS (needed for non-random mid layer initialization)
        pls = PLSRegression(n_components=n_components_pls)
        pls.fit(source_scores[:, :k_src], target_scores[:, :k_tgt])
        
        # Helper to init weight: random if learnable AND random_init, else from PCA/PLS
        def init_weight(shape, pca_pls_data, learnable):
            if learnable and random_init:
                w = torch.empty(shape, device=device)
                nn.init.kaiming_uniform_(w, a=np.sqrt(5))
                return nn.Parameter(w, requires_grad=True)
            else:
                return nn.Parameter(torch.tensor(pca_pls_data, dtype=torch.float32, device=device), requires_grad=learnable)
        
        self.W_enc = init_weight((d_source, k_src), source_loadings[:, :k_src], learn_encoder)
        self.W_mid = init_weight((k_src, k_tgt), pls.coef_, learn_mid)
        self.W_dec = init_weight((k_tgt, d_target), target_loadings[:, :k_tgt].T, learn_decoder)
        
        # Log initialization
        init_info = lambda name, p: f"{name}:{'rand' if (p.requires_grad and random_init) else 'pca/pls'}{'(frozen)' if not p.requires_grad else ''}"
        print(f"Init: {init_info('enc', self.W_enc)}, {init_info('mid', self.W_mid)}, {init_info('dec', self.W_dec)}")
        print(f"Shapes: enc={tuple(self.W_enc.shape)}, mid={tuple(self.W_mid.shape)}, dec={tuple(self.W_dec.shape)}")

        # Dropout layers
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = x.to(self.device).to(torch.float32)
        z = torch.matmul(x - self.source_mean, self.W_enc)
        z = self.dropout(z)
        z = torch.matmul(z, self.W_mid.T)
        z = self.dropout(z)
        return torch.matmul(z, self.W_dec) + self.target_mean

    def get_reg_loss(self):
        """Compute L1/L2 regularization on learnable weights using model's l1_reg/l2_reg."""
        if self.l1_reg == 0 and self.l2_reg == 0:
            return 0.0
        reg = 0.0
        for p in [self.W_enc, self.W_mid, self.W_dec]:
            if p.requires_grad:
                if self.l1_reg > 0:
                    reg = reg + self.l1_reg * p.abs().sum()
                if self.l2_reg > 0:
                    reg = reg + self.l2_reg * (p ** 2).sum()
        return reg
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)