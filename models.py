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
from scipy.sparse.linalg import LinearOperator, svds
import torch
import torch.nn as nn
from loss import create_loss_fn


def compute_reg_loss(parameters, l1_l2_tuple=(0.0, 0.0)):
    """
    Compute L1/L2 regularization loss for given parameters.
    
    Args:
        parameters: Iterable of torch.nn.Parameter or torch.Tensor to regularize.
                    Can be a list/tuple of specific parameters or model.parameters().
        l1_l2_tuple: Tuple of (l1_reg, l2_reg) weights. Default (0.0, 0.0).
    
    Returns:
        Regularization loss (scalar tensor or 0.0 if both weights are 0).
    """
    l1_reg, l2_reg = l1_l2_tuple
    
    if l1_reg == 0 and l2_reg == 0:
        return 0.0
    
    reg = 0.0
    for p in parameters:
        if p.requires_grad:
            if l1_reg > 0:
                reg = reg + l1_reg * p.abs().sum()
            if l2_reg > 0:
                reg = reg + l2_reg * (p ** 2).sum()
    return reg


def get_modality_data(base, device=None, include_scores=True, include_raw_data=False):
    """
    Extract source and target modality data from base dataset object.
    
    Args:
        base: Dataset base object (e.g., HCP_Base) with source/target modalities.
        device: torch device for tensors (default: CPU).
        include_scores: If True, include PCA scores. Default True.
        include_raw_data: If True, include raw upper_triangles arrays. Default False.
    
    Returns:
        dict with keys:
            - source_mean: (d_source,) numpy array
            - source_loadings: (d_source, n_components) numpy array
            - source_scores: (n_train, n_components) numpy array (if include_scores=True)
            - target_mean: (d_target,) numpy array
            - target_loadings: (d_target, n_components) numpy array
            - target_scores: (n_train, n_components) numpy array (if include_scores=True)
            - source_upper_triangles: (n_total, d_source) numpy array (if include_raw_data=True)
            - target_upper_triangles: (n_total, d_target) numpy array (if include_raw_data=True)
            - train_indices: array of training indices (if include_raw_data=True)
    """
    if device is None:
        device = torch.device("cpu")
    
    # Source modality
    if base.source == "SC":
        source_mean = base.sc_train_avg
        source_loadings = base.sc_train_loadings
        source_scores = base.sc_train_scores if include_scores else None
        source_upper_triangles = base.sc_upper_triangles if include_raw_data else None
    elif base.source == "FC":
        source_mean = base.fc_train_avg
        source_loadings = base.fc_train_loadings
        source_scores = base.fc_train_scores if include_scores else None
        source_upper_triangles = base.fc_upper_triangles if include_raw_data else None
    else:
        raise ValueError(f"Unknown source modality: {base.source}")
    
    # Target modality
    if base.target == "SC":
        target_mean = base.sc_train_avg
        target_loadings = base.sc_train_loadings
        target_scores = base.sc_train_scores if include_scores else None
        target_upper_triangles = base.sc_upper_triangles if include_raw_data else None
    elif base.target == "FC":
        target_mean = base.fc_train_avg
        target_loadings = base.fc_train_loadings
        target_scores = base.fc_train_scores if include_scores else None
        target_upper_triangles = base.fc_upper_triangles if include_raw_data else None
    else:
        raise ValueError(f"Unknown target modality: {base.target}")
    
    result = {
        'source_mean': source_mean,
        'source_loadings': source_loadings,
        'target_mean': target_mean,
        'target_loadings': target_loadings,
    }
    
    if include_scores:
        result['source_scores'] = source_scores
        result['target_scores'] = target_scores
    
    if include_raw_data:
        result['source_upper_triangles'] = source_upper_triangles
        result['target_upper_triangles'] = target_upper_triangles
        result['train_indices'] = base.trainvaltest_partition_indices["train"]
    
    return result


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
            out = model(x)
            preds = out[0] if isinstance(out, tuple) else out
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

        # Get modality data using helper function
        data = get_modality_data(base, device=self.device, include_scores=False)
        source_mean = data['source_mean']
        source_loadings = data['source_loadings']
        target_mean = data['target_mean']
        target_loadings = data['target_loadings']
        
        self.source_mean = torch.tensor(source_mean, dtype=torch.float32, device=self.device)
        self.source_loadings = torch.tensor(source_loadings, dtype=torch.float32, device=self.device)
        self.target_mean = torch.tensor(target_mean, dtype=torch.float32, device=self.device)
        self.target_loadings = torch.tensor(target_loadings, dtype=torch.float32, device=self.device)

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
        self.n_components = n_components
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Get raw data using helper function
        data = get_modality_data(base, device=self.device, include_scores=False, include_raw_data=True)
        train_indices = data['train_indices']
        X_train = data['source_upper_triangles'][train_indices].astype(np.float64)
        Y_train = data['target_upper_triangles'][train_indices].astype(np.float64)
        
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

        # Get modality data using helper function (with scores)
        data = get_modality_data(base, device=self.device, include_scores=True)
        source_mean = data['source_mean']
        source_loadings = data['source_loadings']
        source_scores = data['source_scores']
        target_mean = data['target_mean']
        target_loadings = data['target_loadings']
        target_scores = data['target_scores']
        
        self.source_mean = torch.tensor(source_mean, dtype=torch.float32, device=self.device)
        self.source_loadings = torch.tensor(source_loadings, dtype=torch.float32, device=self.device)
        self.source_scores = torch.tensor(source_scores, dtype=torch.float32, device=self.device)  # shape: (n_train, n_components)
        self.target_mean = torch.tensor(target_mean, dtype=torch.float32, device=self.device)
        self.target_loadings = torch.tensor(target_loadings, dtype=torch.float32, device=self.device)
        self.target_scores = torch.tensor(target_scores, dtype=torch.float32, device=self.device)  # shape: (n_train, n_components)

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
        l1_l2_tuple: Tuple of (l1_reg, l2_reg) weights. Default (0.0, 0.001).
        lr: Learning rate. Default 1e-4.
        epochs: Number of training epochs. Default 100.
        loss_fn: Loss type string or module. Default 'mse'.
        loss_alpha: Weight for weighted_mse. Default 0.5.
    """
    def __init__(self, base, n_components_pca_source=256, n_components_pca_target=256, 
                 n_components_pls=64, device=None, learn_means=False,
                 learn_encoder=False, learn_mid=True, learn_decoder=False,
                 random_init=False, dropout=0.3, l1_l2_tuple=(0.0, 0.001),
                 lr=1e-4, epochs=100, loss_fn='demeaned_mse', loss_alpha=0.5):
        super().__init__()
        self.n_components_pca_source = n_components_pca_source
        self.n_components_pca_target = n_components_pca_target
        self.random_init = random_init
        self.dropout_p = dropout
        
        self.l1_reg, self.l2_reg = l1_l2_tuple
        self.l1_l2_tuple = l1_l2_tuple
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

        # Get modality data using helper function (with scores)
        data = get_modality_data(base, device=device, include_scores=True)
        source_mean = data['source_mean']
        source_loadings = data['source_loadings']
        source_scores = data['source_scores']
        target_mean = data['target_mean']
        target_loadings = data['target_loadings']
        target_scores = data['target_scores']

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
        """Compute L1/L2 regularization on learnable weights using global helper."""
        return compute_reg_loss([self.W_enc, self.W_mid, self.W_dec], self.l1_l2_tuple)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_mlp(in_dim, hidden_dims, out_dim, dropout_p=0.1):
    """Build MLP: in_dim -> hidden_dims -> out_dim with ReLU and dropout."""
    layers = []
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
    return nn.Sequential(*layers)


class CrossModalVAE(nn.Module):
    """
    Non-linear cross-modal VAE: MLP encoder -> Gaussian latent (mu, logvar) -> MLP decoder.
    Trained with reconstruction (MSE) + beta * KLD to standard Gaussian prior.
    Optional PCA projection on input (default off).

    Args:
        base: Dataset base object (e.g., HCP_Base) providing source/target means and loadings.
        latent_dim: Latent dimension (e.g. 64 or 128).
        hidden_dims: List of hidden layer sizes for encoder and decoder (e.g. [512, 256]).
        use_pca_encoder: If True, project input to PCA space before encoder. Default False.
        n_pca_components_encoder: Number of PCA components for encoder when use_pca_encoder=True.
        use_pca_decoder: If True, have decoder output PCA scores and reconstruct via target PCA. Default False.
        n_pca_components_decoder: Number of PCA components for decoder when use_pca_decoder=True.
        dropout: Dropout probability. Default 0.1.
        lr: Learning rate. Default 1e-4.
        epochs: Number of training epochs. Default 100.
        loss_fn: Loss type string or module. Default 'vae'.
        beta: KLD weight for VAE loss. Default 1.0.
        l1_l2_tuple: Tuple of (l1_reg, l2_reg) weights. Default (0.0, 0.0).
        device: torch device.

    Note:
        For backward compatibility, the deprecated arguments `use_pca` and `n_pca_components`
        are still accepted and mapped to encoder PCA when the new encoder arguments are not set.
    """
    def __init__(self, base, latent_dim=64, hidden_dims=(512, 256),
                 use_pca=False, n_pca_components=256,
                 use_pca_encoder=None, n_pca_components_encoder=None,
                 use_pca_decoder=False, n_pca_components_decoder=256,
                 dropout=0.1, lr=1e-4, epochs=100,
                 loss_fn='vae', beta=1.0, l1_l2_tuple=(0.0, 0.0), device=None):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Backwards compatibility: if new encoder args not provided, fall back to old ones
        if use_pca_encoder is None:
            use_pca_encoder = use_pca
        if n_pca_components_encoder is None:
            n_pca_components_encoder = n_pca_components
        
        self.use_pca_encoder = bool(use_pca_encoder)
        self.n_pca_components_encoder = int(n_pca_components_encoder)
        self.use_pca_decoder = bool(use_pca_decoder)
        self.n_pca_components_decoder = int(n_pca_components_decoder)
        self.dropout_p = dropout
        self.lr = lr
        self.epochs = epochs
        
        self.l1_reg, self.l2_reg = l1_l2_tuple
        self.l1_l2_tuple = l1_l2_tuple

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Loss function
        if isinstance(loss_fn, str) and loss_fn == 'vae':
            self.loss_fn = create_loss_fn('vae', beta=beta)
        else:
            self.loss_fn = loss_fn
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(device)

        # Get modality data using helper function
        data = get_modality_data(base, device=device, include_scores=False)
        source_mean = data['source_mean']
        source_loadings = data['source_loadings']
        target_mean = data['target_mean']
        target_loadings = data['target_loadings']

        d_source = source_loadings.shape[0]
        d_target = target_mean.shape[0]

        # Buffers: means and optional PCA loadings
        self.register_buffer('source_mean_', torch.tensor(source_mean, dtype=torch.float32, device=device))
        self.register_buffer('target_mean_', torch.tensor(target_mean, dtype=torch.float32, device=device))
        # Encoder PCA
        if self.use_pca_encoder:
            enc_loadings_k = source_loadings[:, :self.n_pca_components_encoder]
            self.register_buffer('source_loadings_enc', torch.tensor(enc_loadings_k, dtype=torch.float32, device=device))
            encoder_in_dim = self.n_pca_components_encoder
            print(f"CrossModalVAE: use_pca_encoder=True, encoder input dim={encoder_in_dim}")
        else:
            self.register_buffer('source_loadings_enc', None)
            encoder_in_dim = d_source
            print(f"CrossModalVAE: use_pca_encoder=False, encoder input dim={encoder_in_dim}")

        # Decoder PCA
        if self.use_pca_decoder:
            dec_loadings_k = target_loadings[:, :self.n_pca_components_decoder]
            self.register_buffer('target_loadings_dec', torch.tensor(dec_loadings_k, dtype=torch.float32, device=device))
            decoder_out_dim = self.n_pca_components_decoder
            print(f"CrossModalVAE: use_pca_decoder=True, decoder output dim={decoder_out_dim}")
        else:
            self.register_buffer('target_loadings_dec', None)
            decoder_out_dim = d_target
            print(f"CrossModalVAE: use_pca_decoder=False, decoder output dim={decoder_out_dim}")

        # Encoder: input -> hidden_dims -> latent_dim*2 (mu and logvar)
        hidden_dims = list(hidden_dims)
        self.encoder = _build_mlp(encoder_in_dim, hidden_dims, latent_dim * 2, dropout_p=dropout)

        # Decoder: latent_dim -> reversed(hidden_dims) -> decoder_out_dim
        decoder_hidden = list(reversed(hidden_dims))
        self.decoder = _build_mlp(latent_dim, decoder_hidden, decoder_out_dim, dropout_p=dropout)

        print(f"CrossModalVAE: latent_dim={latent_dim}, hidden_dims={hidden_dims}, d_target={d_target}")

    def forward(self, x):
        x = x.to(self.device).to(torch.float32)
        x = x - self.source_mean_

        if self.use_pca_encoder and self.source_loadings_enc is not None:
            x = torch.matmul(x, self.source_loadings_enc)

        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        logvar = torch.clamp(logvar, -20.0, 2.0)

        if self.training:
            eps = torch.randn_like(mu, device=mu.device, dtype=mu.dtype)
            std = torch.exp(0.5 * logvar)
            z = mu + eps * std
        else:
            z = mu # deterministic as compared to VAE standard stochastic sampling from prior

        y_latent = self.decoder(z)
        if self.use_pca_decoder and self.target_loadings_dec is not None:
            y_pred = torch.matmul(y_latent, self.target_loadings_dec.t()) + self.target_mean_
        else:
            y_pred = y_latent + self.target_mean_
        
        return (y_pred, mu, logvar)

    def get_reg_loss(self):
        """Compute L1/L2 regularization on learnable parameters using global helper."""
        return compute_reg_loss(self.parameters(), self.l1_l2_tuple)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
