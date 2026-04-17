import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator, svds

from models.architectures.utils import _build_mlp, compute_reg_loss, get_model_input, get_modality_data


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
        if len(getattr(base, "source_modalities", [base.source])) != 1:
            raise ValueError("CrossModalPCA only supports a single source modality.")

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
        if len(getattr(base, "source_modalities", [base.source])) != 1:
            raise ValueError("CrossModal_PLS_SVD only supports a single source modality.")
        
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
    def __init__(self, base, n_components_pca=32, n_components_pls=4, n_components_pca_target=None, device=None):
        super().__init__()
        self.base = base
        self.n_components_pca = n_components_pca
        self.n_components_pls = n_components_pls
        self.n_components_pca_target = int(n_components_pca if n_components_pca_target is None else n_components_pca_target)
        self.source_modalities = list(getattr(base, "source_modalities", [base.source]))
        self.target_modality = getattr(base, "target", None)
        if self.target_modality is None:
            self.target_modality = getattr(base, "target_modalities", [base.target])[0]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        data = get_modality_data(base, device=self.device, include_scores=True)
        target_data = data["target"]
        target_mean = target_data["mean"]
        target_loadings = target_data["loadings"]
        target_scores = target_data["scores"]

        if isinstance(n_components_pca, dict):
            self.n_components_pca_by_modality = {
                modality: int(n_components_pca[modality]) for modality in self.source_modalities
            }
        else:
            shared_dim = int(n_components_pca)
            self.n_components_pca_by_modality = {
                modality: shared_dim for modality in self.source_modalities
            }

        self.source_means = {}
        self.source_loadings_k = {}
        self.source_dims = {}
        source_pls_parts = []
        for modality in self.source_modalities:
            modality_data = data["sources"][modality]
            source_mean = torch.tensor(modality_data["mean"], dtype=torch.float32, device=self.device)
            source_loadings = torch.tensor(modality_data["loadings"], dtype=torch.float32, device=self.device)
            source_scores = torch.tensor(modality_data["scores"], dtype=torch.float32, device=self.device)
            k_mod = self.n_components_pca_by_modality[modality]
            self.source_means[modality] = source_mean
            self.source_loadings_k[modality] = source_loadings[:, :k_mod].to(torch.float32)
            self.source_dims[modality] = k_mod
            source_pls_parts.append(source_scores[:, :k_mod].cpu().numpy())

        self.target_mean = torch.tensor(target_mean, dtype=torch.float32, device=self.device)
        self.target_loadings = torch.tensor(target_loadings, dtype=torch.float32, device=self.device)
        self.target_scores = torch.tensor(target_scores, dtype=torch.float32, device=self.device)

        X_pls = np.concatenate(source_pls_parts, axis=1)
        Y_pls = self.target_scores[:, :self.n_components_pca_target].cpu().numpy()

        self.pls_pca_fusion_model = PLSRegression(n_components=self.n_components_pls)
        self.pls_pca_fusion_model.fit(X_pls, Y_pls)
        print("PLS model fit score: ", self.pls_pca_fusion_model.score(X_pls, Y_pls))
        print(f"Source modalities: {self.source_modalities}")
        print(f"Source PCA dims: {self.n_components_pca_by_modality}")
        print(f"Concatenated source score shape: {X_pls.shape}")
        print("PLS source scores shape: ", self.pls_pca_fusion_model.x_scores_.shape)
        print("PLS target scores shape: ", self.pls_pca_fusion_model.y_scores_.shape)
        print(f"PCA target scores shape: {self.target_scores.shape}")

        for modality in self.source_modalities:
            self.register_buffer(f"source_mean_{modality}", self.source_means[modality].to(torch.float32))
            self.register_buffer(f"source_loadings_k_{modality}", self.source_loadings_k[modality].to(torch.float32))
        self.register_buffer('target_loadings_k', self.target_loadings[:, :self.n_components_pca_target].to(torch.float32))
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
        if isinstance(x, dict):
            source_inputs = x
        elif len(self.source_modalities) == 1:
            source_inputs = {self.source_modalities[0]: x}
        else:
            raise TypeError("Expected dict input for multi-source CrossModal_PCA_PLS.")

        z_parts = []
        for modality in self.source_modalities:
            x_mod = source_inputs[modality].to(self.target_mean_.device).to(torch.float32)
            source_mean = getattr(self, f"source_mean_{modality}")
            source_loadings_k = getattr(self, f"source_loadings_k_{modality}")
            z_parts.append(torch.matmul(x_mod - source_mean, source_loadings_k))
        z = torch.cat(z_parts, dim=1)

        # 3. Run through learned PLS model in PCA space to get y_hat_pca
        #    - Switch to cpu and numpy for sklearn
        z_np = z.detach().cpu().numpy()
        y_hat_pca_np = self.pls_pca_fusion_model.predict(z_np)

        # 4. Convert PLS output back to torch and move to device
        y_hat_pca = torch.tensor(y_hat_pca_np, dtype=torch.float32, device=self.target_mean_.device)

        # 5. Reconstruct with target PCA (inverse transform): (batch, k) @ (k, d) => (batch, d)
        y_hat = torch.matmul(y_hat_pca, self.target_loadings_k.t()) + self.target_mean_
        return y_hat

    def to(self, *args, **kwargs): # for device mismatch issues
        super().to(*args, **kwargs)
        self.target_mean_ = self.target_mean_.to(*args, **kwargs)
        for modality in self.source_modalities:
            setattr(self, f"source_mean_{modality}", getattr(self, f"source_mean_{modality}").to(*args, **kwargs))
            setattr(self, f"source_loadings_k_{modality}", getattr(self, f"source_loadings_k_{modality}").to(*args, **kwargs))
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
        learn_encoder: Make W_enc learnable. Default True.
        learn_mid: Make W_mid learnable. Default True.
        learn_decoder: Make W_dec learnable. Default True.
        random_init: If True, use random initialization instead of PCA/PLS. Default False.
        dropout: Dropout probability (0 = no dropout). Default 0.0.
        l1_l2_tuple: Tuple of (l1_reg, l2_reg) weights. Default (0.0, 0.001).
        **kwargs: Ignored (lr, epochs, loss_fn, loss_alpha passed to Lightning module).
    """
    def __init__(self, base, n_components_pca_source=256, n_components_pca_target=256, 
                 n_components_pls=64, device=None,
                 learn_encoder=False, learn_mid=True, learn_decoder=False,
                 random_init=False, dropout=0.3, l1_l2_tuple=(0.0, 0.001), **kwargs):
        super().__init__()
        # If all learn_* flags are False, nothing will be trainable and Lightning/Adam
        # will error out when calling backward. Instead of silently failing, flip all
        # three to True with a warning so this configuration is still usable.
        if not (learn_encoder or learn_mid or learn_decoder):
            print(
                "CrossModal_PCA_PLS_learnable: learn_encoder=learn_mid=learn_decoder=False; "
                "enabling all three to make the model trainable.",
                flush=True,
            )
            learn_encoder = True
            learn_mid = True
            learn_decoder = True

        self.source_modalities = list(getattr(base, "source_modalities", [base.source]))
        self.target_modality = getattr(base, "target", None)
        if self.target_modality is None:
            self.target_modality = getattr(base, "target_modalities", [base.target])[0]
        self.n_components_pca_source = n_components_pca_source
        self.n_components_pca_target = n_components_pca_target
        self.random_init = random_init
        self.dropout_p = dropout
        
        self.l1_reg, self.l2_reg = l1_l2_tuple
        self.l1_l2_tuple = l1_l2_tuple
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        data = get_modality_data(base, device=device, include_scores=True)
        target_data = data["target"]
        target_mean = target_data["mean"]
        target_loadings = target_data["loadings"]
        target_scores = target_data["scores"]
        d_target = target_loadings.shape[0]
        k_tgt = n_components_pca_target

        if isinstance(n_components_pca_source, dict):
            self.n_components_pca_source_by_modality = {
                modality: int(n_components_pca_source[modality]) for modality in self.source_modalities
            }
        else:
            shared_dim = int(n_components_pca_source)
            self.n_components_pca_source_by_modality = {
                modality: shared_dim for modality in self.source_modalities
            }

        self.source_means = nn.ParameterDict()
        self.source_encoders = nn.ParameterDict()
        source_pls_parts = []
        fused_source_dim = 0

        # Helper to init weight: random if learnable AND random_init, else from PCA/PLS
        def init_weight(shape, pca_pls_data, learnable):
            if learnable and random_init:
                w = torch.empty(shape, device=device)
                nn.init.kaiming_uniform_(w, a=np.sqrt(5))
                return nn.Parameter(w, requires_grad=True)
            else:
                return nn.Parameter(torch.tensor(pca_pls_data, dtype=torch.float32, device=device), requires_grad=learnable)

        # Means (always from data, kept fixed to avoid overfitting)
        self.target_mean = nn.Parameter(
            torch.tensor(target_mean, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "target_latent_encoder",
            torch.tensor(target_loadings[:, :k_tgt], dtype=torch.float32, device=device),
        )
        latent_variance = np.var(target_scores[:, :k_tgt], axis=0, dtype=np.float32)
        latent_variance = np.maximum(latent_variance, 1.0e-8)
        latent_weights = latent_variance / float(np.mean(latent_variance))
        self.register_buffer(
            "latent_loss_weights",
            torch.tensor(latent_weights, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "target_latent_encoder",
            torch.tensor(target_loadings[:, :k_tgt], dtype=torch.float32, device=device),
        )
        latent_variance = np.var(target_scores[:, :k_tgt], axis=0, dtype=np.float32)
        latent_variance = np.maximum(latent_variance, 1.0e-8)
        latent_weights = latent_variance / float(np.mean(latent_variance))
        self.register_buffer(
            "latent_loss_weights",
            torch.tensor(latent_weights, dtype=torch.float32, device=device),
        )
        for modality in self.source_modalities:
            modality_data = data["sources"][modality]
            source_mean = modality_data["mean"]
            source_loadings = modality_data["loadings"]
            source_scores = modality_data["scores"]
            d_source_mod = source_loadings.shape[0]
            k_src_mod = self.n_components_pca_source_by_modality[modality]
            fused_source_dim += k_src_mod

            self.source_means[modality] = nn.Parameter(
                torch.tensor(source_mean, dtype=torch.float32, device=device),
                requires_grad=False,
            )
            self.source_encoders[modality] = init_weight(
                (d_source_mod, k_src_mod),
                source_loadings[:, :k_src_mod],
                learn_encoder,
            )
            source_pls_parts.append(source_scores[:, :k_src_mod])

        # Fit PLS (needed for non-random mid layer initialization).
        # X_pls: concatenated source PCA scores with dimension fused_source_dim.
        # Y_pls: target PCA scores with dimension k_tgt.
        X_pls = np.concatenate(source_pls_parts, axis=1)
        Y_pls = target_scores[:, :k_tgt]

        # Validate that requested n_components_pls is mathematically valid for PLS.
        if n_components_pls > min(fused_source_dim, k_tgt):
            raise ValueError(
                f"CrossModal_PCA_PLS_learnable: n_components_pls={n_components_pls} "
                f"is larger than min(fused_source_dim={fused_source_dim}, "
                f"n_components_pca_target={k_tgt}). Reduce n_components_pls or increase "
                f"the PCA dimensions."
            )

        pls = PLSRegression(n_components=n_components_pls)
        pls.fit(X_pls, Y_pls)

        # sklearn PLSRegression.coef_ has shape (n_targets, n_features) and is used as
        #   y ≈ X @ coef_.T + intercept
        # with X shape (n_samples, n_features=fused_source_dim) and
        # y shape (n_samples, n_targets=k_tgt).
        # We want a mid-layer weight that maps fused source latent (fused_source_dim)
        # to target latent (k_tgt) via z @ W_mid, so W_mid must be (fused_source_dim, k_tgt).
        coef = pls.coef_
        if coef.shape != (k_tgt, fused_source_dim):
            raise ValueError(
                "CrossModal_PCA_PLS_learnable: Unexpected PLS coef_ shape "
                f"{coef.shape}; expected ({k_tgt}, {fused_source_dim}) per "
                "sklearn.PLSRegression documentation."
            )
        W_mid_data = coef.T  # (fused_source_dim, k_tgt)

        self.W_mid = init_weight((fused_source_dim, k_tgt), W_mid_data, learn_mid)
        self.W_dec = init_weight((k_tgt, d_target), target_loadings[:, :k_tgt].T, learn_decoder)
        
        # Log initialization
        init_info = lambda name, p: f"{name}:{'rand' if (p.requires_grad and random_init) else 'pca/pls'}{'(frozen)' if not p.requires_grad else ''}"
        enc_shapes = {modality: tuple(self.source_encoders[modality].shape) for modality in self.source_modalities}
        print(f"Init: encoders={enc_shapes}, {init_info('mid', self.W_mid)}, {init_info('dec', self.W_dec)}")
        print(f"Source PCA dims: {self.n_components_pca_source_by_modality}, mid={tuple(self.W_mid.shape)}, dec={tuple(self.W_dec.shape)}")

        # Dropout layers
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def _resolve_source_inputs(self, x):
        if isinstance(x, dict):
            return x
        if len(self.source_modalities) == 1:
            return {self.source_modalities[0]: x}
        raise TypeError("Expected dict input for multi-source CrossModal_PCA_PLS_learnable.")

    def encode_target_latents(self, y):
        y = y.to(self.target_mean.device).to(torch.float32)
        return torch.matmul(y - self.target_mean, self.target_latent_encoder)

    def predict_target_latents(self, x):
        device = self.target_mean.device
        source_inputs = self._resolve_source_inputs(x)

        z_parts = []
        for modality in self.source_modalities:
            x_mod = source_inputs[modality].to(device).to(torch.float32)
            z_parts.append(torch.matmul(x_mod - self.source_means[modality], self.source_encoders[modality]))
        z = torch.cat(z_parts, dim=1)
        z = self.dropout(z)
        z = torch.matmul(z, self.W_mid)
        z = self.dropout(z)
        return z

    def forward(self, x):
        z = self.predict_target_latents(x)
        return torch.matmul(z, self.W_dec) + self.target_mean

    def get_reg_loss(self):
        """Compute L1/L2 regularization on learnable weights using global helper."""
        return compute_reg_loss(list(self.source_encoders.values()) + [self.W_mid, self.W_dec], self.l1_l2_tuple)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CrossModal_PCA_PLS_CovProjector(nn.Module):
    """
    Learnable source-to-target PCA/PLS backbone with an additive covariate residual branch.

    Architecture:
        z_src = concat_i((x_i - mu_i) @ W_enc_i)
        z_base = dropout(z_src) @ W_mid.T
        z_cov = fu sion(concat_j(encoder_j(cov_j)))
        z_target = z_base + alpha * z_cov
        y_hat = z_target @ W_dec + mu_target
    """
    def __init__(
        self,
        base,
        n_components_pca_source=256,
        n_components_pca_target=256,
        n_components_pls=64,
        device=None,
        learn_encoder=False,
        learn_mid=True,
        learn_decoder=False,
        random_init=False,
        dropout=0.3,
        l1_l2_tuple=(0.0, 0.001),
        cov_sources=("fs_all",),
        cov_projectors=None,
        cov_fusion=None,
        use_target_scores_in_projector=False,
        target_scores_projector=None,
        **kwargs,
    ):
        super().__init__()
        
        self.uses_cov = True
        self.use_target_scores_in_projector = bool(use_target_scores_in_projector)
        self.target_scores_projector = target_scores_projector or {"type": "linear"}
        
        self.source_modalities = list(getattr(base, "source_modalities",[base.source]))
        self.target_modality = getattr(base, "target", None)
        if self.target_modality is None:
            self.target_modality = getattr(base, "target_modalities", [base.target])[0]
        self.n_components_pca_source = n_components_pca_source
        self.n_components_pca_target = n_components_pca_target
        self.random_init = random_init
        self.dropout_p = dropout
        
        self.cov_sources = list(cov_sources) if cov_sources is not None else ["fs_all"]
        self.cov_projectors = cov_projectors or {}
        self.cov_fusion = cov_fusion or {"type": "linear"}
        self.l1_reg, self.l2_reg = l1_l2_tuple
        self.l1_l2_tuple = l1_l2_tuple

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        data = get_modality_data(base, device=device, include_scores=True)
        target_data = data["target"]
        target_mean = target_data["mean"]
        target_loadings = target_data["loadings"]
        target_scores = target_data["scores"]
        d_target = target_loadings.shape[0]
        k_tgt = int(n_components_pca_target)
        cov_dims = getattr(base, "cov_dims", {})

        if isinstance(n_components_pca_source, dict):
            self.n_components_pca_source_by_modality = {
                modality: int(n_components_pca_source[modality]) for modality in self.source_modalities
            }
        else:
            shared_dim = int(n_components_pca_source)
            self.n_components_pca_source_by_modality = {
                modality: shared_dim for modality in self.source_modalities
            }

        self.source_means = nn.ParameterDict()
        self.source_encoders = nn.ParameterDict()
        source_pls_parts = []
        fused_source_dim = 0

        def init_weight(shape, pca_pls_data, learnable):
            if learnable and random_init:
                w = torch.empty(shape, device=device)
                nn.init.kaiming_uniform_(w, a=np.sqrt(5))
                return nn.Parameter(w, requires_grad=True)
            return nn.Parameter(torch.tensor(pca_pls_data, dtype=torch.float32, device=device), requires_grad=learnable)

        # Means are fixed PCA means (not learned) to avoid overfitting.
        self.target_mean = nn.Parameter(
            torch.tensor(target_mean, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        for modality in self.source_modalities:
            modality_data = data["sources"][modality]
            source_mean = modality_data["mean"]
            source_loadings = modality_data["loadings"]
            source_scores = modality_data["scores"]
            d_source_mod = source_loadings.shape[0]
            k_src_mod = self.n_components_pca_source_by_modality[modality]
            fused_source_dim += k_src_mod

            self.source_means[modality] = nn.Parameter(
                torch.tensor(source_mean, dtype=torch.float32, device=device),
                requires_grad=False,
            )
            self.source_encoders[modality] = init_weight(
                (d_source_mod, k_src_mod),
                source_loadings[:, :k_src_mod],
                learn_encoder,
            )
            source_pls_parts.append(source_scores[:, :k_src_mod])

        pls = PLSRegression(n_components=n_components_pls)
        pls.fit(np.concatenate(source_pls_parts, axis=1), target_scores[:, :k_tgt])

        self.W_mid = init_weight((fused_source_dim, k_tgt), pls.coef_, learn_mid)
        self.W_dec = init_weight((k_tgt, d_target), target_loadings[:, :k_tgt].T, learn_decoder)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Covariate projectors (per-source encoders)
        self.cov_encoders = nn.ModuleDict()
        self.cov_embed_dims = {}
        for source in self.cov_sources:
            spec = self.cov_projectors.get(source, {}) or {}
            proj_type = spec.get("type", "linear")
            out_dim = int(spec.get("out_dim", 32))
            self.cov_embed_dims[source] = out_dim
            if proj_type == "embedding":
                vocab_size = int(cov_dims.get(f"{source}_vocab_size", 0))
                if vocab_size <= 0:
                    raise ValueError(f"Embedding projector requested for {source} but vocab size is unavailable.")
                self.cov_encoders[source] = nn.Embedding(vocab_size, out_dim)
            elif proj_type == "mlp":
                in_dim = int(cov_dims.get(source, 0))
                if in_dim <= 0:
                    raise ValueError(f"Covariate source '{source}' is unavailable in base.")
                hidden = spec.get("hidden_dims", [out_dim])
                self.cov_encoders[source] = _build_mlp(
                    in_dim, hidden, out_dim,
                    dropout_p=float(spec.get("dropout", 0.0)),
                    use_layer_norm=bool(spec.get("layer_norm", False)),
                )
            else:
                in_dim = int(cov_dims.get(source, 0))
                if in_dim <= 0:
                    raise ValueError(f"Covariate source '{source}' is unavailable in base.")
                self.cov_encoders[source] = nn.Linear(in_dim, out_dim)

        fused_cov_dim = sum(self.cov_embed_dims.values())
        # Cov fusion output must match target latent dim (z_cov is added to z_base).
        fusion_type = (self.cov_fusion or {}).get("type", "linear")
        if fusion_type == "mlp":
            hidden = (self.cov_fusion or {}).get("hidden_dims", [k_tgt])
            dropout_p = float((self.cov_fusion or {}).get("dropout", 0.0))
            use_ln = bool((self.cov_fusion or {}).get("layer_norm", False))
            self.cov_fusion_net = _build_mlp(fused_cov_dim, hidden, k_tgt, dropout_p=dropout_p, use_layer_norm=use_ln)
        else:
            self.cov_fusion_net = nn.Linear(fused_cov_dim, k_tgt)

        # Sanity-check option: feed true target PCA scores (FC latents)
        # into a projector and add to z_cov (target leakage).
        self.target_scores_proj = None
        if self.use_target_scores_in_projector:
            spec = self.target_scores_projector or {}
            proj_type = spec.get("type", "linear")
            hidden = spec.get("hidden_dims", [k_tgt])
            dropout_p = float(spec.get("dropout", 0.0))

            if proj_type == "mlp":
                use_ln = bool(spec.get("layer_norm", False))
                self.target_scores_proj = _build_mlp(k_tgt, hidden, k_tgt, dropout_p=dropout_p, use_layer_norm=use_ln)
            else:
                self.target_scores_proj = nn.Linear(k_tgt, k_tgt)

        print(
            f"CovProjector init: source_dims={self.n_components_pca_source_by_modality}, "
            f"target_dim={k_tgt}, cov_sources={self.cov_sources}"
            + (", use_target_scores_in_projector=True (sanity-check)" if self.use_target_scores_in_projector else "")
        )

    def _resolve_inputs(self, x, cov=None):
        if isinstance(x, dict) and "y" in x:
            batch = x
            cov = batch.get("cov") if cov is None else cov
            x = get_model_input(batch)

        if isinstance(x, dict):
            source_inputs = x
        elif len(self.source_modalities) == 1:
            source_inputs = {self.source_modalities[0]: x}
        else:
            raise TypeError("Expected dict input for multi-source CrossModal_PCA_PLS_CovProjector.")

        if cov is None:
            raise ValueError("Covariate projector model requires cov input.")

        return source_inputs, cov

    def _encode_sources(self, x):
        device = self.target_mean.device
        z_parts = []
        for modality in self.source_modalities:
            x_mod = x[modality].to(device).to(torch.float32)
            z_parts.append(torch.matmul(x_mod - self.source_means[modality], self.source_encoders[modality]))
        return torch.cat(z_parts, dim=1)

    def _encode_covariates(self, cov):
        device = self.target_mean.device
        z_parts = []
        for source in self.cov_sources:
            if source not in cov:
                raise ValueError(f"Missing covariate source '{source}' in batch.")
            c = cov[source]
            encoder = self.cov_encoders[source]
            if isinstance(encoder, nn.Embedding):
                c = c.long().to(device)
                if c.ndim > 1:
                    c = c.squeeze(-1)
                z = encoder(c)
            else:
                z = encoder(c.to(device).to(torch.float32))
            z_parts.append(z)
        return torch.cat(z_parts, dim=1)

    def _predict_target_latent_from_sources(self, z_src):
        z_base = self.dropout(z_src)
        z_base = torch.matmul(z_base, self.W_mid.T)
        return self.dropout(z_base)

    def _decode_target_latent(self, z_target):
        return torch.matmul(z_target, self.W_dec) + self.target_mean

    def encode_target_latents(self, y):
        y = y.to(self.target_mean.device).to(torch.float32)
        return torch.matmul(y - self.target_mean, self.target_latent_encoder)

    def predict_target_latents(self, x, cov=None, y=None):
        source_inputs, cov = self._resolve_inputs(x, cov=cov)
        z_src = self._encode_sources(source_inputs)
        z_base = self._predict_target_latent_from_sources(z_src)
        z_cov = self.cov_fusion_net(self._encode_covariates(cov))
        if self.use_target_scores_in_projector and self.target_scores_proj is not None and y is not None:
            target_scores = self.encode_target_latents(y)
            z_cov = self.target_scores_proj(target_scores)
        return z_base + z_cov

    def predict_latents(self, batch):
        source_inputs, cov = self._resolve_inputs(batch)
        z_src = self._encode_sources(source_inputs)
        z_base = self._predict_target_latent_from_sources(z_src)
        z_cov = self.cov_fusion_net(self._encode_covariates(cov))
        z_target = z_base + z_cov
        return {
            "z_base": z_base,
            "z_cov": z_cov,
            "z_target": z_target,
        }

    def forward(self, x, cov=None, y=None):
        z_target = self.predict_target_latents(x, cov=cov, y=y)
        return self._decode_target_latent(z_target)

    def get_reg_loss(self):
        params = list(self.source_encoders.values()) + [self.W_mid, self.W_dec]
        params.extend(self.cov_fusion_net.parameters())
        if self.target_scores_proj is not None:
            params.extend(self.target_scores_proj.parameters())
        for encoder in self.cov_encoders.values():
            if isinstance(encoder, nn.Embedding):
                params.append(encoder.weight)
            else:
                params.extend(p for p in encoder.parameters())
        return compute_reg_loss(params, self.l1_l2_tuple)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
