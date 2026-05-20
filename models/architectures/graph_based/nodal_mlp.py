import torch
import torch.nn as nn

from data.graph_adapter import (
    infer_num_nodes_from_upper_triangle_dim,
    get_label_edge_index,
)
from models.architectures.utils import compute_reg_loss


class NodalMLP(nn.Module):
    """
    Graph-free SC->FC edge regression.

    Each node i is described by anatomical features (volume / centroid / r2t)
    and/or the length-N SC row from the raw SC matrix (diagonal preserved in
    its (i, i) position). A shared MLP encoder produces node embeddings h_i.
    For each target FC edge (i, j), the prediction comes from a symmetric
    function of (h_i, h_j), giving bidirectional training by construction.
    """

    SC_ROW_NORMS = {"none", "col_zscore", "row_zscore", "row_l2", "brain_scale"}
    DECODER_SYMMETRIES = {"symmetric", "asymmetric", "concat"}
    DECODER_TYPES = {"mlp", "dot", "bilinear", "linear_beta", "diag_bilinear"}
    ENCODER_TYPES = {"none", "pca", "linear", "mlp", "spectral"}
    # encoder_type='spectral' computes per-subject normalized-Laplacian eigenvectors of the SC
    # graph (a.k.a. "connectome harmonics"). The principled spectral story for SC->FC is the
    # Robinson eigenmode / Atasoy 2016 / Abdelnour 2014 family: FC ≈ low-pass filter on L_sym(SC).
    # Compatible with decoder_type ∈ {dot, diag_bilinear} which are basis-invariant under the
    # per-subject orthogonal eigenframe; population-shared full-bilinear / linear / mlp heads
    # learn axis-specific weights and silently break the basis-invariance — see Robinson et al.,
    # "Eigenmodes of brain activity" (NeuroImage, 2016) for the underlying framing.

    def __init__(
        self,
        base,
        encoder_type: str = "none",
        embedding_dim: int = 64,
        decoder_dims=(256, 128),
        dropout: float = 0.2,
        reg: float = 1e-4,
        decoder_symmetry: str = "symmetric",
        decoder_type: str = "mlp",
        sc_row_norm: str = "brain_scale",
        use_volume: bool = True,
        use_spatial: bool = True,
        use_r2t: bool = True,
        use_sc_row: bool = True,
        device=None,
        **kwargs,
    ):
        super().__init__()

        source_modalities = list(getattr(base, "source_modalities", [base.source]))
        if len(source_modalities) != 1:
            raise ValueError("NodalMLP currently supports exactly one source modality.")
        self.source_modality = source_modalities[0]

        if decoder_symmetry not in self.DECODER_SYMMETRIES:
            raise ValueError(
                f"decoder_symmetry must be one of {sorted(self.DECODER_SYMMETRIES)}, got {decoder_symmetry!r}."
            )
        if decoder_type not in self.DECODER_TYPES:
            raise ValueError(
                f"decoder_type must be one of {sorted(self.DECODER_TYPES)}, got {decoder_type!r}."
            )
        if encoder_type not in self.ENCODER_TYPES:
            raise ValueError(
                f"encoder_type must be one of {sorted(self.ENCODER_TYPES)}, got {encoder_type!r}."
            )
        if encoder_type == "pca" and not use_sc_row:
            raise ValueError("encoder_type='pca' requires use_sc_row=True (PCA acts on the SC-row block).")
        if encoder_type == "spectral" and not use_sc_row:
            raise ValueError("encoder_type='spectral' requires use_sc_row=True (eigendecomposes the SC matrix).")
        if sc_row_norm not in self.SC_ROW_NORMS:
            raise ValueError(
                f"sc_row_norm must be one of {sorted(self.SC_ROW_NORMS)}, got {sc_row_norm!r}."
            )
        if not (use_volume or use_spatial or use_r2t or use_sc_row):
            raise ValueError("At least one feature group must be enabled.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.encoder_type = encoder_type
        # embedding_dim is only consumed by encoder_type in {pca, linear, mlp}; for "none" we
        # store None so logging/search-space introspection sees an unused field rather than a
        # stale default. Mirrors the decoder_dims pattern.
        self.embedding_dim = int(embedding_dim) if encoder_type != "none" else None
        if encoder_type != "none":
            if self.embedding_dim is None or self.embedding_dim <= 0:
                raise ValueError(
                    f"embedding_dim must be a positive integer when encoder_type={encoder_type!r}."
                )
        self.decoder_type = decoder_type
        # decoder_dims only shapes the "mlp" head; probe heads ignore it (stored as () so logging stays clean).
        self.decoder_dims = tuple(int(d) for d in decoder_dims) if decoder_type == "mlp" else ()
        if decoder_type == "mlp" and len(self.decoder_dims) == 0:
            raise ValueError("decoder_dims must be a non-empty list of layer widths when decoder_type='mlp'.")
        self.dropout = float(dropout)
        self.reg = float(reg)
        self.decoder_symmetry = decoder_symmetry
        self.sc_row_norm = sc_row_norm

        self.use_volume = bool(use_volume)
        self.use_spatial = bool(use_spatial)
        self.use_r2t = bool(use_r2t)
        self.use_sc_row = bool(use_sc_row)

        # Contract markers consumed by the training/eval wrappers.
        self.uses_node_features = self.use_volume or self.use_spatial or self.use_r2t
        self.uses_sc_matrix = self.use_sc_row
        self.uses_cov = False

        source_ut_dim = int(base.sc_upper_triangles.shape[1])
        self.num_nodes = infer_num_nodes_from_upper_triangle_dim(source_ut_dim)
        self.num_edges_upper = source_ut_dim

        label_edge_index = get_label_edge_index(self.num_nodes, device=device)
        self.register_buffer("label_edge_index", label_edge_index)

        train_indices = base.trainvaltest_partition_indices["train"]

        # Static anatomical block: pick the subset of parcel_node_features columns.
        # Expected layout: [volume, centroid_x, centroid_y, centroid_z, r2t_0, ...].
        full_node_dim = int(base.parcel_node_features.shape[-1])
        r2t_dim = max(0, full_node_dim - 4)
        static_idx = []
        if self.use_volume:
            static_idx.append(0)
        if self.use_spatial:
            static_idx.extend([1, 2, 3])
        if self.use_r2t and r2t_dim > 0:
            static_idx.extend(range(4, full_node_dim))
        self.static_dim = len(static_idx)
        if self.static_dim > 0:
            self.register_buffer(
                "static_feature_idx", torch.tensor(static_idx, dtype=torch.long)
            )
            train_static = base.parcel_node_features[train_indices][:, :, static_idx]
            static_mean = torch.as_tensor(
                train_static.mean(axis=(0, 1)), dtype=torch.float32
            ).view(1, 1, -1)
            static_std = torch.as_tensor(
                train_static.std(axis=(0, 1)), dtype=torch.float32
            ).view(1, 1, -1)
            static_std = torch.where(static_std == 0, torch.ones_like(static_std), static_std)
            self.register_buffer("static_mean", static_mean)
            self.register_buffer("static_std", static_std)

        # Dynamic SC-row block: length-N vector per node from the raw [N, N] SC matrix.
        # Self-loops stay in their (i, i) cell; nothing is concatenated onto the row.
        # When encoder_type='pca', rows are projected to embedding_dim via a frozen PCA basis
        # fitted on stacked training connectomes; otherwise the raw N-dim row is passed through
        # to the (linear/mlp) encoder or directly to the decoder for encoder_type='none'.
        self._uses_pca_on_sc_row = (self.encoder_type == "pca") and self.use_sc_row
        self._uses_spectral_on_sc_row = (self.encoder_type == "spectral") and self.use_sc_row
        if self._uses_spectral_on_sc_row and self.embedding_dim > self.num_nodes - 1:
            # Drop the trivial constant eigenvector; need at least 1 non-trivial mode left.
            raise ValueError(
                f"encoder_type='spectral' requires embedding_dim <= num_nodes - 1 "
                f"(have num_nodes={self.num_nodes}, embedding_dim={self.embedding_dim})."
            )
        if self.use_sc_row:
            if self._uses_pca_on_sc_row or self._uses_spectral_on_sc_row:
                self.sc_row_dim = self.embedding_dim
            else:
                self.sc_row_dim = self.num_nodes
        else:
            self.sc_row_dim = 0
        if self.use_sc_row and self.sc_row_norm == "col_zscore":
            train_sc = torch.as_tensor(base.sc_matrices[train_indices], dtype=torch.float32)
            self.register_buffer("sc_col_mean", train_sc.mean(dim=0, keepdim=True))
            self.register_buffer(
                "sc_col_std", train_sc.std(dim=0, keepdim=True).clamp_min(1e-8)
            )

        if self._uses_pca_on_sc_row:
            train_sc = torch.as_tensor(base.sc_matrices[train_indices], dtype=torch.float32)
            # Apply the same sc_row_norm that will be used at inference before projecting.
            m = self._apply_sc_norm_to_tensor(train_sc)  # [B_train, N, N]
            stacked = m.reshape(-1, self.num_nodes)       # [B_train*N, N]
            pca_mean = stacked.mean(dim=0)                # [N]
            stacked_centered = stacked - pca_mean.unsqueeze(0)
            # Randomised SVD: O(B*N * K) instead of full SVD — avoids materialising
            # the full (B*N x N) decomposition for large training sets.
            _, _, V = torch.pca_lowrank(stacked_centered, q=self.embedding_dim, niter=4)
            # V: [N, K] — columns are principal directions; store for frozen inference.
            self.register_buffer("pca_mean", pca_mean)       # [N]
            self.register_buffer("pca_projection", V)         # [N, K]

        if self._uses_spectral_on_sc_row:
            # Precompute per-subject normalized-Laplacian eigvecs ONCE for every subject in
            # the dataset (train+val+test) and look them up by SC-fingerprint at forward time.
            # Eliminates the per-batch eigh, which is the dominant forward cost.
            full_sc = torch.as_tensor(base.sc_matrices, dtype=torch.float32, device=device)
            full_sc_norm = self._apply_sc_norm_to_tensor(full_sc)              # [N_total, N, N]
            full_eigvecs = self._laplacian_eigvecs(full_sc_norm, self.embedding_dim)
            # contiguous() because slicing-based views may keep the parent tensor alive.
            self.register_buffer("_spectral_eigvecs_full", full_eigvecs.contiguous())

            # Fingerprint coords: a small deterministic set of (i, j) positions whose values
            # in the *normalized* SC uniquely identify each subject in practice. F=16 entries
            # of float32 → collision probability ≈ 0 for any two distinct SC matrices.
            f_size = min(16, self.num_nodes)
            g = torch.Generator().manual_seed(0)
            fp_rows = torch.randint(0, self.num_nodes, (f_size,), generator=g).to(device).long()
            fp_cols = torch.randint(0, self.num_nodes, (f_size,), generator=g).to(device).long()
            self.register_buffer("_spectral_fp_rows", fp_rows)
            self.register_buffer("_spectral_fp_cols", fp_cols)
            fps = full_sc_norm[:, fp_rows, fp_cols].contiguous()                # [N_total, F]
            self.register_buffer("_spectral_fingerprints_full", fps)

        # feature_dim is the per-node dim *after* any sc-row PCA reduction (when encoder_type='pca')
        # but *before* a learnable linear/mlp encoder. encoder_type 'linear'/'mlp' map this down to
        # embedding_dim; 'none' and 'pca' leave it untouched.
        self.feature_dim = self.static_dim + self.sc_row_dim

        if self.encoder_type == "linear":
            self.encoder = nn.Linear(self.feature_dim, self.embedding_dim)
            self.node_embed_dim = self.embedding_dim
        elif self.encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(self.feature_dim, self.embedding_dim),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.PReLU(),
            )
            self.node_embed_dim = self.embedding_dim
        else:  # "none" or "pca" — both pass through (PCA is applied earlier at sc_row_block time)
            self.encoder = nn.Identity()
            self.node_embed_dim = self.feature_dim

        decoder_mult = {"symmetric": 3, "asymmetric": 4, "concat": 2}[self.decoder_symmetry]
        D = self.node_embed_dim
        if self.decoder_type == "mlp":
            self.edge_head = self._build_decoder_mlp(
                in_dim=decoder_mult * D,
                hidden_dims=self.decoder_dims,
                dropout=self.dropout,
            )
        elif self.decoder_type == "dot":
            # Inherently symmetric in (h_i, h_j); decoder_symmetry is ignored.
            self.edge_scale = nn.Parameter(torch.ones(1))
            self.edge_bias = nn.Parameter(torch.zeros(1))
        elif self.decoder_type == "bilinear":
            # h_i^T W h_j with W symmetrized at apply-time.
            self.edge_W = nn.Parameter(torch.empty(D, D))
            nn.init.xavier_uniform_(self.edge_W)
            self.edge_bias = nn.Parameter(torch.zeros(1))
        elif self.decoder_type == "diag_bilinear":
            # FC[i,j] ≈ sum_k w_k * h_i,k * h_j,k = h_i^T diag(w) h_j.
            # Sign-invariant under joint flips of (h_i,k, h_j,k); strict generalization of dot
            # (recovered at w=1). Mechanistically a learnable per-mode gain on the SC spectrum,
            # i.e. the data-driven analogue of the Robinson 2016 / Atasoy 2016 eigenmode model
            # of FC as a spectral filter on L_sym(SC). Pairs cleanly with encoder_type='spectral'.
            self.edge_w = nn.Parameter(torch.ones(D))
            self.edge_bias = nn.Parameter(torch.zeros(1))
        else:  # "linear_beta"
            # Linear regression on the same symmetric edge featurization the MLP head consumes.
            self.edge_head = nn.Linear(decoder_mult * D, 1)

    @staticmethod
    def _build_decoder_mlp(in_dim, hidden_dims, dropout):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _apply_sc_norm_to_tensor(self, m: torch.Tensor) -> torch.Tensor:
        """Apply sc_row_norm to a [B, N, N] float32 tensor. Used both at init (PCA fitting) and inference."""
        if self.sc_row_norm == "none":
            return m
        if self.sc_row_norm == "col_zscore":
            return (m - self.sc_col_mean) / self.sc_col_std
        if self.sc_row_norm == "row_zscore":
            row_mean = m.mean(dim=-1, keepdim=True)
            row_std = m.std(dim=-1, keepdim=True).clamp_min(1e-8)
            return (m - row_mean) / row_std
        if self.sc_row_norm == "row_l2":
            return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # brain_scale
        return m / m.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def _resolve_input(self, x):
        if isinstance(x, dict):
            if self.source_modality not in x:
                raise ValueError(
                    f"Expected source modality '{self.source_modality}' in input dict keys {list(x.keys())}."
                )
            return x[self.source_modality]
        return x

    def _static_block(self, node_features, bsz, device):
        if self.static_dim == 0:
            return None
        if node_features is None:
            raise ValueError("NodalMLP requires batch['node_features'] when anatomical features are enabled.")
        if node_features.ndim == 2:
            node_features = node_features.unsqueeze(0)
        if node_features.shape[:2] != (bsz, self.num_nodes):
            raise ValueError(
                f"node_features shape mismatch: expected [{bsz}, {self.num_nodes}, F], "
                f"got {tuple(node_features.shape)}"
            )
        feats = node_features.to(device=device, dtype=torch.float32).index_select(
            dim=2, index=self.static_feature_idx.to(device)
        )
        return (feats - self.static_mean.to(device)) / (self.static_std.to(device) + 1e-8)

    def _sc_row_block(self, sc_matrix, bsz, device):
        if not self.use_sc_row:
            return None
        if sc_matrix is None:
            raise ValueError("NodalMLP requires batch['sc_matrix'] when use_sc_row=True.")
        if tuple(sc_matrix.shape) != (bsz, self.num_nodes, self.num_nodes):
            raise ValueError(
                f"sc_matrix shape mismatch: expected [{bsz}, {self.num_nodes}, {self.num_nodes}], "
                f"got {tuple(sc_matrix.shape)}"
            )
        m = sc_matrix.to(device=device, dtype=torch.float32)
        m = self._apply_sc_norm_to_tensor(m)  # [B, N, N]
        if self._uses_pca_on_sc_row:
            # Project each node's normalised SC row onto the frozen PCA basis.
            # [B, N, N] -> [B*N, N] -> [B*N, K] -> [B, N, K]
            m_flat = m.reshape(-1, self.num_nodes)
            m_proj = (m_flat - self.pca_mean) @ self.pca_projection  # [B*N, K]
            return m_proj.reshape(bsz, self.num_nodes, self.embedding_dim)
        if self._uses_spectral_on_sc_row:
            return self._spectral_eigvecs(m)
        return m

    @staticmethod
    def _laplacian_eigvecs(m_norm: torch.Tensor, K: int) -> torch.Tensor:
        """Top-K non-trivial eigvecs of L_sym = I - D^{-1/2} W D^{-1/2} for each subject.

        Used at __init__ time to precompute the per-subject buffer; not called during training.
        Input: already-normalized SC, shape [N_total, N, N]. Output: [N_total, N, K].

        Reference: Robinson et al., "Eigenmodes of brain activity" (NeuroImage, 2016);
        Atasoy et al., "Human brain networks function in connectome-specific harmonic waves"
        (Nature Communications, 2016).
        """
        # Symmetrise defensively (SC is nominally symmetric).
        m_sym = 0.5 * (m_norm + m_norm.transpose(-1, -2))
        # |W| keeps D positive even if sc_row_norm produced signed entries.
        deg = m_sym.abs().sum(dim=-1).clamp_min(1e-8)
        d_inv_sqrt = deg.pow(-0.5)
        norm_adj = m_sym * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
        n = m_sym.shape[-1]
        eye = torch.eye(n, device=m_sym.device, dtype=m_sym.dtype).expand_as(norm_adj)
        L_sym = eye - norm_adj
        # Symmetric eigendecomposition; columns sorted by eigval ascending. Drop the
        # trivial first eigenvector (eigval ≈ 0, constant on each connected component).
        _, eigvecs = torch.linalg.eigh(L_sym)
        return eigvecs[:, :, 1 : 1 + K]

    def _spectral_eigvecs(self, m: torch.Tensor) -> torch.Tensor:
        """Per-subject Laplacian eigvecs via fingerprint lookup against the precomputed buffer.

        `m` is the (already sc_row_norm'd) batch SC matrix [B, N, N]. We fingerprint the same
        deterministic (i, j) entries we registered at init, find the matching row in
        `_spectral_fingerprints_full` via L1 nearest-neighbor, and gather precomputed eigvecs.
        During training, applies independent per-(subject, mode) sign-flip augmentation.

        See Robinson 2016 / Atasoy 2016 references on `_laplacian_eigvecs` for the underlying
        connectome-harmonics formulation.
        """
        # Fingerprint the (already-normalized) batch SC at the registered coords.
        batch_fp = m[:, self._spectral_fp_rows, self._spectral_fp_cols]       # [B, F]
        # L1 distance to each precomputed fingerprint; argmin gives the matching row.
        diffs = (batch_fp.unsqueeze(1) - self._spectral_fingerprints_full.unsqueeze(0)).abs().sum(dim=-1)
        idx = diffs.argmin(dim=1)                                             # [B]
        emb = self._spectral_eigvecs_full[idx]                                # [B, N, K]
        if self.training:
            signs = torch.randint(0, 2, (emb.shape[0], emb.shape[2]), device=emb.device, dtype=emb.dtype)
            signs = signs * 2 - 1
            emb = emb * signs.unsqueeze(1)
        return emb

    def forward(self, x, node_features=None, sc_matrix=None, **kwargs):
        model_device = next(self.parameters()).device
        x_ut = self._resolve_input(x).to(device=model_device, dtype=torch.float32)
        if x_ut.ndim == 1:
            x_ut = x_ut.unsqueeze(0)
        bsz = x_ut.shape[0]

        blocks = []
        s = self._static_block(node_features, bsz, model_device)
        if s is not None:
            blocks.append(s)
        r = self._sc_row_block(sc_matrix, bsz, model_device)
        if r is not None:
            blocks.append(r)
        feats = torch.cat(blocks, dim=-1) if len(blocks) > 1 else blocks[0]  # [B, N, F]

        h = self.encoder(feats)  # [B, N, D]
        h_i = h[:, self.label_edge_index[0], :]
        h_j = h[:, self.label_edge_index[1], :]

        if self.decoder_type == "dot":
            return (h_i * h_j).sum(dim=-1) * self.edge_scale + self.edge_bias

        if self.decoder_type == "bilinear":
            W_sym = 0.5 * (self.edge_W + self.edge_W.t())
            return torch.einsum("bed,df,bef->be", h_i, W_sym, h_j) + self.edge_bias

        if self.decoder_type == "diag_bilinear":
            return (h_i * h_j * self.edge_w).sum(dim=-1) + self.edge_bias

        # mlp + linear_beta share the symmetric edge featurization
        if self.decoder_symmetry == "symmetric":
            edge_feat = torch.cat([h_i + h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)
        elif self.decoder_symmetry == "asymmetric":
            edge_feat = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)
        else:  # "concat"
            edge_feat = torch.cat([h_i, h_j], dim=-1)
        b, e, f = edge_feat.shape
        out = self.edge_head(edge_feat.reshape(b * e, f))  # [B*E, 1]
        return out.view(b, e)

    def get_reg_loss(self):
        if self.reg <= 0:
            return 0.0
        # True squared-L2 (ridge). Regularize all *weight* content — including the
        # decoder heads' learned vectors/scalars (`edge_w` for diag_bilinear,
        # `edge_scale` for dot, `edge_W` for bilinear, encoder/decoder Linear weights).
        # Exclude biases and BatchNorm/PReLU affine params via module-type and name.
        # This keeps decoder-family comparisons fair at fixed `reg`: every head has
        # its content under penalty, not just heads that happen to use 2-D matrices.
        no_reg_ids = set()
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.PReLU)):
                for p in module.parameters(recurse=False):
                    no_reg_ids.add(id(p))
        params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in no_reg_ids:
                continue
            if name.endswith(".bias") or name == "edge_bias":
                continue
            params.append(p)
        return compute_reg_loss(params, l1_l2_tuple=(0.0, self.reg))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
