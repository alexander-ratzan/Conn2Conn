import torch
import torch.nn as nn

from data.graph_adapter import (
    infer_num_nodes_from_upper_triangle_dim,
    get_label_edge_index,
)


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

    def __init__(
        self,
        base,
        hidden_dim: int = 96,
        decoder_dim: int = 64,
        dropout: float = 0.2,
        reg: float = 1e-4,
        use_encoder: bool = True,
        decoder_symmetry: str = "symmetric",
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
        if sc_row_norm not in self.SC_ROW_NORMS:
            raise ValueError(
                f"sc_row_norm must be one of {sorted(self.SC_ROW_NORMS)}, got {sc_row_norm!r}."
            )
        if not (use_volume or use_spatial or use_r2t or use_sc_row):
            raise ValueError("At least one feature group must be enabled.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.hidden_dim = int(hidden_dim)
        self.decoder_dim = int(decoder_dim)
        self.dropout = float(dropout)
        self.reg = float(reg)
        self.use_encoder = bool(use_encoder)
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
        self.sc_row_dim = self.num_nodes if self.use_sc_row else 0
        if self.use_sc_row and self.sc_row_norm == "col_zscore":
            train_sc = torch.as_tensor(base.sc_matrices[train_indices], dtype=torch.float32)
            self.register_buffer("sc_col_mean", train_sc.mean(dim=0, keepdim=True))
            self.register_buffer(
                "sc_col_std", train_sc.std(dim=0, keepdim=True).clamp_min(1e-8)
            )

        self.feature_dim = self.static_dim + self.sc_row_dim

        if self.use_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.PReLU(),
            )
            self.node_embed_dim = self.hidden_dim
        else:
            self.encoder = nn.Identity()
            self.node_embed_dim = self.feature_dim

        decoder_mult = {"symmetric": 3, "asymmetric": 4, "concat": 2}[self.decoder_symmetry]
        self.edge_mlp = nn.Sequential(
            nn.Linear(decoder_mult * self.node_embed_dim, self.decoder_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.PReLU(),
            nn.Linear(self.decoder_dim, 1),
        )

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
        if self.sc_row_norm == "none":
            return m
        if self.sc_row_norm == "col_zscore":
            return (m - self.sc_col_mean.to(device)) / self.sc_col_std.to(device)
        if self.sc_row_norm == "row_zscore":
            row_mean = m.mean(dim=-1, keepdim=True)
            row_std = m.std(dim=-1, keepdim=True).clamp_min(1e-8)
            return (m - row_mean) / row_std
        if self.sc_row_norm == "row_l2":
            return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # brain_scale: per-subject divisor = mean over the full [N, N] matrix.
        return m / m.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

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

        if self.decoder_symmetry == "symmetric":
            edge_feat = torch.cat([h_i + h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)
        elif self.decoder_symmetry == "asymmetric":
            edge_feat = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)
        else:  # "concat"
            edge_feat = torch.cat([h_i, h_j], dim=-1)
        return self.edge_mlp(edge_feat).squeeze(-1)  # [B, E]

    def get_reg_loss(self):
        if self.reg <= 0:
            return 0.0
        l2 = 0.0
        for p in self.parameters():
            if p.requires_grad and p.ndim > 1:
                l2 = l2 + torch.norm(p, p=2)
        return self.reg * l2

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
