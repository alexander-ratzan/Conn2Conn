import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from data.graph_adapter import (
    infer_num_nodes_from_upper_triangle_dim,
    get_label_edge_index,
)


class NodalGNN(nn.Module):
    """
    SC->FC graph model with subject-specific nodal features.

    Differences vs Chen2024GCN:
    - Uses per-subject node features (volume, centroid, r2t) instead of identity by default.
    - Supports feature ablations via use_volume/use_spatial/use_r2t flags.
    - Adds stronger regularization defaults (dropout + residual stack + L2 reg term).
    """

    def __init__(
        self,
        base,
        layer_num: int = 3,
        hidden_dim: int = 96,
        decoder_dim: int = 64,
        dropout: float = 0.35,
        edge_dropout: float = 0.10,
        reg: float = 1e-4,
        use_volume: bool = True,
        use_spatial: bool = True,
        use_r2t: bool = True,
        add_self_loops: bool = True,
        device=None,
        **kwargs,
    ):
        super().__init__()

        source_modalities = list(getattr(base, "source_modalities", [base.source]))
        if len(source_modalities) != 1:
            raise ValueError("NodalGNN currently supports exactly one source modality.")
        self.source_modality = source_modalities[0]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.layer_num = int(layer_num)
        self.hidden_dim = int(hidden_dim)
        self.decoder_dim = int(decoder_dim)
        self.dropout = float(dropout)
        self.edge_dropout = float(edge_dropout)
        self.reg = float(reg)
        self.add_self_loops = bool(add_self_loops)

        self.use_volume = bool(use_volume)
        self.use_spatial = bool(use_spatial)
        self.use_r2t = bool(use_r2t)
        if not (self.use_volume or self.use_spatial or self.use_r2t):
            raise ValueError("At least one nodal feature group must be enabled.")

        # Markers consumed by training/eval wrappers.
        self.uses_node_features = True
        self.uses_cov = False

        source_ut_dim = int(base.sc_upper_triangles.shape[1])
        self.num_nodes = infer_num_nodes_from_upper_triangle_dim(source_ut_dim)
        self.num_edges_upper = source_ut_dim

        # Edge topology templates (reused every forward).
        label_edge_index = get_label_edge_index(self.num_nodes, device=device)
        self.register_buffer("label_edge_index", label_edge_index)
        mp_edge_index_template = torch.cat(
            [label_edge_index, torch.stack([label_edge_index[1], label_edge_index[0]], dim=0)],
            dim=1,
        )
        self.register_buffer("mp_edge_index_template", mp_edge_index_template)

        # Expected node feature layout from HCP_Base:
        # [volume, centroid_x, centroid_y, centroid_z, r2t_0, ..., r2t_k]
        full_node_dim = int(base.parcel_node_features.shape[-1])
        r2t_dim = max(0, full_node_dim - 4)
        feature_idx = []
        if self.use_volume:
            feature_idx.extend([0])
        if self.use_spatial:
            feature_idx.extend([1, 2, 3])
        if self.use_r2t and r2t_dim > 0:
            feature_idx.extend(list(range(4, full_node_dim)))
        if len(feature_idx) == 0:
            raise ValueError(
                "Requested nodal feature groups are unavailable for this dataset "
                f"(full_node_dim={full_node_dim}, r2t_dim={r2t_dim})."
            )
        self.register_buffer("selected_feature_idx", torch.tensor(feature_idx, dtype=torch.long))
        self.node_feat_dim = len(feature_idx)

        # Train-split normalization stats for selected node features.
        train_indices = base.trainvaltest_partition_indices["train"]
        train_node_feats = base.parcel_node_features[train_indices][:, :, feature_idx]
        feat_mean = torch.as_tensor(train_node_feats.mean(axis=(0, 1)), dtype=torch.float32).view(1, 1, -1)
        feat_std = torch.as_tensor(train_node_feats.std(axis=(0, 1)), dtype=torch.float32).view(1, 1, -1)
        feat_std = torch.where(feat_std == 0, torch.ones_like(feat_std), feat_std)
        self.register_buffer("node_feat_mean", feat_mean)
        self.register_buffer("node_feat_std", feat_std)

        # Node encoder from selected nodal features -> hidden embedding.
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
        )

        self.convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.layer_num):
            self.convs.append(
                GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=self.add_self_loops, normalize=True)
            )
            self.prelus.append(nn.PReLU())
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        # Richer edge decoder than Chen baseline.
        # Inputs: [h_i, h_j, |h_i-h_j|, h_i*h_j]
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.decoder_dim),
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

    def _resolve_node_features(self, node_features, batch_size, device):
        if node_features is None:
            raise ValueError("NodalGNN requires batch['node_features'] for forward().")
        if node_features.ndim == 2:
            node_features = node_features.unsqueeze(0)
        if node_features.shape[0] != batch_size:
            raise ValueError(
                f"node_features batch mismatch: expected {batch_size}, got {node_features.shape[0]}"
            )
        if node_features.shape[1] != self.num_nodes:
            raise ValueError(
                f"node_features node mismatch: expected {self.num_nodes}, got {node_features.shape[1]}"
            )
        if node_features.shape[2] <= int(self.selected_feature_idx.max().item()):
            raise ValueError(
                "node_features channel dimension is too small for selected feature indices. "
                f"shape={tuple(node_features.shape)}, max_idx={int(self.selected_feature_idx.max().item())}"
            )

        feats = node_features.to(device=device, dtype=torch.float32).index_select(
            dim=2,
            index=self.selected_feature_idx.to(device),
        )
        feats = (feats - self.node_feat_mean.to(device)) / (self.node_feat_std.to(device) + 1e-8)
        return feats

    def forward(self, x, node_features=None, **kwargs):
        model_device = self.edge_mlp[0].weight.device
        x_ut = self._resolve_input(x).to(device=model_device, dtype=torch.float32)
        if x_ut.ndim == 1:
            x_ut = x_ut.unsqueeze(0)
        bsz = x_ut.shape[0]

        node_feats = self._resolve_node_features(node_features, bsz, model_device)
        h = self.node_encoder(node_feats).reshape(bsz * self.num_nodes, -1)

        # Build sparse SC-weighted graph for the full batch.
        offsets = (torch.arange(bsz, device=model_device, dtype=torch.long) * self.num_nodes).view(bsz, 1, 1)
        mp_edge_template = self.mp_edge_index_template.to(model_device)
        mp_edge_index = (mp_edge_template.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        mp_edge_weight = torch.cat([x_ut, x_ut], dim=1).reshape(-1)
        if self.training and self.edge_dropout > 0:
            keep_prob = 1.0 - self.edge_dropout
            keep_mask = torch.rand_like(mp_edge_weight) < keep_prob
            mp_edge_weight = torch.where(
                keep_mask,
                mp_edge_weight / keep_prob,
                torch.zeros_like(mp_edge_weight),
            )

        for conv, prelu, norm in zip(self.convs, self.prelus, self.norms):
            h_in = h
            h = conv(h, mp_edge_index, mp_edge_weight)
            h = prelu(h)
            h = norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in  # residual

        label_edge_template = self.label_edge_index.to(model_device)
        label_edge_index = (label_edge_template.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        h_i = h[label_edge_index[0]]
        h_j = h[label_edge_index[1]]
        edge_feat = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)
        y_hat = self.edge_mlp(edge_feat).squeeze(-1).reshape(bsz, self.num_edges_upper)
        return y_hat

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

