import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from data.graph_adapter import (
    infer_num_nodes_from_upper_triangle_dim,
    upper_triangle_to_symmetric,
    get_label_edge_index,
)
from models.gnn_features import build_node_features


class Chen2024GCN(nn.Module):
    """
    Chen et al. (2024)-style SC->FC edge regression baseline.

    Pipeline:
    1) Build SC graph from upper-triangle vector input.
    2) Learn node embeddings via stacked GCN layers + PReLU.
    3) For each target edge (i, j), predict FC edge value from [h_i || h_j] via MLP.
    """

    def __init__(
        self,
        base,
        layer_num: int = 2,
        conv_dim: int = 256,
        dnn_dim: int = 64,
        node_feature_type: str = "identity",
        reg: float = 1e-4,
        add_self_loops: bool = True,
        device=None,
        **kwargs,
    ):
        super().__init__()

        source_modalities = list(getattr(base, "source_modalities", [base.source]))
        if len(source_modalities) != 1:
            raise ValueError("Chen2024GCN currently supports exactly one source modality.")
        self.source_modality = source_modalities[0]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.layer_num = int(layer_num)
        self.conv_dim = int(conv_dim)
        self.dnn_dim = int(dnn_dim)
        self.node_feature_type = str(node_feature_type)
        self.reg = float(reg)
        self.add_self_loops = bool(add_self_loops)

        source_ut_dim = int(base.sc_upper_triangles.shape[1])
        self.num_nodes = infer_num_nodes_from_upper_triangle_dim(source_ut_dim)
        self.num_edges_upper = source_ut_dim

        # Cached topology templates:
        # - label_edge_index: upper-triangle FC edges to predict
        # - mp_edge_index_template: symmetric SC edges used for message passing
        label_edge_index = get_label_edge_index(self.num_nodes, device=device)
        self.register_buffer("label_edge_index", label_edge_index)
        mp_edge_index_template = torch.cat(
            [
                label_edge_index,
                torch.stack([label_edge_index[1], label_edge_index[0]], dim=0),
            ],
            dim=1,
        )
        self.register_buffer("mp_edge_index_template", mp_edge_index_template)
        self.register_buffer(
            "identity_node_features",
            torch.eye(self.num_nodes, device=device, dtype=torch.float32),
        )

        # Determine feature dimensionality based on chosen node feature type.
        feature_dim = self.num_nodes if self.node_feature_type == "identity" else self.num_nodes

        self.convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        self.convs.append(
            GCNConv(feature_dim, self.conv_dim, add_self_loops=self.add_self_loops, normalize=True)
        )
        self.prelus.append(nn.PReLU())
        for _ in range(self.layer_num - 1):
            self.convs.append(
                GCNConv(self.conv_dim, self.conv_dim, add_self_loops=self.add_self_loops, normalize=True)
            )
            self.prelus.append(nn.PReLU())

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.conv_dim * 2, self.dnn_dim),
            nn.PReLU(),
            nn.Linear(self.dnn_dim, 1),
        )

    def _resolve_input(self, x):
        if isinstance(x, dict):
            if self.source_modality not in x:
                raise ValueError(
                    f"Expected source modality '{self.source_modality}' in input dict keys {list(x.keys())}."
                )
            return x[self.source_modality]
        return x

    def forward(self, x):
        model_device = self.edge_mlp[0].weight.device
        x_ut = self._resolve_input(x).to(device=model_device, dtype=torch.float32)
        if x_ut.ndim == 1:
            x_ut = x_ut.unsqueeze(0)
        bsz = x_ut.shape[0]
        device = model_device

        # Build node features.
        if self.node_feature_type == "identity":
            h = self.identity_node_features.to(device).unsqueeze(0).expand(bsz, -1, -1)
        else:
            # Optional alternative for experiments; identity is the Chen-style default.
            sc_dense = upper_triangle_to_symmetric(x_ut, self.num_nodes)
            h = build_node_features(sc_dense, feature_type=self.node_feature_type)
        h = h.reshape(bsz * self.num_nodes, -1)

        # Build batched sparse graph from cached topology + per-subject SC edge weights.
        offsets = (torch.arange(bsz, device=device, dtype=torch.long) * self.num_nodes).view(bsz, 1, 1)
        mp_edge_template = self.mp_edge_index_template.to(device)
        mp_edge_index = (mp_edge_template.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        mp_edge_weight = torch.cat([x_ut, x_ut], dim=1).reshape(-1)

        for conv, prelu in zip(self.convs, self.prelus):
            h = prelu(conv(h, mp_edge_index, mp_edge_weight))

        label_edge_template = self.label_edge_index.to(device)
        label_edge_index = (label_edge_template.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        h_i = h[label_edge_index[0]]
        h_j = h[label_edge_index[1]]
        edge_feat = torch.cat([h_i, h_j], dim=-1)  # [B * E, 2*conv_dim]
        y_hat = self.edge_mlp(edge_feat).squeeze(-1).reshape(bsz, self.num_edges_upper)
        return y_hat

    def get_reg_loss(self):
        if self.reg <= 0:
            return 0.0
        # Match the context implementation: L2 regularize MLP linear weights.
        l2 = torch.norm(self.edge_mlp[0].weight, p=2) + torch.norm(self.edge_mlp[2].weight, p=2)
        return self.reg * l2

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
