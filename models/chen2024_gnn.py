import torch
import torch.nn as nn

from data.graph_adapter import (
    infer_num_nodes_from_upper_triangle_dim,
    upper_triangle_to_symmetric,
    get_label_edge_index,
    build_message_passing_adjacency,
)
from models.gnn_features import build_node_features


class DenseGCNLayer(nn.Module):
    """Dense GCN layer: H' = A_norm @ H @ W"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # h: [B, N, Fin], a_norm: [B, N, N]
        return torch.matmul(a_norm, self.lin(h))


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

        # Label edges (target FC upper-triangle) follow Conn2Conn target vectorization order.
        self.register_buffer("label_edge_index", get_label_edge_index(self.num_nodes, device=device))

        # Determine feature dimensionality based on chosen node feature type.
        feature_dim = self.num_nodes if self.node_feature_type == "identity" else self.num_nodes

        self.convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        self.convs.append(DenseGCNLayer(feature_dim, self.conv_dim))
        self.prelus.append(nn.PReLU())
        for _ in range(self.layer_num - 1):
            self.convs.append(DenseGCNLayer(self.conv_dim, self.conv_dim))
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
        x_ut = self._resolve_input(x).to(torch.float32)
        if x_ut.ndim == 1:
            x_ut = x_ut.unsqueeze(0)

        sc_dense = upper_triangle_to_symmetric(x_ut, self.num_nodes)
        a_norm = build_message_passing_adjacency(sc_dense, add_self_loops=self.add_self_loops)
        h = build_node_features(sc_dense, feature_type=self.node_feature_type)

        for conv, prelu in zip(self.convs, self.prelus):
            h = prelu(conv(h, a_norm))

        edge_idx = self.label_edge_index
        h_i = h[:, edge_idx[0], :]
        h_j = h[:, edge_idx[1], :]
        edge_feat = torch.cat([h_i, h_j], dim=-1)
        y_hat = self.edge_mlp(edge_feat).squeeze(-1)
        return y_hat

    def get_reg_loss(self):
        if self.reg <= 0:
            return 0.0
        # Match the context implementation: L2 regularize MLP linear weights.
        l2 = torch.norm(self.edge_mlp[0].weight, p=2) + torch.norm(self.edge_mlp[2].weight, p=2)
        return self.reg * l2

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
