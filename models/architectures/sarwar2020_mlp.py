import torch
import torch.nn as nn

from models.models import get_modality_data, compute_reg_loss


class Sarwar2020MLP(nn.Module):
    """Fully-connected SC->FC baseline inspired by Sarwar et al. (2020)."""

    def __init__(
        self,
        base,
        hidden_dim=1024,
        num_hidden_layers=7,
        dropout=0.5,
        activation_mode="alternating",
        leaky_relu_slope=0.2,
        output_tanh=True,
        l1_l2_tuple=(0.0, 0.0),
        device=None,
        **kwargs,
    ):
        super().__init__()

        source_modalities = list(getattr(base, "source_modalities", [base.source]))
        if len(source_modalities) != 1:
            raise ValueError("Sarwar2020MLP currently supports exactly one source modality.")
        self.source_modality = source_modalities[0]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        data = get_modality_data(base, device=device, include_scores=False)
        self.input_dim = int(data["source_loadings"].shape[0])
        self.output_dim = int(data["target_mean"].shape[0])
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.dropout_p = float(dropout)
        self.activation_mode = str(activation_mode)
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.output_tanh = bool(output_tanh)

        self.l1_l2_tuple = tuple(l1_l2_tuple)

        dims = [self.input_dim] + [self.hidden_dim] * self.num_hidden_layers + [self.output_dim]
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for layer in self.linears:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _get_hidden_activation(self, layer_idx):
        if self.activation_mode == "alternating":
            return (
                nn.LeakyReLU(negative_slope=self.leaky_relu_slope)
                if layer_idx % 2 == 0
                else nn.Tanh()
            )
        if self.activation_mode == "leaky_relu":
            return nn.LeakyReLU(negative_slope=self.leaky_relu_slope)
        if self.activation_mode == "tanh":
            return nn.Tanh()
        raise ValueError(
            f"Unknown activation_mode: {self.activation_mode}. "
            "Choose from {'alternating', 'leaky_relu', 'tanh'}."
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
        device = self.linears[0].weight.device
        h = self._resolve_input(x).to(device).to(torch.float32)

        for i, layer in enumerate(self.linears[:-1]):
            h = layer(h)
            h = self.dropout(h)
            h = self._get_hidden_activation(i)(h)

        y = self.linears[-1](h)
        if self.output_tanh:
            y = torch.tanh(y)
        return y

    def get_reg_loss(self):
        return compute_reg_loss(self.parameters(), self.l1_l2_tuple)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
