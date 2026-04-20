import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression

from models.train.loss import compute_latent_reconstruction_loss
from models.architectures.utils import compute_reg_loss, get_modality_data


class MultiHeadTokenSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        attn_dim,
        value_dim,
        num_heads=1,
        attention_activation="softmax",
        dropout=0.0,
        use_input_as_value=False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.attn_dim = int(attn_dim)
        self.use_input_as_value = bool(use_input_as_value)
        self.value_dim = int(self.input_dim if self.use_input_as_value else value_dim)
        self.num_heads = int(num_heads)
        self.attention_activation = str(attention_activation)
        self.dropout_p = float(dropout)

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}.")
        if self.attn_dim % self.num_heads != 0:
            raise ValueError(
                f"attn_dim={self.attn_dim} must be divisible by num_heads={self.num_heads}."
            )
        if self.value_dim % self.num_heads != 0:
            raise ValueError(
                f"value_dim={self.value_dim} must be divisible by num_heads={self.num_heads}."
            )
        if self.attention_activation not in {"softmax", "identity"}:
            raise ValueError(
                f"Unknown attention_activation='{self.attention_activation}'. "
                "Choose from {'softmax', 'identity'}."
            )

        self.q_head_dim = self.attn_dim // self.num_heads
        self.v_head_dim = self.value_dim // self.num_heads

        self.W_Q = nn.Linear(self.input_dim, self.attn_dim, bias=False)
        self.W_K = nn.Linear(self.input_dim, self.attn_dim, bias=False)
        self.W_V = None if self.use_input_as_value else nn.Linear(self.input_dim, self.value_dim, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

    def forward(self, tokens, key_mask=None):
        """
        key_mask: optional bool/float tensor (batch, num_tokens). True/1 = key is
        visible and allowed to contribute; False/0 = key is excluded from attention.
        Softmax: invalid keys get -inf logits. Identity: invalid keys get zero weight.
        """
        batch_size, num_tokens, _ = tokens.shape
        Q = self.W_Q(tokens).view(batch_size, num_tokens, self.num_heads, self.q_head_dim).transpose(1, 2)
        K = self.W_K(tokens).view(batch_size, num_tokens, self.num_heads, self.q_head_dim).transpose(1, 2)
        V_input = tokens if self.W_V is None else self.W_V(tokens)
        V = V_input.view(batch_size, num_tokens, self.num_heads, self.v_head_dim).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.q_head_dim)
        if key_mask is not None:
            km = key_mask.to(dtype=torch.bool)
            # (batch, 1, 1, num_tokens) broadcasts over heads and queries
            km_b = km.view(batch_size, 1, 1, num_tokens)
            if self.attention_activation == "softmax":
                attn_logits = attn_logits.masked_fill(~km_b, float("-inf"))
                attn = torch.softmax(attn_logits, dim=-1)
            else:
                attn = attn_logits.masked_fill(~km_b, 0.0)
        else:
            if self.attention_activation == "softmax":
                attn = torch.softmax(attn_logits, dim=-1)
            else:
                attn = attn_logits
        attn = self.attn_dropout(attn)
        Z = torch.matmul(attn, V)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.value_dim)
        return Z, attn


class TransformerTokenBlock(nn.Module):
    def __init__(self, token_dim, attn_dim, num_heads=1, dropout=0.0, attention_activation="softmax", ff_hidden_dim=None):
        super().__init__()
        self.token_dim = int(token_dim)
        self.attn_dim = int(attn_dim)
        ff_hidden_dim = int(ff_hidden_dim or (4 * self.token_dim))
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.self_attn = MultiHeadTokenSelfAttention(
            input_dim=self.token_dim,
            attn_dim=self.attn_dim,
            value_dim=self.attn_dim,
            num_heads=num_heads,
            attention_activation=attention_activation,
            dropout=dropout,
        )
        self.attn_out_proj = nn.Linear(self.attn_dim, self.token_dim, bias=False)
        self.attn_resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.token_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden_dim, self.token_dim),
        )
        self.ffn_resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, tokens, key_mask=None):
        attn_in = self.norm1(tokens)
        attn_out, attn = self.self_attn(attn_in, key_mask=key_mask)
        tokens = tokens + self.attn_resid_dropout(self.attn_out_proj(attn_out))
        ffn_in = self.norm2(tokens)
        tokens = tokens + self.ffn_resid_dropout(self.ffn(ffn_in))
        return tokens, attn


class LatentAttnMasked(nn.Module):
    """
    Joint SC/FC latent attention model with masked FC reconstruction.

    The model predicts target PCA coefficients through:
    1. A latent-space backbone (`attention_only`, `none`, `pls_residual`, `linear_residual`).
    2. An optional masked FC attention branch that predicts a residual correction.
    3. Fixed decoding back to full target edge space through the target PCA basis.

    `residual_mode` semantics:
    - `attention_only`: zero backbone, active attention branch
    - `none`: linear backbone only, no attention branch
    - `pls_residual`: PLS backbone plus attention residual
    - `linear_residual`: learned linear backbone plus attention residual
    """

    @staticmethod
    def _normalize_readout_hidden_spec(spec, default_dim):
        if spec is None:
            return [int(default_dim)]
        if isinstance(spec, np.ndarray):
            spec = spec.tolist()
        if isinstance(spec, (list, tuple)):
            return [int(dim) for dim in spec]
        return [int(spec)]

    @staticmethod
    def _build_scalar_readout(in_dim, hidden_dims):
        dims = [int(in_dim)] + [int(dim) for dim in hidden_dims] + [1]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.PReLU())
        return nn.Sequential(*layers)

    @staticmethod
    def _sample_orthonormal_projector(input_dim, output_dim, device):
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")
        if output_dim > input_dim:
            raise ValueError(
                f"Cannot build an orthonormal compression from input_dim={input_dim} "
                f"to output_dim={output_dim} because output_dim must be <= input_dim."
            )
        basis, _ = torch.linalg.qr(torch.randn(input_dim, output_dim, device=device), mode="reduced")
        return basis

    @staticmethod
    def _build_mlp(in_dim, hidden_dims, out_dim):
        dims = [int(in_dim)] + [int(dim) for dim in hidden_dims] + [int(out_dim)]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.PReLU())
        return nn.Sequential(*layers)

    def __init__(
        self,
        base,
        n_components_pca=64,
        n_components_pls=16,
        token_embedding_type="learned",
        token_embedding_dim=None,
        pca_barcode_dim=32,
        attn_dim=16,
        value_dim=16,
        transformer_layers=0,
        num_heads=1,
        readout_type="linear",
        readout_hidden_dim=None,
        attention_activation="softmax",
        residual_mode="attention_only",
        residual_gain_init=1.0e-3,
        zscore_pca_scores=False,
        attention_dropout=0.0,
        sc_token_dropout=0.0,
        reg=1.0e-4,
        normalize_barcodes=True,
        mask_ratio=0.5,
        min_masked_components=1,
        train_use_visible_fc_context=True,
        prior_qk_init_scale=0.25,
        prior_qk_kernel_reg_weight=1.0,
        device=None,
        **kwargs,
    ):
        super().__init__()
        if len(getattr(base, "source_modalities", [base.source])) != 1:
            raise ValueError("LatentAttnMasked currently supports exactly one source modality.")

        self.base = base
        self.source_modality = getattr(base, "source_modalities", [base.source])[0]
        self.target_modality = getattr(base, "target", None) or getattr(base, "target_modalities", [base.target])[0]
        self.n_components_pca = int(n_components_pca)
        self.n_components_pls = int(n_components_pls)
        self.requested_token_embedding_type = str(token_embedding_type).lower()
        token_embedding_aliases = {
            "pls_learned": "pls_encoded",
            "pca_learned": "pca_encoded",
        }
        self.token_embedding_type = token_embedding_aliases.get(
            self.requested_token_embedding_type,
            self.requested_token_embedding_type,
        )
        self.attn_dim = int(attn_dim)
        self.value_dim = int(value_dim)
        self.pca_barcode_dim = int(pca_barcode_dim)
        self.transformer_layers = int(transformer_layers)
        self.num_heads = int(num_heads)
        self.readout_type = str(readout_type)
        self.attention_activation = str(attention_activation)
        self.residual_mode = str(residual_mode)
        self.residual_gain_init = float(residual_gain_init)
        self.zscore_pca_scores = bool(zscore_pca_scores)
        self.attention_dropout_p = float(attention_dropout)
        self.sc_token_dropout_p = float(sc_token_dropout)
        self.reg = float(reg)
        self.normalize_barcodes = bool(normalize_barcodes)
        self.mask_ratio = float(mask_ratio)
        self.min_masked_components = int(min_masked_components)
        self.train_use_visible_fc_context = bool(train_use_visible_fc_context)
        self.prior_qk_init_scale = float(prior_qk_init_scale)
        self.prior_qk_kernel_reg_weight = float(prior_qk_kernel_reg_weight)

        if self.token_embedding_type not in {"learned", "learned_cov_init", "pls_encoded", "pca_encoded"}:
            raise ValueError(
                f"Unknown token_embedding_type='{token_embedding_type}'. "
                "Choose from {'learned', 'learned_cov_init', 'pls_encoded', 'pca_encoded'}."
            )
        if self.token_embedding_type == "learned_cov_init" and self.transformer_layers != 0:
            raise ValueError(
                "token_embedding_type='learned_cov_init' currently supports transformer_layers=0 only."
            )
        if self.readout_type not in {"linear", "mlp", "concat_mlp", "scalar_slot"}:
            raise ValueError(
                f"Unknown readout_type='{self.readout_type}'. Choose from {'linear', 'mlp', 'concat_mlp', 'scalar_slot'}."
            )
        if self.transformer_layers < 0:
            raise ValueError(f"transformer_layers must be >= 0, got {self.transformer_layers}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}.")
        if self.attention_activation not in {"softmax", "identity"}:
            raise ValueError(
                f"Unknown attention_activation='{self.attention_activation}'. "
                "Choose from {'softmax', 'identity'}."
            )
        if self.residual_mode not in {"attention_only", "none", "pls_residual", "linear_residual"}:
            raise ValueError(
                f"Unknown residual_mode='{self.residual_mode}'. "
                "Choose from {'attention_only', 'none', 'pls_residual', 'linear_residual'}."
            )
        if not (0.0 <= self.sc_token_dropout_p < 1.0):
            raise ValueError(f"sc_token_dropout must be in [0, 1), got {self.sc_token_dropout_p}.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        data = get_modality_data(base, device=device, include_scores=True)
        source_data = data["sources"][self.source_modality]
        target_data = data["target"]

        source_mean = source_data["mean"]
        source_loadings = source_data["loadings"]
        source_scores = source_data["scores"]
        target_mean = target_data["mean"]
        target_loadings = target_data["loadings"]
        target_scores = target_data["scores"]
        d_source = source_loadings.shape[0]
        d_target = target_loadings.shape[0]

        max_source_k = source_scores.shape[1]
        max_target_k = target_scores.shape[1]
        if self.n_components_pca > min(max_source_k, max_target_k):
            raise ValueError(
                f"LatentAttnMasked requested n_components_pca={self.n_components_pca}, "
                f"but train PCA scores only support min(source={max_source_k}, target={max_target_k})."
            )
        if (
            self.residual_mode == "pls_residual" or self.token_embedding_type == "pls_encoded"
        ) and self.n_components_pls > self.n_components_pca:
            raise ValueError(
                f"LatentAttnMasked requested n_components_pls={self.n_components_pls}, "
                f"but only {self.n_components_pca} PCA components are retained."
            )
        if self.token_embedding_type == "pca_encoded" and self.pca_barcode_dim > min(d_source, d_target):
            raise ValueError(
                f"LatentAttnMasked requested pca_barcode_dim={self.pca_barcode_dim}, "
                f"but it must be <= min(d_source={d_source}, d_target={d_target})."
            )

        source_scores_k = np.asarray(source_scores[:, :self.n_components_pca], dtype=np.float32)
        target_scores_k = np.asarray(target_scores[:, :self.n_components_pca], dtype=np.float32)
        source_score_mean = source_scores_k.mean(axis=0, dtype=np.float32)
        target_score_mean = target_scores_k.mean(axis=0, dtype=np.float32)
        source_score_std = np.maximum(source_scores_k.std(axis=0, dtype=np.float32), 1.0e-8)
        target_score_std = np.maximum(target_scores_k.std(axis=0, dtype=np.float32), 1.0e-8)
        if self.zscore_pca_scores:
            X_pls = (source_scores_k - source_score_mean) / source_score_std
            Y_pls = (target_scores_k - target_score_mean) / target_score_std
        else:
            X_pls = source_scores_k
            Y_pls = target_scores_k

        if token_embedding_dim is None:
            token_embedding_dim = max(16, min(self.n_components_pca, 64))
        self.token_embedding_dim = int(token_embedding_dim)

        self.register_buffer("source_mean", torch.tensor(source_mean, dtype=torch.float32, device=device))
        self.register_buffer(
            "source_loadings_k",
            torch.tensor(source_loadings[:, :self.n_components_pca], dtype=torch.float32, device=device),
        )
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32, device=device))
        self.register_buffer(
            "target_loadings_k",
            torch.tensor(target_loadings[:, :self.n_components_pca], dtype=torch.float32, device=device),
        )
        self.register_buffer("source_score_mean", torch.tensor(source_score_mean, dtype=torch.float32, device=device))
        self.register_buffer("source_score_std", torch.tensor(source_score_std, dtype=torch.float32, device=device))
        self.register_buffer("target_score_mean", torch.tensor(target_score_mean, dtype=torch.float32, device=device))
        self.register_buffer("target_score_std", torch.tensor(target_score_std, dtype=torch.float32, device=device))
        self.register_buffer("source_pca_scores_train", torch.tensor(source_scores_k, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_scores_train", torch.tensor(target_scores_k, dtype=torch.float32, device=device))
        self.register_buffer("source_pca_scores_train_z", torch.tensor(X_pls, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_scores_train_z", torch.tensor(Y_pls, dtype=torch.float32, device=device))
        self.register_buffer(
            "sc_pca_barcodes",
            torch.tensor(source_loadings[:, :self.n_components_pca].T, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "fc_pca_barcodes",
            torch.tensor(target_loadings[:, :self.n_components_pca].T, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "sc_pca_projector",
            self._sample_orthonormal_projector(d_source, self.pca_barcode_dim, device=device),
        )
        self.register_buffer(
            "fc_pca_projector",
            self._sample_orthonormal_projector(d_target, self.pca_barcode_dim, device=device),
        )

        if self.zscore_pca_scores:
            latent_weights = np.ones(self.n_components_pca, dtype=np.float32)
        else:
            latent_variance = np.var(target_scores_k, axis=0, dtype=np.float32)
            latent_variance = np.maximum(latent_variance, 1.0e-8)
            latent_weights = latent_variance / float(np.mean(latent_variance))
        self.register_buffer("latent_loss_weights", torch.tensor(latent_weights, dtype=torch.float32, device=device))

        self.register_buffer(
            "pls_residual_coef",
            torch.zeros(self.n_components_pca, self.n_components_pca, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "pls_residual_intercept",
            torch.zeros(self.n_components_pca, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "sc_pls_loadings",
            torch.zeros(self.n_components_pca, self.n_components_pls, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "fc_pls_loadings",
            torch.zeros(self.n_components_pca, self.n_components_pls, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "sc_pls_scores",
            torch.zeros(source_scores_k.shape[0], self.n_components_pls, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "fc_pls_scores",
            torch.zeros(target_scores_k.shape[0], self.n_components_pls, dtype=torch.float32, device=device),
        )

        need_pls_fit = (self.residual_mode == "pls_residual") or (self.token_embedding_type == "pls_encoded")
        self.backbone_linear = None
        if self.residual_mode in {"none", "linear_residual"}:
            self.backbone_linear = nn.Linear(self.n_components_pca, self.n_components_pca, bias=True)
        self.residual_linear = self.backbone_linear
        if need_pls_fit:
            pls = PLSRegression(n_components=self.n_components_pls)
            pls.fit(X_pls, Y_pls)
            # Build the exact affine latent map induced by sklearn's predict()
            # instead of relying on coef_ orientation conventions.
            origin = np.zeros((1, self.n_components_pca), dtype=np.float32)
            intercept = np.asarray(pls.predict(origin), dtype=np.float32).reshape(-1)
            eye = np.eye(self.n_components_pca, dtype=np.float32)
            basis_pred = np.asarray(pls.predict(eye), dtype=np.float32)
            coef_use = basis_pred - intercept[None, :]
            self.pls_residual_coef.copy_(torch.tensor(coef_use, dtype=torch.float32, device=device))
            self.pls_residual_intercept.copy_(torch.tensor(intercept, dtype=torch.float32, device=device))
            self.sc_pls_loadings.copy_(torch.tensor(np.asarray(pls.x_loadings_, dtype=np.float32), dtype=torch.float32, device=device))
            self.fc_pls_loadings.copy_(torch.tensor(np.asarray(pls.y_loadings_, dtype=np.float32), dtype=torch.float32, device=device))
            self.sc_pls_scores.copy_(torch.tensor(np.asarray(pls.x_scores_, dtype=np.float32), dtype=torch.float32, device=device))
            self.fc_pls_scores.copy_(torch.tensor(np.asarray(pls.y_scores_, dtype=np.float32), dtype=torch.float32, device=device))

        self.sc_component_embedding = None
        self.fc_component_embedding = None
        self.sc_barcode_encoder = None
        self.fc_barcode_encoder = None
        if self.token_embedding_type in {"learned", "learned_cov_init"}:
            self.sc_component_embedding = nn.Embedding(self.n_components_pca, self.token_embedding_dim)
            self.fc_component_embedding = nn.Embedding(self.n_components_pca, self.token_embedding_dim)
        elif self.token_embedding_type == "pls_encoded":
            self.sc_barcode_encoder = nn.Linear(self.n_components_pls, self.token_embedding_dim, bias=True)
            self.fc_barcode_encoder = nn.Linear(self.n_components_pls, self.token_embedding_dim, bias=True)
        else:
            pca_hidden_dim = max(self.pca_barcode_dim, self.token_embedding_dim)
            self.sc_barcode_encoder = self._build_mlp(
                self.pca_barcode_dim,
                [pca_hidden_dim],
                self.token_embedding_dim,
            )
            self.fc_barcode_encoder = self._build_mlp(
                self.pca_barcode_dim,
                [pca_hidden_dim],
                self.token_embedding_dim,
            )
        self.use_attention_residual = self.residual_mode in {"attention_only", "pls_residual", "linear_residual"}

        token_dim = 1 + self.token_embedding_dim
        self.token_dim = token_dim
        self.transformer_blocks = None
        self.raw_attention = None
        self.raw_attention_value_dim = None
        if self.transformer_layers > 0:
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerTokenBlock(
                        token_dim=self.token_dim,
                        attn_dim=self.attn_dim,
                        num_heads=self.num_heads,
                        dropout=self.attention_dropout_p,
                        attention_activation=self.attention_activation,
                    )
                    for _ in range(self.transformer_layers)
                ]
            )
            self.transformer_output_norm = nn.LayerNorm(self.token_dim)
            self.W_Q = None
            self.W_K = None
            self.W_V = None
        else:
            use_input_as_value = self.readout_type == "scalar_slot"
            raw_value_dim = self.token_dim if use_input_as_value else self.value_dim
            self.raw_attention_value_dim = raw_value_dim
            self.raw_attention = MultiHeadTokenSelfAttention(
                input_dim=self.token_dim,
                attn_dim=self.attn_dim,
                value_dim=raw_value_dim,
                num_heads=self.num_heads,
                attention_activation=self.attention_activation,
                dropout=self.attention_dropout_p,
                use_input_as_value=use_input_as_value,
            )
            self.W_Q = self.raw_attention.W_Q
            self.W_K = self.raw_attention.W_K
            self.W_V = self.raw_attention.W_V
            self.transformer_output_norm = None
        self.readout_context_dim = self.token_dim if self.transformer_layers > 0 else self.raw_attention_value_dim
        self.readout_hidden_dims = self._normalize_readout_hidden_spec(
            readout_hidden_dim,
            default_dim=self.readout_context_dim,
        )
        if self.readout_type == "scalar_slot":
            self.readout_head = None
        elif self.readout_type == "linear":
            self.readout_head = nn.Linear(self.readout_context_dim, 1, bias=True)
        elif self.readout_type == "mlp":
            self.readout_head = self._build_scalar_readout(
                self.readout_context_dim,
                self.readout_hidden_dims,
            )
        else:
            self.readout_head = self._build_scalar_readout(
                self.token_dim + self.readout_context_dim,
                self.readout_hidden_dims,
            )
        self.sc_token_dropout = nn.Dropout(self.sc_token_dropout_p) if self.sc_token_dropout_p > 0 else nn.Identity()
        self.residual_gain = None if not self.use_attention_residual else nn.Parameter(
            torch.tensor([self.residual_gain_init], dtype=torch.float32, device=device)
        )

        self.fc_mask_value = nn.Parameter(torch.zeros(1))

        self.last_attention = None
        self.last_latent_pred = None
        self.last_fc_mask = None
        self.last_residual_base = None
        self.last_latent_delta = None
        self.prior_qk_init_rel_error = None

        self._apply_prior_qk_initialization()

        print(
            f"LatentAttnMasked init | src={self.source_modality} tgt={self.target_modality} "
            f"| k={self.n_components_pca} pls={self.n_components_pls} "
            f"| token_embedding_type={self.token_embedding_type} token_embedding_dim={self.token_embedding_dim} "
            f"| residual_mode={self.residual_mode} residual_gain_init={self.residual_gain_init} "
            f"| attn_dim={self.attn_dim} value_dim={self.value_dim} "
            f"| transformer_layers={self.transformer_layers} num_heads={self.num_heads} "
            f"| readout_type={self.readout_type} readout_hidden_dims={self.readout_hidden_dims} "
            f"| attention_activation={self.attention_activation} "
            f"| zscore_pca_scores={self.zscore_pca_scores} "
            f"| mask_ratio={self.mask_ratio} visible_fc_context={self.train_use_visible_fc_context} "
            f"| sc_token_dropout={self.sc_token_dropout_p}"
            + (
                f" | prior_qk_init_scale={self.prior_qk_init_scale}"
                f" prior_qk_kernel_reg_weight={self.prior_qk_kernel_reg_weight}"
                f" prior_qk_rel_error={self.prior_qk_init_rel_error:.3e}"
                if self.token_embedding_type == "learned_cov_init"
                else ""
            ),
            flush=True,
        )
        self.to(device)

    def encode_source_latents(self, x):
        x = x.to(self.source_mean.device).to(torch.float32)
        c_source = torch.matmul(x - self.source_mean, self.source_loadings_k)
        if self.zscore_pca_scores:
            c_source = (c_source - self.source_score_mean) / self.source_score_std
        return c_source

    def encode_target_latents(self, y):
        y = y.to(self.target_mean.device).to(torch.float32)
        c_target = torch.matmul(y - self.target_mean, self.target_loadings_k)
        if self.zscore_pca_scores:
            c_target = (c_target - self.target_score_mean) / self.target_score_std
        return c_target

    def decode_target_latents(self, c_target_hat):
        if self.zscore_pca_scores:
            c_target_hat = c_target_hat * self.target_score_std + self.target_score_mean
        return torch.matmul(c_target_hat, self.target_loadings_k.t()) + self.target_mean

    def _compute_residual_base(self, c_source):
        if self.residual_mode == "attention_only":
            return torch.zeros_like(c_source)
        if self.residual_mode == "pls_residual":
            return torch.matmul(c_source, self.pls_residual_coef) + self.pls_residual_intercept
        return self.backbone_linear(c_source)

    def predict_backbone_latents(self, x):
        c_source = self.encode_source_latents(x)
        return self._compute_residual_base(c_source)

    def _compute_cross_component_correlation(self):
        """
        Empirical SC/FC cross-component correlation on training PCA scores.

        Returns an (k, k) float64 numpy array C where C[i, j] = corr(sc_score_i, fc_score_j)
        across training subjects. Rows index SC (query side), columns index FC (key side).
        When zscore_pca_scores=True the pre-standardized score buffers are used, otherwise
        the raw PCA score buffers. Mirrors the correlation construction in
        CrossModal_ConditionalGaussian._cov_to_corr.
        """
        if self.zscore_pca_scores:
            sc_np = self.source_pca_scores_train_z.detach().cpu().numpy().astype(np.float64)
            fc_np = self.target_pca_scores_train_z.detach().cpu().numpy().astype(np.float64)
        else:
            sc_np = self.source_pca_scores_train.detach().cpu().numpy().astype(np.float64)
            fc_np = self.target_pca_scores_train.detach().cpu().numpy().astype(np.float64)

        k = int(self.n_components_pca)
        joint = np.concatenate([sc_np, fc_np], axis=1)
        Sigma = np.cov(joint, rowvar=False)
        std = np.sqrt(np.clip(np.diag(Sigma), a_min=1.0e-12, a_max=None))
        denom = np.outer(std, std)
        corr = Sigma / denom
        corr[~np.isfinite(corr)] = 0.0
        np.fill_diagonal(corr, 1.0)
        C = corr[:k, k:]
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(C, dtype=np.float64)

    def _build_qk_init_from_prior(self, C_np):
        """
        Factorize the cross-correlation prior into query/key embedding-space weights.

        Given learned component embeddings A=E_sc in R^{k x d_emb} and B=E_fc in R^{k x d_emb},
        we want (A W_Q)(B W_K)^T / sqrt(d) ~ C, so X Y^T ~ sqrt(d) * pinv(A) C pinv(B)^T =: sqrt(d) M.

        Steps:
            M   = pinv(A) @ C @ pinv(B).T                       -> (d_emb, d_emb)
            M   = U S V^T                                       SVD, d_emb modes
            Wq  = U[:, :r] diag(sqrt(S[:r]))                    -> (d_emb, r)
            Wk  = V[:, :r] diag(sqrt(S[:r])) * sqrt(attn_dim)   -> (d_emb, r)

        with r = min(d_emb, attn_dim). We zero-pad to attn_dim columns when r < attn_dim and
        scale both outputs by sqrt(prior_qk_init_scale) so the initial kernel is approximately
        prior_qk_init_scale * C rather than C itself. Only the first attn_dim columns are
        populated; per-head slicing inside MultiHeadTokenSelfAttention then reads consecutive
        slices of the same (d_emb, attn_dim) matrix.

        Returns numpy float64 (Wq_init, Wk_init) both shaped (d_emb, attn_dim).
        """
        if self.sc_component_embedding is None or self.fc_component_embedding is None:
            raise RuntimeError(
                "learned_cov_init requires sc/fc_component_embedding to be initialized before calling _build_qk_init_from_prior."
            )

        A = self.sc_component_embedding.weight.detach().cpu().numpy().astype(np.float64)
        B = self.fc_component_embedding.weight.detach().cpu().numpy().astype(np.float64)

        d_emb = int(self.token_embedding_dim)
        attn_dim = int(self.attn_dim)
        pinv_A = np.linalg.pinv(A)
        pinv_B = np.linalg.pinv(B)
        M = pinv_A @ C_np @ pinv_B.T

        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        V = Vt.T
        r = int(min(d_emb, attn_dim))
        sqrt_S = np.sqrt(np.clip(S[:r], a_min=0.0, a_max=None))
        Wq_core = U[:, :r] * sqrt_S[None, :]
        Wk_core = V[:, :r] * sqrt_S[None, :] * np.sqrt(float(attn_dim))

        scale = np.sqrt(max(self.prior_qk_init_scale, 0.0))
        Wq_core = scale * Wq_core
        Wk_core = scale * Wk_core

        if r < attn_dim:
            pad = np.zeros((d_emb, attn_dim - r), dtype=np.float64)
            Wq_init = np.concatenate([Wq_core, pad], axis=1)
            Wk_init = np.concatenate([Wk_core, pad], axis=1)
        else:
            Wq_init = Wq_core[:, :attn_dim]
            Wk_init = Wk_core[:, :attn_dim]

        return Wq_init.astype(np.float64), Wk_init.astype(np.float64)

    def _apply_prior_qk_initialization(self):
        """
        Initialize raw_attention.W_Q / W_K from the SC-FC cross-correlation prior.

        Only active when token_embedding_type='learned_cov_init' and the model uses the raw
        single-attention path (transformer_layers=0). Writes only into the embedding columns
        [:, 1:] of W_Q/W_K so the scalar-slot column stays at its default nn.Linear init.
        Logs the prior type, the computed shapes, and the relative Frobenius reconstruction
        error of the initialized kernel with respect to the target prior C. No gradients are
        computed during this routine; nothing is frozen afterwards.
        """
        if self.token_embedding_type != "learned_cov_init":
            return
        if self.transformer_layers != 0 or self.raw_attention is None:
            return
        if self.num_heads != 1:
            print(
                f"LatentAttnMasked prior-init warning | num_heads={self.num_heads} > 1: "
                "init writes a single (d_emb, attn_dim) block; per-head scaling uses q_head_dim, "
                "so the reported relative error assumes aggregate attn_dim scaling.",
                flush=True,
            )

        C_np = self._compute_cross_component_correlation()
        Wq_init, Wk_init = self._build_qk_init_from_prior(C_np)

        weight_device = self.raw_attention.W_Q.weight.device
        weight_dtype = self.raw_attention.W_Q.weight.dtype
        Wq_t = torch.tensor(Wq_init, dtype=weight_dtype, device=weight_device)
        Wk_t = torch.tensor(Wk_init, dtype=weight_dtype, device=weight_device)
        with torch.no_grad():
            self.raw_attention.W_Q.weight[:, 1:].copy_(Wq_t.t())
            self.raw_attention.W_K.weight[:, 1:].copy_(Wk_t.t())

        with torch.no_grad():
            A = self.sc_component_embedding.weight.detach().cpu().numpy().astype(np.float64)
            B = self.fc_component_embedding.weight.detach().cpu().numpy().astype(np.float64)
            kernel = (A @ Wq_init) @ (B @ Wk_init).T / float(np.sqrt(max(self.attn_dim, 1)))
            target = float(self.prior_qk_init_scale) * C_np
            num = float(np.linalg.norm(kernel - target, ord="fro"))
            den = float(np.linalg.norm(target, ord="fro")) + 1.0e-12
            rel_err = num / den
        self.prior_qk_init_rel_error = rel_err

        print(
            f"LatentAttnMasked prior-init | type={self.token_embedding_type} "
            f"| k={self.n_components_pca} d_emb={self.token_embedding_dim} attn_dim={self.attn_dim} num_heads={self.num_heads} "
            f"| C.shape={C_np.shape} Wq.shape={Wq_init.shape} Wk.shape={Wk_init.shape} "
            f"| scale={self.prior_qk_init_scale} kernel_reg_weight={self.prior_qk_kernel_reg_weight} "
            f"| rel_frob_error={rel_err:.3e}",
            flush=True,
        )

    def _sample_fc_mask(self, batch_size, device):
        mask = (torch.rand(batch_size, self.n_components_pca, device=device) < self.mask_ratio)
        min_mask = min(self.min_masked_components, self.n_components_pca)
        if min_mask > 0:
            counts = mask.sum(dim=1)
            needs_fix = counts < min_mask
            if needs_fix.any():
                for row_idx in torch.nonzero(needs_fix, as_tuple=False).flatten():
                    perm = torch.randperm(self.n_components_pca, device=device)
                    mask[row_idx, perm[:min_mask]] = True
        return mask

    def _get_component_embeddings(self, batch_size):
        idx = torch.arange(self.n_components_pca, device=self.source_mean.device)
        if self.token_embedding_type in {"learned", "learned_cov_init"}:
            sc_emb = self.sc_component_embedding(idx)
            fc_emb = self.fc_component_embedding(idx)
        elif self.token_embedding_type == "pls_encoded":
            sc_barcode = self.sc_pls_loadings
            fc_barcode = self.fc_pls_loadings
            if self.normalize_barcodes:
                sc_barcode = torch.nn.functional.normalize(sc_barcode, p=2, dim=-1, eps=1.0e-8)
                fc_barcode = torch.nn.functional.normalize(fc_barcode, p=2, dim=-1, eps=1.0e-8)
            sc_emb = self.sc_barcode_encoder(sc_barcode)
            fc_emb = self.fc_barcode_encoder(fc_barcode)
        else:
            sc_barcode = torch.matmul(self.sc_pca_barcodes, self.sc_pca_projector)
            fc_barcode = torch.matmul(self.fc_pca_barcodes, self.fc_pca_projector)
            if self.normalize_barcodes:
                sc_barcode = torch.nn.functional.normalize(sc_barcode, p=2, dim=-1, eps=1.0e-8)
                fc_barcode = torch.nn.functional.normalize(fc_barcode, p=2, dim=-1, eps=1.0e-8)
            sc_emb = self.sc_barcode_encoder(sc_barcode)
            fc_emb = self.fc_barcode_encoder(fc_barcode)
        sc_emb = sc_emb.unsqueeze(0).expand(batch_size, -1, -1)
        fc_emb = fc_emb.unsqueeze(0).expand(batch_size, -1, -1)
        return sc_emb, fc_emb

    def _build_joint_tokens(self, c_source, c_target_true=None, fc_mask=None):
        batch_size = c_source.shape[0]
        sc_emb, fc_emb = self._get_component_embeddings(batch_size)
        sc_tokens = torch.cat([c_source.unsqueeze(-1), sc_emb], dim=-1)
        if self.training and self.sc_token_dropout_p > 0:
            sc_tokens = self.sc_token_dropout(sc_tokens)

        if c_target_true is None:
            c_target_scalar = self.fc_mask_value.view(1, 1).expand(batch_size, self.n_components_pca)
            fc_mask = torch.ones(batch_size, self.n_components_pca, dtype=torch.bool, device=c_source.device)
        else:
            if fc_mask is None:
                fc_mask = torch.ones(batch_size, self.n_components_pca, dtype=torch.bool, device=c_source.device)
            visible = c_target_true if self.train_use_visible_fc_context else torch.zeros_like(c_target_true)
            mask_fill = self.fc_mask_value.view(1, 1).expand_as(visible)
            c_target_scalar = torch.where(fc_mask, mask_fill, visible)

        fc_tokens = torch.cat([c_target_scalar.unsqueeze(-1), fc_emb], dim=-1)
        return torch.cat([sc_tokens, fc_tokens], dim=1), fc_mask

    def _run_attention_stack(self, tokens):
        if self.transformer_layers > 0:
            attn = None
            hidden = tokens
            for block in self.transformer_blocks:
                hidden, attn = block(hidden)
            hidden = self.transformer_output_norm(hidden)
            return hidden, attn
        return self.raw_attention(tokens)

    def predict_target_latents(self, x, y=None, return_attention=False, return_mask=False, force_all_masked=None):
        c_source = self.encode_source_latents(x)
        c_target_true = None if y is None else self.encode_target_latents(y)
        c_target_base = self._compute_residual_base(c_source)

        if not self.use_attention_residual:
            fc_mask = torch.ones(c_source.shape[0], self.n_components_pca, dtype=torch.bool, device=c_source.device)
            c_target_delta = torch.zeros_like(c_target_base)
            c_target_hat = c_target_base
            self.last_attention = None
            self.last_latent_pred = c_target_hat.detach()
            self.last_fc_mask = fc_mask.detach()
            self.last_residual_base = c_target_base.detach()
            self.last_latent_delta = c_target_delta.detach()

            outputs = [c_target_hat]
            if return_attention:
                outputs.append(None)
            if return_mask:
                outputs.append(fc_mask)
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

        if force_all_masked is None:
            force_all_masked = (y is None)

        if force_all_masked:
            fc_mask = torch.ones(c_source.shape[0], self.n_components_pca, dtype=torch.bool, device=c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=None, fc_mask=fc_mask)
        else:
            fc_mask = self._sample_fc_mask(c_source.shape[0], c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=c_target_true, fc_mask=fc_mask)

        Z, attn = self._run_attention_stack(tokens)
        fc_context = Z[:, self.n_components_pca:, :]
        fc_tokens = tokens[:, self.n_components_pca:, :]
        if self.readout_type == "scalar_slot":
            c_target_delta = fc_context[..., 0]
        elif self.readout_type == "concat_mlp":
            fc_readout_input = torch.cat([fc_tokens, fc_context], dim=-1)
            c_target_delta = self.readout_head(fc_readout_input).squeeze(-1)
        else:
            c_target_delta = self.readout_head(fc_context).squeeze(-1)
        if self.residual_gain is not None:
            c_target_delta = self.residual_gain * c_target_delta
        c_target_hat = c_target_base + c_target_delta

        self.last_attention = attn.detach() if attn is not None else None
        self.last_latent_pred = c_target_hat.detach()
        self.last_fc_mask = fc_mask.detach()
        self.last_residual_base = c_target_base.detach()
        self.last_latent_delta = c_target_delta.detach()

        outputs = [c_target_hat]
        if return_attention:
            outputs.append(attn)
        if return_mask:
            outputs.append(fc_mask)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def inspect_attention_state(self, x, y=None, force_all_masked=None):
        c_source = self.encode_source_latents(x)
        c_target_true = None if y is None else self.encode_target_latents(y)
        c_target_base = self._compute_residual_base(c_source)

        if not self.use_attention_residual:
            fc_mask = torch.ones(c_source.shape[0], self.n_components_pca, dtype=torch.bool, device=c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=None if force_all_masked or y is None else c_target_true, fc_mask=fc_mask)
            c_target_delta = torch.zeros_like(c_target_base)
            c_target_hat = c_target_base
            return {
                "c_source": c_source,
                "c_target_true": c_target_true,
                "c_target_base": c_target_base,
                "c_target_delta": c_target_delta,
                "tokens_pre": tokens,
                "tokens_post": tokens,
                "attention": None,
                "fc_mask": fc_mask,
                "c_target_hat": c_target_hat,
            }

        if force_all_masked is None:
            force_all_masked = (y is None)

        if force_all_masked:
            fc_mask = torch.ones(c_source.shape[0], self.n_components_pca, dtype=torch.bool, device=c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=None, fc_mask=fc_mask)
        else:
            fc_mask = self._sample_fc_mask(c_source.shape[0], c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=c_target_true, fc_mask=fc_mask)

        token_context, attn = self._run_attention_stack(tokens)
        fc_context = token_context[:, self.n_components_pca:, :]
        fc_tokens = tokens[:, self.n_components_pca:, :]
        if self.readout_type == "scalar_slot":
            c_target_delta = fc_context[..., 0]
        elif self.readout_type == "concat_mlp":
            fc_readout_input = torch.cat([fc_tokens, fc_context], dim=-1)
            c_target_delta = self.readout_head(fc_readout_input).squeeze(-1)
        else:
            c_target_delta = self.readout_head(fc_context).squeeze(-1)
        if self.residual_gain is not None:
            c_target_delta = self.residual_gain * c_target_delta
        c_target_hat = c_target_base + c_target_delta

        return {
            "c_source": c_source,
            "c_target_true": c_target_true,
            "c_target_base": c_target_base,
            "c_target_delta": c_target_delta,
            "tokens_pre": tokens,
            "tokens_post": token_context,
            "attention": attn,
            "fc_mask": fc_mask,
            "c_target_hat": c_target_hat,
        }

    def forward(self, x):
        c_target_hat = self.predict_target_latents(x, y=None, force_all_masked=True)
        return self.decode_target_latents(c_target_hat)

    def forward_backbone_only(self, x):
        return self.decode_target_latents(self.predict_backbone_latents(x))

    def compute_latent_loss(self, batch, loss_type):
        x = batch["x"] if "x" in batch else batch["x_modalities"]
        y = batch["y"]
        c_true = self.encode_target_latents(y)
        c_hat, fc_mask = self.predict_target_latents(x, y=y, return_mask=True, force_all_masked=False)
        return compute_latent_reconstruction_loss(
            c_hat,
            c_true,
            loss_type,
            weights=self.latent_loss_weights,
            mask=fc_mask,
        )

    def get_reg_loss(self):
        if self.reg <= 0:
            return 0.0
        params = []
        if self.sc_component_embedding is not None:
            params.extend([self.sc_component_embedding.weight, self.fc_component_embedding.weight])
        if self.sc_barcode_encoder is not None:
            params.extend([p for p in self.sc_barcode_encoder.parameters() if p.requires_grad])
            params.extend([p for p in self.fc_barcode_encoder.parameters() if p.requires_grad])
        if self.use_attention_residual and self.raw_attention is not None:
            params.extend([self.raw_attention.W_Q.weight, self.raw_attention.W_K.weight])
            if self.raw_attention.W_V is not None:
                params.append(self.raw_attention.W_V.weight)
        if self.use_attention_residual and self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                params.extend([p for p in block.parameters() if p.requires_grad])
            params.extend([p for p in self.transformer_output_norm.parameters() if p.requires_grad])
        if self.use_attention_residual and self.readout_head is not None:
            params.extend([p for p in self.readout_head.parameters() if p.requires_grad])
        if self.residual_linear is not None:
            params.extend([p for p in self.residual_linear.parameters() if p.requires_grad])
        return compute_reg_loss(params, l1_l2_tuple=(0.0, self.reg))
