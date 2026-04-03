import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression

from models.models import compute_reg_loss, get_modality_data


class LatentAttnTranslation(nn.Module):
    """
    Single-head latent translation model built on top of source/target PCA spaces.

    The model predicts target PCA coefficients from source PCA coefficients through:
    1. An optional residual backbone in latent space (`none`, `pls_residual`, `linear_residual`).
    2. A learned token-attention residual branch that corrects that backbone prediction.
    3. Fixed decoding back to full target edge space through the target PCA basis.
    """

    def __init__(
        self,
        base,
        n_components_pca=64,
        n_components_pls=16,
        token_embedding_type="learned",
        token_embedding_dim=None,
        learnable_pls_projection=False,
        attn_dim=16,
        value_dim=16,
        readout_type="linear",
        readout_hidden_dim=None,
        attention_activation="softmax",
        residual_mode="none",
        zscore_pca_scores=False,
        attention_dropout=0.0,
        reg=1.0e-4,
        normalize_barcodes=True,
        device=None,
        **kwargs,
    ):
        super().__init__()
        if len(getattr(base, "source_modalities", [base.source])) != 1:
            raise ValueError("LatentAttnTranslation currently supports exactly one source modality.")

        self.base = base
        self.source_modality = getattr(base, "source_modalities", [base.source])[0]
        self.target_modality = getattr(base, "target", None) or getattr(base, "target_modalities", [base.target])[0]
        self.n_components_pca = int(n_components_pca)
        self.n_components_pls = int(n_components_pls)
        self.requested_token_embedding_type = str(token_embedding_type)
        self.token_embedding_type = "learned"
        self.learnable_pls_projection = False
        self.attn_dim = int(attn_dim)
        self.value_dim = int(value_dim)
        self.readout_type = str(readout_type)
        self.attention_activation = str(attention_activation)
        self.residual_mode = str(residual_mode)
        self.zscore_pca_scores = bool(zscore_pca_scores)
        self.attention_dropout_p = float(attention_dropout)
        self.reg = float(reg)
        self.normalize_barcodes = False

        if self.readout_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown readout_type='{self.readout_type}'. Choose from {{'linear', 'mlp'}}.")
        if self.attention_activation not in {"softmax", "identity"}:
            raise ValueError(
                f"Unknown attention_activation='{self.attention_activation}'. Choose from {{'softmax', 'identity'}}."
            )
        if self.residual_mode not in {"none", "pls_residual", "linear_residual"}:
            raise ValueError(
                f"Unknown residual_mode='{self.residual_mode}'. "
                "Choose from {'none', 'pls_residual', 'linear_residual'}."
            )

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
                f"LatentAttnTranslation requested n_components_pca={self.n_components_pca}, "
                f"but train PCA scores only support min(source={max_source_k}, target={max_target_k})."
            )
        if self.residual_mode == "pls_residual" and self.n_components_pls > self.n_components_pca:
            raise ValueError(
                f"LatentAttnTranslation requested n_components_pls={self.n_components_pls}, "
                f"but only {self.n_components_pca} PCA components are retained."
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

        if self.zscore_pca_scores:
            latent_weights = np.ones(self.n_components_pca, dtype=np.float32)
        else:
            latent_variance = np.var(target_scores_k, axis=0, dtype=np.float32)
            latent_variance = np.maximum(latent_variance, 1.0e-8)
            latent_weights = latent_variance / float(np.mean(latent_variance))
        self.register_buffer("latent_loss_weights", torch.tensor(latent_weights, dtype=torch.float32, device=device))

        self.component_embedding = nn.Embedding(self.n_components_pca, self.token_embedding_dim)

        self.residual_linear = None
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

        if self.residual_mode == "linear_residual":
            self.residual_linear = nn.Linear(self.n_components_pca, self.n_components_pca, bias=True)
        elif self.residual_mode == "pls_residual":
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

        token_dim = 1 + self.token_embedding_dim
        self.W_Q = nn.Linear(token_dim, self.attn_dim, bias=False)
        self.W_K = nn.Linear(token_dim, self.attn_dim, bias=False)
        self.W_V = nn.Linear(token_dim, self.value_dim, bias=False)
        if readout_hidden_dim is None:
            readout_hidden_dim = self.value_dim
        self.readout_hidden_dim = int(readout_hidden_dim)
        if self.readout_type == "linear":
            self.readout_head = nn.Linear(self.value_dim, 1, bias=True)
        else:
            self.readout_head = nn.Sequential(
                nn.Linear(self.value_dim, self.readout_hidden_dim),
                nn.PReLU(),
                nn.Linear(self.readout_hidden_dim, self.readout_hidden_dim),
                nn.PReLU(),
                nn.Linear(self.readout_hidden_dim, 1),
            )
        self.attn_dropout = nn.Dropout(self.attention_dropout_p) if self.attention_dropout_p > 0 else nn.Identity()
        if self.residual_mode == "none":
            self.residual_gain = None
        else:
            self.residual_gain = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
            if self.readout_type == "linear":
                nn.init.zeros_(self.readout_head.weight)
                nn.init.zeros_(self.readout_head.bias)
            else:
                final_layer = self.readout_head[-1]
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)

        self.last_attention = None
        self.last_latent_pred = None
        self.last_residual_base = None
        self.last_latent_delta = None

        print(
            f"LatentAttnTranslation init | src={self.source_modality} tgt={self.target_modality} "
            f"| d_source={d_source} d_target={d_target} | k={self.n_components_pca} "
            f"| token_embedding_type={self.token_embedding_type} token_embedding_dim={self.token_embedding_dim} "
            f"| residual_mode={self.residual_mode} "
            f"| attn_dim={self.attn_dim} value_dim={self.value_dim} "
            f"| readout_type={self.readout_type} readout_hidden_dim={self.readout_hidden_dim} "
            f"| attention_activation={self.attention_activation} "
            f"| zscore_pca_scores={self.zscore_pca_scores}",
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

    def _get_component_embeddings(self, batch_size):
        idx = torch.arange(self.n_components_pca, device=self.source_mean.device)
        emb = self.component_embedding(idx)
        return emb.unsqueeze(0).expand(batch_size, -1, -1)

    def _build_tokens(self, c_source):
        batch_size = c_source.shape[0]
        emb_batch = self._get_component_embeddings(batch_size)
        return torch.cat([c_source.unsqueeze(-1), emb_batch], dim=-1)

    def _compute_residual_base(self, c_source):
        if self.residual_mode == "none":
            return torch.zeros_like(c_source)
        if self.residual_mode == "pls_residual":
            return torch.matmul(c_source, self.pls_residual_coef) + self.pls_residual_intercept
        return self.residual_linear(c_source)

    def predict_backbone_latents(self, x):
        c_source = self.encode_source_latents(x)
        return self._compute_residual_base(c_source)

    def predict_target_latents(self, x, return_attention=False):
        c_source = self.encode_source_latents(x)
        tokens = self._build_tokens(c_source)
        c_target_base = self._compute_residual_base(c_source)

        Q = self.W_Q(tokens)
        K = self.W_K(tokens)
        V = self.W_V(tokens)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        if self.attention_activation == "softmax":
            attn = torch.softmax(attn_logits, dim=-1)
        else:
            attn = attn_logits
        attn = self.attn_dropout(attn)
        Z = torch.matmul(attn, V)
        c_target_delta = self.readout_head(Z).squeeze(-1)
        if self.residual_gain is not None:
            c_target_delta = self.residual_gain * c_target_delta
        c_target_hat = c_target_base + c_target_delta

        self.last_attention = attn.detach()
        self.last_latent_pred = c_target_hat.detach()
        self.last_residual_base = c_target_base.detach()
        self.last_latent_delta = c_target_delta.detach()
        if return_attention:
            return c_target_hat, attn, Z, c_target_base, c_target_delta
        return c_target_hat

    def decode_target_latents(self, c_target_hat):
        if self.zscore_pca_scores:
            c_target_hat = c_target_hat * self.target_score_std + self.target_score_mean
        return torch.matmul(c_target_hat, self.target_loadings_k.t()) + self.target_mean

    def forward(self, x):
        c_target_hat = self.predict_target_latents(x, return_attention=False)
        return self.decode_target_latents(c_target_hat)

    def forward_backbone_only(self, x):
        return self.decode_target_latents(self.predict_backbone_latents(x))

    def get_reg_loss(self):
        if self.reg <= 0:
            return 0.0
        params = [self.W_Q.weight, self.W_K.weight, self.W_V.weight, self.component_embedding.weight]
        params.extend([p for p in self.readout_head.parameters() if p.requires_grad])
        if self.residual_linear is not None:
            params.extend([p for p in self.residual_linear.parameters() if p.requires_grad])
        return compute_reg_loss(params, l1_l2_tuple=(0.0, self.reg))
