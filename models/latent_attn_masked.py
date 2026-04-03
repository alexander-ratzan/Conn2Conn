import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression

from models.loss import compute_latent_reconstruction_loss
from models.models import compute_reg_loss, get_modality_data


class LatentAttnMasked(nn.Module):
    """
    Joint SC/FC latent attention model with masked FC reconstruction.

    The model predicts target PCA coefficients through:
    1. An optional latent-space backbone (`none`, `pls_residual`, `linear_residual`).
    2. A masked FC attention branch that predicts a residual correction.
    3. Fixed decoding back to full target edge space through the target PCA basis.
    """

    def __init__(
        self,
        base,
        n_components_pca=64,
        n_components_pls=16,
        token_embedding_type="learned",
        token_embedding_dim=None,
        attn_dim=16,
        value_dim=16,
        readout_type="linear",
        readout_hidden_dim=None,
        attention_activation="softmax",
        residual_mode="none",
        residual_gain_init=1.0e-3,
        zscore_pca_scores=False,
        attention_dropout=0.0,
        sc_token_dropout=0.0,
        reg=1.0e-4,
        normalize_barcodes=True,
        mask_ratio=0.5,
        min_masked_components=1,
        train_use_visible_fc_context=True,
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
        self.requested_token_embedding_type = str(token_embedding_type)
        self.token_embedding_type = "learned"
        self.attn_dim = int(attn_dim)
        self.value_dim = int(value_dim)
        self.readout_type = str(readout_type)
        self.attention_activation = str(attention_activation)
        self.residual_mode = str(residual_mode)
        self.residual_gain_init = float(residual_gain_init)
        self.zscore_pca_scores = bool(zscore_pca_scores)
        self.attention_dropout_p = float(attention_dropout)
        self.sc_token_dropout_p = float(sc_token_dropout)
        self.reg = float(reg)
        self.normalize_barcodes = False
        self.mask_ratio = float(mask_ratio)
        self.min_masked_components = int(min_masked_components)
        self.train_use_visible_fc_context = bool(train_use_visible_fc_context)

        if self.readout_type not in {"linear", "mlp", "concat_mlp", "scalar_slot"}:
            raise ValueError(
                f"Unknown readout_type='{self.readout_type}'. Choose from {'linear', 'mlp', 'concat_mlp', 'scalar_slot'}."
            )
        if self.attention_activation not in {"softmax", "identity"}:
            raise ValueError(
                f"Unknown attention_activation='{self.attention_activation}'. "
                "Choose from {'softmax', 'identity'}."
            )
        if self.residual_mode not in {"none", "pls_residual", "linear_residual"}:
            raise ValueError(
                f"Unknown residual_mode='{self.residual_mode}'. "
                "Choose from {'none', 'pls_residual', 'linear_residual'}."
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

        max_source_k = source_scores.shape[1]
        max_target_k = target_scores.shape[1]
        if self.n_components_pca > min(max_source_k, max_target_k):
            raise ValueError(
                f"LatentAttnMasked requested n_components_pca={self.n_components_pca}, "
                f"but train PCA scores only support min(source={max_source_k}, target={max_target_k})."
            )
        if self.residual_mode == "pls_residual" and self.n_components_pls > self.n_components_pca:
            raise ValueError(
                f"LatentAttnMasked requested n_components_pls={self.n_components_pls}, "
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

        self.residual_linear = None
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

        self.sc_component_embedding = nn.Embedding(self.n_components_pca, self.token_embedding_dim)
        self.fc_component_embedding = nn.Embedding(self.n_components_pca, self.token_embedding_dim)

        token_dim = 1 + self.token_embedding_dim
        self.W_Q = nn.Linear(token_dim, self.attn_dim, bias=False)
        self.W_K = nn.Linear(token_dim, self.attn_dim, bias=False)
        self.W_V = None if self.readout_type == "scalar_slot" else nn.Linear(token_dim, self.value_dim, bias=False)
        if readout_hidden_dim is None:
            readout_hidden_dim = self.value_dim
        self.readout_hidden_dim = int(readout_hidden_dim)
        if self.readout_type == "scalar_slot":
            self.readout_head = None
        elif self.readout_type == "linear":
            self.readout_head = nn.Linear(self.value_dim, 1, bias=True)
        elif self.readout_type == "mlp":
            self.readout_head = nn.Sequential(
                nn.Linear(self.value_dim, self.readout_hidden_dim),
                nn.PReLU(),
                nn.Linear(self.readout_hidden_dim, self.readout_hidden_dim),
                nn.PReLU(),
                nn.Linear(self.readout_hidden_dim, 1),
            )
        else:
            self.readout_head = nn.Sequential(
                nn.Linear(token_dim + self.value_dim, self.readout_hidden_dim),
                nn.PReLU(),
                nn.Linear(self.readout_hidden_dim, self.readout_hidden_dim),
                nn.PReLU(),
                nn.Linear(self.readout_hidden_dim, 1),
            )
        self.attn_dropout = nn.Dropout(self.attention_dropout_p) if self.attention_dropout_p > 0 else nn.Identity()
        self.sc_token_dropout = nn.Dropout(self.sc_token_dropout_p) if self.sc_token_dropout_p > 0 else nn.Identity()
        self.residual_gain = None if self.residual_mode == "none" else nn.Parameter(
            torch.tensor([self.residual_gain_init], dtype=torch.float32, device=device)
        )

        self.fc_mask_value = nn.Parameter(torch.zeros(1))

        self.last_attention = None
        self.last_latent_pred = None
        self.last_fc_mask = None
        self.last_residual_base = None
        self.last_latent_delta = None

        print(
            f"LatentAttnMasked init | src={self.source_modality} tgt={self.target_modality} "
            f"| k={self.n_components_pca} pls={self.n_components_pls} "
            f"| token_embedding_type={self.token_embedding_type} token_embedding_dim={self.token_embedding_dim} "
            f"| residual_mode={self.residual_mode} residual_gain_init={self.residual_gain_init} "
            f"| attn_dim={self.attn_dim} value_dim={self.value_dim} "
            f"| readout_type={self.readout_type} readout_hidden_dim={self.readout_hidden_dim} "
            f"| attention_activation={self.attention_activation} "
            f"| zscore_pca_scores={self.zscore_pca_scores} "
            f"| mask_ratio={self.mask_ratio} visible_fc_context={self.train_use_visible_fc_context} "
            f"| sc_token_dropout={self.sc_token_dropout_p}",
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
        if self.residual_mode == "none":
            return torch.zeros_like(c_source)
        if self.residual_mode == "pls_residual":
            return torch.matmul(c_source, self.pls_residual_coef) + self.pls_residual_intercept
        return self.residual_linear(c_source)

    def predict_backbone_latents(self, x):
        c_source = self.encode_source_latents(x)
        return self._compute_residual_base(c_source)

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
        sc_emb = self.sc_component_embedding(idx)
        fc_emb = self.fc_component_embedding(idx)
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

    def _run_attention(self, tokens):
        Q = self.W_Q(tokens)
        K = self.W_K(tokens)
        V = tokens if self.W_V is None else self.W_V(tokens)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        if self.attention_activation == "softmax":
            attn = torch.softmax(attn_logits, dim=-1)
        else:
            attn = attn_logits
        attn = self.attn_dropout(attn)
        Z = torch.matmul(attn, V)
        return Z, attn

    def predict_target_latents(self, x, y=None, return_attention=False, return_mask=False, force_all_masked=None):
        c_source = self.encode_source_latents(x)
        c_target_true = None if y is None else self.encode_target_latents(y)
        c_target_base = self._compute_residual_base(c_source)

        if force_all_masked is None:
            force_all_masked = (y is None)

        if force_all_masked:
            fc_mask = torch.ones(c_source.shape[0], self.n_components_pca, dtype=torch.bool, device=c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=None, fc_mask=fc_mask)
        else:
            fc_mask = self._sample_fc_mask(c_source.shape[0], c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=c_target_true, fc_mask=fc_mask)

        Z, attn = self._run_attention(tokens)
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

        self.last_attention = attn.detach()
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

        if force_all_masked is None:
            force_all_masked = (y is None)

        if force_all_masked:
            fc_mask = torch.ones(c_source.shape[0], self.n_components_pca, dtype=torch.bool, device=c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=None, fc_mask=fc_mask)
        else:
            fc_mask = self._sample_fc_mask(c_source.shape[0], c_source.device)
            tokens, fc_mask = self._build_joint_tokens(c_source, c_target_true=c_target_true, fc_mask=fc_mask)

        token_context, attn = self._run_attention(tokens)
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
        params = [self.W_Q.weight, self.W_K.weight, self.sc_component_embedding.weight, self.fc_component_embedding.weight]
        if self.W_V is not None:
            params.append(self.W_V.weight)
        if self.readout_head is not None:
            params.extend([p for p in self.readout_head.parameters() if p.requires_grad])
        if self.residual_linear is not None:
            params.extend([p for p in self.residual_linear.parameters() if p.requires_grad])
        return compute_reg_loss(params, l1_l2_tuple=(0.0, self.reg))
