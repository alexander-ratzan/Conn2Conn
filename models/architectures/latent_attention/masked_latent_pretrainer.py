"""SSL pretrainer for joint SC/FC PCA-latent reconstruction.

Masks PCA components independently per modality and reconstructs the held-out
coefficients from the remaining visible SC+FC context. Shares the tokenization,
attention stack, and per-modality component embeddings with LatentAttnMasked,
but drops the residual backbone and FC-only readout in favor of symmetric
SC/FC reconstruction heads.

Transfer to LatentAttnMasked: see `export_to_latent_attn_masked`.
"""
import numpy as np
import torch
import torch.nn as nn

from models.train.loss import compute_latent_reconstruction_loss
from models.architectures.utils import compute_reg_loss, get_modality_data
from models.architectures.latent_attention.latent_attn_masked import (
    MultiHeadTokenSelfAttention,
    TransformerTokenBlock,
)


class MaskedLatentPretrainer(nn.Module):
    """
    Joint SC/FC masked-PCA reconstruction pretrainer.

    Token layout (per subject, left-to-right):
        [CLS?, SC_0, ..., SC_{k-1}, FC_0, ..., FC_{k-1}]
    where CLS is present iff `use_covariates_cls=True` and is initialized by a
    cov projector. SC/FC tokens are `[scalar_slot || component_embedding]` with
    separate embedding tables per modality. Masked SC/FC scalar slots are
    replaced by `sc_mask_value` / `fc_mask_value` learned constants.

    Supervision: latent MSE on masked positions, summed across both modalities.
    """

    def __init__(
        self,
        base,
        n_components_pca=64,
        token_embedding_dim=16,
        attn_dim=16,
        value_dim=16,
        transformer_layers=0,
        num_heads=1,
        readout_type="linear",
        readout_hidden_dim=None,
        attention_activation="softmax",
        zscore_pca_scores=True,
        attention_dropout=0.0,
        reg=1.0e-4,
        sc_mask_ratio=0.5,
        fc_mask_ratio=0.5,
        min_masked_components_per_modality=1,
        use_covariates_cls=False,
        cov_projector_hidden_dim=None,
        loss_weighting="per_modality_mean",
        device=None,
        **kwargs,
    ):
        super().__init__()
        if len(getattr(base, "source_modalities", [base.source])) != 1:
            raise ValueError("MaskedLatentPretrainer currently supports exactly one source modality.")

        self.base = base
        self.source_modality = getattr(base, "source_modalities", [base.source])[0]
        self.target_modality = getattr(base, "target", None) or getattr(base, "target_modalities", [base.target])[0]
        self.n_components_pca = int(n_components_pca)
        self.token_embedding_dim = int(token_embedding_dim)
        self.attn_dim = int(attn_dim)
        self.value_dim = int(value_dim)
        self.transformer_layers = int(transformer_layers)
        self.num_heads = int(num_heads)
        self.readout_type = str(readout_type)
        self.attention_activation = str(attention_activation)
        self.zscore_pca_scores = bool(zscore_pca_scores)
        self.attention_dropout_p = float(attention_dropout)
        self.reg = float(reg)
        self.sc_mask_ratio = float(sc_mask_ratio)
        self.fc_mask_ratio = float(fc_mask_ratio)
        self.min_masked_components = int(min_masked_components_per_modality)
        self.use_covariates_cls = bool(use_covariates_cls)
        self.cov_projector_hidden_dim = cov_projector_hidden_dim
        self.loss_weighting = str(loss_weighting)
        self.uses_cov = self.use_covariates_cls

        if self.readout_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown readout_type='{self.readout_type}'. Choose from {{'linear', 'mlp'}}.")
        if self.attention_activation not in {"softmax", "identity"}:
            raise ValueError(f"Unknown attention_activation='{self.attention_activation}'.")
        if not (0.0 < self.sc_mask_ratio <= 1.0):
            raise ValueError(f"sc_mask_ratio must be in (0, 1], got {self.sc_mask_ratio}.")
        if not (0.0 < self.fc_mask_ratio <= 1.0):
            raise ValueError(f"fc_mask_ratio must be in (0, 1], got {self.fc_mask_ratio}.")
        if self.loss_weighting not in {"per_modality_mean", "sum"}:
            raise ValueError(f"Unknown loss_weighting='{self.loss_weighting}'.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        data = get_modality_data(base, device=device, include_scores=True)
        src = data["sources"][self.source_modality]
        tgt = data["target"]
        max_src_k = src["scores"].shape[1]
        max_tgt_k = tgt["scores"].shape[1]
        if self.n_components_pca > min(max_src_k, max_tgt_k):
            raise ValueError(
                f"n_components_pca={self.n_components_pca} exceeds min(src={max_src_k}, tgt={max_tgt_k})."
            )

        src_scores_k = np.asarray(src["scores"][:, : self.n_components_pca], dtype=np.float32)
        tgt_scores_k = np.asarray(tgt["scores"][:, : self.n_components_pca], dtype=np.float32)
        src_score_mean = src_scores_k.mean(axis=0, dtype=np.float32)
        tgt_score_mean = tgt_scores_k.mean(axis=0, dtype=np.float32)
        src_score_std = np.maximum(src_scores_k.std(axis=0, dtype=np.float32), 1.0e-8)
        tgt_score_std = np.maximum(tgt_scores_k.std(axis=0, dtype=np.float32), 1.0e-8)

        self.register_buffer("source_mean", torch.tensor(src["mean"], dtype=torch.float32, device=device))
        self.register_buffer(
            "source_loadings_k",
            torch.tensor(src["loadings"][:, : self.n_components_pca], dtype=torch.float32, device=device),
        )
        self.register_buffer("target_mean", torch.tensor(tgt["mean"], dtype=torch.float32, device=device))
        self.register_buffer(
            "target_loadings_k",
            torch.tensor(tgt["loadings"][:, : self.n_components_pca], dtype=torch.float32, device=device),
        )
        self.register_buffer("source_score_mean", torch.tensor(src_score_mean, dtype=torch.float32, device=device))
        self.register_buffer("source_score_std", torch.tensor(src_score_std, dtype=torch.float32, device=device))
        self.register_buffer("target_score_mean", torch.tensor(tgt_score_mean, dtype=torch.float32, device=device))
        self.register_buffer("target_score_std", torch.tensor(tgt_score_std, dtype=torch.float32, device=device))

        if self.zscore_pca_scores:
            sc_weights = np.ones(self.n_components_pca, dtype=np.float32)
            fc_weights = np.ones(self.n_components_pca, dtype=np.float32)
        else:
            sc_var = np.maximum(np.var(src_scores_k, axis=0, dtype=np.float32), 1.0e-8)
            fc_var = np.maximum(np.var(tgt_scores_k, axis=0, dtype=np.float32), 1.0e-8)
            sc_weights = sc_var / float(np.mean(sc_var))
            fc_weights = fc_var / float(np.mean(fc_var))
        self.register_buffer("sc_latent_weights", torch.tensor(sc_weights, dtype=torch.float32, device=device))
        self.register_buffer("fc_latent_weights", torch.tensor(fc_weights, dtype=torch.float32, device=device))

        self.sc_component_embedding = nn.Embedding(self.n_components_pca, self.token_embedding_dim)
        self.fc_component_embedding = nn.Embedding(self.n_components_pca, self.token_embedding_dim)

        self.sc_mask_value = nn.Parameter(torch.zeros(1))
        self.fc_mask_value = nn.Parameter(torch.zeros(1))

        self.token_dim = 1 + self.token_embedding_dim
        self.transformer_blocks = None
        self.raw_attention = None
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
            readout_in_dim = self.token_dim
        else:
            self.raw_attention = MultiHeadTokenSelfAttention(
                input_dim=self.token_dim,
                attn_dim=self.attn_dim,
                value_dim=self.value_dim,
                num_heads=self.num_heads,
                attention_activation=self.attention_activation,
                dropout=self.attention_dropout_p,
            )
            self.transformer_output_norm = None
            readout_in_dim = self.value_dim
        self.readout_in_dim = readout_in_dim

        if readout_hidden_dim is None:
            readout_hidden_dim = readout_in_dim
        self.readout_hidden_dim = int(readout_hidden_dim)
        self.sc_readout_head = self._build_readout(readout_in_dim, self.readout_hidden_dim, self.readout_type)
        self.fc_readout_head = self._build_readout(readout_in_dim, self.readout_hidden_dim, self.readout_type)

        self.cov_projector = None
        self._cov_dim = None

        self.last_attention = None
        self.last_reconstructions = None

        print(
            f"MaskedLatentPretrainer init | src={self.source_modality} tgt={self.target_modality} "
            f"| k={self.n_components_pca} token_dim={self.token_dim} "
            f"| attn_dim={self.attn_dim} value_dim={self.value_dim} num_heads={self.num_heads} "
            f"| transformer_layers={self.transformer_layers} readout={self.readout_type} "
            f"| zscore={self.zscore_pca_scores} sc_p={self.sc_mask_ratio} fc_p={self.fc_mask_ratio} "
            f"| use_cov_cls={self.use_covariates_cls}",
            flush=True,
        )
        self.to(device)

    @staticmethod
    def _build_readout(in_dim, hidden_dim, readout_type):
        if readout_type == "linear":
            return nn.Linear(in_dim, 1, bias=True)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _init_cov_projector(self, cov_dim, device):
        if self.cov_projector_hidden_dim is None:
            self.cov_projector = nn.Linear(int(cov_dim), self.token_dim, bias=True).to(device)
        else:
            hidden = int(self.cov_projector_hidden_dim)
            self.cov_projector = nn.Sequential(
                nn.Linear(int(cov_dim), hidden),
                nn.GELU(),
                nn.Linear(hidden, self.token_dim),
            ).to(device)
        self._cov_dim = int(cov_dim)

    def encode_source_latents(self, x):
        x = x.to(self.source_mean.device).to(torch.float32)
        c = torch.matmul(x - self.source_mean, self.source_loadings_k)
        if self.zscore_pca_scores:
            c = (c - self.source_score_mean) / self.source_score_std
        return c

    def encode_target_latents(self, y):
        y = y.to(self.target_mean.device).to(torch.float32)
        c = torch.matmul(y - self.target_mean, self.target_loadings_k)
        if self.zscore_pca_scores:
            c = (c - self.target_score_mean) / self.target_score_std
        return c

    def decode_target_latents(self, c_target_hat):
        if self.zscore_pca_scores:
            c_target_hat = c_target_hat * self.target_score_std + self.target_score_mean
        return torch.matmul(c_target_hat, self.target_loadings_k.t()) + self.target_mean

    def decode_source_latents(self, c_source_hat):
        if self.zscore_pca_scores:
            c_source_hat = c_source_hat * self.source_score_std + self.source_score_mean
        return torch.matmul(c_source_hat, self.source_loadings_k.t()) + self.source_mean

    def _sample_modality_mask(self, batch_size, mask_ratio, device):
        mask = torch.rand(batch_size, self.n_components_pca, device=device) < mask_ratio
        min_mask = min(self.min_masked_components, self.n_components_pca)
        if min_mask > 0:
            counts = mask.sum(dim=1)
            needs_fix = counts < min_mask
            if needs_fix.any():
                for row_idx in torch.nonzero(needs_fix, as_tuple=False).flatten():
                    perm = torch.randperm(self.n_components_pca, device=device)
                    mask[row_idx, perm[:min_mask]] = True
        return mask

    def _sample_joint_mask(self, batch_size, device):
        sc_mask = self._sample_modality_mask(batch_size, self.sc_mask_ratio, device)
        fc_mask = self._sample_modality_mask(batch_size, self.fc_mask_ratio, device)
        return sc_mask, fc_mask

    def _build_pretrain_tokens(self, c_source, c_target, sc_mask, fc_mask, cov=None):
        batch_size = c_source.shape[0]
        device = c_source.device
        idx = torch.arange(self.n_components_pca, device=device)
        sc_emb = self.sc_component_embedding(idx).unsqueeze(0).expand(batch_size, -1, -1)
        fc_emb = self.fc_component_embedding(idx).unsqueeze(0).expand(batch_size, -1, -1)

        sc_fill = self.sc_mask_value.view(1, 1).expand_as(c_source)
        fc_fill = self.fc_mask_value.view(1, 1).expand_as(c_target)
        sc_scalar = torch.where(sc_mask, sc_fill, c_source)
        fc_scalar = torch.where(fc_mask, fc_fill, c_target)

        sc_tokens = torch.cat([sc_scalar.unsqueeze(-1), sc_emb], dim=-1)
        fc_tokens = torch.cat([fc_scalar.unsqueeze(-1), fc_emb], dim=-1)
        tokens = torch.cat([sc_tokens, fc_tokens], dim=1)

        if self.use_covariates_cls:
            if cov is None:
                raise ValueError("use_covariates_cls=True requires 'cov' in the forward call / batch.")
            cov = cov.to(device).to(torch.float32)
            if self.cov_projector is None:
                self._init_cov_projector(cov.shape[-1], device=device)
            cls = self.cov_projector(cov).unsqueeze(1)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens

    def _run_attention_stack(self, tokens):
        if self.transformer_layers > 0:
            attn = None
            hidden = tokens
            for block in self.transformer_blocks:
                hidden, attn = block(hidden)
            hidden = self.transformer_output_norm(hidden)
            return hidden, attn
        return self.raw_attention(tokens)

    def predict_reconstructions(self, x, y, cov=None, joint_mask=None):
        c_source = self.encode_source_latents(x)
        c_target = self.encode_target_latents(y)
        batch_size = c_source.shape[0]
        device = c_source.device

        if joint_mask is None:
            sc_mask, fc_mask = self._sample_joint_mask(batch_size, device)
        else:
            sc_mask, fc_mask = joint_mask
            sc_mask = sc_mask.to(device=device, dtype=torch.bool)
            fc_mask = fc_mask.to(device=device, dtype=torch.bool)

        tokens = self._build_pretrain_tokens(c_source, c_target, sc_mask, fc_mask, cov=cov)
        Z, attn = self._run_attention_stack(tokens)

        offset = 1 if self.use_covariates_cls else 0
        sc_out = Z[:, offset : offset + self.n_components_pca, :]
        fc_out = Z[:, offset + self.n_components_pca :, :]
        c_source_hat = self.sc_readout_head(sc_out).squeeze(-1)
        c_target_hat = self.fc_readout_head(fc_out).squeeze(-1)

        self.last_attention = attn.detach() if attn is not None else None
        self.last_reconstructions = {
            "c_source_true": c_source.detach(),
            "c_target_true": c_target.detach(),
            "c_source_hat": c_source_hat.detach(),
            "c_target_hat": c_target_hat.detach(),
            "sc_mask": sc_mask.detach(),
            "fc_mask": fc_mask.detach(),
        }
        return c_source_hat, c_target_hat, c_source, c_target, sc_mask, fc_mask

    def compute_latent_loss(self, batch, loss_type):
        if loss_type not in {"latent_mse", "latent_weighted_mse"}:
            raise ValueError(
                f"MaskedLatentPretrainer supports loss_type in {{'latent_mse', 'latent_weighted_mse'}}, got '{loss_type}'."
            )
        x = batch["x"] if "x" in batch else batch["x_modalities"]
        y = batch["y"]
        cov = batch.get("cov") if self.use_covariates_cls else None
        c_source_hat, c_target_hat, c_source, c_target, sc_mask, fc_mask = self.predict_reconstructions(
            x, y, cov=cov, joint_mask=None
        )
        sc_weights = self.sc_latent_weights if loss_type == "latent_weighted_mse" else None
        fc_weights = self.fc_latent_weights if loss_type == "latent_weighted_mse" else None
        sc_loss = compute_latent_reconstruction_loss(c_source_hat, c_source, loss_type, weights=sc_weights, mask=sc_mask)
        fc_loss = compute_latent_reconstruction_loss(c_target_hat, c_target, loss_type, weights=fc_weights, mask=fc_mask)
        if self.loss_weighting == "per_modality_mean":
            return 0.5 * (sc_loss + fc_loss)
        return sc_loss + fc_loss

    def forward(self, x, cov=None):
        """
        Returns decoded FC edge-space reconstruction under the "downstream" mask
        pattern: all FC components held out, all SC components visible.

        This exists to satisfy the Lightning training loop's edge-space aux
        metrics (pearson_r, demeaned_r) and gives a consistent diagnostic of
        the model's FC-generation quality over epochs. The pretraining
        objective itself is computed via `compute_latent_loss`; use
        `forward_full` or `reconstruct` for full introspection under random
        joint masks.
        """
        c_source = self.encode_source_latents(x)
        batch_size = c_source.shape[0]
        device = c_source.device
        c_target_placeholder = torch.zeros(batch_size, self.n_components_pca, device=device, dtype=torch.float32)
        sc_mask = torch.zeros(batch_size, self.n_components_pca, dtype=torch.bool, device=device)
        fc_mask = torch.ones(batch_size, self.n_components_pca, dtype=torch.bool, device=device)
        tokens = self._build_pretrain_tokens(c_source, c_target_placeholder, sc_mask, fc_mask, cov=cov)
        Z, attn = self._run_attention_stack(tokens)
        offset = 1 if self.use_covariates_cls else 0
        fc_out = Z[:, offset + self.n_components_pca :, :]
        c_target_hat = self.fc_readout_head(fc_out).squeeze(-1)
        self.last_attention = attn.detach() if attn is not None else None
        return self.decode_target_latents(c_target_hat)

    def forward_full(self, x, y, cov=None, joint_mask=None):
        """Full dict output with both modality reconstructions and masks."""
        c_source_hat, c_target_hat, c_source, c_target, sc_mask, fc_mask = self.predict_reconstructions(
            x, y, cov=cov, joint_mask=joint_mask
        )
        return {
            "c_source": c_source,
            "c_target": c_target,
            "c_source_hat": c_source_hat,
            "c_target_hat": c_target_hat,
            "sc_recon_edges": self.decode_source_latents(c_source_hat),
            "fc_recon_edges": self.decode_target_latents(c_target_hat),
            "sc_mask": sc_mask,
            "fc_mask": fc_mask,
            "attention": self.last_attention,
        }

    def reconstruct(self, batch, joint_mask=None):
        """Notebook-friendly inspection helper. Returns the full state dict."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = batch["x"] if "x" in batch else batch["x_modalities"]
            y = batch["y"]
            cov = batch.get("cov") if self.use_covariates_cls else None
            out = self.predict_reconstructions(x, y, cov=cov, joint_mask=joint_mask)
        if was_training:
            self.train()
        c_source_hat, c_target_hat, c_source, c_target, sc_mask, fc_mask = out
        return {
            "c_source": c_source,
            "c_target": c_target,
            "c_source_hat": c_source_hat,
            "c_target_hat": c_target_hat,
            "sc_mask": sc_mask,
            "fc_mask": fc_mask,
            "attention": self.last_attention,
        }

    def per_component_reconstruction_error(self, batch, joint_mask=None, squared=True):
        """(sc_err, fc_err) each shape (k,): mean error per component on masked positions only."""
        out = self.reconstruct(batch, joint_mask=joint_mask)
        sc_diff = (out["c_source_hat"] - out["c_source"]).to(torch.float32)
        fc_diff = (out["c_target_hat"] - out["c_target"]).to(torch.float32)
        if squared:
            sc_diff = sc_diff.pow(2)
            fc_diff = fc_diff.pow(2)
        else:
            sc_diff = sc_diff.abs()
            fc_diff = fc_diff.abs()
        sc_mask_f = out["sc_mask"].to(torch.float32)
        fc_mask_f = out["fc_mask"].to(torch.float32)
        sc_err = (sc_diff * sc_mask_f).sum(dim=0) / sc_mask_f.sum(dim=0).clamp_min(1.0)
        fc_err = (fc_diff * fc_mask_f).sum(dim=0) / fc_mask_f.sum(dim=0).clamp_min(1.0)
        return sc_err, fc_err

    def get_reg_loss(self):
        if self.reg <= 0:
            return 0.0
        params = [self.sc_component_embedding.weight, self.fc_component_embedding.weight]
        if self.raw_attention is not None:
            params.extend([self.raw_attention.W_Q.weight, self.raw_attention.W_K.weight])
            if self.raw_attention.W_V is not None:
                params.append(self.raw_attention.W_V.weight)
        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                params.extend([p for p in block.parameters() if p.requires_grad])
            params.extend([p for p in self.transformer_output_norm.parameters() if p.requires_grad])
        params.extend([p for p in self.sc_readout_head.parameters() if p.requires_grad])
        params.extend([p for p in self.fc_readout_head.parameters() if p.requires_grad])
        if self.cov_projector is not None:
            params.extend([p for p in self.cov_projector.parameters() if p.requires_grad])
        return compute_reg_loss(params, l1_l2_tuple=(0.0, self.reg))

    def export_to_latent_attn_masked(self, downstream_model):
        """
        Copy compatible weights into a fresh LatentAttnMasked instance.

        Transferred (if shapes match):
        - sc_component_embedding.weight, fc_component_embedding.weight
        - raw_attention.W_Q/W_K/W_V OR transformer_blocks
        - fc_mask_value
        - fc_readout_head (only if downstream readout_type is compatible)

        Returns a dict of keys that were actually copied.
        """
        copied = []
        skipped = []

        def _try_copy(name, src, dst):
            if src is None or dst is None:
                skipped.append((name, "missing on one side"))
                return
            if src.shape != dst.shape:
                skipped.append((name, f"shape {tuple(src.shape)} != {tuple(dst.shape)}"))
                return
            with torch.no_grad():
                dst.copy_(src.to(dst.device))
            copied.append(name)

        _try_copy(
            "sc_component_embedding",
            getattr(self.sc_component_embedding, "weight", None),
            getattr(getattr(downstream_model, "sc_component_embedding", None), "weight", None),
        )
        _try_copy(
            "fc_component_embedding",
            getattr(self.fc_component_embedding, "weight", None),
            getattr(getattr(downstream_model, "fc_component_embedding", None), "weight", None),
        )
        _try_copy("fc_mask_value", self.fc_mask_value.data, downstream_model.fc_mask_value.data)

        if self.raw_attention is not None and getattr(downstream_model, "raw_attention", None) is not None:
            _try_copy("raw_attention.W_Q", self.raw_attention.W_Q.weight, downstream_model.raw_attention.W_Q.weight)
            _try_copy("raw_attention.W_K", self.raw_attention.W_K.weight, downstream_model.raw_attention.W_K.weight)
            if self.raw_attention.W_V is not None and downstream_model.raw_attention.W_V is not None:
                _try_copy("raw_attention.W_V", self.raw_attention.W_V.weight, downstream_model.raw_attention.W_V.weight)

        if self.transformer_blocks is not None and getattr(downstream_model, "transformer_blocks", None) is not None:
            if len(self.transformer_blocks) == len(downstream_model.transformer_blocks):
                for i, (src_block, dst_block) in enumerate(zip(self.transformer_blocks, downstream_model.transformer_blocks)):
                    src_sd = src_block.state_dict()
                    dst_sd = dst_block.state_dict()
                    for key in src_sd:
                        if key in dst_sd and src_sd[key].shape == dst_sd[key].shape:
                            with torch.no_grad():
                                dst_sd[key].copy_(src_sd[key].to(dst_sd[key].device))
                            copied.append(f"transformer_blocks.{i}.{key}")
                        else:
                            skipped.append((f"transformer_blocks.{i}.{key}", "shape mismatch"))

        # FC readout head: only copy if shapes match perfectly (downstream may use concat_mlp/scalar_slot).
        if hasattr(downstream_model, "readout_head") and downstream_model.readout_head is not None:
            try:
                src_sd = self.fc_readout_head.state_dict()
                dst_sd = downstream_model.readout_head.state_dict()
                shape_ok = all(k in dst_sd and src_sd[k].shape == dst_sd[k].shape for k in src_sd)
                if shape_ok and set(src_sd.keys()) == set(dst_sd.keys()):
                    for k in src_sd:
                        with torch.no_grad():
                            dst_sd[k].copy_(src_sd[k].to(dst_sd[k].device))
                        copied.append(f"fc_readout_head.{k}")
                else:
                    skipped.append(("fc_readout_head", "incompatible readout head structure"))
            except Exception as e:
                skipped.append(("fc_readout_head", f"copy failed: {e}"))

        return {"copied": copied, "skipped": skipped}
