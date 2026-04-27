"""Drop-in MLP baseline for MaskedLatentPretrainer.

Same contract (SC+FC PCA scalars, masked positions replaced by learned
sc_mask_value/fc_mask_value, masked-MSE objective, forward(x) zero-shot
SC->FC probe) but the attention stack is replaced by a flat MLP encoder
and two per-modality heads. Intended as a diagnostic baseline to isolate
whether the 0.5 masked-MSE plateau on the attention path is a capacity
issue or the PCA-decorrelation structural limit.

Key toggles:
- `nonlinear=True`: MLP encoder with PReLU + Dropout.
- `nonlinear=False`: pure linear encoder. If `low_rank_dim` is set, the
  linear map is factorized as two linear layers with a bottleneck of that
  rank (no bias on the first); composition has rank <= low_rank_dim.
"""
import numpy as np
import torch
import torch.nn as nn

from models.train.loss import compute_latent_reconstruction_loss
from models.architectures.utils import compute_reg_loss, get_modality_data


class MaskedMLPPretrainer(nn.Module):
    def __init__(
        self,
        base,
        n_components_pca=64,
        hidden_dim=128,
        num_hidden_layers=1,
        dropout=0.0,
        nonlinear=True,
        low_rank_dim=None,
        readout_type="linear",
        readout_hidden_dim=None,
        zscore_pca_scores=True,
        reg=1.0e-4,
        sc_mask_ratio=0.5,
        fc_mask_ratio=0.5,
        min_masked_components_per_modality=1,
        use_covariates=False,
        cov_projector_hidden_dim=None,
        loss_weighting="per_modality_mean",
        device=None,
        **kwargs,
    ):
        super().__init__()
        if len(getattr(base, "source_modalities", [base.source])) != 1:
            raise ValueError("MaskedMLPPretrainer currently supports exactly one source modality.")

        self.base = base
        self.source_modality = getattr(base, "source_modalities", [base.source])[0]
        self.target_modality = getattr(base, "target", None) or getattr(base, "target_modalities", [base.target])[0]
        self.n_components_pca = int(n_components_pca)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.dropout_p = float(dropout)
        self.nonlinear = bool(nonlinear)
        self.low_rank_dim = int(low_rank_dim) if low_rank_dim is not None else None
        self.readout_type = str(readout_type)
        self.readout_hidden_dim = readout_hidden_dim
        self.zscore_pca_scores = bool(zscore_pca_scores)
        self.reg = float(reg)
        self.sc_mask_ratio = float(sc_mask_ratio)
        self.fc_mask_ratio = float(fc_mask_ratio)
        self.min_masked_components = int(min_masked_components_per_modality)
        self.use_covariates = bool(use_covariates)
        self.cov_projector_hidden_dim = cov_projector_hidden_dim
        self.loss_weighting = str(loss_weighting)
        self.uses_cov = self.use_covariates

        if self.readout_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown readout_type='{self.readout_type}'. Choose from {{'linear', 'mlp'}}.")
        if self.num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {self.num_hidden_layers}.")
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

        # Learned mask-fill scalars (one per modality), mirroring attention path.
        self.sc_mask_value = nn.Parameter(torch.zeros(1))
        self.fc_mask_value = nn.Parameter(torch.zeros(1))

        self.flat_dim = 2 * self.n_components_pca  # [sc_scalars || fc_scalars]
        self.cov_projector = None
        self._cov_dim = None
        self._cov_out_dim = None

        # Encoder path: flat vector -> hidden_dim
        encoder_in_dim = self.flat_dim  # cov (if any) appended at forward time, see _maybe_extend_in_dim
        self.encoder = self._build_encoder(encoder_in_dim, self.hidden_dim)

        # Per-modality decoder heads: hidden_dim -> k
        self.sc_readout_head = self._build_head(self.hidden_dim, self.n_components_pca)
        self.fc_readout_head = self._build_head(self.hidden_dim, self.n_components_pca)

        self.last_reconstructions = None

        print(
            f"MaskedMLPPretrainer init | src={self.source_modality} tgt={self.target_modality} "
            f"| k={self.n_components_pca} hidden_dim={self.hidden_dim} num_hidden_layers={self.num_hidden_layers} "
            f"| nonlinear={self.nonlinear} low_rank_dim={self.low_rank_dim} readout={self.readout_type} "
            f"| zscore={self.zscore_pca_scores} sc_p={self.sc_mask_ratio} fc_p={self.fc_mask_ratio} "
            f"| use_cov={self.use_covariates}",
            flush=True,
        )
        self.to(device)

    def _build_encoder(self, in_dim, out_dim):
        if not self.nonlinear:
            if self.low_rank_dim is None:
                return nn.Linear(in_dim, out_dim, bias=True)
            # Two linear layers with a bottleneck; composition rank <= low_rank_dim.
            return nn.Sequential(
                nn.Linear(in_dim, self.low_rank_dim, bias=False),
                nn.Linear(self.low_rank_dim, out_dim, bias=True),
            )
        layers = [nn.Linear(in_dim, out_dim), nn.PReLU()]
        if self.dropout_p > 0:
            layers.append(nn.Dropout(self.dropout_p))
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(out_dim, out_dim))
            layers.append(nn.PReLU())
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
        return nn.Sequential(*layers)

    def _build_head(self, in_dim, out_dim):
        if self.readout_type == "linear":
            return nn.Linear(in_dim, out_dim, bias=True)
        hidden = int(self.readout_hidden_dim) if self.readout_hidden_dim is not None else in_dim
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.PReLU(),
            nn.Linear(hidden, out_dim),
        )

    def _init_cov_projector(self, cov_dim, device):
        if self.cov_projector_hidden_dim is None:
            cov_out_dim = int(cov_dim)
            self.cov_projector = nn.Identity().to(device)
        else:
            hidden = int(self.cov_projector_hidden_dim)
            cov_out_dim = hidden
            self.cov_projector = nn.Sequential(
                nn.Linear(int(cov_dim), hidden),
                nn.GELU(),
            ).to(device)
        self._cov_dim = int(cov_dim)
        self._cov_out_dim = cov_out_dim
        # Rebuild encoder to absorb the appended cov features.
        new_in = self.flat_dim + cov_out_dim
        self.encoder = self._build_encoder(new_in, self.hidden_dim).to(device)

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

    def _build_flat_input(self, c_source, c_target, sc_mask, fc_mask, cov=None):
        sc_fill = self.sc_mask_value.view(1, 1).expand_as(c_source)
        fc_fill = self.fc_mask_value.view(1, 1).expand_as(c_target)
        sc_scalar = torch.where(sc_mask, sc_fill, c_source)
        fc_scalar = torch.where(fc_mask, fc_fill, c_target)
        flat = torch.cat([sc_scalar, fc_scalar], dim=-1)
        if self.use_covariates:
            if cov is None:
                raise ValueError("use_covariates=True requires 'cov' in the forward call / batch.")
            cov = cov.to(flat.device).to(torch.float32)
            if self.cov_projector is None:
                self._init_cov_projector(cov.shape[-1], device=flat.device)
            flat = torch.cat([flat, self.cov_projector(cov)], dim=-1)
        return flat

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

        flat = self._build_flat_input(c_source, c_target, sc_mask, fc_mask, cov=cov)
        h = self.encoder(flat)
        c_source_hat = self.sc_readout_head(h)
        c_target_hat = self.fc_readout_head(h)

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
                f"MaskedMLPPretrainer supports loss_type in {{'latent_mse', 'latent_weighted_mse'}}, got '{loss_type}'."
            )
        x = batch["x"] if "x" in batch else batch["x_modalities"]
        y = batch["y"]
        cov = batch.get("cov") if self.use_covariates else None
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
        """Zero-shot SC->FC probe: all FC masked, all SC visible, decode to FC edge space."""
        c_source = self.encode_source_latents(x)
        batch_size = c_source.shape[0]
        device = c_source.device
        c_target_placeholder = torch.zeros(batch_size, self.n_components_pca, device=device, dtype=torch.float32)
        sc_mask = torch.zeros(batch_size, self.n_components_pca, dtype=torch.bool, device=device)
        fc_mask = torch.ones(batch_size, self.n_components_pca, dtype=torch.bool, device=device)
        flat = self._build_flat_input(c_source, c_target_placeholder, sc_mask, fc_mask, cov=cov)
        h = self.encoder(flat)
        c_target_hat = self.fc_readout_head(h)
        return self.decode_target_latents(c_target_hat)

    def forward_full(self, x, y, cov=None, joint_mask=None):
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
            "attention": None,
        }

    def reconstruct(self, batch, joint_mask=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = batch["x"] if "x" in batch else batch["x_modalities"]
            y = batch["y"]
            cov = batch.get("cov") if self.use_covariates else None
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
            "attention": None,
        }

    def per_component_reconstruction_error(self, batch, joint_mask=None, squared=True):
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
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        params.extend([p for p in self.sc_readout_head.parameters() if p.requires_grad])
        params.extend([p for p in self.fc_readout_head.parameters() if p.requires_grad])
        if self.cov_projector is not None:
            params.extend([p for p in self.cov_projector.parameters() if p.requires_grad])
        return compute_reg_loss(params, l1_l2_tuple=(0.0, self.reg))
