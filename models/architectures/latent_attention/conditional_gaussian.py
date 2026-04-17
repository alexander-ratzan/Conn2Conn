import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf, OAS

from models.models import get_modality_data


class CrossModal_ConditionalGaussian(nn.Module):
    """
    Closed-form SC->FC baseline derived from the conditional mean of a joint Gaussian.

    Supported fitting domains:
    - `pca`: conditional Gaussian in retained PCA latent space
    - `raw_edges`: ridge-equivalent conditional mean in raw edge space using a dual/sample-space solve
    """

    def __init__(
        self,
        base,
        n_components_pca=128,
        lambda_ridge=1.0e-3,
        covariance_estimator="empirical",
        zscore_pca_scores=False,
        fit_domain="pca",
        use_covariates=False,
        cov_sources=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        if len(getattr(base, "source_modalities", [base.source])) != 1:
            raise ValueError("CrossModal_ConditionalGaussian currently supports exactly one source modality.")

        self.base = base
        self.source_modality = getattr(base, "source_modalities", [base.source])[0]
        self.target_modality = getattr(base, "target", None) or getattr(base, "target_modalities", [base.target])[0]
        self.n_components_pca = int(n_components_pca)
        self.lambda_ridge = float(lambda_ridge)
        self.covariance_estimator = str(covariance_estimator)
        self.zscore_pca_scores = bool(zscore_pca_scores)
        self.fit_domain = str(fit_domain)
        self.use_covariates = bool(use_covariates)
        self.cov_sources = list(cov_sources) if cov_sources is not None else []
        self.uses_cov = self.use_covariates

        if self.lambda_ridge < 0:
            raise ValueError(f"lambda_ridge must be non-negative, got {self.lambda_ridge}.")
        if self.covariance_estimator not in {"empirical", "ledoit_wolf", "oas"}:
            raise ValueError(
                f"Unknown covariance_estimator='{self.covariance_estimator}'. "
                "Choose from {'empirical', 'ledoit_wolf', 'oas'}."
            )
        if self.fit_domain not in {"pca", "raw_edges"}:
            raise ValueError(
                f"Unknown fit_domain='{self.fit_domain}'. Choose from {'pca', 'raw_edges'}."
            )
        if self.fit_domain == "raw_edges" and self.covariance_estimator != "empirical":
            raise ValueError(
                "Raw-edge conditional Gaussian currently supports covariance_estimator='empirical' only. "
                "Shrinkage estimators are only implemented for fit_domain='pca' in v1."
            )
        if self.use_covariates and not self.cov_sources:
            raise ValueError("use_covariates=True requires a non-empty cov_sources list.")

        if device is None:
            default_device = "cpu" if self.fit_domain == "raw_edges" else ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(default_device)
        self.device = device

        if self.fit_domain == "pca":
            self._init_pca_domain(base, device)
        else:
            self._init_raw_edge_domain(base, device)
        self.to(device)

    def _get_full_covariate_matrix(self, base):
        if not self.use_covariates:
            return None
        cov_parts = []
        for source in self.cov_sources:
            if source == "fs_all":
                cov_parts.append(np.asarray(base.fs_features_z, dtype=np.float64))
            elif source == "fs_volumes":
                if base.fs_volumes_z is None:
                    raise ValueError("fs_volumes requested but FS volume covariates are unavailable.")
                cov_parts.append(np.asarray(base.fs_volumes_z, dtype=np.float64))
            elif source == "age":
                cov_parts.append(np.asarray(base.age_z, dtype=np.float64))
            elif source == "sex":
                cov_parts.append(np.asarray(base.sex_oh, dtype=np.float64))
            elif source == "race_eth":
                cov_parts.append(np.asarray(base.race_eth_oh, dtype=np.float64))
            else:
                raise ValueError(f"Unknown covariate source '{source}'.")
        return np.concatenate(cov_parts, axis=1)

    def _prepare_cov_batch(self, cov):
        if not self.use_covariates:
            return None
        if cov is None:
            raise ValueError("Covariate-conditioned Gaussian model requires cov input at prediction time.")
        device_ref = getattr(self, "source_mean", None)
        if device_ref is None:
            device_ref = getattr(self, "raw_source_mean", None)
        if device_ref is None:
            raise RuntimeError("Could not determine device reference for covariate batch preparation.")
        cov_parts = []
        for source in self.cov_sources:
            if source not in cov:
                raise ValueError(f"Missing covariate source '{source}' in batch.")
            cov_parts.append(cov[source].to(device_ref.device).to(torch.float32))
        return torch.cat(cov_parts, dim=1)

    def _cov_to_corr(self, cov):
        std = np.sqrt(np.clip(np.diag(cov), a_min=1.0e-12, a_max=None))
        denom = np.outer(std, std)
        corr = cov / denom
        corr[~np.isfinite(corr)] = 0.0
        np.fill_diagonal(corr, 1.0)
        return corr

    def _register_shared_pca_buffers(self, source_mean, source_loadings, target_mean, target_loadings, device):
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

    def _init_pca_domain(self, base, device):
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
                f"CrossModal_ConditionalGaussian requested n_components_pca={self.n_components_pca}, "
                f"but train PCA scores only support min(source={max_source_k}, target={max_target_k})."
            )

        source_scores_k = np.asarray(source_scores[:, :self.n_components_pca], dtype=np.float64)
        target_scores_k = np.asarray(target_scores[:, :self.n_components_pca], dtype=np.float64)
        source_score_mean = source_scores_k.mean(axis=0)
        target_score_mean = target_scores_k.mean(axis=0)
        source_score_std = np.maximum(source_scores_k.std(axis=0), 1.0e-8)
        target_score_std = np.maximum(target_scores_k.std(axis=0), 1.0e-8)

        if self.zscore_pca_scores:
            Z_sc = (source_scores_k - source_score_mean) / source_score_std
            Z_fc = (target_scores_k - target_score_mean) / target_score_std
        else:
            Z_sc = source_scores_k
            Z_fc = target_scores_k

        full_cov = self._get_full_covariate_matrix(base)
        cov_train = None
        if self.use_covariates:
            train_indices = np.asarray(base.trainvaltest_partition_indices["train"])
            cov_train = np.asarray(full_cov[train_indices], dtype=np.float64)
            Z_sc = np.concatenate([Z_sc, cov_train], axis=1)

        mu_sc = Z_sc.mean(axis=0)
        mu_fc = Z_fc.mean(axis=0)
        Z_joint = np.concatenate([Z_sc, Z_fc], axis=1)

        if self.covariance_estimator == "empirical":
            Sigma = np.cov(Z_joint, rowvar=False)
            shrinkage_value = np.nan
        elif self.covariance_estimator == "ledoit_wolf":
            est = LedoitWolf().fit(Z_joint)
            Sigma = est.covariance_
            shrinkage_value = float(est.shrinkage_)
        else:
            est = OAS().fit(Z_joint)
            Sigma = est.covariance_
            shrinkage_value = float(est.shrinkage_)

        k = self.n_components_pca
        source_dim = Z_sc.shape[1]
        Sigma_11 = Sigma[:source_dim, :source_dim]
        Sigma_12 = Sigma[:source_dim, source_dim:]
        Sigma_21 = Sigma[source_dim:, :source_dim]
        Sigma_22 = Sigma[source_dim:, source_dim:]
        reg_eye = self.lambda_ridge * np.eye(source_dim, dtype=np.float64)
        Sigma_11_reg = Sigma_11 + reg_eye
        W_t = np.linalg.solve(Sigma_11_reg.T, Sigma_21.T)
        W = W_t.T
        b = mu_fc - W @ mu_sc
        Sigma_cond = Sigma_22 - W @ Sigma_12
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
        cond_var_fit = np.clip(np.diag(Sigma_cond), a_min=0.0, a_max=None)

        if self.zscore_pca_scores:
            D_t = np.diag(target_score_std)
            Sigma_cond_latent = D_t @ Sigma_cond @ D_t
            cond_var_latent = np.clip(np.diag(Sigma_cond_latent), a_min=0.0, a_max=None)
        else:
            Sigma_cond_latent = Sigma_cond.copy()
            cond_var_latent = cond_var_fit.copy()

        self._register_shared_pca_buffers(source_mean, source_loadings, target_mean, target_loadings, device)
        self.register_buffer("source_score_mean", torch.tensor(source_score_mean, dtype=torch.float32, device=device))
        self.register_buffer("source_score_std", torch.tensor(source_score_std, dtype=torch.float32, device=device))
        self.register_buffer("target_score_mean", torch.tensor(target_score_mean, dtype=torch.float32, device=device))
        self.register_buffer("target_score_std", torch.tensor(target_score_std, dtype=torch.float32, device=device))
        self.register_buffer("mu_sc", torch.tensor(mu_sc, dtype=torch.float32, device=device))
        self.register_buffer("mu_fc", torch.tensor(mu_fc, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_joint", torch.tensor(Sigma, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_11", torch.tensor(Sigma_11, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_12", torch.tensor(Sigma_12, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_21", torch.tensor(Sigma_21, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_22", torch.tensor(Sigma_22, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_11_reg", torch.tensor(Sigma_11_reg, dtype=torch.float32, device=device))
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32, device=device))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_fc_given_sc_fit", torch.tensor(Sigma_cond, dtype=torch.float32, device=device))
        self.register_buffer("Sigma_fc_given_sc", torch.tensor(Sigma_cond_latent, dtype=torch.float32, device=device))
        self.register_buffer("conditional_var_fit", torch.tensor(cond_var_fit, dtype=torch.float32, device=device))
        self.register_buffer("conditional_var", torch.tensor(cond_var_latent, dtype=torch.float32, device=device))
        self.register_buffer("conditional_std", torch.tensor(np.sqrt(cond_var_latent), dtype=torch.float32, device=device))
        self.register_buffer("joint_correlation", torch.tensor(self._cov_to_corr(Sigma), dtype=torch.float32, device=device))
        self.register_buffer("source_correlation", torch.tensor(self._cov_to_corr(Sigma_11), dtype=torch.float32, device=device))
        self.register_buffer("target_correlation", torch.tensor(self._cov_to_corr(Sigma_22), dtype=torch.float32, device=device))
        self.register_buffer("cross_correlation", torch.tensor(self._cov_to_corr(Sigma)[k:, :k], dtype=torch.float32, device=device))
        self.register_buffer("source_pca_scores_train", torch.tensor(source_scores_k, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_scores_train", torch.tensor(target_scores_k, dtype=torch.float32, device=device))
        self.register_buffer("source_pca_scores_train_fit", torch.tensor(Z_sc, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_scores_train_fit", torch.tensor(Z_fc, dtype=torch.float32, device=device))
        if cov_train is not None:
            self.register_buffer("covariates_train_fit", torch.tensor(cov_train, dtype=torch.float32, device=device))

        self.condition_number = float(np.linalg.cond(Sigma_11_reg))
        self.shrinkage_value = shrinkage_value
        self.feature_dim = int(source_dim)
        self.source_latent_dim = int(k)
        self.covariate_dim = int(0 if cov_train is None else cov_train.shape[1])
        self.n_train = int(Z_sc.shape[0])

        print(
            f"CrossModal_ConditionalGaussian init | domain=pca | src={self.source_modality} tgt={self.target_modality} "
            f"| k={self.n_components_pca} | cov_estimator={self.covariance_estimator} "
            f"| lambda_ridge={self.lambda_ridge} | zscore_pca_scores={self.zscore_pca_scores} "
            f"| use_covariates={self.use_covariates} cov_dim={self.covariate_dim} "
            f"| cond(Sigma_11+lamI)={self.condition_number:.3e}"
            + (f" | shrinkage={self.shrinkage_value:.4f}" if np.isfinite(self.shrinkage_value) else ""),
            flush=True,
        )

    def _init_raw_edge_domain(self, base, device):
        data = get_modality_data(base, device=device, include_scores=True, include_raw_data=True)
        source_data = data["sources"][self.source_modality]
        target_data = data["target"]
        train_indices = np.asarray(target_data["train_indices"])

        X_train = np.asarray(source_data["upper_triangles"][train_indices], dtype=np.float64)
        Y_train = np.asarray(target_data["upper_triangles"][train_indices], dtype=np.float64)
        cov_train = None
        if self.use_covariates:
            full_cov = self._get_full_covariate_matrix(base)
            cov_train = np.asarray(full_cov[train_indices], dtype=np.float64)
            X_train = np.concatenate([X_train, cov_train], axis=1)
        source_mean = X_train.mean(axis=0)
        target_mean = Y_train.mean(axis=0)
        X_centered = X_train - source_mean
        Y_centered = Y_train - target_mean

        n_train = X_centered.shape[0]
        gram = X_centered @ X_centered.T
        gram_reg = gram + self.lambda_ridge * np.eye(n_train, dtype=np.float64)
        gram_reg_inv = np.linalg.inv(gram_reg)
        # Dual ridge predictor: y = mu_y + k(x)^T (K + lambda I)^-1 Y_centered
        dual_coef = gram_reg_inv @ Y_centered
        Y_hat_train = gram @ dual_coef + target_mean
        residuals = Y_train - Y_hat_train
        residual_var_edges = np.var(residuals, axis=0, dtype=np.float64)
        residual_std_edges = np.sqrt(np.clip(residual_var_edges, a_min=0.0, a_max=None))

        # Project residual uncertainty into the target PCA basis for component-level summaries.
        target_loadings = np.asarray(target_data["loadings"][:, :self.n_components_pca], dtype=np.float64)
        residual_scores = residuals @ target_loadings
        residual_component_var = np.var(residual_scores, axis=0, dtype=np.float64)
        residual_component_std = np.sqrt(np.clip(residual_component_var, a_min=0.0, a_max=None))
        target_score_std = np.maximum(np.asarray(target_data["scores"][:, :self.n_components_pca], dtype=np.float64).std(axis=0), 1.0e-8)
        uncertainty_ratio = residual_component_std / target_score_std
        explained_fraction = 1.0 - (residual_component_std ** 2) / np.clip(target_score_std ** 2, 1.0e-8, None)

        self._register_shared_pca_buffers(source_data["mean"], source_data["loadings"], target_data["mean"], target_data["loadings"], device)
        self.register_buffer("raw_source_mean", torch.tensor(source_mean, dtype=torch.float32, device=device))
        self.register_buffer("raw_target_mean", torch.tensor(target_mean, dtype=torch.float32, device=device))
        self.register_buffer("X_train_centered", torch.tensor(X_centered, dtype=torch.float32, device=device))
        self.register_buffer("dual_coef", torch.tensor(dual_coef, dtype=torch.float32, device=device))
        self.register_buffer("gram_train", torch.tensor(gram, dtype=torch.float32, device=device))
        self.register_buffer("gram_train_reg", torch.tensor(gram_reg, dtype=torch.float32, device=device))
        self.register_buffer("residual_var_edges", torch.tensor(residual_var_edges, dtype=torch.float32, device=device))
        self.register_buffer("residual_std_edges", torch.tensor(residual_std_edges, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_residual_var", torch.tensor(residual_component_var, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_residual_std", torch.tensor(residual_component_std, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_target_std", torch.tensor(target_score_std, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_uncertainty_ratio", torch.tensor(uncertainty_ratio, dtype=torch.float32, device=device))
        self.register_buffer("target_pca_explained_fraction", torch.tensor(explained_fraction, dtype=torch.float32, device=device))
        self.register_buffer("source_pca_scores_train", torch.tensor(np.asarray(source_data["scores"][:, :self.n_components_pca], dtype=np.float64), dtype=torch.float32, device=device))
        self.register_buffer("target_pca_scores_train", torch.tensor(np.asarray(target_data["scores"][:, :self.n_components_pca], dtype=np.float64), dtype=torch.float32, device=device))
        if cov_train is not None:
            self.register_buffer("covariates_train_fit", torch.tensor(cov_train, dtype=torch.float32, device=device))

        small_k = min(self.n_components_pca, n_train)
        X_lat_small = np.asarray(source_data["scores"][:, :small_k], dtype=np.float64)
        Y_lat_small = np.asarray(target_data["scores"][:, :small_k], dtype=np.float64)
        joint_small = np.concatenate([X_lat_small, Y_lat_small], axis=1)
        Sigma_small = np.cov(joint_small, rowvar=False)
        self.register_buffer("Sigma_joint", torch.tensor(Sigma_small, dtype=torch.float32, device=device))
        self.register_buffer("joint_correlation", torch.tensor(self._cov_to_corr(Sigma_small), dtype=torch.float32, device=device))
        self.register_buffer("source_correlation", torch.tensor(self._cov_to_corr(Sigma_small[:small_k, :small_k]), dtype=torch.float32, device=device))
        self.register_buffer("target_correlation", torch.tensor(self._cov_to_corr(Sigma_small[small_k:, small_k:]), dtype=torch.float32, device=device))
        self.register_buffer("cross_correlation", torch.tensor(self._cov_to_corr(Sigma_small)[small_k:, :small_k], dtype=torch.float32, device=device))

        self.condition_number = float(np.linalg.cond(gram_reg))
        self.shrinkage_value = np.nan
        self.feature_dim = int(X_centered.shape[1])
        self.source_latent_dim = int(self.n_components_pca)
        self.covariate_dim = int(0 if cov_train is None else cov_train.shape[1])
        self.n_train = int(n_train)

        print(
            f"CrossModal_ConditionalGaussian init | domain=raw_edges | src={self.source_modality} tgt={self.target_modality} "
            f"| edges={self.feature_dim} | n_train={self.n_train} | lambda_ridge={self.lambda_ridge} "
            f"| use_covariates={self.use_covariates} cov_dim={self.covariate_dim} "
            f"| cond(K+lamI)={self.condition_number:.3e}",
            flush=True,
        )

    def encode_source_latents(self, x):
        x = x.to(self.source_mean.device).to(torch.float32)
        z_sc = torch.matmul(x - self.source_mean, self.source_loadings_k)
        if self.zscore_pca_scores:
            z_sc = (z_sc - self.source_score_mean) / self.source_score_std
        return z_sc

    def encode_target_latents(self, y):
        y = y.to(self.target_mean.device).to(torch.float32)
        z_fc = torch.matmul(y - self.target_mean, self.target_loadings_k)
        if self.zscore_pca_scores:
            z_fc = (z_fc - self.target_score_mean) / self.target_score_std
        return z_fc

    def predict_target_latents(self, x, cov=None):
        if self.fit_domain != "pca":
            raise RuntimeError("predict_target_latents is only defined for fit_domain='pca'.")
        z_sc = self.encode_source_latents(x)
        cov_batch = self._prepare_cov_batch(cov)
        if cov_batch is not None:
            z_sc = torch.cat([z_sc, cov_batch], dim=1)
        return torch.matmul(z_sc, self.W.t()) + self.b

    def predict_target_uncertainty(self):
        if self.fit_domain == "pca":
            return {
                "conditional_covariance": self.Sigma_fc_given_sc,
                "conditional_variance": self.conditional_var,
                "conditional_std": self.conditional_std,
            }
        return {
            "edge_residual_variance": self.residual_var_edges,
            "edge_residual_std": self.residual_std_edges,
            "target_pca_residual_variance": self.target_pca_residual_var,
            "target_pca_residual_std": self.target_pca_residual_std,
            "target_pca_uncertainty_ratio": self.target_pca_uncertainty_ratio,
            "target_pca_explained_fraction": self.target_pca_explained_fraction,
        }

    def decode_target_latents(self, z_fc_hat):
        if self.zscore_pca_scores:
            z_fc_hat = z_fc_hat * self.target_score_std + self.target_score_mean
        return torch.matmul(z_fc_hat, self.target_loadings_k.t()) + self.target_mean

    def _predict_raw_edges(self, x, cov=None):
        x = x.to(self.raw_source_mean.device).to(torch.float32)
        cov_batch = self._prepare_cov_batch(cov)
        if cov_batch is not None:
            x = torch.cat([x, cov_batch], dim=1)
        x_centered = x - self.raw_source_mean
        k_x = torch.matmul(x_centered, self.X_train_centered.t())
        return torch.matmul(k_x, self.dual_coef) + self.raw_target_mean

    def forward(self, x, cov=None):
        if self.fit_domain == "pca":
            return self.decode_target_latents(self.predict_target_latents(x, cov=cov))
        return self._predict_raw_edges(x, cov=cov)

    def get_diagnostics(self):
        out = {
            "fit_domain": self.fit_domain,
            "condition_number": self.condition_number,
            "shrinkage_value": self.shrinkage_value,
            "feature_dim": self.feature_dim,
            "source_latent_dim": self.source_latent_dim,
            "covariate_dim": self.covariate_dim,
            "n_train": self.n_train,
            "use_covariates": self.use_covariates,
            "cov_sources": list(self.cov_sources),
            "Sigma_joint": self.Sigma_joint,
            "joint_correlation": self.joint_correlation,
            "source_correlation": self.source_correlation,
            "target_correlation": self.target_correlation,
            "cross_correlation": self.cross_correlation,
            "source_pca_scores_train": self.source_pca_scores_train,
            "target_pca_scores_train": self.target_pca_scores_train,
        }
        if self.fit_domain == "pca":
            out.update(
                {
                    "mu_sc": self.mu_sc,
                    "mu_fc": self.mu_fc,
                    "Sigma_11": self.Sigma_11,
                    "Sigma_12": self.Sigma_12,
                    "Sigma_21": self.Sigma_21,
                    "Sigma_22": self.Sigma_22,
                    "Sigma_11_reg": self.Sigma_11_reg,
                    "W": self.W,
                    "b": self.b,
                    "Sigma_fc_given_sc_fit": self.Sigma_fc_given_sc_fit,
                    "Sigma_fc_given_sc": self.Sigma_fc_given_sc,
                    "conditional_variance": self.conditional_var,
                    "conditional_std": self.conditional_std,
                    "source_pca_scores_train_fit": self.source_pca_scores_train_fit,
                    "target_pca_scores_train_fit": self.target_pca_scores_train_fit,
                }
            )
        else:
            out.update(
                {
                    "raw_source_mean": self.raw_source_mean,
                    "raw_target_mean": self.raw_target_mean,
                    "gram_train": self.gram_train,
                    "gram_train_reg": self.gram_train_reg,
                    "residual_var_edges": self.residual_var_edges,
                    "residual_std_edges": self.residual_std_edges,
                    "target_pca_residual_var": self.target_pca_residual_var,
                    "target_pca_residual_std": self.target_pca_residual_std,
                    "target_pca_target_std": self.target_pca_target_std,
                    "target_pca_uncertainty_ratio": self.target_pca_uncertainty_ratio,
                    "target_pca_explained_fraction": self.target_pca_explained_fraction,
                }
            )
        return out
