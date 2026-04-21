import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from models.architectures.utils import get_model_input
from models.utils import get_batch_cov
from models.eval.evaluator_viz_markdown import EvaluatorVizMarkdownMixin
from models.eval.metrics import (
    compute_basic_regression_metrics,
    compute_corr_matrix,
    compute_demeaned_pearson_r,
    compute_pearson_r,
    compute_prediction_norm_ratio,
    compute_prediction_variance_ratio,
)

def evaluate_model(model, data_loader, target_train_mean, device):
    """
    Evaluate model predictions over a data loader.

    Returns batch-averaged MSE, correlation, demeaned correlation, variance ratio,
    and norm ratio metrics.
    """
    model.eval()
    total_mse = 0.0
    total_pearson_r = 0.0
    total_demeaned_r = 0.0
    total_variance_ratio = 0.0
    total_norm_ratio = 0.0
    n_batches = 0

    target_mean_tensor = torch.tensor(target_train_mean, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in data_loader:
            x = get_model_input(batch)
            y = batch["y"].to(device)

            kwargs = {}
            if getattr(model, "uses_cov", False):
                kwargs["cov"] = get_batch_cov(batch)
                if getattr(model, "use_target_scores_in_projector", False) and "y" in batch:
                    kwargs["y"] = batch["y"]
            if getattr(model, "uses_node_features", False) and "node_features" in batch:
                kwargs["node_features"] = batch["node_features"]
            if getattr(model, "uses_sc_matrix", False) and "sc_matrix" in batch:
                kwargs["sc_matrix"] = batch["sc_matrix"]

            out = model(x, **kwargs) if kwargs else model(x)
            y_pred = out[0] if isinstance(out, tuple) else out

            total_mse += F.mse_loss(y_pred, y).item()
            total_pearson_r += compute_pearson_r(y_pred, y)
            total_demeaned_r += compute_demeaned_pearson_r(y_pred, y, target_mean_tensor)
            total_variance_ratio += compute_prediction_variance_ratio(y_pred, y)
            total_norm_ratio += compute_prediction_norm_ratio(y_pred, y)
            n_batches += 1

    return {
        "mse": total_mse / n_batches,
        "pearson_r": total_pearson_r / n_batches,
        "demeaned_r": total_demeaned_r / n_batches,
        "variance_ratio": total_variance_ratio / n_batches,
        "norm_ratio": total_norm_ratio / n_batches,
    }


class Evaluator(EvaluatorVizMarkdownMixin):
    """
    Evaluation class for prediction vs target analysis.

    Args:
        preds: torch.Tensor or np.ndarray, predictions (n_subjects x n_features)
        targets: torch.Tensor or np.ndarray, targets (n_subjects x n_features)
        dataset_partition: Dataset partition (e.g., train/val/test split) with .indices attribute
        dataset: Full dataset object with train_mean attributes (fc_train_avg or sc_train_avg)
    """
    
    def __init__(self, preds, targets, dataset_partition, dataset):
        # Convert to numpy arrays
        self.preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
        self.targets = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    
        # Store dataset references
        self.dataset_partition = dataset_partition
        self.dataset = dataset
        self.subject_indices = dataset_partition.indices
        self.numrois = dataset.fc_matrices.shape[1]
        
        # Get training mean and std (depends on target modality)
        # Use upper_triangles (vectorized) since that's what the model predicts
        train_indices = dataset.trainvaltest_partition_indices["train"]
        if dataset.target == "FC":
            self.train_mean = dataset.fc_train_avg
            train_data = dataset.fc_upper_triangles[train_indices]
        elif dataset.target == "SC":
            self.train_mean = dataset.sc_train_avg
            train_data = dataset.sc_upper_triangles[train_indices]
        elif dataset.target == "SC_r2t":
            self.train_mean = dataset.sc_r2t_corr_train_avg
            train_data = dataset.sc_r2t_corr_upper_triangles[train_indices]
        else:
            raise ValueError(f"Unknown target modality: {dataset.target}")
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.cpu().numpy()
        self.train_std = np.std(train_data, axis=0)
        # Compute correlation matrices (targets, preds) so rows=targets, cols=preds
        # This answers: "for each target, which prediction matches best?"
        self.corr_matrix = compute_corr_matrix(self.targets, self.preds)
        self.corr_matrix_demeaned = compute_corr_matrix(
            self.targets - self.train_mean, 
            self.preds - self.train_mean
        )
    
        self._metrics = self._compute_metrics()
        
        # PCA objects (computed lazily when evaluate_pca_structure is called)
        self._pca_preds = None
        self._pca_targets = None    
        self._spd_matrix_cache = {}
        self._fc_matrix_cache = {}
    

    def _compute_metrics(self):
        """Compute all evaluation metrics."""
        return compute_basic_regression_metrics(
            self.preds,
            self.targets,
            corr_matrix=self.corr_matrix,
            corr_matrix_demeaned=self.corr_matrix_demeaned,
        )

    
    def _fit_pca(self):
        """Fit PCA models for predictions and targets (subject-mode PCA)."""
        # Mean center
        targets_centered = self.targets - self.train_mean  # (n, d)
        preds_centered = self.preds - self.train_mean  # (n, d)
        
        # Transpose to features x subjects (d, n) for subject-mode PCA
        targets_transposed = targets_centered.T
        preds_transposed = preds_centered.T
        
        # PCA for targets
        self._pca_targets = PCA()
        self._pca_targets.fit(targets_transposed)
        
        # PCA for predictions
        self._pca_preds = PCA()
        self._pca_preds.fit(preds_transposed)
        
        # Orthonormality check for bases (across subjects)
        B_targets = self._pca_targets.components_
        B_preds = self._pca_preds.components_
        assert np.allclose(B_targets @ B_targets.T, np.eye(B_targets.shape[0]), atol=1e-6), \
            "B_targets is not orthonormal"
        assert np.allclose(B_preds @ B_preds.T, np.eye(B_preds.shape[0]), atol=1e-6), \
            "B_preds is not orthonormal"
