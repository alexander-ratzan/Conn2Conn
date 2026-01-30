"""
PCA + PCA^-1
PCA + PLS + PCA^-1
PLS
Kraken
Kraken + learnable PCA
non-linear Kraken
cov-VAE/CLIP pretraining + MLP + Kraken learnable PCA
GNN
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from scipy import stats
import torch
import torch.nn as nn


def predict_from_loader(model, data_loader, device=None):
    """Generate predictions from a model using a data loader."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"]
            y = batch["y"]
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets


class CrossModalPCA(nn.Module):
    """
    Cross-modal PCA model implemented in PyTorch.
    Maps source modality to target modality using PCA decomposition from training data.
    Given a base data object (which provides population mean and loadings for each modality), 
    it projects from source modality into source PCA space, truncates to num_components, 
    then reconstructs (inverse transforms) via the target modality's PCA.

    Typical intended use is for source and target in {'SC', 'FC'}.

    Args:
        base: Dataset base object (e.g., HCP_Base) providing PCA means/loadings for both modalities on train partition.
        num_components: Number of components to use (int).
        source: 'SC' or 'FC' (modality for input)
        target: 'SC' or 'FC' (modality for output)
    """
    def __init__(self, base, num_components=256,device=None):
        super().__init__()
        self.base = base
        self.num_components = num_components

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Select appropriate means, loadings for source and target
        if base.source == "SC":
            self.source_mean = torch.tensor(base.sc_train_avg, dtype=torch.float32, device=self.device)
            self.source_loadings = torch.tensor(base.sc_train_loadings, dtype=torch.float32, device=self.device)
        elif base.source == "FC":
            self.source_mean = torch.tensor(base.fc_train_avg, dtype=torch.float32, device=self.device)
            self.source_loadings = torch.tensor(base.fc_train_loadings, dtype=torch.float32, device=self.device)

        if base.target == "SC":
            self.target_mean = torch.tensor(base.sc_train_avg, dtype=torch.float32, device=self.device)
            self.target_loadings = torch.tensor(base.sc_train_loadings, dtype=torch.float32, device=self.device)
        elif base.target == "FC":
            self.target_mean = torch.tensor(base.fc_train_avg, dtype=torch.float32, device=self.device)
            self.target_loadings = torch.tensor(base.fc_train_loadings, dtype=torch.float32, device=self.device)

        print(f"Source mean shape: {self.source_mean.shape}")
        print(f"Source loadings shape: {self.source_loadings.shape}")
        print(f"Target mean shape: {self.target_mean.shape}")
        print(f"Target loadings shape: {self.target_loadings.shape}")
        
        # Only keep first k components, and explicitly ensure float32 dtype to avoid matmul dtype mismatch
        self.register_buffer('source_loadings_k', self.source_loadings[:, :self.num_components].to(torch.float32))
        self.register_buffer('target_loadings_k', self.target_loadings[:, :self.num_components].to(torch.float32))
        self.register_buffer('source_mean_', self.source_mean.to(torch.float32))
        self.register_buffer('target_mean_', self.target_mean.to(torch.float32))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor(batch_size, d)): source modality tensor.

        Returns:
            y_hat (torch.Tensor(batch_size, d)): predicted target modality.
        """
        x = x.to(self.device).to(torch.float32)

        # 1. Center input
        x_centered = x - self.source_mean_

        # 2. Project into truncated source PCA space (batch @ (d, k)) => (batch, k)
        z = torch.matmul(x_centered, self.source_loadings_k)

        # 3. Reconstruct with target PCA (inverse transform): (batch, k) @ (k, d) => (batch, d)
        y_hat = torch.matmul(z, self.target_loadings_k.t()) + self.target_mean_

        return y_hat

    def to(self, *args, **kwargs): # for device mismatch issues
        super().to(*args, **kwargs)
        self.source_mean_ = self.source_mean_.to(*args, **kwargs)
        self.target_mean_ = self.target_mean_.to(*args, **kwargs)
        self.source_loadings_k = self.source_loadings_k.to(*args, **kwargs)
        self.target_loadings_k = self.target_loadings_k.to(*args, **kwargs)
        return self


def run_pca_and_plot(X_train, X_val=None, X_test=None, 
                     train_ids=None, val_ids=None, test_ids=None,
                     modality="SC", parcellation="Glasser",
                     var_threshold=0.95, marker_size=2, random_state=None,
                     max_components_for_bar=100, reconstruct_and_plot=True):
    """
    Run PCA on the input train data X_train, print shape info, and plot cumulative explained variance.
    Additionally, computes and plots reconstruction R^2 for training, validation, and test datasets 
    using PCA fit on training data. Also, reconstruct and output for all subjects with error bands.

    Args:
        X_train (np.ndarray): Training data of shape (n_train_samples, n_features).
        X_val (np.ndarray or None): Validation data of shape (n_val_samples, n_features).
        X_test (np.ndarray or None): Test data of shape (n_test_samples, n_features).
        train_ids (list of str/int, optional): Subject IDs for train set.
        val_ids (list of str/int, optional): Subject IDs for val set.
        test_ids (list of str/int, optional): Subject IDs for test set.
        modality (str): Modality name (e.g., "SC", "FC"). Default: "SC".
        parcellation (str): Parcellation name (e.g., "Glasser", "S456"). Default: "Glasser".
        var_threshold (float): The cumulative variance threshold to annotate.
        marker_size (int): Marker size for the plot points.
        random_state (int or None): Random seed used for reproducibility.
        max_components_for_bar (int): Maximum number of components to show in bar chart.
        reconstruct_and_plot (bool): Whether to reconstruct and plot the PCA results.
    Returns:
        pca (PCA object): The fitted PCA object.
        scores_dict (dict): Contains 'train', 'val', 'test' scores.
        loadings (np.ndarray): The PCA loadings (n_components x n_features).
    """
    np.random.seed(random_state)

    # Fit PCA on the training set only
    pca = PCA()
    train_scores = pca.fit_transform(X_train)
    loadings = pca.components_

    print("Loadings shape (n_components, n_features):", loadings.shape)
    print("Train scores shape (n_train_samples, n_components):", train_scores.shape)

    # Plot 1: Side-by-side cumulative and per-component variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    var_per_comp = pca.explained_variance_ratio_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Left: Cumulative variance
    ax1.plot(np.arange(1, len(cumvar) + 1), cumvar, marker="o", markersize=marker_size)
    k_thresh = np.searchsorted(cumvar, var_threshold) + 1
    ax1.axhline(var_threshold, color='red', linestyle='--',
                label=f'{int(var_threshold*100)}% Variance (n_comps={k_thresh})')
    ax1.set_xlabel("Number of components")
    ax1.set_ylabel("Cumulative explained variance ratio")
    ax1.set_title("Cumulative Variance Explained")
    ax1.grid(True)
    ax1.legend()
    
    # Right: Per-component variance (bar chart)
    n_components_to_plot = min(max_components_for_bar, len(var_per_comp))
    ax2.bar(np.arange(1, n_components_to_plot + 1), var_per_comp[:n_components_to_plot], 
            alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel("Component number")
    ax2.set_ylabel("Explained variance ratio")
    ax2.set_title(f"Variance Explained per Component (first {n_components_to_plot})")
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Cumulative Variance Explained by PCA: {modality} ({parcellation})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    if reconstruct_and_plot:
        # Compute R^2 for different numbers of principal components for train/val/test using train PCA
        ks = np.array([16, 64, 128, 256, 512])
        r2s = {'train': [], 'val': [], 'test': []}

        # Helper for reconstruction
        def reconstruct(pca, scores, n_components):
            loadings_k = pca.components_[:n_components, :]
            return np.dot(scores[:, :n_components], loadings_k) + pca.mean_

        # Train set
        for k in ks:
            X_hat_train = reconstruct(pca, train_scores, k)
            r2_train = r2_score(X_train, X_hat_train, multioutput='uniform_average')
            r2s['train'].append(r2_train)

        # Validation set
        if X_val is not None:
            val_scores = pca.transform(X_val)
            for k in ks:
                X_hat_val = reconstruct(pca, val_scores, k)
                r2_val = r2_score(X_val, X_hat_val, multioutput='uniform_average')
                r2s['val'].append(r2_val)
        else:
            r2s['val'] = None

        # Test set
        if X_test is not None:
            test_scores = pca.transform(X_test)
            for k in ks:
                X_hat_test = reconstruct(pca, test_scores, k)
                r2_test_ = r2_score(X_test, X_hat_test, multioutput='uniform_average')
                r2s['test'].append(r2_test_)
        else:
            r2s['test'] = None

        # Print results for key values
        for special_k in [128, 256, 512]:
            if special_k in ks:
                idx = list(ks).index(special_k)
                line = f"{special_k} components: "
                line += f"Train R^2: {r2s['train'][idx]:.5f}  "
                if r2s['val'] is not None:
                    line += f"Val R^2: {r2s['val'][idx]:.5f}  "
                if r2s['test'] is not None:
                    line += f"Test R^2: {r2s['test'][idx]:.5f}"
                print(line)

        # Plot 3: Individualized reconstruction with error bands for ALL subjects
        def compute_all_subject_r2s(X, scores, pca, ks):
            """
            Compute R^2 for all subjects across different k values.
            Reconstructions are computed vectorized for all subjects at once.
            """
            n_subjects = X.shape[0]
            n_ks = len(ks)
            r2s_all = np.zeros((n_subjects, n_ks))
            
            # Compute reconstructions and R^2 for each k value
            for k_idx, k in enumerate(ks):
                # Reconstruct all subjects at once: (n_subjects, k) @ (k, n_features) -> (n_subjects, n_features)
                X_hat = np.dot(scores[:, :k], loadings[:k, :]) + pca.mean_
                
                # Compute R^2 per subject (vectorized)
                # R^2 = 1 - (SS_res / SS_tot) for each subject
                ss_res = np.sum((X - X_hat) ** 2, axis=1)  # (n_subjects,)
                ss_tot = np.sum((X - X.mean(axis=0, keepdims=True)) ** 2, axis=1)  # (n_subjects,)
                r2s_all[:, k_idx] = 1 - (ss_res / (ss_tot + 1e-10))  # Add small epsilon to avoid division by zero
            
            return r2s_all

        plt.figure(figsize=(9, 6))
        colors = {'train': 'navy', 'val': 'darkorange', 'test': 'forestgreen'}
        
        # Compute R^2 for all subjects in each split
        if X_train is not None:
            r2s_train_all = compute_all_subject_r2s(X_train, train_scores, pca, ks)
            means_train = np.mean(r2s_train_all, axis=0)
            stds_train = np.std(r2s_train_all, axis=0)
            # 95% confidence interval (using t-distribution)
            n_train = r2s_train_all.shape[0]
            ci_train = stats.t.interval(0.95, n_train - 1, loc=means_train, scale=stds_train / np.sqrt(n_train))
            ci_train_lower = ci_train[0]
            ci_train_upper = ci_train[1]
            
            plt.plot(ks, means_train, 'o-', label='Train', color=colors['train'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_train_lower, ci_train_upper, alpha=0.2, color=colors['train'])
        
        if X_val is not None and val_ids is not None:
            val_scores = pca.transform(X_val)
            r2s_val_all = compute_all_subject_r2s(X_val, val_scores, pca, ks)
            means_val = np.mean(r2s_val_all, axis=0)
            stds_val = np.std(r2s_val_all, axis=0)
            n_val = r2s_val_all.shape[0]
            ci_val = stats.t.interval(0.95, n_val - 1, loc=means_val, scale=stds_val / np.sqrt(n_val))
            ci_val_lower = ci_val[0]
            ci_val_upper = ci_val[1]
            
            plt.plot(ks, means_val, 'o-', label='Validation', color=colors['val'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_val_lower, ci_val_upper, alpha=0.2, color=colors['val'])
        
        if X_test is not None and test_ids is not None:
            test_scores = pca.transform(X_test)
            r2s_test_all = compute_all_subject_r2s(X_test, test_scores, pca, ks)
            means_test = np.mean(r2s_test_all, axis=0)
            stds_test = np.std(r2s_test_all, axis=0)
            n_test = r2s_test_all.shape[0]
            ci_test = stats.t.interval(0.95, n_test - 1, loc=means_test, scale=stds_test / np.sqrt(n_test))
            ci_test_lower = ci_test[0]
            ci_test_upper = ci_test[1]
            
            plt.plot(ks, means_test, 'o-', label='Test', color=colors['test'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_test_lower, ci_test_upper, alpha=0.2, color=colors['test'])
        
        plt.xlabel("Number of PCA components", fontsize=12)
        plt.ylabel("Individual $R^2$ (true vs reconstructed)", fontsize=12)
        plt.title(f"PCA Reconstruction $R^2$ Across All Subjects: {modality} ({parcellation})\n(Mean ± 95% Confidence Interval)", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

        # Plot 4: Pearson correlation with error bands for ALL subjects
        def compute_all_subject_pearsonr(X, scores, pca, ks):
            """
            Compute Pearson correlation for all subjects across different k values.
            Reconstructions are computed vectorized for all subjects at once.
            """
            n_subjects = X.shape[0]
            n_ks = len(ks)
            pearsonr_all = np.zeros((n_subjects, n_ks))
            
            # Compute reconstructions and Pearson correlation for each k value
            for k_idx, k in enumerate(ks):
                # Reconstruct all subjects at once: (n_subjects, k) @ (k, n_features) -> (n_subjects, n_features)
                X_hat = np.dot(scores[:, :k], loadings[:k, :]) + pca.mean_
                
                # Compute Pearson correlation per subject (vectorized)
                # Center the data
                X_centered = X - X.mean(axis=1, keepdims=True)
                X_hat_centered = X_hat - X_hat.mean(axis=1, keepdims=True)
                
                # Compute correlation: r = sum((X - X_mean) * (X_hat - X_hat_mean)) / sqrt(sum((X - X_mean)^2) * sum((X_hat - X_hat_mean)^2))
                numerator = np.sum(X_centered * X_hat_centered, axis=1)  # (n_subjects,)
                denom_X = np.sqrt(np.sum(X_centered ** 2, axis=1))  # (n_subjects,)
                denom_X_hat = np.sqrt(np.sum(X_hat_centered ** 2, axis=1))  # (n_subjects,)
                pearsonr_all[:, k_idx] = numerator / (denom_X * denom_X_hat + 1e-10)  # Add small epsilon to avoid division by zero
            
            return pearsonr_all

        plt.figure(figsize=(9, 6))
        colors = {'train': 'navy', 'val': 'darkorange', 'test': 'forestgreen'}
        
        # Compute Pearson correlation for all subjects in each split
        if X_train is not None:
            pearsonr_train_all = compute_all_subject_pearsonr(X_train, train_scores, pca, ks)
            means_train = np.mean(pearsonr_train_all, axis=0)
            stds_train = np.std(pearsonr_train_all, axis=0)
            # 95% confidence interval (using t-distribution)
            n_train = pearsonr_train_all.shape[0]
            ci_train = stats.t.interval(0.95, n_train - 1, loc=means_train, scale=stds_train / np.sqrt(n_train))
            ci_train_lower = ci_train[0]
            ci_train_upper = ci_train[1]
            
            plt.plot(ks, means_train, 'o-', label='Train', color=colors['train'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_train_lower, ci_train_upper, alpha=0.2, color=colors['train'])
        
        if X_val is not None and val_ids is not None:
            val_scores = pca.transform(X_val)
            pearsonr_val_all = compute_all_subject_pearsonr(X_val, val_scores, pca, ks)
            means_val = np.mean(pearsonr_val_all, axis=0)
            stds_val = np.std(pearsonr_val_all, axis=0)
            n_val = pearsonr_val_all.shape[0]
            ci_val = stats.t.interval(0.95, n_val - 1, loc=means_val, scale=stds_val / np.sqrt(n_val))
            ci_val_lower = ci_val[0]
            ci_val_upper = ci_val[1]
            
            plt.plot(ks, means_val, 'o-', label='Validation', color=colors['val'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_val_lower, ci_val_upper, alpha=0.2, color=colors['val'])
        
        if X_test is not None and test_ids is not None:
            test_scores = pca.transform(X_test)
            pearsonr_test_all = compute_all_subject_pearsonr(X_test, test_scores, pca, ks)
            means_test = np.mean(pearsonr_test_all, axis=0)
            stds_test = np.std(pearsonr_test_all, axis=0)
            n_test = pearsonr_test_all.shape[0]
            ci_test = stats.t.interval(0.95, n_test - 1, loc=means_test, scale=stds_test / np.sqrt(n_test))
            ci_test_lower = ci_test[0]
            ci_test_upper = ci_test[1]
            
            plt.plot(ks, means_test, 'o-', label='Test', color=colors['test'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_test_lower, ci_test_upper, alpha=0.2, color=colors['test'])
        
        plt.xlabel("Number of PCA components", fontsize=12)
        plt.ylabel("Pearson $r$ (true vs reconstructed)", fontsize=12)
        plt.title(f"PCA Reconstruction Pearson Correlation Across All Subjects: {modality} ({parcellation})\n(Mean ± 95% Confidence Interval)", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

        # Plot 5: Demeaned correlation with error bands for ALL subjects
        # This evaluates how well we match non-mean patterns at individual subject level
        def compute_all_subject_demeaned_corr(X, scores, pca, ks):
            """
            Compute demeaned correlation for all subjects across different k values.
            Correlation is computed between (X_subj - μ_train) and (X̂_subj - μ_train), 
            where μ_train is the training set mean (pca.mean_).
            This evaluates how well we capture deviations from the training mean pattern.
            """
            n_subjects = X.shape[0]
            n_ks = len(ks)
            demeaned_corr_all = np.zeros((n_subjects, n_ks))
            
            # Training set mean (used for demeaning all splits)
            mu_train = pca.mean_  # (n_features,) - mean computed from training data only
            
            # Compute reconstructions and demeaned correlation for each k value
            for k_idx, k in enumerate(ks):
                # Reconstruct all subjects at once: (n_subjects, k) @ (k, n_features) -> (n_subjects, n_features)
                X_hat = np.dot(scores[:, :k], loadings[:k, :]) + mu_train
                
                # Demean both original and reconstructed data by subtracting training set mean
                X_demeaned = X - mu_train  # (n_subjects, n_features)
                X_hat_demeaned = X_hat - mu_train  # (n_subjects, n_features)
                
                # Compute correlation per subject (vectorized)
                # r = sum((X - μ_train) * (X_hat - μ_train)) / sqrt(sum((X - μ_train)^2) * sum((X_hat - μ_train)^2))
                numerator = np.sum(X_demeaned * X_hat_demeaned, axis=1)  # (n_subjects,)
                denom_X = np.sqrt(np.sum(X_demeaned ** 2, axis=1))  # (n_subjects,)
                denom_X_hat = np.sqrt(np.sum(X_hat_demeaned ** 2, axis=1))  # (n_subjects,)
                demeaned_corr_all[:, k_idx] = numerator / (denom_X * denom_X_hat + 1e-10)  # Add small epsilon to avoid division by zero
            
            return demeaned_corr_all

        plt.figure(figsize=(9, 6))
        colors = {'train': 'navy', 'val': 'darkorange', 'test': 'forestgreen'}
        
        # Compute demeaned correlation for all subjects in each split
        if X_train is not None:
            demeaned_corr_train_all = compute_all_subject_demeaned_corr(X_train, train_scores, pca, ks)
            means_train = np.mean(demeaned_corr_train_all, axis=0)
            stds_train = np.std(demeaned_corr_train_all, axis=0)
            # 95% confidence interval (using t-distribution)
            n_train = demeaned_corr_train_all.shape[0]
            ci_train = stats.t.interval(0.95, n_train - 1, loc=means_train, scale=stds_train / np.sqrt(n_train))
            ci_train_lower = ci_train[0]
            ci_train_upper = ci_train[1]
            
            plt.plot(ks, means_train, 'o-', label='Train', color=colors['train'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_train_lower, ci_train_upper, alpha=0.2, color=colors['train'])
        
        if X_val is not None and val_ids is not None:
            val_scores = pca.transform(X_val)
            demeaned_corr_val_all = compute_all_subject_demeaned_corr(X_val, val_scores, pca, ks)
            means_val = np.mean(demeaned_corr_val_all, axis=0)
            stds_val = np.std(demeaned_corr_val_all, axis=0)
            n_val = demeaned_corr_val_all.shape[0]
            ci_val = stats.t.interval(0.95, n_val - 1, loc=means_val, scale=stds_val / np.sqrt(n_val))
            ci_val_lower = ci_val[0]
            ci_val_upper = ci_val[1]
            
            plt.plot(ks, means_val, 'o-', label='Validation', color=colors['val'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_val_lower, ci_val_upper, alpha=0.2, color=colors['val'])
        
        if X_test is not None and test_ids is not None:
            test_scores = pca.transform(X_test)
            demeaned_corr_test_all = compute_all_subject_demeaned_corr(X_test, test_scores, pca, ks)
            means_test = np.mean(demeaned_corr_test_all, axis=0)
            stds_test = np.std(demeaned_corr_test_all, axis=0)
            n_test = demeaned_corr_test_all.shape[0]
            ci_test = stats.t.interval(0.95, n_test - 1, loc=means_test, scale=stds_test / np.sqrt(n_test))
            ci_test_lower = ci_test[0]
            ci_test_upper = ci_test[1]
            
            plt.plot(ks, means_test, 'o-', label='Test', color=colors['test'], linewidth=2, markersize=8)
            plt.fill_between(ks, ci_test_lower, ci_test_upper, alpha=0.2, color=colors['test'])
        
        plt.xlabel("Number of PCA components", fontsize=12)
        plt.ylabel("Demeaned Pearson $r$ ($X_{subj} - \\mu_{train}$ vs $\\hat{X}_{subj} - \\mu_{train}$)", fontsize=12)
        plt.title(f"PCA Reconstruction Demeaned Correlation Across All Subjects: {modality} ({parcellation})\n(Mean ± 95% Confidence Interval)", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

        return pca, {'train': train_scores, 'val': (pca.transform(X_val) if X_val is not None else None), 'test': (pca.transform(X_test) if X_test is not None else None)}, loadings
