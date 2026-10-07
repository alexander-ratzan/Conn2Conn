"""Spec v3 I5 check E: sha256 of FC -> SC CrossModal_PCA_PLS test predictions (plain FC source, default cohort,
seed 0) for the checkout given as argv[1], so the tree before I5 (a HEAD worktree) and after can be compared.

    python scripts/experiments/joint_taskfc2sc/checks/fingerprint_fc2sc.py <repo_root>
"""
import hashlib
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.hcp_dataset import HCP_Base  # noqa: E402
from models.registry import build_model  # noqa: E402

base = HCP_Base(parcellation="Glasser", hemi="both", source="FC", target="SC", shuffle_seed=0, data_load_mode="precomputed")
model = build_model(base, "CrossModal_PCA_PLS", {"n_components_pca": 128, "n_components_pca_target": 32, "n_components_pls": 16})
x = torch.as_tensor(base.fc_upper_triangles[base.trainvaltest_partition_indices["test"]])
pred = model(x).detach().cpu().numpy().astype(np.float32)
print(f"FINGERPRINT n={len(base.metadata_df)} {hashlib.sha256(pred.tobytes()).hexdigest()}", flush=True)
