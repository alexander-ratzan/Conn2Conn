"""Shared helpers for tractography_predict experiments.

Adds two source representations on top of the standard load_seed_split:
  - r2t_flat:    flattened region-to-tract profile (n_subj, 360 * 66 = 23,760)
  - r2t_corr_tri: upper-triangle of the (360, 360) corrcoef of r2t rows (n_subj, 64,620)

Both come from base.sc_r2t_matrices (loaded from r2t_matrices.npy in the SC cache).
The HCP_Base computes sc_r2t_corr_matrices inline at load time when r2t is available.
"""
from pathlib import Path
import sys
import numpy as np

# Locate _setup.py and reuse it (data loading + closed-form predictors).
for _cand in [Path(__file__).resolve().parent.parent / "further_exploration",
              Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration"),
              Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration")]:
    if (_cand / "_setup.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError("could not locate further_exploration/_setup.py")
from _setup import (load_seed_split, pca_pls_predict, combined_predict,
                    br_per_component_predict, fit_basis_ols, full_panel_eval,
                    pair_indices_by_relation, demeaned_cosine_pair_sim,
                    extract_pair_sims, auc_vs_unrelated,
                    PARCELLATION, DATA_LOAD_MODE, CROSSMODAL_PCA_CONFIG,
                    PCA, BayesianRidge, LinearRegression, PLSRegression)

N_REGIONS = 360
N_TRACTS  = 66
N_EDGES   = N_REGIONS * (N_REGIONS - 1) // 2   # 64620
N_R2T_FLAT = N_REGIONS * N_TRACTS              # 23760

_triu_i, _triu_j = np.triu_indices(N_REGIONS, k=1)


def load_seed_split_with_r2t(seed: int) -> dict:
    """Same as load_seed_split, plus r2t_flat and r2t_corr_tri train/test arrays.

    base.sc_r2t_matrices: shape (n_subj, 360, 66).
    base.sc_r2t_corr_matrices: shape (n_subj, 360, 360), computed inline by HCP_Base.

    Both are aligned to the canonical subject intersection (same ordering as
    fc_matrices / sc_matrices).
    """
    split = load_seed_split(seed=seed)
    base = split["base"]
    if not hasattr(base, "sc_r2t_matrices") or base.sc_r2t_matrices is None:
        raise RuntimeError(
            "base.sc_r2t_matrices is None — confirm the SC cache directory contains "
            "r2t_matrices.npy. (Cache path: "
            f"{getattr(base, 'precompute_cache_root', '?')})"
        )
    r2t = np.asarray(base.sc_r2t_matrices, dtype=np.float32)  # (n_subj_full, 360, 66)
    # base loads at the full population, then the canonical intersection is sliced;
    # we need to align r2t to the canonical fc/sc subject ordering.
    # The HCP_Base loader does NOT slice sc_r2t_matrices on the canonical intersection
    # automatically — we must do it here using fc/sc subject ordering.
    canonical_ids = list(base.metadata_df["subject"])
    # sc_r2t_matrices is indexed by sc_subject_ids; align via id->index map.
    sc_id_to_idx = {sid: i for i, sid in enumerate(base.sc_subject_ids)}
    canonical_sc_idx = np.array([sc_id_to_idx[sid] for sid in canonical_ids], dtype=int)
    r2t_canonical = r2t[canonical_sc_idx]      # (n_canonical_subj, 360, 66)

    if hasattr(base, "sc_r2t_corr_matrices") and base.sc_r2t_corr_matrices is not None:
        r2t_corr = np.asarray(base.sc_r2t_corr_matrices, dtype=np.float32)[canonical_sc_idx]
    else:
        # Recompute if not available; corrcoef of r2t rows = (360, 360) symmetric.
        r2t_corr = np.stack(
            [np.nan_to_num(np.corrcoef(mat), nan=0.0) for mat in r2t_canonical],
            axis=0,
        ).astype(np.float32)

    tr = split["train_idx"]
    te = split["test_idx"]

    split["r2t_flat_train"]     = r2t_canonical[tr].reshape(len(tr), -1)       # (n_tr, 23760)
    split["r2t_flat_test"]      = r2t_canonical[te].reshape(len(te), -1)
    split["r2t_corr_tri_train"] = r2t_corr[tr][:, _triu_i, _triu_j].astype(np.float32)
    split["r2t_corr_tri_test"]  = r2t_corr[te][:, _triu_i, _triu_j].astype(np.float32)
    # Per-subject raw r2t (for E4 PC-localization later).
    split["r2t_train"]          = r2t_canonical[tr]
    split["r2t_test"]           = r2t_canonical[te]
    return split


def source_train_test(split, name):
    """Pull (X_train, X_test) for a named source representation."""
    M = {
        "SC":         (split["SC_train"],         split["SC_test"]),
        "r2t":        (split["r2t_flat_train"],   split["r2t_flat_test"]),
        "r2t_corr":   (split["r2t_corr_tri_train"], split["r2t_corr_tri_test"]),
        "SC_r2t":     (np.concatenate([split["SC_train"],       split["r2t_flat_train"]], axis=1),
                       np.concatenate([split["SC_test"],        split["r2t_flat_test"]],  axis=1)),
        "kitchen_sink": (
            np.concatenate([split["r2t_flat_train"], split["SC_train"],
                            split["bv_train"],       split["demo_train"]], axis=1),
            np.concatenate([split["r2t_flat_test"],  split["SC_test"],
                            split["bv_test"],        split["demo_test"]],  axis=1),
        ),
        "FC":         (split["FC_train"],         split["FC_test"]),
    }
    if name not in M:
        raise ValueError(f"unknown source rep: {name!r}; available {sorted(M)}")
    return M[name]


def target_train_test(split, name):
    """Pull (Y_train, Y_test, Y_train_mean) for a named target."""
    if name == "FC":
        return split["FC_train"], split["FC_test"], split["FC_train"].mean(axis=0)
    if name == "SC":
        return split["SC_train"], split["SC_test"], split["SC_train"].mean(axis=0)
    if name == "r2t":
        return split["r2t_flat_train"], split["r2t_flat_test"], split["r2t_flat_train"].mean(axis=0)
    if name == "r2t_corr":
        return split["r2t_corr_tri_train"], split["r2t_corr_tri_test"], split["r2t_corr_tri_train"].mean(axis=0)
    raise ValueError(f"unknown target: {name!r}")
