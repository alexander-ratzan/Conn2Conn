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
    # HCP_Base slices sc_r2t_matrices to the canonical subject set during init
    # (hcp_dataset.py L235-237), so it is already aligned 1:1 with sc_matrices and
    # with the post-canonical metadata_df ordering. Use it directly.
    r2t_canonical = np.asarray(base.sc_r2t_matrices, dtype=np.float32)  # (n_canon, 360, 66)
    # Same for sc_r2t_corr_matrices (computed inline by HCP_Base on the already-sliced r2t).
    if hasattr(base, "sc_r2t_corr_matrices") and base.sc_r2t_corr_matrices is not None:
        r2t_corr = np.asarray(base.sc_r2t_corr_matrices, dtype=np.float32)
    else:
        r2t_corr = np.stack(
            [np.nan_to_num(np.corrcoef(mat), nan=0.0) for mat in r2t_canonical],
            axis=0,
        ).astype(np.float32)
    # r2t_canonical rows are indexed in the same canonical ordering as
    # base.sc_matrices / base.sc_upper_triangles, so train_idx/test_idx index
    # into it directly (same as SC_train = sc_upper_triangles[train_idx]).
    # Canonical set is the full sample (~957); the train/test partition is a
    # SUBSET of canonical, with val taking the remainder.
    assert r2t_canonical.shape[0] == base.sc_upper_triangles.shape[0], (
        f"r2t row count {r2t_canonical.shape[0]} != sc canonical "
        f"{base.sc_upper_triangles.shape[0]}"
    )

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


def block_pca_pls_predict(blocks_train, blocks_test, Y_train,
                          k_per_block=256, k_tgt=256, k_pls=64, max_iter=2000):
    """Scale-fair combined predictor: PCA each source block to its OWN latent space,
    concat the latents, then PLS -> inverse-PCA(Y).

    Fixes the naive-concat bug where a high-magnitude / high-dimensional block
    dominates a single shared PCA, making the other blocks invisible. Each block
    gets an equal k_per_block latent budget regardless of raw scale or width.

    blocks_train / blocks_test: list of (n_subj, d_block) arrays.
    Low-dim blocks (e.g. bv 16-dim, demo) are passed through raw if narrower than
    k_per_block (PCA can't make more comps than features).
    """
    Z_tr_parts, Z_te_parts = [], []
    for Xtr, Xte in zip(blocks_train, blocks_test):
        d = Xtr.shape[1]
        if d <= k_per_block:
            # Narrow block: standardize on train, pass through raw.
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True)
            sd = np.where(sd > 1e-8, sd, 1.0)
            Z_tr_parts.append((Xtr - mu) / sd)
            Z_te_parts.append((Xte - mu) / sd)
        else:
            p = PCA(n_components=k_per_block, random_state=0).fit(Xtr)
            Z_tr_parts.append(p.transform(Xtr))
            Z_te_parts.append(p.transform(Xte))
    Z_tr = np.concatenate(Z_tr_parts, axis=1)
    Z_te = np.concatenate(Z_te_parts, axis=1)

    pca_tgt = PCA(n_components=k_tgt, random_state=0).fit(Y_train)
    Y_tgt_tr = pca_tgt.transform(Y_train)
    pls = PLSRegression(n_components=k_pls, scale=True, max_iter=max_iter).fit(Z_tr, Y_tgt_tr)
    Y_tgt_te_pred = pls.predict(Z_te)
    return pca_tgt.inverse_transform(Y_tgt_te_pred).astype(np.float32)


def source_blocks(split, name):
    """Return list of source blocks (train_list, test_list) for a combined rep.
    Single-block reps return a one-element list (so block_pca_pls_predict works
    uniformly)."""
    SCtr, SCte = split["SC_train"], split["SC_test"]
    r2tr, r2te = split["r2t_flat_train"], split["r2t_flat_test"]
    bvtr, bvte = split["bv_train"], split["bv_test"]
    dmtr, dmte = split["demo_train"], split["demo_test"]
    M = {
        "SC":           ([SCtr], [SCte]),
        "r2t":          ([r2tr], [r2te]),
        "SC_r2t":       ([SCtr, r2tr], [SCte, r2te]),
        "kitchen_sink": ([SCtr, r2tr, bvtr, dmtr], [SCte, r2te, bvte, dmte]),
    }
    if name not in M:
        raise ValueError(f"unknown combined rep: {name!r}; available {sorted(M)}")
    return M[name]


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
