"""Shared setup for further_exploration notebooks.

Single source of truth for:
  - Data loading (Sim instantiation at a given shuffle_seed)
  - Closed-form predictors (pca_pls_predict, combined_predict, br_per_component_predict, fit_basis_ols)
  - Full-panel evaluator (full_panel_eval)
  - Phase 2 Analysis 1 family-structure helpers (pair indices, demeaned cosine, AUC vs unrelated)

These match STEP 3/6.0/8.1 of `../model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb`.
If a helper drifts from the main notebook, the MAIN NOTEBOOK is canonical — update here, then
re-run the dependent notebook.

Typical usage in a notebook:
    import sys; sys.path.insert(0, str(Path.cwd()))
    from _setup import *
    split = load_seed_split(seed=0)
    base, train_idx, test_idx = split["base"], split["train_idx"], split["test_idx"]
    FC_tr, FC_te = split["FC_train"], split["FC_test"]
    SC_tr, SC_te = split["SC_train"], split["SC_test"]
    bv_tr, bv_te = split["bv_train"], split["bv_test"]
    demo_tr, demo_te = split["demo_train"], split["demo_test"]
    bvdemo_tr, bvdemo_te = split["bvdemo_train"], split["bvdemo_test"]
"""
from __future__ import annotations
from pathlib import Path
import sys

# --- Project root on path ---
_REPO_ROOT = Path(__file__).resolve()
while _REPO_ROOT.name != "Conn2Conn" and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score

import main
import models.configs  # noqa: F401
import models.architectures  # noqa: F401
import models.eval.metrics as _metrics
from main import Sim

PARCELLATION = "Glasser"
DATA_LOAD_MODE = "precomputed"
CROSSMODAL_PCA_CONFIG = _REPO_ROOT / "models/configs/CrossModalPCA.yml"


# ============================================================================
# Data loading
# ============================================================================
def load_seed_split(seed: int = 0, source: str = "FC", target: str = "SC") -> dict:
    """Build Sim at the given shuffle_seed and return the standard train/test split.

    Returns dict with: base, train_idx, test_idx, FC_train/test, SC_train/test,
    bv_train/test, demo_train/test, bvdemo_train/test.
    """
    sim = Sim(
        model_name="CrossModalPCA",
        config_path=str(CROSSMODAL_PCA_CONFIG),
        source=source, target=target,
        parcellation=PARCELLATION, shuffle_seed=seed,
        data_load_mode=DATA_LOAD_MODE,
    )
    base = sim.base
    train_idx = base.trainvaltest_partition_indices["train"]
    test_idx  = base.trainvaltest_partition_indices["test"]

    FC_train = np.asarray(base.fc_upper_triangles[train_idx], dtype=np.float32)
    FC_test  = np.asarray(base.fc_upper_triangles[test_idx],  dtype=np.float32)
    SC_train = np.asarray(base.sc_upper_triangles[train_idx], dtype=np.float32)
    SC_test  = np.asarray(base.sc_upper_triangles[test_idx],  dtype=np.float32)

    bv_train = base.fs_volumes_z[train_idx]
    bv_test  = base.fs_volumes_z[test_idx]
    demo_train = np.concatenate([
        base.age_z[train_idx], base.sex_oh[train_idx], base.race_eth_oh[train_idx]
    ], axis=1).astype(np.float32)
    demo_test = np.concatenate([
        base.age_z[test_idx], base.sex_oh[test_idx], base.race_eth_oh[test_idx]
    ], axis=1).astype(np.float32)
    bvdemo_train = np.concatenate([bv_train, demo_train], axis=1)
    bvdemo_test  = np.concatenate([bv_test,  demo_test ], axis=1)

    return dict(
        sim=sim, base=base,
        train_idx=train_idx, test_idx=test_idx,
        FC_train=FC_train, FC_test=FC_test,
        SC_train=SC_train, SC_test=SC_test,
        bv_train=bv_train, bv_test=bv_test,
        demo_train=demo_train, demo_test=demo_test,
        bvdemo_train=bvdemo_train, bvdemo_test=bvdemo_test,
    )


# ============================================================================
# Closed-form predictors (mirror STEP 6.0 of main notebook)
# ============================================================================
def fit_basis_ols(X_train, X_test, Y_train):
    reg = LinearRegression().fit(X_train, Y_train)
    return (reg.predict(X_train).astype(np.float32),
            reg.predict(X_test ).astype(np.float32))


def pca_pls_predict(X_src_train, X_src_test, Y_train,
                    k_src=256, k_tgt=256, k_pls=64, max_iter=2000):
    pca_src = PCA(n_components=k_src, random_state=0).fit(X_src_train)
    pca_tgt = PCA(n_components=k_tgt, random_state=0).fit(Y_train)
    Z_src_tr = pca_src.transform(X_src_train)
    Z_src_te = pca_src.transform(X_src_test)
    Z_tgt_tr = pca_tgt.transform(Y_train)
    pls = PLSRegression(n_components=k_pls, scale=True, max_iter=max_iter).fit(Z_src_tr, Z_tgt_tr)
    Z_tgt_te_pred = pls.predict(Z_src_te)
    return pca_tgt.inverse_transform(Z_tgt_te_pred).astype(np.float32)


def br_per_component_predict(X_src_train, X_src_test, Y_train,
                             k_src=256, k_tgt=256, max_iter=300):
    pca_src = PCA(n_components=k_src, random_state=0).fit(X_src_train)
    pca_tgt = PCA(n_components=k_tgt, random_state=0).fit(Y_train)
    Z_src_tr = pca_src.transform(X_src_train)
    Z_src_te = pca_src.transform(X_src_test)
    Z_tgt_tr = pca_tgt.transform(Y_train)
    Z_tgt_te_pred = np.zeros((Z_src_te.shape[0], k_tgt), dtype=np.float32)
    for k in range(k_tgt):
        m = BayesianRidge(max_iter=max_iter).fit(Z_src_tr, Z_tgt_tr[:, k])
        Z_tgt_te_pred[:, k] = m.predict(Z_src_te)
    return pca_tgt.inverse_transform(Z_tgt_te_pred).astype(np.float32)


def combined_predict(X_xmod_train, X_xmod_test, Y_train, basis_train, basis_test,
                     k_src=256, k_tgt=256, max_iter=300):
    """PCA(xmod) -> concat with basis -> BR per target PCA component -> inverse-PCA."""
    pca_src = PCA(n_components=k_src, random_state=0).fit(X_xmod_train)
    pca_tgt = PCA(n_components=k_tgt, random_state=0).fit(Y_train)
    Z_src_tr = pca_src.transform(X_xmod_train)
    Z_src_te = pca_src.transform(X_xmod_test)
    Z_tgt_tr = pca_tgt.transform(Y_train)
    X_full_tr = np.concatenate([Z_src_tr, basis_train], axis=1)
    X_full_te = np.concatenate([Z_src_te, basis_test ], axis=1)
    Z_tgt_te_pred = np.zeros((X_full_te.shape[0], k_tgt), dtype=np.float32)
    for k in range(k_tgt):
        br = BayesianRidge(max_iter=max_iter).fit(X_full_tr, Z_tgt_tr[:, k])
        Z_tgt_te_pred[:, k] = br.predict(X_full_te)
    return pca_tgt.inverse_transform(Z_tgt_te_pred).astype(np.float32)


# ============================================================================
# Full panel evaluator (mse, r2, pearson, demeaned_pearson, top1_acc, avg_rank)
# ============================================================================
def full_panel_eval(y_pred, y_true, target_train_mean_vec):
    yp = np.asarray(y_pred, dtype=np.float32)
    yt = np.asarray(y_true, dtype=np.float32)
    mu = np.asarray(target_train_mean_vec, dtype=np.float32)
    cc_raw = _metrics.compute_corr_matrix(
        torch.tensor(yt, dtype=torch.float32),
        torch.tensor(yp, dtype=torch.float32),
    )
    if hasattr(cc_raw, "cpu"):
        cc_raw = cc_raw.cpu().numpy()
    panel = _metrics.compute_basic_regression_metrics(
        torch.tensor(yp, dtype=torch.float32),
        torch.tensor(yt, dtype=torch.float32),
        corr_matrix=torch.tensor(cc_raw, dtype=torch.float32),
        corr_matrix_demeaned=None,
    )
    if isinstance(panel, dict):
        panel = {k: float(v) if hasattr(v, "item") else float(v) for k, v in panel.items()}
    yp_dm = yp - mu
    yt_dm = yt - mu
    num   = (yp_dm * yt_dm).sum(axis=1)
    den_p = np.sqrt((yp_dm ** 2).sum(axis=1))
    den_t = np.sqrt((yt_dm ** 2).sum(axis=1))
    panel["demeaned_pearson"] = float((num / (den_p * den_t + 1e-10)).mean())
    return panel


# ============================================================================
# Phase 2 Analysis 1 helpers — family structure pair analysis (mirror STEP 8.1)
# ============================================================================
def pair_indices_by_relation(metadata_df, test_indices, rng, age_tol_yrs=3.0):
    """Return dict of (i, j) pairs (LOCAL test-set indices) bucketed by relation.

    Categories: MZ, DZ, sibling (same Family_ID, not both MZ/DZ),
    unrelated_matched (different Family_ID, same sex, within age tolerance).
    """
    md = metadata_df.iloc[test_indices].reset_index(drop=True)
    n  = len(md)
    fam = md["Family_ID"].values
    rel = md["Family_Relation"].values
    age = md["age"].values.astype(float)
    sex = md["sex"].values

    pairs = {"MZ": [], "DZ": [], "sibling": []}
    for i in range(n):
        for j in range(i + 1, n):
            if pd.isna(fam[i]) or pd.isna(fam[j]) or fam[i] != fam[j]:
                continue
            if rel[i] == "MZ" and rel[j] == "MZ":
                pairs["MZ"].append((i, j))
            elif rel[i] == "DZ" and rel[j] == "DZ":
                pairs["DZ"].append((i, j))
            else:
                pairs["sibling"].append((i, j))

    unrelated_all = []
    for i in range(n):
        for j in range(i + 1, n):
            if pd.isna(fam[i]) or pd.isna(fam[j]) or fam[i] != fam[j]:
                if sex[i] == sex[j] and abs(age[i] - age[j]) <= age_tol_yrs:
                    unrelated_all.append((i, j))

    n_related = len(pairs["MZ"]) + len(pairs["DZ"]) + len(pairs["sibling"])
    if n_related == 0 or len(unrelated_all) == 0:
        matched = []
    else:
        n_sample = min(n_related, len(unrelated_all))
        idx = rng.choice(len(unrelated_all), size=n_sample, replace=False)
        matched = [unrelated_all[k] for k in idx]
    pairs["unrelated_matched"] = matched
    return pairs


def demeaned_cosine_pair_sim(X_test, train_mean_vec):
    """Symmetric (N_test, N_test) matrix of cos(x_i - mu, x_j - mu)."""
    X = np.asarray(X_test, dtype=np.float32)
    mu = np.asarray(train_mean_vec, dtype=np.float32)
    Z = X - mu
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.maximum(norms, 1e-12)
    return Zn @ Zn.T


def extract_pair_sims(sim_matrix, pairs_by_rel):
    out = {}
    for rel, pair_list in pairs_by_rel.items():
        if not pair_list:
            out[rel] = np.array([], dtype=np.float32)
            continue
        idx = np.asarray(pair_list)
        out[rel] = sim_matrix[idx[:, 0], idx[:, 1]].astype(np.float32)
    return out


def auc_vs_unrelated(sims_by_rel):
    """AUC of (relation vs unrelated_matched). Returns dict {rel: auc}."""
    ref = sims_by_rel.get("unrelated_matched", np.array([]))
    if ref.size < 2:
        return {}
    out = {}
    for rel in ("MZ", "DZ", "sibling"):
        scores = sims_by_rel.get(rel, np.array([]))
        if scores.size < 2:
            out[rel] = float("nan")
            continue
        y = np.concatenate([np.ones(scores.size), np.zeros(ref.size)])
        s = np.concatenate([scores, ref])
        out[rel] = float(roc_auc_score(y, s))
    return out


# ============================================================================
# Convenience: results directory
# ============================================================================
RESULTS_ROOT = _REPO_ROOT / "notebooks-FC_to_SC-experimental/model_overviews/results/local_results/further_exploration"


def results_dir(notebook_name: str) -> Path:
    p = RESULTS_ROOT / notebook_name
    p.mkdir(parents=True, exist_ok=True)
    return p
