"""Shared helpers for the F6/F7/F8 family-structure + mechanism grid.

Reuses the project's single source of truth (`further_exploration/_setup.py`, imported via
the main grid's `_grid_common`) for data loading and the family-structure pair helpers — so
every number matches the notebooks exactly. Adds ONLY the few aggregation helpers that live
in the notebook cells (STEP 8.1 / 8.3) and never made it into `_setup`:

  - zscore_by_unrelated, perm_p_auc, bootstrap_auc   (copied verbatim from STEP 8.1, cell 49)
  - fdr_bh                                            (copied verbatim from STEP 8.3, cell 51)

plus the per-seed variant builder (STEP 8.2, cell 50) and the aggregation driver (STEP 8.3),
both faithful ports. The notebook is canonical; if a helper drifts, fix the notebook first.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score               # noqa: E402  (lightweight; no torch)

# The data layer (_grid_common -> _setup -> main/torch) is HEAVY and only available on the
# compute host. Import it LAZILY so the pure aggregation helpers below (and their tests)
# run anywhere, e.g. on a laptop with no torch. Data-dependent functions call _data().
_HERE = Path(__file__).resolve()
REPRO_ROOT = _HERE.parent.parent                       # .../Conn2Conn/reproduction
_DATA = {}


def _data():
    """Lazily import + cache the data layer. Raises only when actually loading data."""
    if not _DATA:
        if str(REPRO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPRO_ROOT))
        import _grid_common as gc                        # noqa: E402  (also puts _setup on path)
        from _setup import (                             # noqa: E402  (verbatim notebook mirrors)
            load_seed_split, fit_basis_ols, pca_pls_predict, combined_predict,
            pair_indices_by_relation, demeaned_cosine_pair_sim, extract_pair_sims,
            auc_vs_unrelated as _auc,
        )
        _DATA.update(dict(
            gc=gc, load_seed_split=load_seed_split, fit_basis_ols=fit_basis_ols,
            pca_pls_predict=pca_pls_predict, combined_predict=combined_predict,
            pair_indices_by_relation=pair_indices_by_relation,
            demeaned_cosine_pair_sim=demeaned_cosine_pair_sim,
            extract_pair_sims=extract_pair_sims, auc_vs_unrelated=_auc,
            set_parcellation=gc.set_parcellation, PARCELLATIONS=gc.PARCELLATIONS,
        ))
    return _DATA


def auc_vs_unrelated(sims_by_rel):
    """AUC of (relation vs unrelated_matched). Self-contained (no data layer) so it is
    importable + testable without torch; identical to _setup.auc_vs_unrelated."""
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

# --- the F6 variant set, relations, pairing tolerance (STEP 8.2 constants) ------------------
FAM_VARIANTS = [
    "obs_SC", "obs_FC",
    "pred_SC_raw", "pred_SC_resid_bvdemo",
    "combined_pred_SC", "bvdemo_to_SC",
    "pred_FC_raw", "pred_FC_resid_bvdemo",
]
RELATIONS = ["MZ", "DZ", "sibling", "unrelated_matched"]
PAIR_AGE_TOL = 3.0   # years
PARCELLATIONS = ["Glasser", "4S456Parcels"]   # torch-free constant (matches _grid_common)


# ============================================================================
# Aggregation helpers — VERBATIM from STEP 8.1 (cell 49) and STEP 8.3 (cell 51),
# de-underscored and using the public roc_auc_score. Behaviour identical.
# ============================================================================
def zscore_by_unrelated(sims_by_rel):
    """Z-score every relation bucket by mean+std of the unrelated_matched bucket."""
    ref = sims_by_rel.get("unrelated_matched", np.array([]))
    if ref.size < 2:
        return sims_by_rel
    mu, sd = float(ref.mean()), float(ref.std(ddof=1))
    if sd == 0:
        return sims_by_rel
    return {rel: (v - mu) / sd for rel, v in sims_by_rel.items()}


def perm_p_auc(sims_by_rel, n_perm=10_000, rng=None):
    """Permutation null on AUC by shuffling rel-vs-unrelated labels. {rel: p_two_sided}."""
    if rng is None:
        rng = np.random.default_rng(0)
    ref = sims_by_rel.get("unrelated_matched", np.array([]))
    if ref.size < 2:
        return {}
    out = {}
    for rel in ("MZ", "DZ", "sibling"):
        scores = sims_by_rel.get(rel, np.array([]))
        if scores.size < 2:
            out[rel] = float("nan")
            continue
        s = np.concatenate([scores, ref])
        y = np.concatenate([np.ones(scores.size), np.zeros(ref.size)])
        obs = roc_auc_score(y, s)
        null = np.empty(n_perm, dtype=np.float32)
        for k in range(n_perm):
            yk = rng.permutation(y)
            null[k] = roc_auc_score(yk, s)
        out[rel] = float((np.abs(null - 0.5) >= abs(obs - 0.5)).mean())
    return out


def bootstrap_auc(sims_by_rel, n_boot=1000, rng=None):
    """Per-pair bootstrap on AUC; {rel: (auc_lo, auc_hi)} 95% CI."""
    if rng is None:
        rng = np.random.default_rng(0)
    ref = sims_by_rel.get("unrelated_matched", np.array([]))
    if ref.size < 2:
        return {}
    out = {}
    for rel in ("MZ", "DZ", "sibling"):
        scores = sims_by_rel.get(rel, np.array([]))
        if scores.size < 2:
            out[rel] = (float("nan"), float("nan"))
            continue
        s = np.concatenate([scores, ref])
        y = np.concatenate([np.ones(scores.size), np.zeros(ref.size)])
        n = len(s)
        bs = np.empty(n_boot, dtype=np.float32)
        for k in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                bs[k] = roc_auc_score(y[idx], s[idx])
            except ValueError:
                bs[k] = np.nan
        bs = bs[~np.isnan(bs)]
        if bs.size < 10:
            out[rel] = (float("nan"), float("nan"))
        else:
            out[rel] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR. Returns (reject_bool_array, padj_array).
    Drop-in for statsmodels multipletests(..., method='fdr_bh')."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    padj = np.empty(n)
    padj[order] = adj
    return padj <= alpha, padj


# ============================================================================
# Per-seed variant builder — faithful port of STEP 8.2 (cell 50).
# Consumes a load_seed_split() dict; returns {variant: array}, {variant: train_mean}.
# ============================================================================
def build_family_variants(split):
    """Build the 8 connectome variants on TEST in their natural prediction space."""
    D = _data()
    fit_basis_ols = D["fit_basis_ols"]; pca_pls_predict = D["pca_pls_predict"]
    combined_predict = D["combined_predict"]
    SC_tr, SC_te = split["SC_train"], split["SC_test"]
    FC_tr, FC_te = split["FC_train"], split["FC_test"]
    Xbd_tr, Xbd_te = split["bvdemo_train"], split["bvdemo_test"]

    # bv+demo residuals (target-side), exactly as STEP 8.2
    SC_bd_tr_pred, SC_bd_te_pred = fit_basis_ols(Xbd_tr, Xbd_te, SC_tr)
    FC_bd_tr_pred, FC_bd_te_pred = fit_basis_ols(Xbd_tr, Xbd_te, FC_tr)
    SC_res_tr = (SC_tr - SC_bd_tr_pred).astype(np.float32)
    FC_res_tr = (FC_tr - FC_bd_tr_pred).astype(np.float32)

    var, mean = {}, {}
    var["obs_SC"], mean["obs_SC"] = SC_te, SC_tr.mean(axis=0)
    var["obs_FC"], mean["obs_FC"] = FC_te, FC_tr.mean(axis=0)
    var["pred_SC_raw"] = pca_pls_predict(FC_tr, FC_te, SC_tr)
    mean["pred_SC_raw"] = SC_tr.mean(axis=0)
    var["pred_SC_resid_bvdemo"] = pca_pls_predict(FC_tr, FC_te, SC_res_tr)
    mean["pred_SC_resid_bvdemo"] = SC_res_tr.mean(axis=0)
    var["combined_pred_SC"] = combined_predict(FC_tr, FC_te, SC_tr, Xbd_tr, Xbd_te)
    mean["combined_pred_SC"] = SC_tr.mean(axis=0)
    var["bvdemo_to_SC"], mean["bvdemo_to_SC"] = SC_bd_te_pred, SC_tr.mean(axis=0)
    var["pred_FC_raw"] = pca_pls_predict(SC_tr, SC_te, FC_tr)
    mean["pred_FC_raw"] = FC_tr.mean(axis=0)
    var["pred_FC_resid_bvdemo"] = pca_pls_predict(SC_tr, SC_te, FC_res_tr)
    mean["pred_FC_resid_bvdemo"] = FC_res_tr.mean(axis=0)
    return var, mean


def pair_sims_for_seed(split, seed):
    """Build variants + extract per-pair sims bucketed by relation, for ONE seed.
    Mirrors STEP 8.2: pair rng = default_rng(42 + seed)."""
    D = _data()
    var, mean = build_family_variants(split)
    rng = np.random.default_rng(42 + seed)
    pairs_by_rel = D["pair_indices_by_relation"](
        split["base"].metadata_df, split["test_idx"], rng, PAIR_AGE_TOL)
    sims = {}
    for v in FAM_VARIANTS:
        sim_mat = D["demeaned_cosine_pair_sim"](var[v], mean[v])
        sims[v] = D["extract_pair_sims"](sim_mat, pairs_by_rel)
    pair_counts = {r: len(p) for r, p in pairs_by_rel.items()}
    return sims, pair_counts


# ============================================================================
# Aggregation driver — faithful port of STEP 8.3 (cell 51).
# Input: pooled {variant: {relation: sims_array}} (concatenated across seeds).
# Output: DataFrame with the exact columns of aggregate_auc.csv.
# Uses the notebook's RNG seeds: perm default_rng(42), boot default_rng(43).
# ============================================================================
def aggregate_family(pooled, n_perm=10_000, n_boot=1000):
    rng_perm = np.random.default_rng(42)
    rng_boot = np.random.default_rng(43)
    records = []
    for v in FAM_VARIANTS:
        aucs = auc_vs_unrelated(pooled[v])
        pvals = perm_p_auc(pooled[v], n_perm=n_perm, rng=rng_perm)
        bootci = bootstrap_auc(pooled[v], n_boot=n_boot, rng=rng_boot)
        for rel in ("MZ", "DZ", "sibling"):
            records.append({
                "variant": v, "relation": rel,
                "n_pairs": int(pooled[v][rel].size),
                "auc": aucs.get(rel, np.nan),
                "auc_lo": bootci.get(rel, (np.nan, np.nan))[0],
                "auc_hi": bootci.get(rel, (np.nan, np.nan))[1],
                "p_perm": pvals.get(rel, np.nan),
            })
    df = pd.DataFrame(records)
    mask = df["p_perm"].notna()
    if mask.any():
        reject, padj = fdr_bh(df.loc[mask, "p_perm"].values)
        df.loc[mask, "p_fdr"] = padj
        df.loc[mask, "sig_fdr"] = reject
    return df


def pool_seed_sims(seed_sims_list):
    """Concatenate per-seed {variant:{rel:arr}} dicts into one pooled dict (STEP 8.3 step 1)."""
    pooled = {v: {r: [] for r in RELATIONS} for v in FAM_VARIANTS}
    for sims in seed_sims_list:
        for v in FAM_VARIANTS:
            for r in RELATIONS:
                arr = sims[v].get(r, np.array([]))
                if arr.size:
                    pooled[v][r].append(arr)
    return {v: {r: (np.concatenate(lst) if lst else np.array([])) for r, lst in by.items()}
            for v, by in pooled.items()}
