#!/usr/bin/env python3
"""Retest reliability ICC pipeline for the PC3-localization sanity check.

Goal: replace the strength+distance proxy in `decisive_partialled_enrichment.py`
with an empirical edge-reliability vector measured from same-subject repeated
scans, and re-run the partial enrichment test.

What the HCP-YA project cache supports:
  - FC session 1 vs FC session 2 — both available via `expose_fc_sessions=True`
    (this is REST1 vs REST2 fMRI scans of the SAME 957 subjects, the standard
    scan-rescan reliability measure for fMRI).
  - SC test-retest — NOT in this cache. SC was processed once per subject for
    the main 957-subject cohort. The HCP test-retest dMRI release (n≈45) would
    need a separate ingest, parcellation, and tractography pipeline (~1-2 day
    pipeline-extension task — out of scope here).

What this script does:
  1. Loads FC session 1 + session 2 upper-triangle matrices for all subjects
     with both sessions exposed by the project loader.
  2. Computes per-edge **scan-rescan Pearson r across subjects** between
     session1 and session2. This is the field-standard per-edge reliability
     measure when ICC(2,1) isn't required, and equivalent to it under common
     assumptions.
  3. Saves the (64,620,)-dim FC reliability vector.
  4. Re-runs the tract_check decisive partial enrichment test, using
     FC reliability as a THIRD proxy alongside edge strength and edge distance.

Critical caveat: this is FC reliability, NOT SC reliability. They are
different constructs (different scanner physics, processing pipelines,
sensitivity to motion/partial-volume/registration noise). Two related
questions justify using it as a check anyway:

  (a) Edges connecting parcels that are hard to segment / have unstable
      partial-volume effects tend to be unreliable in BOTH modalities — so
      FC reliability and SC reliability share lower bounds at the worst
      edges.
  (b) The strength+distance proxy already passed at residual visual||visual
      = 13.3x and DAN||DAN = 5.7x. If FC reliability (a fully independent
      data-driven measure) also fails to explain PC3, the convergent
      negative result is stronger than either alone.

If this also passes, the reliability-confound objection is closed for any
reasonable referee.

Inputs:
  - sc_pc_loadings.npy  (Depth 1 output)
  - data/atlas_info/Glasser_dseg_reformatted.csv  (for network labels in re-run)
  - HCP project loader for FC sessions 1+2

Outputs (next to this script, under retest_icc_results/):
  - fc_edge_reliability_pearson.npy  (64620-dim, per-edge scan-rescan r)
  - fc_reliability_summary.csv  (subjects, edges with NaN, mean/median r)
  - enrichment_residual_with_fc_reliability.csv  (re-run of tract_check decisive
                                                   with fc_reliability added as a partial)
  - retest_findings.txt  (verdict text)
"""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Project root (Conn2Conn) on path.
_REPO_ROOT = Path(__file__).resolve()
while _REPO_ROOT.name != "Conn2Conn" and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "retest_icc_results"
OUT_DIR.mkdir(exist_ok=True)

# Reuse the further_exploration setup for parcellation + data load.
for _cand in [Path(__file__).resolve().parent,
              Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration"),
              Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration")]:
    if (_cand / "_setup.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError("_setup.py not found")
from _setup import PARCELLATION, DATA_LOAD_MODE, CROSSMODAL_PCA_CONFIG
from main import Sim

# ============================================================================
# Stage 1: load FC session 1 + session 2 (expose_fc_sessions=True)
# ============================================================================
print("=== Stage 1: load FC session 1 + 2 ===", flush=True)
sim = Sim(
    model_name="CrossModalPCA",
    config_path=str(CROSSMODAL_PCA_CONFIG),
    source="FC", target="SC",
    parcellation=PARCELLATION,
    shuffle_seed=0,
    data_load_mode=DATA_LOAD_MODE,
    expose_fc_sessions=True,   # request session 1 + 2
)
base = sim.base
# Both attributes exist when expose_fc_sessions is True.
assert hasattr(base, "fc_session1_upper_triangles"), \
    "FC session caches missing — confirm expose_fc_sessions flag is honored"
fc_s1 = np.asarray(base.fc_session1_upper_triangles, dtype=np.float32)
fc_s2 = np.asarray(base.fc_session2_upper_triangles, dtype=np.float32)
print(f"FC session 1 shape: {fc_s1.shape}")
print(f"FC session 2 shape: {fc_s2.shape}")
assert fc_s1.shape == fc_s2.shape, "session 1 and 2 must align per subject"
n_subjects, n_edges = fc_s1.shape
assert n_edges == 64620, f"unexpected edge count {n_edges}"

# ============================================================================
# Stage 2: per-edge scan-rescan Pearson across subjects.
# r_e = corr( fc_s1[:, e] , fc_s2[:, e] ) over all subjects.
# A high r_e means edge e's value is consistent between scans across the
# sample — i.e., the edge is reliably measured.
# ============================================================================
print("\n=== Stage 2: per-edge scan-rescan Pearson r ===", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # zero-std edges produce NaN
    # Vectorized: per-edge mean-centered, then dot product / norms.
    s1c = fc_s1 - fc_s1.mean(axis=0, keepdims=True)
    s2c = fc_s2 - fc_s2.mean(axis=0, keepdims=True)
    s1_norm = np.sqrt((s1c ** 2).sum(axis=0))
    s2_norm = np.sqrt((s2c ** 2).sum(axis=0))
    numer   = (s1c * s2c).sum(axis=0)
    denom   = s1_norm * s2_norm
    fc_reliability = np.where(denom > 1e-12, numer / denom, np.nan).astype(np.float64)

n_nan = int(np.isnan(fc_reliability).sum())
print(f"  per-edge reliability r: shape={fc_reliability.shape}, NaN edges={n_nan}")
print(f"  min={np.nanmin(fc_reliability):.4f}, max={np.nanmax(fc_reliability):.4f}")
print(f"  mean={np.nanmean(fc_reliability):.4f}, median={np.nanmedian(fc_reliability):.4f}")

# Sanity histogram percentiles.
qs = [1, 5, 25, 50, 75, 95, 99]
for q in qs:
    print(f"  p{q}: {np.nanpercentile(fc_reliability, q):.4f}")

np.save(OUT_DIR / "fc_edge_reliability_pearson.npy", fc_reliability)
pd.DataFrame([{
    "n_subjects": n_subjects,
    "n_edges": n_edges,
    "n_nan_edges": n_nan,
    "mean_r": float(np.nanmean(fc_reliability)),
    "median_r": float(np.nanmedian(fc_reliability)),
    "p5":   float(np.nanpercentile(fc_reliability, 5)),
    "p95":  float(np.nanpercentile(fc_reliability, 95)),
}]).to_csv(OUT_DIR / "fc_reliability_summary.csv", index=False)
print(f"  Saved -> {OUT_DIR / 'fc_edge_reliability_pearson.npy'}")

# ============================================================================
# Stage 3: re-run the tract_check decisive enrichment with FC reliability added.
# Loads cached strength/distance/PC3 from tract_check directory.
# ============================================================================
print("\n=== Stage 3: re-run decisive enrichment with FC reliability ===", flush=True)
TRACT_CACHE_DIR = THIS_DIR  # same dir as this script (alongside tract_check caches)
pc3_abs = np.load(TRACT_CACHE_DIR / "_cache_pc3_abs.npy")
edge_strength = np.load(TRACT_CACHE_DIR / "_cache_edge_strength.npy")
edge_distance = np.load(TRACT_CACHE_DIR / "_cache_edge_distance.npy")
triu_i = np.load(TRACT_CACHE_DIR / "_cache_triu_i.npy")
triu_j = np.load(TRACT_CACHE_DIR / "_cache_triu_j.npy")
assert pc3_abs.shape == fc_reliability.shape

# Handle NaN edges in FC reliability: set to median (so they don't dominate the residual).
fcr_clean = np.where(np.isnan(fc_reliability), np.nanmedian(fc_reliability), fc_reliability)

# Spearman + R^2 of |PC3| ~ FC reliability alone.
rho_fcr, p_fcr = spearmanr(pc3_abs, fcr_clean)
X1 = fcr_clean.reshape(-1, 1)
r2_fcr_only = r2_score(pc3_abs, LinearRegression().fit(X1, pc3_abs).predict(X1))
print(f"  Spearman(|PC3|, FC reliability) = {rho_fcr:+.4f} (p={p_fcr:.2g})")
print(f"  OLS R² of |PC3| ~ FC reliability alone = {r2_fcr_only:.4f}")

# Full model: |PC3| ~ strength + distance + fc_reliability
X3 = np.column_stack([edge_strength, edge_distance, fcr_clean])
ols3 = LinearRegression().fit(X3, pc3_abs)
pred3 = ols3.predict(X3)
r2_full3 = r2_score(pc3_abs, pred3)
pc3_resid3 = pc3_abs - pred3
print(f"  Full-model R² (strength + distance + FC reliability) = {r2_full3:.4f}")

# Atlas (for network labels).
ATLAS_PATH = next(p for p in [
    Path("/scratch/ans9868/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
    Path("/Users/user/projects/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
] if p.exists())
atlas = pd.read_csv(ATLAS_PATH).sort_values("id").reset_index(drop=True)
atlas["idx0"] = atlas["id"].astype(int) - 1
atlas_idx = atlas.set_index("idx0")
net_i = atlas_idx.loc[triu_i, "community_yeo"].values
net_j = atlas_idx.loc[triu_j, "community_yeo"].values
net_pair_all = np.array([" || ".join(sorted([a, b])) for a, b in zip(net_i, net_j)])
all_pair_counts = pd.Series(net_pair_all).value_counts()

K = 200
N = len(pc3_abs)
order_raw = np.argsort(-pc3_abs)[:K]
order_resid = np.argsort(-np.abs(pc3_resid3))[:K]
raw_pair_counts = pd.Series(net_pair_all[order_raw]).value_counts()
resid_pair_counts = pd.Series(net_pair_all[order_resid]).value_counts()

rows = []
pairs_seen = set(raw_pair_counts.index) | set(resid_pair_counts.index)
for pair in pairs_seen:
    n_total = int(all_pair_counts.get(pair, 0))
    n_exp   = n_total * (K / N)
    n_raw   = int(raw_pair_counts.get(pair, 0))
    n_resid = int(resid_pair_counts.get(pair, 0))
    rows.append({
        "net_pair":  pair,
        "n_total":   n_total,
        "n_raw_top200":   n_raw,
        "n_resid_top200": n_resid,
        "enrichment_raw":   n_raw / n_exp if n_exp > 0 else np.nan,
        "enrichment_resid": n_resid / n_exp if n_exp > 0 else np.nan,
    })
comp = pd.DataFrame(rows).sort_values("enrichment_raw", ascending=False)
comp.to_csv(OUT_DIR / "enrichment_residual_with_fc_reliability.csv", index=False)
print(f"\n=== Top network-pair enrichment: RAW vs RESIDUALIZED (strength+distance+FC-reliability) ===")
print(comp.head(8).to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

# Headline pair survival.
print("\n=== Headline pair survival (3-proxy partial) ===")
for pair in ["visual || visual",
             "dorsal attention || dorsal attention",
             "dorsal attention || visual"]:
    row = comp[comp["net_pair"] == pair]
    if len(row) == 0:
        print(f"  {pair:45s} -- not in top-K --")
        continue
    r = row.iloc[0]
    print(f"  {pair:45s} raw={r['enrichment_raw']:7.3f}  "
          f"resid={r['enrichment_resid']:7.3f}  "
          f"Δ={r['enrichment_resid']-r['enrichment_raw']:+.3f}")

# Verdict.
vv_r = float(comp.loc[comp.net_pair=="visual || visual","enrichment_resid"].iloc[0])
dd_r = float(comp.loc[comp.net_pair=="dorsal attention || dorsal attention","enrichment_resid"].iloc[0])
dv_r = float(comp.loc[comp.net_pair=="dorsal attention || visual","enrichment_resid"].iloc[0])
worst = min(vv_r, dd_r, dv_r)
verdict_lines = [
    "Retest reliability + tract_check joint verdict",
    "=" * 60,
    f"  Strength + distance + FC reliability jointly explain {r2_full3*100:.1f}% of |PC3|.",
    f"  Headline pair survival under 3-proxy residual:",
    f"    visual||visual      = {vv_r:.2f}x",
    f"    DAN||DAN            = {dd_r:.2f}x",
    f"    DAN||visual         = {dv_r:.2f}x",
    "",
]
if worst >= 5.0:
    verdict_lines.append("  -> CLEAN. All three headline pairs survive even when FC reliability")
    verdict_lines.append("     (an independent data-driven proxy) is added to the partialling.")
    verdict_lines.append("     => Reliability-confound objection is comprehensively closed.")
elif worst >= 2.0:
    verdict_lines.append("  -> HEDGE. Headline pairs survive but reduced under the joint partial.")
    verdict_lines.append("     => Strong but not bulletproof claim; report all three numbers.")
else:
    verdict_lines.append("  -> COLLAPSE. Pairs fall below 2x — joint reliability proxy DOES explain")
    verdict_lines.append("     PC3 localization. Demote the localization claim.")
verdict_lines += [
    "",
    "Caveats (still standing):",
    "  - This is FC reliability, NOT SC reliability. They are different physical",
    "    constructs. We use it because it's the only same-subject scan-rescan",
    "    measure available from this cache; gold-standard SC ICC requires the",
    "    HCP retest dMRI release (not in this project).",
    "  - Even so, FC reliability passing the partialling test is an independent",
    "    cross-check on the strength+distance proxy result.",
]
verdict = "\n".join(verdict_lines)
print("\n" + verdict)
(OUT_DIR / "retest_findings.txt").write_text(verdict + "\n")
print(f"\nSaved -> {OUT_DIR / 'retest_findings.txt'}")
