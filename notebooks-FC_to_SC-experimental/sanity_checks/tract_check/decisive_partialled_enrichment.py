#!/usr/bin/env python3
"""THE LOAD-BEARING TEST: does visual/DAN enrichment survive partialling reliability?

Residualize |PC3| on [edge_strength, edge_distance] via OLS. Re-rank edges on the
residual. Re-run network enrichment (Yeo7 pairs) on the top-200 residual edges.

Comparison: the original top-200 enrichment from Depth 1 Section D was:
  - visual || visual            : 11.74x
  - dorsal attention || dorsal  :  8.16x
  - dorsal attention || visual  :  4.86x

After residualization:
  - Visual/DAN stays at high enrichment (>5x)  -> biological claim CLEAN.
  - Visual/DAN partly survives (2-5x)          -> hedge as 'partly attributable'.
  - Visual/DAN collapses to ~1x                 -> reliability artifact; demote claim.

Reads:
  - _cache_pc3_abs.npy, _cache_edge_strength.npy, _cache_edge_distance.npy
  - _cache_triu_i.npy, _cache_triu_j.npy
  - Glasser atlas for network labels.

Writes:
  - stdout (capture via shell redirect)
  - enrichment_residual_top200.csv (full residualized enrichment table)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

THIS_DIR = Path(__file__).resolve().parent

# Load caches.
pc3_abs        = np.load(THIS_DIR / "_cache_pc3_abs.npy")
edge_strength  = np.load(THIS_DIR / "_cache_edge_strength.npy")
edge_distance  = np.load(THIS_DIR / "_cache_edge_distance.npy")
triu_i         = np.load(THIS_DIR / "_cache_triu_i.npy")
triu_j         = np.load(THIS_DIR / "_cache_triu_j.npy")
assert pc3_abs.shape == (64620,)
print(f"Loaded caches: pc3_abs={pc3_abs.shape}, "
      f"strength={edge_strength.shape}, distance={edge_distance.shape}")

# Atlas (for network labels in enrichment).
ATLAS_PATH = next(p for p in [
    Path("/scratch/ans9868/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
    Path("/Users/user/projects/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
] if p.exists())
atlas = pd.read_csv(ATLAS_PATH).sort_values("id").reset_index(drop=True)
atlas["idx0"] = atlas["id"].astype(int) - 1
atlas_idx = atlas.set_index("idx0")
NETWORK_COL = "community_yeo"

# Residualize |PC3| on [strength, distance].
X = np.column_stack([edge_strength, edge_distance])
y = pc3_abs
ols = LinearRegression().fit(X, y)
pred = ols.predict(X)
r2_full = r2_score(y, pred)
pc3_resid = y - pred
print(f"\nFull-model OLS  |PC3| ~ strength + distance")
print(f"  R²                          = {r2_full:.4f}")
print(f"  coef[strength]              = {ols.coef_[0]:+.6f}")
print(f"  coef[distance]              = {ols.coef_[1]:+.6f}")
print(f"  intercept                   = {ols.intercept_:+.6f}")
print(f"  residual std / signal std   = {pc3_resid.std() / y.std():.4f}")

# Per-edge network pair labels.
net_i = atlas_idx.loc[triu_i, NETWORK_COL].values
net_j = atlas_idx.loc[triu_j, NETWORK_COL].values
net_pair_all = np.array([" || ".join(sorted([a, b])) for a, b in zip(net_i, net_j)])
all_pair_counts = pd.Series(net_pair_all).value_counts()
N_EDGES = len(pc3_abs)

# === Original top-200 enrichment (baseline reference) ===
K = 200
order_raw = np.argsort(-pc3_abs)
top_raw = order_raw[:K]
raw_pair_counts = pd.Series(net_pair_all[top_raw]).value_counts()

# === Residualized top-200 enrichment ===
# Residual magnitudes can be negative; we rank on |residual|.
order_resid = np.argsort(-np.abs(pc3_resid))
top_resid = order_resid[:K]
resid_pair_counts = pd.Series(net_pair_all[top_resid]).value_counts()

# Build comparison table.
rows = []
all_pairs_seen = set(raw_pair_counts.index) | set(resid_pair_counts.index)
for pair in all_pairs_seen:
    n_total = int(all_pair_counts.get(pair, 0))
    n_expected = n_total * (K / N_EDGES)
    n_raw   = int(raw_pair_counts.get(pair, 0))
    n_resid = int(resid_pair_counts.get(pair, 0))
    rows.append({
        "net_pair":      pair,
        "n_total":       n_total,
        "n_expected_K":  n_expected,
        "n_raw_top200":  n_raw,
        "n_resid_top200": n_resid,
        "enrichment_raw":   n_raw / n_expected if n_expected > 0 else np.nan,
        "enrichment_resid": n_resid / n_expected if n_expected > 0 else np.nan,
    })
comp = pd.DataFrame(rows).sort_values("enrichment_raw", ascending=False)
print(f"\n=== Top network-pair enrichment: RAW vs RESIDUALIZED (K={K}, seed-0 PC3) ===")
print(comp.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

# Save full table for the writeup.
OUT_CSV = THIS_DIR / "enrichment_residual_top200.csv"
comp.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

# Spotlight the three Depth-1.1 headline pairs.
headline_pairs = ["visual || visual",
                  "dorsal attention || dorsal attention",
                  "dorsal attention || visual"]
print("\n=== Headline pair survival ===")
print(f"  {'pair':45s}  raw_enr   resid_enr  ratio")
for pair in headline_pairs:
    row = comp[comp["net_pair"] == pair]
    if len(row) == 0:
        print(f"  {pair:45s}  -- not in top-K either ranking --")
        continue
    r = row.iloc[0]
    raw_e   = r["enrichment_raw"]
    resid_e = r["enrichment_resid"]
    ratio = resid_e / raw_e if raw_e > 0 else float("nan")
    print(f"  {pair:45s}  {raw_e:7.3f}  {resid_e:7.3f}  {ratio:+.3f}")

# === VERDICT ===
print("\n" + "=" * 70)
print("RESIDUALIZED ENRICHMENT VERDICT")
print("=" * 70)
print(f"  Reliability proxy explains {r2_full*100:.1f}% of |PC3| variance.")
print()

vis_vis_resid = comp.loc[comp["net_pair"] == "visual || visual", "enrichment_resid"]
vis_vis_resid = float(vis_vis_resid.iloc[0]) if len(vis_vis_resid) else float("nan")
dan_dan_resid = comp.loc[comp["net_pair"] == "dorsal attention || dorsal attention", "enrichment_resid"]
dan_dan_resid = float(dan_dan_resid.iloc[0]) if len(dan_dan_resid) else float("nan")
dan_vis_resid = comp.loc[comp["net_pair"] == "dorsal attention || visual", "enrichment_resid"]
dan_vis_resid = float(dan_vis_resid.iloc[0]) if len(dan_vis_resid) else float("nan")

best_headline = max([x for x in [vis_vis_resid, dan_dan_resid, dan_vis_resid] if not np.isnan(x)],
                     default=float("nan"))
worst_headline = min([x for x in [vis_vis_resid, dan_dan_resid, dan_vis_resid] if not np.isnan(x)],
                      default=float("nan"))

if not np.isnan(best_headline) and worst_headline >= 5.0:
    print("  -> OUTCOME 1: CLEAN. All three headline pairs survive partialling at >=5x.")
    print(f"     vis||vis residual enr = {vis_vis_resid:.2f}x")
    print(f"     dan||dan residual enr = {dan_dan_resid:.2f}x")
    print(f"     dan||vis residual enr = {dan_vis_resid:.2f}x")
    print("     => Write: 'PC3's visual/DAN localization is not explained by edge")
    print("        reliability (survives partialling streamline density and inter-region")
    print("        distance; residual enrichment {:.1f}x).'".format(best_headline))
elif not np.isnan(best_headline) and best_headline >= 2.0:
    print("  -> OUTCOME 2: HEDGE. At least one headline pair survives at 2-5x; others lower.")
    print(f"     vis||vis residual enr = {vis_vis_resid:.2f}x")
    print(f"     dan||dan residual enr = {dan_dan_resid:.2f}x")
    print(f"     dan||vis residual enr = {dan_vis_resid:.2f}x")
    print("     => Write: 'Concentrates in visual/DAN cortex, partially but not fully")
    print("        attributable to higher reconstruction reliability in these short")
    print("        posterior connections.'")
else:
    print("  -> OUTCOME 3: COLLAPSE. Headline pairs drop to ~1x after partialling.")
    print(f"     vis||vis residual enr = {vis_vis_resid:.2f}x")
    print(f"     dan||dan residual enr = {dan_dan_resid:.2f}x")
    print(f"     dan||vis residual enr = {dan_vis_resid:.2f}x")
    print("     => PC3 was tracking reliability. Demote the localization claim.")
