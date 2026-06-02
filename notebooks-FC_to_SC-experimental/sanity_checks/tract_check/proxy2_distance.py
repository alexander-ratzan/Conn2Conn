#!/usr/bin/env python3
"""Proxy 2: does PC3 favor short (more reliable) connections?

Longer streamlines accumulate more tracking error → less reliable reconstruction. So
short edges are more reliable proxies. Distance = euclidean(centroid_i, centroid_j)
using Glasser MNI coords from `data/atlas_info/Glasser_dseg_reformatted.csv`.

  Spearman(|PC3 loadings|, edge_distance)
  OLS R² of |PC3 loadings| ~ edge_distance

Strong NEGATIVE Spearman = PC3 favors short edges → distance/reliability confound.

Reads:
  - data/atlas_info/Glasser_dseg_reformatted.csv (mni_x, mni_y, mni_z columns)
  - _cache_pc3_abs.npy (from proxy1_strength.py)

Writes only to stdout — capture via shell redirect.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

THIS_DIR = Path(__file__).resolve().parent

# Locate atlas.
ATLAS_PATH = next(p for p in [
    Path("/scratch/ans9868/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
    Path("/Users/user/projects/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
] if p.exists())
print(f"Loading atlas from: {ATLAS_PATH}")
atlas = pd.read_csv(ATLAS_PATH)
assert len(atlas) == 360, f"Expected 360 Glasser regions, got {len(atlas)}"
# 1-indexed id; map to 0-indexed.
atlas["idx0"] = atlas["id"].astype(int) - 1
atlas = atlas.sort_values("idx0").reset_index(drop=True)
coords = atlas[["mni_x", "mni_y", "mni_z"]].values.astype(np.float64)
print(f"Centroid coords: shape={coords.shape}, ranges per axis:")
for i, ax in enumerate(["x", "y", "z"]):
    print(f"  mni_{ax}: [{coords[:, i].min():+.1f}, {coords[:, i].max():+.1f}]  "
          f"mean={coords[:, i].mean():+.1f}")

# Upper triangle.
N = 360
triu_i, triu_j = np.triu_indices(N, k=1)
assert len(triu_i) == 64620

# Edge distance.
diff = coords[triu_i] - coords[triu_j]
edge_distance = np.linalg.norm(diff, axis=1)
print(f"\nedge_distance: shape={edge_distance.shape}, "
      f"min={edge_distance.min():.2f} mm, max={edge_distance.max():.2f} mm, "
      f"median={np.median(edge_distance):.2f} mm")

# Load cached |PC3|.
PC3_ABS_PATH = THIS_DIR / "_cache_pc3_abs.npy"
if not PC3_ABS_PATH.exists():
    raise SystemExit(f"Missing cache {PC3_ABS_PATH}. Run proxy1_strength.py first.")
pc3_abs = np.load(PC3_ABS_PATH)
assert pc3_abs.shape == (64620,)

# Spearman(|PC3|, edge_distance).
rho_dist, p_dist = spearmanr(pc3_abs, edge_distance)
print(f"\nSpearman(|PC3|, edge_distance) = {rho_dist:+.4f}   (p = {p_dist:.4g})")

# OLS R^2: |PC3| ~ edge_distance.
X = edge_distance.reshape(-1, 1)
y = pc3_abs
ols = LinearRegression().fit(X, y)
pred = ols.predict(X)
r2 = r2_score(y, pred)
print(f"OLS R² of |PC3| ~ edge_distance = {r2:+.4f}")
print(f"OLS coef = {ols.coef_[0]:+.6f}   intercept = {ols.intercept_:+.6f}")

pearson = float(np.corrcoef(pc3_abs, edge_distance)[0, 1])
print(f"Pearson(|PC3|, edge_distance) = {pearson:+.4f}")

# Verdict cue.
print("\n--- proxy 2 verdict cue ---")
if rho_dist > -0.15 and r2 < 0.10:
    print(f"  rho={rho_dist:+.3f} (small) AND R²={r2:.3f} (low)")
    print("  -> PC3 is roughly orthogonal to edge distance.")
    print("     DISTANCE PROXY does NOT explain PC3's localization.")
elif rho_dist <= -0.30 or r2 >= 0.30:
    print(f"  rho={rho_dist:+.3f} (strongly negative) OR R²={r2:.3f} (high)")
    print("  -> PC3 strongly favors short edges.")
    print("     RELIABILITY CONFOUND IS LIVE on distance. Run decisive partialled test.")
else:
    print(f"  rho={rho_dist:+.3f}, R²={r2:.3f}  (intermediate negative)")
    print("  -> Some short-edge preference, not dominant.")
    print("     Run the decisive partialled test to settle it.")

# Cache for downstream.
np.save(THIS_DIR / "_cache_edge_distance.npy", edge_distance)
np.save(THIS_DIR / "_cache_triu_i.npy", triu_i)
np.save(THIS_DIR / "_cache_triu_j.npy", triu_j)
print(f"\nCached -> {THIS_DIR / '_cache_edge_distance.npy'}")
print(f"Cached -> {THIS_DIR / '_cache_triu_i.npy'}, _cache_triu_j.npy")
