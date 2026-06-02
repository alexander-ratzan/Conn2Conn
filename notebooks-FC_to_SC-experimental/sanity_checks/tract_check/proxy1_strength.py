#!/usr/bin/env python3
"""Proxy 1: is PC3 just living in the high-streamline edges?

For each upper-triangle edge of the 360x360 Glasser SC matrix, compute the train-set
mean SC weight (a proxy for tractography reconstruction reliability — higher weight
edges are reconstructed more consistently). Then:

  Spearman(|PC3 loadings|, edge_strength)
  OLS R² of |PC3 loadings| ~ edge_strength

A strong positive Spearman or high R² says PC3's emphasis tracks edge weight —
reliability confound is live. A near-zero Spearman + low R² says PC3 is roughly
orthogonal to edge strength — the biological claim survives this proxy.

Reads:
  - Saved Depth 1 PC loadings (sc_pc_loadings.npy): (10, 64620), PC3 = row 2
  - Seed-0 SC_train (via _setup.load_seed_split)

Writes only to stdout — capture via shell redirect.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Locate _setup.py (same pattern as the further_exploration notebooks).
for _cand in [Path.cwd(),
              Path.cwd() / "notebooks-FC_to_SC-experimental" / "further_exploration",
              Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration"),
              Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration")]:
    if (_cand / "_setup.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError(f"_setup.py not found from cwd={Path.cwd()}")
from _setup import load_seed_split

# Locate Depth 1 PC loadings.
LOADINGS_PATH = next(p for p in [
    Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/model_overviews/results/local_results/further_exploration/depth1_spectral_mechanism/sc_pc_loadings.npy"),
    Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/model_overviews/results/local_results/further_exploration/depth1_spectral_mechanism/sc_pc_loadings.npy"),
] if p.exists())
print(f"Loading PC loadings from: {LOADINGS_PATH}")

pc_loadings = np.load(LOADINGS_PATH)
assert pc_loadings.shape == (10, 64620), f"Expected (10, 64620), got {pc_loadings.shape}"
pc3 = pc_loadings[2].astype(np.float64)
pc3_abs = np.abs(pc3)
print(f"PC3 loading vector: shape={pc3.shape}, "
      f"||·||={np.linalg.norm(pc3):.4f}, max|·|={pc3_abs.max():.4f}")

# Build seed-0 split → edge strength.
print("Loading seed 0 split for edge strength...", flush=True)
split = load_seed_split(seed=0)
SC_train = split["SC_train"]
print(f"SC_train shape: {SC_train.shape}")
edge_strength = SC_train.mean(axis=0).astype(np.float64)
print(f"edge_strength: shape={edge_strength.shape}, "
      f"min={edge_strength.min():.4f}, max={edge_strength.max():.4f}, "
      f"mean={edge_strength.mean():.4f}, std={edge_strength.std():.4f}")

# Spearman(|PC3|, edge_strength).
rho_strength, p_strength = spearmanr(pc3_abs, edge_strength)
print(f"\nSpearman(|PC3|, edge_strength) = {rho_strength:+.4f}   (p = {p_strength:.4g})")

# OLS R^2: |PC3| ~ edge_strength.
X = edge_strength.reshape(-1, 1)
y = pc3_abs
ols = LinearRegression().fit(X, y)
pred = ols.predict(X)
r2 = r2_score(y, pred)
print(f"OLS R² of |PC3| ~ edge_strength = {r2:+.4f}")
print(f"OLS coef = {ols.coef_[0]:+.6f}   intercept = {ols.intercept_:+.6f}")

# Pearson too for completeness.
pearson = float(np.corrcoef(pc3_abs, edge_strength)[0, 1])
print(f"Pearson(|PC3|, edge_strength) = {pearson:+.4f}")

# Verdict cue.
print("\n--- proxy 1 verdict cue ---")
if abs(rho_strength) < 0.15 and r2 < 0.10:
    print(f"  rho={rho_strength:+.3f} (low) AND R²={r2:.3f} (low)")
    print("  -> PC3 is roughly orthogonal to edge strength.")
    print("     STRENGTH PROXY does NOT explain PC3's localization.")
elif abs(rho_strength) >= 0.30 or r2 >= 0.30:
    print(f"  rho={rho_strength:+.3f} (moderate/high) OR R²={r2:.3f} (high)")
    print("  -> PC3 tracks edge strength substantially.")
    print("     RELIABILITY CONFOUND IS LIVE. Run decisive_partialled_enrichment.py.")
else:
    print(f"  rho={rho_strength:+.3f}, R²={r2:.3f}  (intermediate)")
    print("  -> Some structure-strength alignment, but not dominant.")
    print("     Run the decisive partialled test to settle it.")

# Persist the strength vector + abs(PC3) for downstream scripts (no need to recompute).
OUT_DIR = Path(__file__).resolve().parent
np.save(OUT_DIR / "_cache_edge_strength.npy", edge_strength)
np.save(OUT_DIR / "_cache_pc3_abs.npy",       pc3_abs)
np.save(OUT_DIR / "_cache_pc3_signed.npy",    pc3)
print(f"\nCached -> {OUT_DIR / '_cache_edge_strength.npy'}")
print(f"Cached -> {OUT_DIR / '_cache_pc3_abs.npy'}")
print(f"Cached -> {OUT_DIR / '_cache_pc3_signed.npy'}")
