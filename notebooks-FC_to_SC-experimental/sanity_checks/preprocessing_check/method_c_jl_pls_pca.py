#!/usr/bin/env python3
"""Method C — JL→PLS→PCA. Data-blind random input reduction.

For each seed, replace the input PCA(256) with a Johnson-Lindenstrauss random
projection (also to 256 dims), keep PLS(64) and the output PCA(256)+inverse-PCA
identical to the main model. Then ask: does the cross-modal asymmetry hold?

Three JL variants (per-seed fresh random matrix):
  - Gaussian dense:           sklearn GaussianRandomProjection
  - Sparse Achlioptas auto:   sklearn SparseRandomProjection (density = 1/sqrt(p))
  - Sparse 1/3 (original):    sklearn SparseRandomProjection (density = 1/3)

If A (learned PCA) and all C variants agree on asymmetry magnitude, the FC PCA
basis is not doing anything privileged — any random projection captures the
cross-modal signal equally well.

Note: we keep the output (target) PCA so the regression target stays in the same
low-dim space the main model uses. The reduction-axis question is specifically
about the INPUT preprocessing.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.random_projection import (GaussianRandomProjection,
                                        SparseRandomProjection)

for _cand in [Path.cwd(),
              Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration"),
              Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration")]:
    if (_cand / "_setup.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError(f"_setup.py not found from cwd={Path.cwd()}")
from _setup import load_seed_split, full_panel_eval

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
K_REDUCE = 256
K_PLS = 64
MAX_ITER = 2000


def make_jl(variant, seed):
    if variant == "gaussian_dense":
        return GaussianRandomProjection(n_components=K_REDUCE, random_state=seed)
    if variant == "sparse_auto":
        return SparseRandomProjection(n_components=K_REDUCE, density="auto",
                                       random_state=seed)
    if variant == "sparse_third":
        return SparseRandomProjection(n_components=K_REDUCE, density=1.0/3.0,
                                       random_state=seed)
    raise ValueError(f"unknown variant {variant!r}")


def jl_pls_pca_predict(X_src_train, X_src_test, Y_train, jl_variant, seed):
    """Replace input PCA with JL; keep target PCA + inverse-PCA exactly as main model."""
    jl = make_jl(jl_variant, seed)
    Z_src_tr = jl.fit_transform(X_src_train)
    Z_src_te = jl.transform(X_src_test)

    pca_tgt = PCA(n_components=K_REDUCE, random_state=0).fit(Y_train)
    Z_tgt_tr = pca_tgt.transform(Y_train)

    pls = PLSRegression(n_components=K_PLS, scale=True, max_iter=MAX_ITER)
    pls.fit(Z_src_tr, Z_tgt_tr)
    Z_tgt_te_pred = pls.predict(Z_src_te)
    return pca_tgt.inverse_transform(Z_tgt_te_pred).astype(np.float32)


VARIANTS = ["gaussian_dense", "sparse_auto", "sparse_third"]

rows = []
for variant in VARIANTS:
    print(f"\n========== variant {variant} ==========")
    for seed in range(N_SEEDS):
        sp = load_seed_split(seed=seed)
        FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
        SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
        SC_train_mean = SC_tr.mean(axis=0)
        FC_train_mean = FC_tr.mean(axis=0)

        print(f"[seed {seed:2d}] FC->SC ...", flush=True)
        pred_fs = jl_pls_pca_predict(FC_tr, FC_te, SC_tr, variant, seed)
        panel_fs = full_panel_eval(pred_fs, SC_te, SC_train_mean)

        print(f"[seed {seed:2d}] SC->FC ...", flush=True)
        pred_sf = jl_pls_pca_predict(SC_tr, SC_te, FC_tr, variant, seed)
        panel_sf = full_panel_eval(pred_sf, FC_te, FC_train_mean)

        ratio = panel_fs["demeaned_pearson"] / max(panel_sf["demeaned_pearson"], 1e-9)
        print(f"   FC->SC dp = {panel_fs['demeaned_pearson']:.4f}  "
              f"SC->FC dp = {panel_sf['demeaned_pearson']:.4f}  "
              f"ratio = {ratio:.3f}")

        rows.append({"method": "JL_PLS_PCA", "jl_variant": variant, "seed": seed,
                     "direction": "FC->SC",
                     "demeaned_pearson": panel_fs["demeaned_pearson"],
                     "top1_acc": panel_fs.get("top1_acc", np.nan),
                     "avg_rank": panel_fs.get("avg_rank", np.nan)})
        rows.append({"method": "JL_PLS_PCA", "jl_variant": variant, "seed": seed,
                     "direction": "SC->FC",
                     "demeaned_pearson": panel_sf["demeaned_pearson"],
                     "top1_acc": panel_sf.get("top1_acc", np.nan),
                     "avg_rank": panel_sf.get("avg_rank", np.nan)})

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "method_c_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'method_c_results.csv'}")
print("\nSummary by (variant, direction):")
print(df.groupby(["jl_variant", "direction"])["demeaned_pearson"]
        .agg(["median", "min", "max"])
        .to_string(float_format=lambda x: f"{x:.4f}"))
