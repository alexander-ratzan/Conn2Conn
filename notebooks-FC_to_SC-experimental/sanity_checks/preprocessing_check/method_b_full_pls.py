#!/usr/bin/env python3
"""Method B — Full PLS (no reduction). PLSRegression directly on raw 64,620-dim
edge vectors, both directions, 10 seeds.

If A (PCA→PLS→PCA) and B agree on the asymmetry, the learned PCA preprocessing
is neither hiding nor injecting cross-modal signal — the reduction is just a
computational efficiency choice.

Notes on PLSRegression at this scale:
  - n ≈ 683 train subjects, p = 64,620 edges (both X and Y)
  - K_PLS = 64 components (matches main model)
  - NIPALS iterations are O(n p) per component → 64 × few × 683 × 64620 ≈ 10–50 G ops per fit.
    Should complete in ~30–90 s per direction on a modern CPU.
  - scale=True matches the main model. With p≫n some edges may have near-zero std;
    sklearn handles that by setting their post-scale value to 0 (np.divide handles it
    via where=). If you see RuntimeWarning about div-by-zero, that's the cause and is
    benign.
"""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

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
K_PLS = 64
MAX_ITER = 2000


def full_pls_predict(X_train, X_test, Y_train, k_pls=K_PLS, max_iter=MAX_ITER):
    """Plain PLSRegression on raw features. Bypasses self.coef_ at predict
    time (sklearn's PLSRegression assembles coef_ of shape (p_x, p_y) =
    (64620, 64620) = 33 GB at float64; predict() then does X @ coef_, which
    OOMs at 64 GB allocation). Instead we compute the prediction manually via
    the rotations + loadings, which keeps memory at O(n p)."""
    pls = PLSRegression(n_components=k_pls, scale=True, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # divide-by-zero from zero-std edges
        pls.fit(X_train, Y_train)
        # Manual predict, avoiding self.coef_ (the OOM trigger):
        #   X_test_scaled = (X_test - _x_mean) / _x_std
        #   T_test = X_test_scaled @ x_rotations_                  (n_test, k)
        #   Y_pred_scaled = T_test @ y_loadings_.T                 (n_test, p_y)
        #   Y_pred = Y_pred_scaled * _y_std + _y_mean
        x_mean = pls._x_mean
        x_std  = pls._x_std
        y_mean = pls._y_mean
        y_std  = pls._y_std
        Xte_scaled = (X_test.astype(np.float64) - x_mean) / x_std
        T_test     = Xte_scaled @ pls.x_rotations_
        Y_pred_sc  = T_test @ pls.y_loadings_.T
        pred       = Y_pred_sc * y_std + y_mean
    return pred.astype(np.float32)


rows = []
for seed in range(N_SEEDS):
    sp = load_seed_split(seed=seed)
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    SC_train_mean = SC_tr.mean(axis=0)
    FC_train_mean = FC_tr.mean(axis=0)
    print(f"\n[seed {seed}] FC_tr={FC_tr.shape}, SC_tr={SC_tr.shape}", flush=True)

    print(f"[seed {seed}] full PLS FC->SC ...", flush=True)
    pred_fs = full_pls_predict(FC_tr, FC_te, SC_tr)
    panel_fs = full_panel_eval(pred_fs, SC_te, SC_train_mean)
    print(f"  demeaned_pearson = {panel_fs['demeaned_pearson']:.4f}")

    print(f"[seed {seed}] full PLS SC->FC ...", flush=True)
    pred_sf = full_pls_predict(SC_tr, SC_te, FC_tr)
    panel_sf = full_panel_eval(pred_sf, FC_te, FC_train_mean)
    print(f"  demeaned_pearson = {panel_sf['demeaned_pearson']:.4f}")

    rows.append({"method": "FULL_PLS", "jl_variant": "", "seed": seed,
                 "direction": "FC->SC",
                 "demeaned_pearson": panel_fs["demeaned_pearson"],
                 "top1_acc": panel_fs.get("top1_acc", np.nan),
                 "avg_rank": panel_fs.get("avg_rank", np.nan)})
    rows.append({"method": "FULL_PLS", "jl_variant": "", "seed": seed,
                 "direction": "SC->FC",
                 "demeaned_pearson": panel_sf["demeaned_pearson"],
                 "top1_acc": panel_sf.get("top1_acc", np.nan),
                 "avg_rank": panel_sf.get("avg_rank", np.nan)})

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "method_b_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'method_b_results.csv'}")
print("\nSummary (median across seeds):")
print(df.groupby("direction")["demeaned_pearson"].agg(["median", "min", "max"])
      .to_string(float_format=lambda x: f"{x:.4f}"))
