"""Spec v3 E2 check: do the PLS "maximum iterations reached" warnings change the selected results?

Refits each version's per-k_task val-selected configuration (results/<parcellation>/<version>/tables/selected.csv)
for one seed with PLSRegression(max_iter=5000, tol=1e-8) instead of the defaults (500, 1e-6), and compares val / test
metrics with the grid's. Prints CONV_OK if every |difference| <= 0.002 demeaned r.

    python scripts/experiments/joint_taskfc2sc/checks/check_convergence.py --seed 0
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import grid  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402

TOL = 0.002


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = grid.load_cfg()
    base = grid.build_base(cfg, args.seed)
    ok, rows = True, []
    for version in cfg["versions"]:
        sel = pd.read_csv(HERE.parent / cfg["output_dir"] / cfg["parcellation"] / version / "tables" / "selected.csv")
        sel = sel[sel["seed"] == args.seed]
        rest, task_blocks, target, _ = grid.block_scores(cfg, base, "raw" if version == "raw" else version)
        for _, r in sel.iterrows():
            kr, kt, ks, k = int(r["k_rest"]), int(r["k_task"]), int(r["k_sc"]), int(r["n_pls"])
            if version != "raw" and kt == 0:
                continue  # shared rest-only cell, checked under raw
            x = {s: np.concatenate([rest[s][:, :kr]] + [b[:, :kt] for b in task_blocks[kr][s]] if kt else [rest[s][:, :kr]],
                                   axis=1) for s in ("train", "val", "test")}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pls = PLSRegression(n_components=k, scale=True, max_iter=5000, tol=1e-8).fit(
                    x["train"], target["train_scores"][:, :ks])
            n_warn = sum("iterations" in str(m.message) for m in caught)
            out = {}
            for s in ("val", "test"):
                pred = (pls.predict(x[s]) @ target["loadings"][:, :ks].T + target["mean"]).astype(np.float32)
                out[s] = float(grid.score(pred, target["true"][s], target["mean"].astype(np.float32))["demeaned_pearson"])
            d = max(abs(out["val"] - r["val_demeaned_pearson"]), abs(out["test"] - r["test_demeaned_pearson"]))
            ok &= d <= TOL
            rows.append(dict(version=version, k_task=kt, k_rest=kr, k_sc=ks, n_pls=k, grid_val=r["val_demeaned_pearson"],
                             refit_val=out["val"], grid_test=r["test_demeaned_pearson"], refit_test=out["test"],
                             max_abs_diff=d, warnings_at_5000=n_warn))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("CONV_OK" if ok else "CONV_DIFF", flush=True)


if __name__ == "__main__":
    main()
