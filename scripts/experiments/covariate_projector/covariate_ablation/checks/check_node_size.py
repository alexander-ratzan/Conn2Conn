"""E1 follow-up (node_size.py) checks, seed 0 (CPU, ~96 GB):
1. the data_viz edgewise refactor gives tables identical to git HEAD's (with the new Glasser-safe atlas reader);
2. the `distance` multifeature model equals the 1-feature distance regression;
3. predict_edgewise_multifeature_regression on the fit partition reproduces the fitted R2;
4. node_size block summary + held-out rows run for normalised SC.

  python scripts/experiments/covariate_projector/covariate_ablation/checks/check_node_size.py
"""
import importlib.util, subprocess, sys, tempfile, time
import numpy as np
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[5])
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts/experiments/covariate_projector/covariate_ablation")
import node_size as ns
from conditions import load_config
from data import data_viz as new

src = subprocess.run(["git", "-C", REPO, "show", "HEAD:data/data_viz.py"], capture_output=True, text=True, check=True).stdout
p = tempfile.mkdtemp() + "/data_viz_head.py"; open(p, "w").write(src)
spec = importlib.util.spec_from_file_location("data_viz_head", p); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
old._atlas_roi_df = new._atlas_roi_df  # HEAD's atlas reader assumes 4S network columns (absent for Glasser)

cfg = load_config()
t = time.time(); base = ns.load_base(cfg, "invnodevol", 0); print("base", time.time() - t, "s; train", len(base.trainvaltest_partition_indices["train"]), flush=True)
for y, m in [("FC", "distance_log_product_volume_log_volume_mismatch"), ("SC", "distance_log_product_volume_log_volume_mismatch_sc")]:
    t = time.time(); a = old.compute_edgewise_multifeature_regression(base, "train", y, m); b = new.compute_edgewise_multifeature_regression(base, "train", y, m)
    num = [c for c in a.columns if a[c].dtype.kind == "f"]
    print("refactor identical", y, m, a.columns.equals(b.columns) and all(np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) for c in num), f"{time.time()-t:.0f}s", flush=True)
d_old = old.compute_edgewise_distance_regression(base, "train", "SC")
d_new = new.compute_edgewise_multifeature_regression(base, "train", "SC", "distance")
print("distance model matches 1-feature regression: max |r2 diff|", float(np.nanmax(np.abs(d_old.r2.to_numpy() - d_new.r2.to_numpy()))), flush=True)
fits = ns.fit_models(base, "SC")
print(ns.block_summary(ns.volume_block(fits), "SC (invnodevol)"), flush=True)
for r in ns.heldout_rows(base, fits, "SC (invnodevol)", "SC", 0): print(r, flush=True)
# train-fit sanity: predicting train reproduces the fitted r2 (mean over edges)
yp, yt = new.predict_edgewise_multifeature_regression(base, fits["volume"], "train")
ss = ((yt - yt.mean(0)) ** 2).sum(0); r2 = 1 - ((yt - yp) ** 2).sum(0) / np.where(ss > 0, ss, np.nan)
print("predict(train) r2 vs fitted r2 max diff", float(np.nanmax(np.abs(r2 - fits["volume"].r2.to_numpy()))), flush=True)
