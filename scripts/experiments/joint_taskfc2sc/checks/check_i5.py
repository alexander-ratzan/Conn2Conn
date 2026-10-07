"""Spec v3 I5 acceptance checks (CPU, seed 0, one grid cell): the grid runner equals the main.py path.

A. raw cell (k_rest 256, k_task 16, k_sc 32, n_pls 16): grid.py metrics == Sim(CrossModal_PCA_PLS, FC + 7 FC_<task>)
B. predictive-residual cell (same): grid.py == Sim(CrossModal_PCA_PLS_TaskResid)
C. rest-only cell (k_task 0): grid.py == Sim(CrossModal_PCA_PLS, source FC only, same cohort)
D. TaskResid internals: train residual blocks uncorrelated with the rest scores; a huge ridge penalty reproduces raw
E. (check_i5.sh + fingerprint_fc2sc.py) a plain FC source gives identical predictions before and after I5

    python scripts/experiments/joint_taskfc2sc/checks/check_i5.py            # prints CHECK_OK / CHECK_FAILED
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP))

CELL = dict(k_rest=256, k_task=16, k_sc=32)
NPLS = 16
KEYS = ("pearson", "demeaned_pearson", "avg_rank", "top1_acc", "mse")
TOL = 1e-4


def sim_metrics(model, source, model_cfg, tasks, seed=0):
    from main import Sim
    data = {"fc_conditions": list(tasks), "expose_fc_sessions": True}
    sim = Sim(model_name=model, config_overrides={"model": model_cfg, "data": data}, source=source, target="SC",
              shuffle_seed=seed, data_load_mode="precomputed")
    out = sim.run_single(mode="dev")
    return {s: {k: float(out["metrics"][s][k]) for k in KEYS} for s in ("val", "test")}, sim


def grid_row(cfg, version, cell, npls):
    """Run grid.py's cell code in-process on a one-cell grid and return {split: metrics}."""
    import grid
    g = copy.deepcopy(cfg)
    g["grid"].update(k_rest=[cell["k_rest"]], k_task=[cell["k_task"]], k_sc=[cell["k_sc"]], n_pls=[npls])
    base = grid.build_base(g, 0)
    rest, task_blocks, target, _ = grid.block_scores(g, base, version)
    kr, kt, ks = cell["k_rest"], cell["k_task"], cell["k_sc"]
    x = {s: np.concatenate([rest[s][:, :kr]] + [b[:, :kt] for b in task_blocks[kr][s]] if kt else [rest[s][:, :kr]], axis=1)
         for s in ("train", "val", "test")}
    from sklearn.cross_decomposition import PLSRegression
    pls = PLSRegression(n_components=min(128, x["train"].shape[1]), scale=True).fit(x["train"], target["train_scores"][:, :ks])
    out = {}
    for s in ("val", "test"):
        pred = (grid.truncated_predict(pls, x[s], npls) @ target["loadings"][:, :ks].T + target["mean"]).astype(np.float32)
        m = grid.score(pred, target["true"][s], target["mean"].astype(np.float32))
        out[s] = {k: float(m[k]) for k in KEYS}
    return out, base, rest


def compare(name, a, b):
    worst = max(abs(a[s][k] - b[s][k]) for s in ("val", "test") for k in KEYS if k != "mse")
    ok = worst <= TOL
    print(f"{name}: {'OK  ' if ok else 'FAIL'} max |diff| {worst:.2e}  grid val dr {a['val']['demeaned_pearson']:.4f} "
          f"test dr {a['test']['demeaned_pearson']:.4f} | main.py val dr {b['val']['demeaned_pearson']:.4f} "
          f"test dr {b['test']['demeaned_pearson']:.4f}", flush=True)
    return ok


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    import grid
    cfg = grid.load_cfg()
    tasks = cfg["task_conditions"]
    src = "+".join(["FC"] + [f"FC_{t}" for t in tasks])
    ok = True

    g_raw, base, rest = grid_row(cfg, "raw", CELL, NPLS)
    dims = {"FC": CELL["k_rest"], **{f"FC_{t}": CELL["k_task"] for t in tasks}}
    s_raw, _ = sim_metrics("CrossModal_PCA_PLS", src, {"n_components_pca": dims, "n_components_pls": NPLS,
                                                        "n_components_pca_target": CELL["k_sc"]}, tasks)
    ok &= compare("A raw cell", g_raw, s_raw)

    g_res, _, _ = grid_row(cfg, "predictive_resid", CELL, NPLS)
    s_res, sim_res = sim_metrics("CrossModal_PCA_PLS_TaskResid", src, {"k_rest": CELL["k_rest"], "k_task": CELL["k_task"],
                                 "n_components_pls": NPLS, "n_components_pca_target": CELL["k_sc"]}, tasks)
    ok &= compare("B predictive-residual cell", g_res, s_res)

    cell0 = dict(CELL, k_task=0)
    g_0, _, _ = grid_row(cfg, "raw", cell0, NPLS)
    s_0, _ = sim_metrics("CrossModal_PCA_PLS", "FC", {"n_components_pca": CELL["k_rest"], "n_components_pls": NPLS,
                                                      "n_components_pca_target": CELL["k_sc"]}, tasks)
    ok &= compare("C rest-only cell (FC source alone)", g_0, s_0)

    # D: residual blocks vs rest scores (train), and the infinite-penalty limit
    from models.architectures.crossmodal_pca_pls import CrossModal_PCA_PLS_TaskResid, fit_rest_residualizer
    from models.architectures.utils import get_modality_data
    data = get_modality_data(base, include_scores=True, include_raw_data=True)
    tr = np.asarray(base.trainvaltest_partition_indices["train"])
    z = np.asarray(data["sources"]["FC"]["scores"])[:, :CELL["k_rest"]]
    worst = 0.0
    for t in tasks:
        d = data["sources"][f"FC_{t}"]
        y = np.asarray(d["upper_triangles"])[tr] - np.asarray(d["mean"])
        coef, alpha = fit_rest_residualizer(z, y)
        r = y - z @ coef
        # correlation of each residual PC score with each rest score (train)
        from data.dataset_utils import population_mean_pca
        _, _, rs = population_mean_pca(r)
        rs = np.asarray(rs)[:, :CELL["k_task"]]
        c = np.corrcoef(np.concatenate([z, rs], axis=1).T)[:z.shape[1], z.shape[1]:]
        worst = max(worst, float(np.abs(c).max()))
        print(f"  {t:10s} ridge alpha {alpha:.3g}; max |corr(rest score, residual PC)| = {np.abs(c).max():.3f}", flush=True)
    d_ok = worst < 0.2
    print(f"D1 residual blocks ~ uncorrelated with rest scores (max |r| {worst:.3f} < 0.2; ridge shrinkage keeps a "
          f"little): {'OK' if d_ok else 'FAIL'}", flush=True)
    ok &= d_ok
    big = CrossModal_PCA_PLS_TaskResid(sim_res.base, k_rest=CELL["k_rest"], k_task=CELL["k_task"], n_components_pls=NPLS,
                                       n_components_pca_target=CELL["k_sc"], resid_alphas=[1e12])
    from models.architectures.crossmodal_pca_pls import CrossModal_PCA_PLS
    raw = CrossModal_PCA_PLS(sim_res.base, n_components_pca=dims, n_components_pls=NPLS, n_components_pca_target=CELL["k_sc"])
    te = np.asarray(sim_res.base.trainvaltest_partition_indices["test"])
    xin = {"FC": torch.as_tensor(sim_res.base.fc_upper_triangles[te])}
    xin.update({f"FC_{t}": torch.as_tensor(sim_res.base.fc_condition_upper_triangles[t][te]) for t in tasks})
    diff = float((big(xin) - raw(xin)).abs().max() / raw(xin).abs().max())
    d2 = diff < 1e-3
    print(f"D2 TaskResid with alpha 1e12 == raw CrossModal_PCA_PLS: max rel. diff {diff:.1e} {'OK' if d2 else 'FAIL'}", flush=True)
    ok &= d2
    print("CHECK_OK" if ok else "CHECK_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
