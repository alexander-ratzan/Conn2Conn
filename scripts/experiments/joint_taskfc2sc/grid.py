"""Spec v3 E2 grid runner (CPU): every (k_rest, k_task, k_sc) cell of one seed x version, scored for every n_pls.

One HCP_Base per seed (E0's matched cohort; sources FC + FC_<task> for the 7 tasks, target SC). Each block's PCA is
the library's (HCP_Base / get_modality_data) and is sliced per cell. Per cell one PLSRegression(scale=True) is fit at
min(max n_pls, D) components and truncated to every n_pls (exact, checks/bench_pls_grid.py). Predictions are decoded
through the SC PCA and scored with models.eval.metrics.compute_basic_regression_metrics exactly as the Evaluator does.
  raw:              task block = PCA of the task FC (CrossModal_PCA_PLS with a per-block dict)
  predictive_resid: task block = PCA of the task FC minus its ridge prediction from the subject's k_rest rest scores
                    (models.architectures.crossmodal_pca_pls.fit_rest_residualizer; CrossModal_PCA_PLS_TaskResid)
Rows go to runs/<version>/seed<S>.csv as each cell finishes; a rerun skips cells already written.

    python scripts/experiments/joint_taskfc2sc/grid.py --seed 0 --version raw [--cells k_rest=256,k_task=16,k_sc=32]
"""
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from numpy.linalg import pinv  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402

from data.dataset_utils import population_mean_pca  # noqa: E402
from data.hcp_dataset import HCP_Base  # noqa: E402
from models.architectures.crossmodal_pca_pls import fit_rest_residualizer  # noqa: E402
from models.architectures.utils import get_modality_data  # noqa: E402
from models.eval.metrics import compute_basic_regression_metrics, compute_corr_matrix  # noqa: E402

SPLITS = ("val", "test")
METRIC_KEYS = ("mse", "r2", "pearson", "demeaned_pearson", "avg_rank", "top1_acc")


def load_cfg():
    return yaml.safe_load((HERE / "config.yml").read_text())


def build_base(cfg, seed):
    tasks = cfg["task_conditions"]
    base = HCP_Base(parcellation=cfg["parcellation"], hemi="both", source="+".join(["FC"] + [f"FC_{t}" for t in tasks]),
                    target="SC", shuffle_seed=seed, data_load_mode="precomputed", fc_conditions=list(tasks),
                    expose_fc_sessions=True)
    n = len(base.metadata_df)
    if n != cfg["cohort"]:
        raise SystemExit(f"cohort {n} != config cohort {cfg['cohort']}")
    return base


def block_scores(cfg, base, version):
    """{split: rest scores (n, max k_rest)}, {k_rest: {split: [task score blocks (n, max k_task)]}} and target data.
    Train scores are the library's PCA scores; val / test are projections, as CrossModal_PCA_PLS.forward does."""
    idx = {s: np.asarray(base.trainvaltest_partition_indices[s]) for s in ("train", "val", "test")}
    data = get_modality_data(base, include_scores=True, include_raw_data=True)
    kr, kt = max(cfg["grid"]["k_rest"]), max(cfg["grid"]["k_task"])

    def project(d, k):
        raw, mean, load = np.asarray(d["upper_triangles"]), np.asarray(d["mean"]), np.asarray(d["loadings"])[:, :k]
        out = {s: (raw[idx[s]] - mean) @ load for s in SPLITS}
        out["train"] = np.asarray(d["scores"])[:, :k]
        return out

    rest = project(data["sources"]["FC"], kr)
    tasks = [f"FC_{t}" for t in cfg["task_conditions"]]
    task_blocks, alphas = {}, {}
    if version == "raw":
        raw_blocks = [project(data["sources"][m], kt) for m in tasks]
        task_blocks = {k: {s: [b[s] for b in raw_blocks] for s in ("train", *SPLITS)} for k in cfg["grid"]["k_rest"]}
    else:
        for k in cfg["grid"]["k_rest"]:
            task_blocks[k] = {s: [] for s in ("train", *SPLITS)}
            for m in tasks:
                d = data["sources"][m]
                raw, mean = np.asarray(d["upper_triangles"]), np.asarray(d["mean"])
                coef, alphas[(k, m)] = fit_rest_residualizer(rest["train"][:, :k], raw[idx["train"]] - mean)
                resid = {s: raw[idx[s]] - mean - rest[s][:, :k] @ coef for s in ("train", *SPLITS)}
                r_mean, r_load, r_scores = population_mean_pca(resid["train"])
                task_blocks[k]["train"].append(np.asarray(r_scores)[:, :kt])
                for s in SPLITS:
                    task_blocks[k][s].append((resid[s] - r_mean) @ np.asarray(r_load)[:, :kt])
            print(f"  residual blocks k_rest={k}: ridge alphas "
                  f"{ {m[3:]: f'{alphas[(k, m)]:.3g}' for m in tasks} }", flush=True)
    tgt = data["target"]
    target = dict(mean=np.asarray(tgt["mean"]), loadings=np.asarray(tgt["loadings"]), train_scores=np.asarray(tgt["scores"]),
                  true={s: np.asarray(base.sc_upper_triangles[idx[s]], dtype=np.float32) for s in SPLITS})
    return rest, task_blocks, target, alphas


def truncated_predict(pls, x, k):
    """First-k-component prediction of a fitted PLSRegression (sklearn's coef_ algebra on truncated factors)."""
    w, p, q = pls.x_weights_[:, :k], pls.x_loadings_[:, :k], pls.y_loadings_[:, :k]
    coef = ((w @ pinv(p.T @ w) @ q.T) * pls._y_std).T / pls._x_std
    return (x - pls._x_mean) @ coef.T + pls.intercept_


def block_vip_shares(pls, k, sizes):
    """Share of the first-k-component explained Y sum of squares carried by each input block: sum over the block's
    columns of VIP_j^2 / n_inputs, i.e. sum_a SSY_a * sum_{j in block} w_ja^2 / ||w_a||^2, normalised over blocks."""
    w, t, q = pls.x_weights_[:, :k], pls.x_scores_[:, :k], pls.y_loadings_[:, :k]
    ssy = (q ** 2).sum(0) * (t ** 2).sum(0)                    # explained Y SS per component
    contrib = (w ** 2 / (w ** 2).sum(0)) @ ssy                 # per input column
    edges = np.cumsum([0] + list(sizes))
    shares = np.array([contrib[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    return shares / shares.sum()


def score(pred, true, train_mean):
    """Evaluator metrics (models/eval/evaluator.py: corr matrices rows = targets; demeaned uses the target train mean)."""
    cm = compute_corr_matrix(true, pred)
    cmd = compute_corr_matrix(true - train_mean, pred - train_mean)
    return compute_basic_regression_metrics(pred, true, corr_matrix=cm, corr_matrix_demeaned=cmd)


def cells(cfg, version, only=None):
    g = cfg["grid"]
    for kr in g["k_rest"]:
        for kt in g["k_task"]:
            if version != "raw" and kt == 0:
                continue  # rest-only is the same model in both versions: scored once, under raw
            for ks in g["k_sc"]:
                c = dict(k_rest=kr, k_task=kt, k_sc=ks)
                if only is None or all(c[k] == v for k, v in only.items()):
                    yield c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--version", choices=["raw", "predictive_resid"], required=True)
    ap.add_argument("--cells", help="comma list key=value restricting the grid, e.g. k_rest=256,k_task=16")
    ap.add_argument("--out", help="CSV path (default runs/<version>/seed<S>.csv)")
    args = ap.parse_args()
    cfg = load_cfg()
    only = {k: int(v) for k, v in (kv.split("=") for kv in args.cells.split(","))} if args.cells else None
    out = Path(args.out) if args.out else HERE / cfg["runs_dir"] / args.version / f"seed{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(int(r["k_rest"]), int(r["k_task"]), int(r["k_sc"])) for r in csv.DictReader(f)}
    todo = [c for c in cells(cfg, args.version, only) if (c["k_rest"], c["k_task"], c["k_sc"]) not in done]
    print(f"seed {args.seed} {args.version}: {len(todo)} cells to run ({len(done)} already in {out})", flush=True)
    if not todo:
        return
    t0 = time.time()
    base = build_base(cfg, args.seed)
    rest, task_blocks, target, alphas = block_scores(cfg, base, args.version)
    print(f"data + blocks ready in {time.time() - t0:.0f} s", flush=True)
    npls_grid = cfg["grid"]["n_pls"]
    block_names = ["rest"] + list(cfg["task_conditions"])
    fields = ["version", "seed", "k_rest", "k_task", "k_sc", "n_pls", "n_inputs", "fit_s", "n_conv_warnings",
              "resid_alpha_median"] + [f"{s}_{m}" for s in SPLITS for m in METRIC_KEYS] + [f"vip_{b}" for b in block_names]
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for c in todo:
            kr, kt, ks = c["k_rest"], c["k_task"], c["k_sc"]
            blocks = task_blocks[kr]
            x = {s: np.concatenate([rest[s][:, :kr]] + [b[:, :kt] for b in blocks[s]] if kt else [rest[s][:, :kr]], axis=1)
                 for s in ("train", *SPLITS)}
            d = x["train"].shape[1]
            y = target["train_scores"][:, :ks]
            t1 = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pls = PLSRegression(n_components=min(max(npls_grid), d), scale=True).fit(x["train"], y)
            fit_s = time.time() - t1
            n_warn = sum("iterations" in str(m.message) for m in caught)
            load = target["loadings"][:, :ks]
            alpha_med = (float(np.median([a for (k, _), a in alphas.items() if k == kr])) if alphas and kt else "")
            for k in [k for k in npls_grid if k <= d]:
                row = dict(version=args.version, seed=args.seed, k_rest=kr, k_task=kt, k_sc=ks, n_pls=k, n_inputs=d,
                           fit_s=round(fit_s, 2), n_conv_warnings=n_warn, resid_alpha_median=alpha_med)
                sizes = [kr] + ([kt] * len(cfg["task_conditions"]) if kt else [])
                shares = block_vip_shares(pls, k, sizes)
                row.update({f"vip_{b}": (float(shares[i]) if i < len(shares) else 0.0) for i, b in enumerate(block_names)})
                for s in SPLITS:
                    pred = (truncated_predict(pls, x[s], k) @ load.T + target["mean"]).astype(np.float32)
                    m = score(pred, target["true"][s], target["mean"].astype(np.float32))
                    row.update({f"{s}_{key}": float(m[key]) for key in METRIC_KEYS})
                w.writerow(row)
            f.flush()
            print(f"  k_rest={kr} k_task={kt} k_sc={ks} D={d}: fit {fit_s:.1f} s, conv warnings {n_warn}, "
                  f"cell {time.time() - t1:.1f} s", flush=True)
    print(f"seed {args.seed} {args.version}: done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
