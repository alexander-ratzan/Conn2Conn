#!/usr/bin/env python3
"""Reconstruction grid runner (parameterized).

For each (parcellation, seed, estimator, input_set, target) cell:
  - load data via _setup, HARD-ASSERT against the frozen split (BP-2)
  - predict target connectome from the input set (low-dim cap baked into the estimator)
  - score with the project metric panel (mse/r2/pearson/demeaned_pearson/top1_acc/avg_rank)
  - log to offline W&B with FLAT keys + append a row to the CSV mirror (source of truth)

SMOKE DEFAULTS (validate plumbing first): inputs {bv, bv+demo} x targets {SC, FC} x
estimator pca_pls x seed 0 x both parcellations. The handoff artifacts (pred_*),
oracle rows, BR/KR estimators, and connectome+bv+demo (BP-1) are Phase B — not here.

Examples:
  python run_reconstruction_grid.py --smoke
  python run_reconstruction_grid.py --parcellations 4S456Parcels --seeds 0 \
      --inputs bv bv+demo --targets SC FC --estimators pca_pls
"""
from pathlib import Path
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import (  # noqa: E402
    load_split_checked, build_input, build_target, ESTIMATORS, full_panel_eval,
    flat_metric_keys, append_csv, git_commit, config_hash,
    OUTPUTS_DIR, WANDB_PROJECT, PANEL_METRICS,
)

TASK = "reconstruction"


def run_cell(parc, seed, estimator, input_set, target, commit, csv_path, use_wandb):
    sp = load_split_checked(seed=seed, parc=parc)
    X_tr, X_te = build_input(sp, input_set)
    Y_tr, Y_te = build_target(sp, target)
    mu = Y_tr.mean(axis=0)

    pred = ESTIMATORS[estimator](X_tr, X_te, Y_tr)
    panel = full_panel_eval(pred, Y_te, mu)

    dims = {"task": TASK, "parcellation": parc, "seed": int(seed),
            "estimator": estimator, "input_set": input_set, "source": input_set,
            "target": target, "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
            "n_src_feat": int(X_tr.shape[1]), "n_tgt_feat": int(Y_tr.shape[1])}
    chash = config_hash({k: dims[k] for k in
                         ("task", "parcellation", "seed", "estimator", "input_set", "target")})

    row = dict(dims)
    row.update({m: float(panel[m]) for m in PANEL_METRICS if m in panel})
    row.update({"git_commit": commit, "config_hash": chash, "data_load_mode": "precomputed"})
    append_csv(csv_path, row)

    if use_wandb:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, mode="offline",
                         name=f"{TASK}-{parc}-s{seed}-{estimator}-{input_set}-to-{target}",
                         tags=[TASK, f"parcellation:{parc}", f"estimator:{estimator}",
                               f"input:{input_set}", f"target:{target}", f"seed:{seed}",
                               "reproduction_2026_06", "smoke"],
                         config={**dims, "git_commit": commit, "config_hash": chash})
        flat = flat_metric_keys(TASK, input_set, target, panel)
        flat.update({m: float(panel[m]) for m in PANEL_METRICS if m in panel})  # also bare cols
        run.log(flat)
        run.summary.update(flat)
        run.finish()

    print(f"[recon] {parc} s{seed} {estimator} {input_set}->{target} | "
          f"demeaned_r={panel.get('demeaned_pearson'):+.4f} pearson={panel.get('pearson'):+.4f} "
          f"top1={panel.get('top1_acc'):.3f} avg_rank={panel.get('avg_rank'):.1f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcellations", nargs="+", default=["Glasser", "4S456Parcels"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--inputs", nargs="+", default=["bv", "bv+demo"])
    ap.add_argument("--targets", nargs="+", default=["SC", "FC"])
    ap.add_argument("--estimators", nargs="+", default=["pca_pls"])
    ap.add_argument("--csv", default=str(OUTPUTS_DIR / "reconstruction.csv"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke subset: seed 0, both parc, bv/bv+demo, SC/FC, pca_pls")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [0]
        args.inputs = ["bv", "bv+demo"]
        args.targets = ["SC", "FC"]
        args.estimators = ["pca_pls"]
        args.csv = str(OUTPUTS_DIR / "smoke_reconstruction.csv")

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(OUTPUTS_DIR))
    # redirect wandb-core's cache/config/log dirs off the :ro overlay HOME (/root) -> scratch,
    # otherwise wandb-core errors "mkdir /root/.cache/wandb: disk quota exceeded" (non-fatal noise)
    _wb_cache = OUTPUTS_DIR / "wandb_cache"
    _wb_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(_wb_cache))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(_wb_cache / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(_wb_cache / "xdg"))
    use_wandb = not args.no_wandb
    commit = git_commit()
    csv_path = Path(args.csv)
    if csv_path.exists():
        csv_path.unlink()  # fresh table per invocation (CSV is regenerated, not accumulated)

    n = 0
    for parc in args.parcellations:
        for seed in args.seeds:
            for estimator in args.estimators:
                for input_set in args.inputs:
                    for target in args.targets:
                        run_cell(parc, seed, estimator, input_set, target,
                                 commit, csv_path, use_wandb)
                        n += 1
    print(f"\n[recon] DONE {n} cells -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
