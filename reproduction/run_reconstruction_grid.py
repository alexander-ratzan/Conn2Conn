#!/usr/bin/env python3
"""Reconstruction grid runner (Phase B — full estimator + input axes).

For each (parcellation, seed) the split is loaded ONCE (Sim build is expensive) and asserted
against the frozen split (BP-2). Then every (input->target) pair x estimator-variant is run:
  - matrix inputs (bv/demo/bv+demo/FC/SC) go through the estimator's `single` fn (low-dim cap)
  - block inputs (FC+bv+demo / SC+bv+demo) go through the `block` fn (BP-1 per-block scaling)
  - kernel_ridge expands to a 3x3 (gamma_mult x alpha) = 9 variant rows
  - score with the project metric panel; PER-CELL completeness assertion (all 6 metrics finite)
  - log to offline W&B (FLAT keys) + append to the CSV mirror (source of truth)

Reconstruction table (explicit claim-driven pairs):
  FC->SC, SC->FC            asymmetry (F1)
  bv->{SC,FC}, demo->{SC,FC}, bv+demo->{SC,FC}   modality dissociation + baseline (F2/C1)
  FC+bv+demo->SC, SC+bv+demo->FC                 does connectome add over subject-info
  FC->FC, SC->SC            within-modality oracle (Ceiling B)

NOTE: the handoff artifacts (pred_*, recon_per_subject, split_index) are written by
make_handoff_artifacts.py (fixed PCA->PLS) — a separate, single-responsibility step.
"""
from pathlib import Path
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import (  # noqa: E402
    load_split_checked, build_input, build_target, build_blocks, is_block_input,
    ESTIMATOR_SPECS, variant_tag, full_panel_eval, flat_metric_keys, append_csv,
    git_commit, config_hash, OUTPUTS_DIR, WANDB_PROJECT, PANEL_METRICS,
)

TASK = "reconstruction"

# Full reconstruction table — explicit (input, target) pairs.
RECON_PAIRS = [
    ("FC", "SC"), ("SC", "FC"),                                   # asymmetry
    ("bv", "SC"), ("bv", "FC"), ("demo", "SC"), ("demo", "FC"),   # dissociation
    ("bv+demo", "SC"), ("bv+demo", "FC"),                         # baseline C1
    ("FC+bv+demo", "SC"), ("SC+bv+demo", "FC"),                   # connectome adds (BP-1 block)
    ("FC", "FC"), ("SC", "SC"),                                   # oracle / Ceiling B
]


def assert_cell_complete(panel: dict, ctx: str):
    """Per-cell write-time assertion: every expected metric present + finite (no expected
    NaN in reconstruction -> any non-finite is a HARD FAIL)."""
    for m in PANEL_METRICS:
        if m not in panel:
            raise AssertionError(f"{ctx}: metric {m!r} MISSING from panel")
        if not np.isfinite(panel[m]):
            raise AssertionError(f"{ctx}: metric {m!r} is non-finite ({panel[m]}) — unexpected")


def run_pair(parc, seed, sp, estimators, input_set, target, commit, csv_path, use_wandb):
    block = is_block_input(input_set)
    if block:
        src_tr, src_te = build_blocks(sp, input_set)
        n_src_feat = int(sum(b.shape[1] for b in src_tr))
        n_train = int(src_tr[0].shape[0]); n_test = int(src_te[0].shape[0])
    else:
        src_tr, src_te = build_input(sp, input_set)
        n_src_feat = int(src_tr.shape[1])
        n_train = int(src_tr.shape[0]); n_test = int(src_te.shape[0])
    Y_tr, Y_te = build_target(sp, target)
    mu = Y_tr.mean(axis=0)

    rows = 0
    for estimator in estimators:
        for spec in ESTIMATOR_SPECS[estimator]:
            fn = spec["block"] if block else spec["single"]
            pred = fn(src_tr, src_te, Y_tr)
            panel = full_panel_eval(pred, Y_te, mu)
            vtag = variant_tag(estimator, spec["params"])
            ctx = f"{parc} s{seed} {vtag} {input_set}->{target}"
            assert_cell_complete(panel, ctx)

            dims = {"task": TASK, "parcellation": parc, "seed": int(seed),
                    "estimator": estimator, "variant": vtag, "input_set": input_set,
                    "source": input_set, "target": target, "is_block": bool(block),
                    "n_train": n_train, "n_test": n_test,
                    "n_src_feat": n_src_feat, "n_tgt_feat": int(Y_tr.shape[1]),
                    **{f"hp_{k}": v for k, v in spec["params"].items()}}
            chash = config_hash({k: dims[k] for k in
                                 ("task", "parcellation", "seed", "variant", "input_set", "target")})
            row = dict(dims)
            row.update({m: float(panel[m]) for m in PANEL_METRICS})
            row.update({"status": "ok", "git_commit": commit, "config_hash": chash,
                        "data_load_mode": "precomputed"})
            append_csv(csv_path, row)

            if use_wandb:
                import wandb
                run = wandb.init(project=WANDB_PROJECT, mode="offline",
                                 name=f"{TASK}-{parc}-s{seed}-{vtag}-{input_set}-to-{target}",
                                 tags=[TASK, f"parcellation:{parc}", f"estimator:{estimator}",
                                       f"variant:{vtag}", f"input:{input_set}",
                                       f"target:{target}", f"seed:{seed}", "reproduction_2026_06"],
                                 config={**dims, "git_commit": commit, "config_hash": chash})
                flat = flat_metric_keys(TASK, input_set, target, panel)
                flat.update({m: float(panel[m]) for m in PANEL_METRICS})
                run.log(flat); run.summary.update(flat); run.finish()

            print(f"[recon] {ctx} | demeaned_r={panel['demeaned_pearson']:+.4f} "
                  f"pearson={panel['pearson']:+.4f} top1={panel['top1_acc']:.3f} "
                  f"avg_rank={panel['avg_rank']:.3f}", flush=True)
            rows += 1
    return rows


def parse_pairs(specs):
    out = []
    for s in specs:
        a, b = s.split(":")
        out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcellations", nargs="+", default=["Glasser", "4S456Parcels"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="input:target pairs; default = full RECON_PAIRS")
    ap.add_argument("--estimators", nargs="+", default=["pca_pls", "bayesian_ridge", "kernel_ridge"])
    ap.add_argument("--csv", default=str(OUTPUTS_DIR / "reconstruction.csv"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--phaseb-smoke", action="store_true",
                    help="seed 0, Glasser, {FC:SC,SC:FC,FC+bv+demo:SC,FC:FC} x all 3 estimators")
    args = ap.parse_args()

    if args.phaseb_smoke:
        args.seeds = [0]; args.parcellations = ["Glasser"]
        args.pairs = ["FC:SC", "SC:FC", "FC+bv+demo:SC", "FC:FC"]
        args.estimators = ["pca_pls", "bayesian_ridge", "kernel_ridge"]
        args.csv = str(OUTPUTS_DIR / "phaseb_smoke_reconstruction.csv")

    pairs = parse_pairs(args.pairs) if args.pairs else RECON_PAIRS

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(OUTPUTS_DIR))
    _wb_cache = OUTPUTS_DIR / "wandb_cache"; _wb_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(_wb_cache))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(_wb_cache / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(_wb_cache / "xdg"))

    use_wandb = not args.no_wandb
    commit = git_commit()
    csv_path = Path(args.csv)
    if csv_path.exists():
        csv_path.unlink()

    total = 0
    for parc in args.parcellations:
        for seed in args.seeds:
            sp = load_split_checked(seed=seed, parc=parc)   # load ONCE, assert frozen (BP-2)
            for (input_set, target) in pairs:
                total += run_pair(parc, seed, sp, args.estimators, input_set, target,
                                  commit, csv_path, use_wandb)
            print(f"[recon] ({parc}, seed{seed}) done", flush=True)
    print(f"\n[recon] DONE {total} cells -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
