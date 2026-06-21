#!/usr/bin/env python3
"""Downstream grid runner (Phase C) — cognition (results) + sex/age (leak-checks).

HARD ORDERING (BP-2): requires the handoff artifacts (make_handoff_artifacts.py) for every
(parc, seed) it runs — pred_SC/pred_FC + subject_ids. It loads them, HARD-ASSERTS they exist,
and JOINS ON subject_id against its own frozen split before using the imputation rows. Never
re-derives the split; never assumes row order.

Inputs: bv+demo (baseline), obs_FC, obs_SC, obs_FC+obs_SC, pred_SC, pred_FC, and the
combined obs/pred +bv+demo blocks (BP-1 per-block scaling). Targets: Cog{Total,Fluid,Cryst}
(+ sex/age leak-checks). Metrics lead with lift_over_bvdemo + paired permutation p.
"""
from pathlib import Path
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import (  # noqa: E402
    load_split_checked, load_scalar_target, is_leak_target, _block_latents,
    SCALAR_ESTIMATOR_SPECS, variant_tag, scalar_regression_metrics, sex_balanced_accuracy,
    residualize_on_baseline, paired_permutation_lift_p, append_csv,
    git_commit, config_hash, OUTPUTS_DIR, WANDB_PROJECT,
)

TASK = "downstream"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"

DOWNSTREAM_INPUTS = [
    "bv+demo", "obs_FC", "obs_SC", "obs_FC+obs_SC", "pred_SC", "pred_FC",
    "obs_FC+bv+demo", "obs_SC+bv+demo", "pred_SC+bv+demo", "pred_FC+bv+demo",
]
# inputs that legitimately CONTAIN bv+demo -> leak-guardrail exempt (flagged)
CONTAINS_BVDEMO = {"bv+demo", "obs_FC+bv+demo", "obs_SC+bv+demo",
                   "pred_SC+bv+demo", "pred_FC+bv+demo"}
COG_RESULT_TARGETS = ["CogTotal", "CogFluid", "CogCryst"]


def load_handoff(parc, seed, sp):
    """Load pred_* artifacts and ALIGN to sp's train/test subject order by subject_id (BP-2).
    Hard-fails if artifacts are missing or subject sets don't match."""
    d = ARTIFACTS_DIR / parc / f"seed{seed}"
    need = ["pred_SC_train.npy", "pred_SC_test.npy", "pred_FC_train.npy", "pred_FC_test.npy",
            "subject_ids_train.npy", "subject_ids_test.npy"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"handoff artifacts missing for ({parc}, seed{seed}): {missing} — "
            f"run make_handoff_artifacts.py first (BP-2 hard gate).")
    a_tr = np.load(d / "subject_ids_train.npy").astype(int)
    a_te = np.load(d / "subject_ids_test.npy").astype(int)
    s_tr = np.asarray(sp["train_ids"], int); s_te = np.asarray(sp["test_ids"], int)
    if set(a_tr) != set(s_tr) or set(a_te) != set(s_te):
        raise AssertionError(f"({parc}, seed{seed}): handoff subject set != split subject set (BP-2)")
    # reorder artifact rows to the split's subject order (join on subject_id)
    pos_tr = {int(s): i for i, s in enumerate(a_tr)}
    pos_te = {int(s): i for i, s in enumerate(a_te)}
    idx_tr = np.array([pos_tr[int(s)] for s in s_tr])
    idx_te = np.array([pos_te[int(s)] for s in s_te])
    out = {}
    for mod in ("SC", "FC"):
        out[f"pred_{mod}_train"] = np.load(d / f"pred_{mod}_train.npy")[idx_tr].astype(np.float32)
        out[f"pred_{mod}_test"] = np.load(d / f"pred_{mod}_test.npy")[idx_te].astype(np.float32)
    return out


def build_downstream_input(sp, hand, name):
    """Return (X_train, X_test) MATRICES for a downstream input (block inputs reduced to
    per-block latents, BP-1)."""
    single = {
        "bv+demo": ("bvdemo_train", "bvdemo_test"),
        "obs_FC": ("FC_train", "FC_test"),
        "obs_SC": ("SC_train", "SC_test"),
    }
    if name in single:
        a, b = single[name]
        return np.asarray(sp[a], np.float32), np.asarray(sp[b], np.float32)
    if name in ("pred_SC", "pred_FC"):
        mod = name.split("_")[1]
        return hand[f"pred_{mod}_train"], hand[f"pred_{mod}_test"]
    # block inputs -> per-block latents
    block_map = {
        "obs_FC+obs_SC": (["FC_train", "SC_train"], ["FC_test", "SC_test"]),
        "obs_FC+bv+demo": (["FC_train", "bvdemo_train"], ["FC_test", "bvdemo_test"]),
        "obs_SC+bv+demo": (["SC_train", "bvdemo_train"], ["SC_test", "bvdemo_test"]),
    }
    if name in block_map:
        tr_keys, te_keys = block_map[name]
        btr = [np.asarray(sp[k], np.float32) for k in tr_keys]
        bte = [np.asarray(sp[k], np.float32) for k in te_keys]
        return _block_latents(btr, bte)
    if name in ("pred_SC+bv+demo", "pred_FC+bv+demo"):
        mod = name.split("_")[1].split("+")[0]
        btr = [hand[f"pred_{mod}_train"], np.asarray(sp["bvdemo_train"], np.float32)]
        bte = [hand[f"pred_{mod}_test"], np.asarray(sp["bvdemo_test"], np.float32)]
        return _block_latents(btr, bte)
    raise ValueError(f"unknown downstream input {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcellations", nargs="+", default=["Glasser", "4S456Parcels"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--inputs", nargs="+", default=DOWNSTREAM_INPUTS)
    ap.add_argument("--targets", nargs="+", default=COG_RESULT_TARGETS + ["sex", "age"])
    ap.add_argument("--estimators", nargs="+", default=["pca_pls", "bayesian_ridge", "kernel_ridge"])
    ap.add_argument("--csv", default=str(OUTPUTS_DIR / "downstream.csv"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--phasec-smoke", action="store_true",
                    help="seed0 Glasser; inputs {bv+demo,obs_FC,pred_SC,obs_FC+bv+demo}; "
                         "targets {CogCryst,sex}; all 3 estimators")
    args = ap.parse_args()

    if args.phasec_smoke:
        args.seeds = [0]; args.parcellations = ["Glasser"]
        args.inputs = ["bv+demo", "obs_FC", "pred_SC", "obs_FC+bv+demo"]
        args.targets = ["CogCryst", "sex"]
        args.csv = str(OUTPUTS_DIR / "phasec_smoke_downstream.csv")

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(OUTPUTS_DIR))
    _wb = OUTPUTS_DIR / "wandb_cache"; _wb.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(_wb))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(_wb / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(_wb / "xdg"))

    use_wandb = not args.no_wandb
    commit = git_commit()
    csv_path = Path(args.csv)
    if csv_path.exists():
        csv_path.unlink()

    total = 0
    for parc in args.parcellations:
        for seed in args.seeds:
            sp = load_split_checked(seed=seed, parc=parc)        # assert frozen (BP-2)
            hand = load_handoff(parc, seed, sp)                  # hard gate + subject_id join
            bvd_tr = np.asarray(sp["bvdemo_train"], np.float32)
            bvd_te = np.asarray(sp["bvdemo_test"], np.float32)
            for target in args.targets:
                y_tr, y_te = load_scalar_target(sp, target)
                leak = is_leak_target(target)
                # residualized reference only for cognition results
                if not leak:
                    yr_tr, yr_te = residualize_on_baseline(y_tr, y_te, bvd_tr, bvd_te)
                for estimator in args.estimators:
                    for spec in SCALAR_ESTIMATOR_SPECS[estimator]:
                        fn = spec["fn"]; vtag = variant_tag(estimator, spec["params"])
                        # baseline (bv+demo) prediction for lift, this (target, variant)
                        base_pred = fn(bvd_tr, bvd_te, y_tr)
                        for input_set in args.inputs:
                            X_tr, X_te = build_downstream_input(sp, hand, input_set)
                            pred = fn(X_tr, X_te, y_tr)
                            row = {"task": TASK, "parcellation": parc, "seed": int(seed),
                                   "estimator": estimator, "variant": vtag,
                                   "input_set": input_set, "target": target,
                                   "is_leak_target": leak,
                                   "contains_bvdemo": input_set in CONTAINS_BVDEMO,
                                   **{f"hp_{k}": v for k, v in spec["params"].items()}}
                            metrics = {}
                            if target == "sex":
                                metrics["balanced_acc"] = sex_balanced_accuracy(pred, y_te)
                                metrics["lift_over_bvdemo"] = (
                                    metrics["balanced_acc"] - sex_balanced_accuracy(base_pred, y_te))
                            else:
                                m = scalar_regression_metrics(pred, y_te)
                                metrics.update(m)
                                base_r = scalar_regression_metrics(base_pred, y_te)["pearson"]
                                metrics["lift_over_bvdemo"] = m["pearson"] - base_r
                                metrics["lift_perm_p"] = paired_permutation_lift_p(pred, base_pred, y_te)
                                if not leak:
                                    pred_resid = fn(X_tr, X_te, yr_tr)
                                    metrics["residualized_pearson"] = scalar_regression_metrics(
                                        pred_resid, yr_te)["pearson"]
                            row.update({k: (float(v) if v is not None else np.nan)
                                        for k, v in metrics.items()})
                            chash = config_hash({k: row[k] for k in
                                                 ("task", "parcellation", "seed", "variant",
                                                  "input_set", "target")})
                            row.update({"git_commit": commit, "config_hash": chash})
                            append_csv(csv_path, row)

                            if use_wandb:
                                import wandb
                                run = wandb.init(
                                    project=WANDB_PROJECT, mode="offline",
                                    name=f"{TASK}-{parc}-s{seed}-{vtag}-{input_set}-{target}",
                                    tags=[TASK, f"parcellation:{parc}", f"estimator:{estimator}",
                                          f"input:{input_set}", f"target:{target}",
                                          f"seed:{seed}", "reproduction_2026_06"],
                                    config={**row})
                                flat = {f"metrics/{TASK}/{input_set}/{target}/{k}": float(v)
                                        for k, v in metrics.items()
                                        if v is not None and np.isfinite(v)}
                                run.log(flat); run.summary.update(flat); run.finish()

                            extra = (f"bal_acc={metrics.get('balanced_acc'):.3f}" if target == "sex"
                                     else f"r={metrics.get('pearson'):+.3f} lift={metrics.get('lift_over_bvdemo'):+.3f} "
                                          f"p={metrics.get('lift_perm_p')}")
                            print(f"[down] {parc} s{seed} {vtag} {input_set}->{target} | {extra}",
                                  flush=True)
                            total += 1
            print(f"[down] ({parc}, seed{seed}) done", flush=True)
    print(f"\n[down] DONE {total} cells -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
