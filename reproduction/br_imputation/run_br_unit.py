#!/usr/bin/env python3
"""BR-only imputation + downstream — per (parcellation, seed) UNIT runner.

This is the isolated "Bayesian-ridge SOTA" run (see dev-notes/BR-only/PLAN.md). It uses
BayesianRidge in BOTH estimator slots and writes to its OWN outputs/ — it never touches the
spine grid's reconstruction.csv / downstream.csv / artifacts/.

Per (parc, seed):
  1. IMPUTE connectomes with capped BayesianRidge (NOT PLS):
       pred_SC_test  = BR(FC_tr, FC_te, SC_tr)     pred_SC_train = BR(FC_tr, FC_tr, SC_tr)  [in-sample]
       pred_FC_test  = BR(SC_tr, SC_te, FC_tr)     pred_FC_train = BR(SC_tr, SC_tr, FC_tr)  [in-sample]
     -> saved under outputs/artifacts/{parc}/seed{seed}/ (+ subject_ids for the BP-2 join).
     (in-sample train imputation is a DELIBERATE, known choice — see PLAN.md §6; not a leak.)
  2. DOWNSTREAM with bayesian_ridge_scalar ONLY over the 18-input set (PLAN §3), targets
     Cog{Total,Fluid,Cryst} + sex/age. Metrics: lift_over_bvdemo + perm-p + residualized.
     -> appended to the --csv (per-seed part; finalize merges).

Reuses _grid_common verbatim so the math matches the rest of the project.
"""
from pathlib import Path
import sys
import argparse
import numpy as np

# import the spine grid's shared helpers (single source of truth)
REPRO = Path(__file__).resolve().parent.parent          # .../Conn2Conn/reproduction
sys.path.insert(0, str(REPRO))
from _grid_common import (  # noqa: E402
    load_split_checked, capped_bayesian_ridge, bayesian_ridge_scalar,
    load_scalar_target, is_leak_target, _block_latents,
    scalar_regression_metrics, sex_balanced_accuracy, residualize_on_baseline,
    paired_permutation_lift_p, per_subject_metrics, append_csv, git_commit, config_hash,
)

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
ARTIFACTS = OUTPUTS / "artifacts"
PARTS = OUTPUTS / "parts"
TASK = "downstream_br"
ESTIMATOR = "bayesian_ridge"   # both slots; downstream tag

COG_TARGETS = ["CogTotal", "CogFluid", "CogCryst"]
LEAK_TARGETS = ["sex", "age"]

# ---- the 18-input registry (PLAN §3). pred_* are BR-IMPUTED here. ------------
# +bv+demo blocks use the single bvdemo block (matches the spine grid's per-block scaling).
BR_INPUTS = [
    # carried over (10)
    "bv+demo", "obs_FC", "obs_SC", "obs_FC+obs_SC", "pred_SC", "pred_FC",
    "obs_FC+bv+demo", "obs_SC+bv+demo", "pred_SC+bv+demo", "pred_FC+bv+demo",
    # new (8)
    "pred_FC+bv", "pred_SC+bv", "obs_SC+pred_FC", "obs_FC+pred_SC",
    "obs_FC+pred_SC+bv", "obs_FC+pred_SC+bv+demo", "obs_SC+pred_FC+bv+demo", "everything",
]

# leak classes for the sex/age guardrail (PLAN §9).
# any input carrying bv and/or demo -> subject-info -> EXEMPT_FLAGGED if it exceeds threshold.
CONTAINS_SUBJECT_INFO = {
    "bv+demo", "obs_FC+bv+demo", "obs_SC+bv+demo", "pred_SC+bv+demo", "pred_FC+bv+demo",
    "pred_FC+bv", "pred_SC+bv", "obs_FC+pred_SC+bv",
    "obs_FC+pred_SC+bv+demo", "obs_SC+pred_FC+bv+demo", "everything",
}
# pure connectome inputs -> predicting sex/age is real biology -> EXPECTED_SIGNAL if it exceeds.
CONNECTOME_ONLY = {
    "obs_FC", "obs_SC", "obs_FC+obs_SC", "pred_SC", "pred_FC",
    "obs_SC+pred_FC", "obs_FC+pred_SC",
}
assert set(BR_INPUTS) == CONTAINS_SUBJECT_INFO | CONNECTOME_ONLY
assert not (CONTAINS_SUBJECT_INFO & CONNECTOME_ONLY)


def impute_and_save(parc, seed, sp, commit):
    """BR-impute pred_{SC,FC}_{train(in-sample),test}; save artifacts; return in-memory dict."""
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    hand = {
        "pred_SC_test":  capped_bayesian_ridge(FC_tr, FC_te, SC_tr),
        "pred_SC_train": capped_bayesian_ridge(FC_tr, FC_tr, SC_tr),   # in-sample (PLAN §6)
        "pred_FC_test":  capped_bayesian_ridge(SC_tr, SC_te, FC_tr),
        "pred_FC_train": capped_bayesian_ridge(SC_tr, SC_tr, FC_tr),   # in-sample
    }
    d = ARTIFACTS / parc / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    for k, v in hand.items():
        np.save(d / f"{k}.npy", v)
    np.save(d / "subject_ids_train.npy", np.asarray(sp["train_ids"], np.int64))
    np.save(d / "subject_ids_test.npy",  np.asarray(sp["test_ids"], np.int64))
    # per-subject recon metrics (test), both directions — for inspection / comparison vs PLS spine
    ps_sc = per_subject_metrics(hand["pred_SC_test"], SC_te, SC_tr.mean(0))
    ps_fc = per_subject_metrics(hand["pred_FC_test"], FC_te, FC_tr.mean(0))
    print(f"[br-impute] {parc} s{seed}: BR pred_SC {hand['pred_SC_test'].shape} "
          f"pred_FC {hand['pred_FC_test'].shape} | "
          f"FC->SC demeaned_r(mean)={ps_sc['demeaned_pearson'].mean():+.4f} "
          f"SC->FC demeaned_r(mean)={ps_fc['demeaned_pearson'].mean():+.4f}", flush=True)
    return hand


def build_br_input(sp, hand, name):
    """Return (X_train, X_test) matrices for a downstream input. Block inputs are reduced to
    per-block latents (BP-1) first; the scalar BR then PCAs on top (mirrors the spine grid)."""
    def f32(key): return np.asarray(sp[key], np.float32)
    # singles (raw matrix; scalar BR does its own PCA)
    singles = {
        "bv+demo": ("bvdemo_train", "bvdemo_test"),
        "obs_FC": ("FC_train", "FC_test"),
        "obs_SC": ("SC_train", "SC_test"),
    }
    if name in singles:
        a, b = singles[name]; return f32(a), f32(b)
    if name in ("pred_SC", "pred_FC"):
        mod = name.split("_")[1]
        return hand[f"pred_{mod}_train"], hand[f"pred_{mod}_test"]
    # block inputs -> list of (train, test) arrays, then _block_latents
    def P(mod, part): return hand[f"pred_{mod}_{part}"]   # imputed connectome block
    blocks = {
        "obs_FC+obs_SC":   ([f32("FC_train"), f32("SC_train")], [f32("FC_test"), f32("SC_test")]),
        "obs_FC+bv+demo":  ([f32("FC_train"), f32("bvdemo_train")], [f32("FC_test"), f32("bvdemo_test")]),
        "obs_SC+bv+demo":  ([f32("SC_train"), f32("bvdemo_train")], [f32("SC_test"), f32("bvdemo_test")]),
        "pred_SC+bv+demo": ([P("SC", "train"), f32("bvdemo_train")], [P("SC", "test"), f32("bvdemo_test")]),
        "pred_FC+bv+demo": ([P("FC", "train"), f32("bvdemo_train")], [P("FC", "test"), f32("bvdemo_test")]),
        "pred_FC+bv":      ([P("FC", "train"), f32("bv_train")], [P("FC", "test"), f32("bv_test")]),
        "pred_SC+bv":      ([P("SC", "train"), f32("bv_train")], [P("SC", "test"), f32("bv_test")]),
        "obs_SC+pred_FC":  ([f32("SC_train"), P("FC", "train")], [f32("SC_test"), P("FC", "test")]),
        "obs_FC+pred_SC":  ([f32("FC_train"), P("SC", "train")], [f32("FC_test"), P("SC", "test")]),
        "obs_FC+pred_SC+bv": ([f32("FC_train"), P("SC", "train"), f32("bv_train")],
                              [f32("FC_test"), P("SC", "test"), f32("bv_test")]),
        "obs_FC+pred_SC+bv+demo": ([f32("FC_train"), P("SC", "train"), f32("bvdemo_train")],
                                   [f32("FC_test"), P("SC", "test"), f32("bvdemo_test")]),
        "obs_SC+pred_FC+bv+demo": ([f32("SC_train"), P("FC", "train"), f32("bvdemo_train")],
                                   [f32("SC_test"), P("FC", "test"), f32("bvdemo_test")]),
        "everything": ([f32("FC_train"), f32("SC_train"), P("FC", "train"), P("SC", "train"), f32("bvdemo_train")],
                       [f32("FC_test"), f32("SC_test"), P("FC", "test"), P("SC", "test"), f32("bvdemo_test")]),
    }
    if name in blocks:
        btr, bte = blocks[name]
        return _block_latents(btr, bte)
    raise ValueError(f"unknown BR input {name!r}")


def run_unit(parc, seed, csv_path, commit):
    sp = load_split_checked(seed=seed, parc=parc)          # assert frozen (BP-2)
    hand = impute_and_save(parc, seed, sp, commit)
    bvd_tr = np.asarray(sp["bvdemo_train"], np.float32)
    bvd_te = np.asarray(sp["bvdemo_test"], np.float32)
    n = 0
    for target in COG_TARGETS + LEAK_TARGETS:
        y_tr, y_te = load_scalar_target(sp, target)
        leak = is_leak_target(target)
        if not leak:
            yr_tr, yr_te = residualize_on_baseline(y_tr, y_te, bvd_tr, bvd_te)
        base_pred = bayesian_ridge_scalar(bvd_tr, bvd_te, y_tr)        # baseline for lift
        for input_set in BR_INPUTS:
            X_tr, X_te = build_br_input(sp, hand, input_set)
            pred = bayesian_ridge_scalar(X_tr, X_te, y_tr)
            row = {"task": TASK, "parcellation": parc, "seed": int(seed),
                   "estimator": ESTIMATOR, "variant": ESTIMATOR, "input_set": input_set,
                   "target": target, "is_leak_target": leak,
                   "contains_subject_info": input_set in CONTAINS_SUBJECT_INFO,
                   "is_connectome_only": input_set in CONNECTOME_ONLY}
            metrics = {}
            if target == "sex":
                metrics["balanced_acc"] = sex_balanced_accuracy(pred, y_te)
                metrics["lift_over_bvdemo"] = (metrics["balanced_acc"]
                                               - sex_balanced_accuracy(base_pred, y_te))
            else:
                m = scalar_regression_metrics(pred, y_te); metrics.update(m)
                base_r = scalar_regression_metrics(base_pred, y_te)["pearson"]
                metrics["lift_over_bvdemo"] = m["pearson"] - base_r
                metrics["lift_perm_p"] = paired_permutation_lift_p(pred, base_pred, y_te)
                if not leak:
                    pred_resid = bayesian_ridge_scalar(X_tr, X_te, yr_tr)
                    metrics["residualized_pearson"] = scalar_regression_metrics(pred_resid, yr_te)["pearson"]
            row.update({k: (float(v) if v is not None else np.nan) for k, v in metrics.items()})
            row["config_hash"] = config_hash({k: row[k] for k in
                                              ("task", "parcellation", "seed", "input_set", "target")})
            row["git_commit"] = commit
            append_csv(csv_path, row)
            extra = (f"bal_acc={metrics.get('balanced_acc'):.3f}" if target == "sex"
                     else f"r={metrics.get('pearson'):+.3f} lift={metrics.get('lift_over_bvdemo'):+.3f} "
                          f"p={metrics.get('lift_perm_p')}")
            print(f"[br-down] {parc} s{seed} {input_set}->{target} | {extra}", flush=True)
            n += 1
    print(f"[br-down] ({parc}, seed{seed}) done: {n} rows -> {csv_path}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", default="Glasser")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--csv", default=None, help="default: outputs/parts/down_br_<parc>_s<seed>.csv")
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else PARTS / f"down_br_{args.parc}_s{args.seed}.csv"
    if csv_path.exists():
        csv_path.unlink()
    commit = git_commit()
    run_unit(args.parc, args.seed, csv_path, commit)


if __name__ == "__main__":
    main()
