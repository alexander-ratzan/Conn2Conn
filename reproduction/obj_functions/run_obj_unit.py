#!/usr/bin/env python3
"""Objective-functions Phase 1 runner — ONE Glasser seed, all estimators, all 3 axes.

For each estimator {BR, PLS, obj1a, obj2c, obj2c_raw} build pred_SC and score:
  - RECONSTRUCTION: full_panel_eval on test (demeaned_pearson, avg_rank, top1, pearson)
  - IDENTITY: demeaned-cosine pair sims of pred_SC_resid_bvdemo, bucketed by relation (-> sibling AUC in finalize)
  - COGNITION: bayesian_ridge_scalar lift over bv+demo for pred_SC and pred_SC+bv+demo, on Cog{Total,Fluid,Cryst}

Writes per-seed parts: recon_s{seed}.csv, cog_s{seed}.csv, family/obj_family_s{seed}.npz.

    python run_obj_unit.py --seed 0      # Glasser only
"""
from pathlib import Path
import sys
import argparse
import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
FAM = REPRO / "family_mechanism"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPRO)); sys.path.insert(0, str(FAM))
import _obj_estimators as oe          # noqa: E402
import _fm_common as fm               # noqa: E402
from _grid_common import (            # noqa: E402
    load_split_checked, full_panel_eval, load_scalar_target, residualize_on_baseline,
    bayesian_ridge_scalar, scalar_regression_metrics, paired_permutation_lift_p,
    _block_latents, append_csv, git_commit,
)

PARTS = HERE / "outputs" / "parts"
COG_TARGETS = ["CogCryst", "CogTotal", "CogFluid"]   # CogCryst = supervised target (lead)
REL = fm.RELATIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    seed = args.seed
    (PARTS / "family").mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    D = fm._data(); D["set_parcellation"]("Glasser")
    sp = load_split_checked(seed=seed, parc="Glasser")
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    Xbd_tr, Xbd_te = sp["bvdemo_train"], sp["bvdemo_test"]
    mu = SC_tr.mean(0)

    # supervision target: CogCryst raw + bv+demo-residual (train), for obj2c
    cog_tr, cog_te = load_scalar_target(sp, "CogCryst")
    cr_tr, _ = residualize_on_baseline(cog_tr, cog_te, np.asarray(Xbd_tr, np.float64),
                                       np.asarray(Xbd_te, np.float64))
    ctx = {"c_raw_tr": cog_tr, "c_resid_tr": cr_tr}

    # SC residualized over bv+demo (for the identity/family variant), via OLS basis
    fit_basis_ols = D["fit_basis_ols"]
    SC_bd_tr, _ = fit_basis_ols(Xbd_tr, Xbd_te, SC_tr)
    SC_res_tr = (SC_tr - SC_bd_tr).astype(np.float32)
    mu_res = SC_res_tr.mean(0)

    # family pairing (same rng convention as br_family / notebook)
    rng = np.random.default_rng(42 + seed)
    pairs = D["pair_indices_by_relation"](sp["base"].metadata_df, sp["test_idx"], rng, fm.PAIR_AGE_TOL)

    EST = oe.estimator_registry()
    recon_csv = PARTS / f"recon_s{seed}.csv"
    cog_csv = PARTS / f"cog_s{seed}.csv"
    for p in (recon_csv, cog_csv):
        if p.exists(): p.unlink()
    fam_save = {"seed": np.array([seed])}

    # bv+demo cognition baseline (shared) per target
    base_pred = {}
    for t in COG_TARGETS:
        y_tr, y_te = load_scalar_target(sp, t)
        base_pred[t] = (y_tr, y_te, bayesian_ridge_scalar(Xbd_tr, Xbd_te, y_tr))

    for name, fn in EST.items():
        # --- build pred_SC (test + in-sample train) and pred_SC_resid (test) ---
        pred_te = fn(FC_tr, FC_te, SC_tr, ctx)
        pred_tr = fn(FC_tr, FC_tr, SC_tr, ctx)
        pred_res_te = fn(FC_tr, FC_te, SC_res_tr, ctx)

        # (1) RECONSTRUCTION
        panel = full_panel_eval(pred_te, SC_te, mu)
        append_csv(recon_csv, {"seed": seed, "estimator": name, "git_commit": commit,
                               **{m: float(panel[m]) for m in
                                  ["demeaned_pearson", "pearson", "avg_rank", "top1_acc", "mse", "r2"]}})

        # (2) IDENTITY (family pair sims of pred_SC_resid_bvdemo)
        sim_mat = D["demeaned_cosine_pair_sim"](pred_res_te, mu_res)
        sims = D["extract_pair_sims"](sim_mat, pairs)
        for r in REL:
            fam_save[f"{name}__{r}"] = sims.get(r, np.array([], np.float32))

        # (3) COGNITION (pred_SC, pred_SC+bv+demo) lift over bv+demo, 3 targets
        for t in COG_TARGETS:
            y_tr, y_te, bp = base_pred[t]
            base_r = scalar_regression_metrics(bp, y_te)["pearson"]
            for inp, (Xtr, Xte) in {
                "pred_SC": (pred_tr, pred_te),
                "pred_SC+bv+demo": _block_latents([pred_tr, np.asarray(Xbd_tr, np.float32)],
                                                  [pred_te, np.asarray(Xbd_te, np.float32)]),
            }.items():
                pr = bayesian_ridge_scalar(Xtr, Xte, y_tr)
                mm = scalar_regression_metrics(pr, y_te)
                append_csv(cog_csv, {"seed": seed, "estimator": name, "input_set": inp, "target": t,
                                     "pearson": mm["pearson"], "lift_over_bvdemo": mm["pearson"] - base_r,
                                     "lift_perm_p": paired_permutation_lift_p(pr, bp, y_te),
                                     "git_commit": commit})
        print(f"[obj s{seed}] {name}: recon dr={panel['demeaned_pearson']:+.3f} "
              f"| sib-pairs={len(sims.get('sibling', []))} done", flush=True)

    np.savez_compressed(PARTS / "family" / f"obj_family_s{seed}.npz", **fam_save)
    print(f"[obj s{seed}] wrote parts -> {PARTS}", flush=True)


if __name__ == "__main__":
    main()
