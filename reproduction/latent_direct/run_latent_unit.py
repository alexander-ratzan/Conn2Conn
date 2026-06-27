#!/usr/bin/env python3
"""Latent-direct runner — ONE Glasser seed, 3 sources × objectives × {cognition, identity}.

Architecture under test:  source -> PCA -> [objective] -> task-tuned LATENT -> classify
(NO inverse-PCA, NO re-PCA). Compare to reproduction/obj_functions (same objectives WITH the
inverse-PCA round-trip) to isolate what the "PCA glow-up" costs.

Arms:
  FC2SC : src=FC, tgt=SC  — 5 cross-modal objectives (BR,PLS,obj1a,obj2c,obj2c_raw)
  SC2FC : src=SC, tgt=FC  — same 5 (reverse direction)
  FCSC  : observed both — concat z-scored FC/SC PCA latents; {plain, obj2c, obj2c_raw}; cognition only

Tasks:
  cognition: BayesianRidge DIRECTLY on the latent -> lift over bv+demo (+ paired-perm p), 3 targets
  identity : sibling AUC from demeaned-cosine pairs of the (resid) latent  [cross-modal arms only]

Parts: cog_s{seed}.csv, family/lat_family_s{seed}.npz
    python run_latent_unit.py --seed 0
"""
from pathlib import Path
import sys
import argparse
import numpy as np
from sklearn.linear_model import BayesianRidge

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
FAM = REPRO / "family_mechanism"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPRO)); sys.path.insert(0, str(FAM))
import _latent_estimators as le         # noqa: E402
import _obj_estimators as oe            # noqa: E402
import _fm_common as fm                 # noqa: E402
from _grid_common import (              # noqa: E402
    load_split_checked, load_scalar_target, residualize_on_baseline,
    bayesian_ridge_scalar, scalar_regression_metrics, paired_permutation_lift_p,
    _block_latents, append_csv, git_commit,
)

PARTS = HERE / "outputs" / "parts"
COG = ["CogCryst", "CogTotal", "CogFluid"]
REL = fm.RELATIONS


def br_direct(Xtr, Xte, y_tr):
    """Downstream classifier DIRECTLY on the latent — no PCA, no re-PCA (the whole point)."""
    Xtr = np.asarray(Xtr, np.float64); Xte = np.asarray(Xte, np.float64)
    ok = ~np.isnan(y_tr)
    return BayesianRidge(max_iter=500).fit(Xtr[ok], y_tr[ok]).predict(Xte)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    seed = ap.parse_args().seed
    (PARTS / "family").mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    D = fm._data(); D["set_parcellation"]("Glasser")
    sp = load_split_checked(seed=seed, parc="Glasser")
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    Xbd_tr, Xbd_te = sp["bvdemo_train"], sp["bvdemo_test"]

    cog_tr, cog_te = load_scalar_target(sp, "CogCryst")
    cr_tr, _ = residualize_on_baseline(cog_tr, cog_te, np.asarray(Xbd_tr, np.float64),
                                       np.asarray(Xbd_te, np.float64))
    ctx = {"c_raw_tr": cog_tr, "c_resid_tr": cr_tr}
    fit_basis_ols = D["fit_basis_ols"]

    rng = np.random.default_rng(42 + seed)
    pairs = D["pair_indices_by_relation"](sp["base"].metadata_df, sp["test_idx"], rng, fm.PAIR_AGE_TOL)

    cog_csv = PARTS / f"cog_s{seed}.csv"
    if cog_csv.exists(): cog_csv.unlink()
    fam_save = {"seed": np.array([seed])}

    base_pred = {}
    for t in COG:
        y_tr, y_te = load_scalar_target(sp, t)
        base_pred[t] = (y_tr, y_te, bayesian_ridge_scalar(Xbd_tr, Xbd_te, y_tr))

    def score_cog(arm, est, lat_tr, lat_te):
        for t in COG:
            y_tr, y_te, bp = base_pred[t]
            base_r = scalar_regression_metrics(bp, y_te)["pearson"]
            pr = br_direct(lat_tr, lat_te, y_tr)
            mm = scalar_regression_metrics(pr, y_te)
            append_csv(cog_csv, {"seed": seed, "arm": arm, "estimator": est, "target": t,
                                 "pearson": mm["pearson"], "lift_over_bvdemo": mm["pearson"] - base_r,
                                 "lift_perm_p": paired_permutation_lift_p(pr, bp, y_te),
                                 "git_commit": commit})

    # ---- cross-modal arms: FC->SC and SC->FC ----
    for arm, (S_tr, S_te, T_tr, T_te) in {
        "FC2SC": (FC_tr, FC_te, SC_tr, SC_te),
        "SC2FC": (SC_tr, SC_te, FC_tr, FC_te),
    }.items():
        T_bd_tr, _ = fit_basis_ols(Xbd_tr, Xbd_te, T_tr)
        T_res_tr = (T_tr - T_bd_tr).astype(np.float32)
        for est, fn in le.LATENT_ESTIMATORS.items():
            lat_tr = fn(S_tr, S_tr, T_tr, ctx)          # in-sample train latent
            lat_te = fn(S_tr, S_te, T_tr, ctx)          # test latent
            score_cog(arm, est, lat_tr, lat_te)
            # identity: residualized-target latent, demeaned-cosine pairs
            lr_tr = fn(S_tr, S_tr, T_res_tr, ctx); lr_te = fn(S_tr, S_te, T_res_tr, ctx)
            sims = D["extract_pair_sims"](
                D["demeaned_cosine_pair_sim"](lr_te, lr_tr.mean(0)), pairs)
            for r in REL:
                fam_save[f"{arm}__{est}__{r}"] = sims.get(r, np.array([], np.float32))
            print(f"[lat s{seed}] {arm} {est} done", flush=True)

    # ---- observed-both arm: FC&SC concat latents (cognition reference) ----
    Zc_tr, Zc_te = _block_latents([np.asarray(FC_tr, np.float32), np.asarray(SC_tr, np.float32)],
                                  [np.asarray(FC_te, np.float32), np.asarray(SC_te, np.float32)])
    score_cog("FCSC", "plain", Zc_tr, Zc_te)
    for est, ckey in [("obj2c", "c_resid_tr"), ("obj2c_raw", "c_raw_tr")]:
        beta = oe._oof_cog_beta(np.asarray(Zc_tr, np.float64), np.asarray(ctx[ckey], np.float64))
        a = beta**2 / (np.mean(beta**2) + oe.EPS)
        score_cog("FCSC", est, Zc_tr * a, Zc_te * a)
    print(f"[lat s{seed}] FCSC done", flush=True)

    np.savez_compressed(PARTS / "family" / f"lat_family_s{seed}.npz", **fam_save)
    print(f"[lat s{seed}] wrote parts -> {PARTS}", flush=True)


if __name__ == "__main__":
    main()
