#!/usr/bin/env python3
"""Diagnostic: obj1a_ungated (amplitude restoration WITHOUT the reliability gate), ONE Glasser seed.
Computes the 3 axes (recon dr / sibling AUC / CogCryst lift) for this one estimator and writes a
single-row CSV. Compare against gated obj1a + BR + PLS from scorecard.csv.

    python run_diag_ungated.py --seed 0
"""
from pathlib import Path
import sys
import argparse
import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
FAM = REPRO / "family_mechanism"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPRO)); sys.path.insert(0, str(FAM))
import _obj_estimators as oe       # noqa: E402
import _fm_common as fm            # noqa: E402
from _grid_common import (         # noqa: E402
    load_split_checked, full_panel_eval, load_scalar_target,
    bayesian_ridge_scalar, scalar_regression_metrics, append_csv, git_commit,
)

PARTS = HERE / "outputs" / "parts"


def auc(pos, neg):
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1.0
    return (r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    seed = ap.parse_args().seed
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"diag_ungated_s{seed}.csv"
    if out.exists(): out.unlink()

    D = fm._data(); D["set_parcellation"]("Glasser")
    sp = load_split_checked(seed=seed, parc="Glasser")
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    Xbd_tr, Xbd_te = sp["bvdemo_train"], sp["bvdemo_test"]
    mu = SC_tr.mean(0)

    fn = oe.obj1a_ungated_restore
    pred_te = fn(FC_tr, FC_te, SC_tr)
    pred_tr = fn(FC_tr, FC_tr, SC_tr)

    # reconstruction
    recon = full_panel_eval(pred_te, SC_te, mu)["demeaned_pearson"]

    # identity (sibling AUC of pred_SC_resid_bvdemo)
    SC_bd_tr, _ = D["fit_basis_ols"](Xbd_tr, Xbd_te, SC_tr)
    SC_res_tr = (SC_tr - SC_bd_tr).astype(np.float32)
    pred_res = fn(FC_tr, FC_te, SC_res_tr)
    rng = np.random.default_rng(42 + seed)
    pairs = D["pair_indices_by_relation"](sp["base"].metadata_df, sp["test_idx"], rng, fm.PAIR_AGE_TOL)
    sims = D["extract_pair_sims"](D["demeaned_cosine_pair_sim"](pred_res, SC_res_tr.mean(0)), pairs)
    sib = auc(sims.get("sibling", np.array([])), sims.get("unrelated_matched", np.array([])))

    # cognition (CogCryst lift over bv+demo)
    y_tr, y_te = load_scalar_target(sp, "CogCryst")
    base = bayesian_ridge_scalar(Xbd_tr, Xbd_te, y_tr)
    pr = bayesian_ridge_scalar(pred_tr, pred_te, y_tr)
    lift = scalar_regression_metrics(pr, y_te)["pearson"] - scalar_regression_metrics(base, y_te)["pearson"]

    append_csv(out, {"seed": seed, "estimator": "obj1a_ungated", "recon_dr": float(recon),
                     "sib_auc": float(sib), "cog_cryst_lift": float(lift), "git_commit": git_commit()})
    print(f"[diag s{seed}] obj1a_ungated: recon={recon:+.3f} sib_AUC={sib:.3f} cogCryst={lift:+.3f}", flush=True)


if __name__ == "__main__":
    main()
