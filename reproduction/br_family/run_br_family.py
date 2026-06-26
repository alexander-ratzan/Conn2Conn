#!/usr/bin/env python3
"""BR-family (F6/F7 heritability) runner — ONE Glasser seed unit.

Faithful port of family_mechanism/run_f6_family.py, with ONE change: the imputed connectome
variants (pred_SC_raw, pred_SC_resid_bvdemo, pred_FC_raw, pred_FC_resid_bvdemo) are built with
**capped BayesianRidge** instead of PLS. The 4 estimator-independent variants (obs_SC, obs_FC,
combined_pred_SC [already BR], bvdemo_to_SC [OLS]) are unchanged → a built-in sanity check
(their AUCs must match the spine PLS family run / notebook).

Question: does the stronger BR reconstructor carry MORE heritable family signal (sibling AUC)
than PLS, and does it change the F7 predictor/identifier collapse?

    python run_br_family.py --seed 0      # Glasser only

Output: per-seed npz of demeaned-cosine pair sims bucketed by relation -> outputs/parts/Glasser/.
Pooled aggregation (AUC + bootstrap + perm + FDR) runs in finalize_br_family.py.
"""
from pathlib import Path
import argparse
import sys
import numpy as np

_HERE = Path(__file__).resolve().parent
FAM = _HERE.parent / "family_mechanism"
sys.path.insert(0, str(FAM))           # reuse the F6/F7 helpers verbatim
sys.path.insert(0, str(_HERE.parent))  # _grid_common (capped_bayesian_ridge)
import _fm_common as fm                 # noqa: E402

OUT_PARTS = _HERE / "outputs" / "parts"
_NB_GLASSER_SEED0_COUNTS = {"MZ": 33, "DZ": 13, "sibling": 125, "unrelated_matched": 171}


def build_family_variants_br(split):
    """Port of fm.build_family_variants with pca_pls_predict -> capped_bayesian_ridge (BR)
    for the four imputed variants. obs_*/combined/bvdemo_to_SC unchanged."""
    D = fm._data()
    from _grid_common import capped_bayesian_ridge as BR   # noqa: E402 (torch host only)
    fit_basis_ols = D["fit_basis_ols"]; combined_predict = D["combined_predict"]
    SC_tr, SC_te = split["SC_train"], split["SC_test"]
    FC_tr, FC_te = split["FC_train"], split["FC_test"]
    Xbd_tr, Xbd_te = split["bvdemo_train"], split["bvdemo_test"]

    SC_bd_tr_pred, SC_bd_te_pred = fit_basis_ols(Xbd_tr, Xbd_te, SC_tr)
    FC_bd_tr_pred, FC_bd_te_pred = fit_basis_ols(Xbd_tr, Xbd_te, FC_tr)
    SC_res_tr = (SC_tr - SC_bd_tr_pred).astype(np.float32)
    FC_res_tr = (FC_tr - FC_bd_tr_pred).astype(np.float32)

    var, mean = {}, {}
    var["obs_SC"], mean["obs_SC"] = SC_te, SC_tr.mean(axis=0)
    var["obs_FC"], mean["obs_FC"] = FC_te, FC_tr.mean(axis=0)
    var["pred_SC_raw"] = BR(FC_tr, FC_te, SC_tr)
    mean["pred_SC_raw"] = SC_tr.mean(axis=0)
    var["pred_SC_resid_bvdemo"] = BR(FC_tr, FC_te, SC_res_tr)
    mean["pred_SC_resid_bvdemo"] = SC_res_tr.mean(axis=0)
    var["combined_pred_SC"] = combined_predict(FC_tr, FC_te, SC_tr, Xbd_tr, Xbd_te)  # already BR
    mean["combined_pred_SC"] = SC_tr.mean(axis=0)
    var["bvdemo_to_SC"], mean["bvdemo_to_SC"] = SC_bd_te_pred, SC_tr.mean(axis=0)
    var["pred_FC_raw"] = BR(SC_tr, SC_te, FC_tr)
    mean["pred_FC_raw"] = FC_tr.mean(axis=0)
    var["pred_FC_resid_bvdemo"] = BR(SC_tr, SC_te, FC_res_tr)
    mean["pred_FC_resid_bvdemo"] = FC_res_tr.mean(axis=0)
    return var, mean


def pair_sims_for_seed_br(split, seed):
    """Mirror fm.pair_sims_for_seed but with the BR variant builder. Pair rng = 42 + seed."""
    D = fm._data()
    var, mean = build_family_variants_br(split)
    rng = np.random.default_rng(42 + seed)
    pairs_by_rel = D["pair_indices_by_relation"](
        split["base"].metadata_df, split["test_idx"], rng, fm.PAIR_AGE_TOL)
    sims = {}
    for v in fm.FAM_VARIANTS:
        sim_mat = D["demeaned_cosine_pair_sim"](var[v], mean[v])
        sims[v] = D["extract_pair_sims"](sim_mat, pairs_by_rel)
    return sims, {r: len(p) for r, p in pairs_by_rel.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", default="Glasser", choices=fm.PARCELLATIONS)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    D = fm._data()
    D["set_parcellation"](args.parc)
    split = D["load_seed_split"](seed=args.seed, source="FC", target="SC")

    sims, pair_counts = pair_sims_for_seed_br(split, args.seed)
    print(f"[BR-family {args.parc} seed{args.seed}] pair counts: {pair_counts}", flush=True)

    # self-check (estimator-independent): Glasser seed-0 pair counts must match the notebook
    if args.parc == "Glasser" and args.seed == 0:
        for r, n in _NB_GLASSER_SEED0_COUNTS.items():
            assert pair_counts.get(r) == n, (
                f"Glasser/seed0 pair-count drift {r}: got {pair_counts.get(r)} expected {n}")
        print("  self-check OK: Glasser/seed0 pair counts match the notebook.", flush=True)

    out_dir = OUT_PARTS / args.parc
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {"seed": np.array([args.seed]),
            "pair_count_keys": np.array(list(pair_counts.keys())),
            "pair_counts": np.array(list(pair_counts.values()))}
    for v in fm.FAM_VARIANTS:
        for r in fm.RELATIONS:
            save[f"{v}__{r}"] = sims[v].get(r, np.array([], dtype=np.float32))
    out_path = out_dir / f"family_seed{args.seed}.npz"
    np.savez_compressed(out_path, **save)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
