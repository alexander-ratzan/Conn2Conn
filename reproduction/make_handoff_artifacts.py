#!/usr/bin/env python3
"""Write the reconstruction->downstream HANDOFF ARTIFACTS (the fragile seam, made explicit).

Per (parcellation, seed), using the FIXED imputation estimator (capped PCA->PLS, NOT the
swept estimator — the imputed connectome is a fixed derived input, not a swept object), write:

  pred_SC_test  = PCA-PLS(FC_tr, FC_te, SC_tr)     (n_test,  n_edges)
  pred_SC_train = PCA-PLS(FC_tr, FC_tr, SC_tr)     (n_train, n_edges)  in-sample, same call
  pred_FC_test  = PCA-PLS(SC_tr, SC_te, FC_tr)
  pred_FC_train = PCA-PLS(SC_tr, SC_tr, FC_tr)
  subject_ids_{train,test}.npy                     for the BP-2 subject_id join
  recon_per_subject.csv                            per-subject 6 metrics, both directions
  split_index.json                                 {train,val,test}_ids (cross-checked vs frozen)

All keyed by (parcellation, seed) under outputs/artifacts/{parc}/seed{seed}/.
Downstream HARD-ASSERTS these exist and joins on subject_id before running imputation rows.
"""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import (  # noqa: E402
    load_split_checked, capped_pca_pls, per_subject_metrics, subject_ids_for,
    split_json_path, OUTPUTS_DIR, git_commit, config_hash,
)

ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"


def artifact_dir(parc, seed):
    d = ARTIFACTS_DIR / parc / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_for(parc, seed, commit):
    sp = load_split_checked(seed=seed, parc=parc)   # asserts vs frozen (BP-2)
    base = sp["base"]
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    train_ids = sp["train_ids"]; test_ids = sp["test_ids"]
    d = artifact_dir(parc, seed)

    # --- imputed connectomes (fixed capped PCA->PLS; train via in-sample, same call) ---
    pred_SC_test  = capped_pca_pls(FC_tr, FC_te, SC_tr)
    pred_SC_train = capped_pca_pls(FC_tr, FC_tr, SC_tr)
    pred_FC_test  = capped_pca_pls(SC_tr, SC_te, FC_tr)
    pred_FC_train = capped_pca_pls(SC_tr, SC_tr, FC_tr)
    np.save(d / "pred_SC_test.npy",  pred_SC_test)
    np.save(d / "pred_SC_train.npy", pred_SC_train)
    np.save(d / "pred_FC_test.npy",  pred_FC_test)
    np.save(d / "pred_FC_train.npy", pred_FC_train)
    np.save(d / "subject_ids_train.npy", np.asarray(train_ids, np.int64))
    np.save(d / "subject_ids_test.npy",  np.asarray(test_ids, np.int64))

    # --- per-subject metrics (test split), both directions ---
    ps_sc = per_subject_metrics(pred_SC_test, SC_te, SC_tr.mean(0))
    ps_sc.insert(0, "direction", "FC->SC"); ps_sc.insert(0, "subject", test_ids)
    ps_fc = per_subject_metrics(pred_FC_test, FC_te, FC_tr.mean(0))
    ps_fc.insert(0, "direction", "SC->FC"); ps_fc.insert(0, "subject", test_ids)
    ps = pd.concat([ps_sc, ps_fc], ignore_index=True)
    ps.insert(0, "seed", seed); ps.insert(0, "parcellation", parc)
    ps.to_csv(d / "recon_per_subject.csv", index=False)

    # --- split_index, cross-checked against frozen ---
    frozen = json.loads(split_json_path(seed).read_text())
    part = base.trainvaltest_partition_indices
    split_index = {
        "parcellation": parc, "seed": seed, "git_commit": commit,
        "config_hash": config_hash({"parc": parc, "seed": seed}),
        "train_ids": train_ids, "val_ids": subject_ids_for(base, part["val"]),
        "test_ids": test_ids,
        "matches_frozen": bool(train_ids == [int(x) for x in frozen["train_ids"]]
                               and test_ids == [int(x) for x in frozen["test_ids"]]),
    }
    assert split_index["matches_frozen"], f"{parc} s{seed}: split_index != frozen (BP-2)"
    (d / "split_index.json").write_text(json.dumps(split_index))

    print(f"[handoff] {parc} s{seed}: wrote pred_* (SC {pred_SC_test.shape}, FC {pred_FC_test.shape}) "
          f"+ per-subject({len(ps)}) + split_index | "
          f"FC->SC demeaned_r(mean)={ps_sc['demeaned_pearson'].mean():+.4f} "
          f"SC->FC demeaned_r(mean)={ps_fc['demeaned_pearson'].mean():+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcellations", nargs="+", default=["Glasser", "4S456Parcels"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()
    commit = git_commit()
    for parc in args.parcellations:
        for seed in args.seeds:
            write_for(parc, seed, commit)
    print(f"\n[handoff] DONE -> {ARTIFACTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
