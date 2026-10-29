"""Spec v4 E5 section 1 inputs for the notebook (CPU): residual similarity between views and the data-view arrays.

  Residual similarity, per seed (test split, E2's per-seed cache): for views a, b, S_ij = corr(r_i^a, r_j^b) with
  r = x - train mean; within-subject residual r (mean diagonal), average rank and top-1 (rows = b), as the
  deviation-similarity EDA, including SC.  -> tables/descriptives_pairs.csv
  Data view: the raw edges of the first `n_subjects` seed-0 test subjects in every view and every view's train mean
  (also the ground truth of the H3 rest examples).  -> OUT/render/data_view_seed0.npz

    python scripts/experiments/cca_analysis/run_descriptives.py
"""
import numpy as np
import pandas as pd

from e5_common import CFG, HERE, OUT, VIEWS, SeedData
from models.eval.metrics import compute_corr_matrix, corr_avg_rank, corr_topn_accuracy


def main():
    rows = []
    for seed in CFG["seeds"]:
        sd = SeedData(seed, "main")
        R = {v: sd.edges[("test", v)].astype(np.float64) - sd.train_mean[v].astype(np.float64) for v in VIEWS}
        for a in VIEWS:
            for b in VIEWS:
                if a == b:
                    continue
                S = compute_corr_matrix(R[b], R[a])                 # rows: true subject in b; columns: subjects in a
                rows.append(dict(seed=seed, source=a, target=b, within_r=float(np.mean(np.diag(S))),
                                 avg_rank=float(corr_avg_rank(cc=S)), top1_acc=float(corr_topn_accuracy(cc=S, topn=1))))
        if seed == CFG["composite"]["examples"]["seed"]:
            n = CFG["composite"]["examples"]["n_subjects"]
            (OUT / "render").mkdir(parents=True, exist_ok=True)
            np.savez(OUT / "render" / f"data_view_seed{seed}.npz",
                     test_index=sd.splits["test"][:n], subject_ids=sd.subject_ids[sd.splits["test"][:n]],
                     **{f"edges__{v}": sd.edges[("test", v)][:n] for v in VIEWS},
                     **{f"train_mean__{v}": sd.train_mean[v] for v in VIEWS})
        print(f"seed {seed} done", flush=True)
    (HERE / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "tables" / "descriptives_pairs.csv", index=False)
    print("wrote tables/descriptives_pairs.csv and render/data_view_seed0.npz")


if __name__ == "__main__":
    main()
