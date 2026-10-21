"""Spec v4 E1.0c: how much each FC view (rest vs each task) contributes to the pooled basis's variance spectrum.

For pooled component j with train eigenvalue lambda_j = sum_m ||X_m w_j||^2 (X_m = view m's train-centered block,
optionally divided by its first singular value or its total norm), view m's share of component j is ||X_m w_j||^2 / lambda_j. With 8
equally weighted views, an even share is 1/8. Reports, per seed and pooling variant:
  - each view's share of the total pooled train variance;
  - each view's share of the variance in component bands (1-10, 11-50, ...) and in the top-k components;
  - each view's share of held-out (val + test) variance captured in the top-k components, from the same projection.

    python scripts/experiments/joint_flow_matching/checks/pooled_block_contributions.py [--seeds 0 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basis_spectrum import FC_VIEWS, full_basis  # noqa: E402
from pooled_basis_variance import FC_TASK_CONDITIONS, HCP_Base  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "pooled_basis_variance"
BANDS = ((1, 10), (11, 50), (51, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048), (2049, None))
TOP_K = (10, 50, 128, 256, 512, 1024)


def run_seed(seed):
    b = HCP_Base(parcellation="Glasser", hemi="both", source="FC", target="SC", data_load_mode="precomputed",
                 shuffle_seed=seed, fc_conditions=["rest", *FC_TASK_CONDITIONS])
    tr = np.asarray(b.trainvaltest_partition_indices["train"])
    ho = np.concatenate([b.trainvaltest_partition_indices["val"], b.trainvaltest_partition_indices["test"]])
    Xtr, Xho, s1 = {}, {}, {}
    for v in FC_VIEWS:
        a = b.fc_condition_upper_triangles[v]
        m = a[tr].astype(np.float64).mean(0)
        Xtr[v] = a[tr].astype(np.float64) - m
        Xho[v] = a[ho].astype(np.float64) - m
        s1[v] = np.sqrt(np.linalg.eigvalsh(Xtr[v] @ Xtr[v].T)[-1])
    rows = []
    frob = {v: np.linalg.norm(Xtr[v]) for v in FC_VIEWS}
    for variant, div in (("scaled", s1), ("frob", frob), ("unscaled", {v: 1.0 for v in FC_VIEWS})):
        tr_blocks = {v: Xtr[v] / div[v] for v in FC_VIEWS}
        ho_blocks = {v: Xho[v] / div[v] for v in FC_VIEWS}
        W, evals = full_basis(np.vstack([tr_blocks[v] for v in FC_VIEWS]))
        r = W.shape[1]
        comp = np.stack([np.square(tr_blocks[v] @ W).sum(0) for v in FC_VIEWS])      # (views, r) train variance
        comp_ho = np.stack([np.square(ho_blocks[v] @ W).sum(0) for v in FC_VIEWS])   # held-out variance
        total = np.array([np.square(tr_blocks[v]).sum() for v in FC_VIEWS])
        for i, v in enumerate(FC_VIEWS):
            base = dict(seed=seed, variant=variant, view=v)
            rows.append(dict(base, measure="share_of_total_train_variance", band="all", value=total[i] / total.sum()))
            for lo, hi in BANDS:
                hi_ = r if hi is None else min(hi, r)
                sl = slice(lo - 1, hi_)
                rows.append(dict(base, measure="share_of_band_train_variance", band=f"{lo}-{hi_}",
                                 value=comp[i, sl].sum() / comp[:, sl].sum()))
            for k in TOP_K:
                rows.append(dict(base, measure="share_of_topk_train_variance", band=f"top{k}",
                                 value=comp[i, :k].sum() / comp[:, :k].sum()))
                rows.append(dict(base, measure="share_of_topk_heldout_variance", band=f"top{k}",
                                 value=comp_ho[i, :k].sum() / comp_ho[:, :k].sum()))
        lead = comp[:, :10] / comp[:, :10].sum(0)
        print(f"seed {seed} {variant}: rest share of components 1-10:", np.round(lead[0], 2), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = ap.parse_args()
    df = pd.DataFrame([r for s in args.seeds for r in run_seed(s)])
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "block_contributions.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for measure in df.measure.unique():
        t = df[df.measure == measure].groupby(["variant", "band", "view"]).value.mean().unstack("view").round(3)
        print(f"\n{measure} (mean over seeds; even share = {1 / len(FC_VIEWS):.3f})\n", t[list(FC_VIEWS)])


if __name__ == "__main__":
    main()
