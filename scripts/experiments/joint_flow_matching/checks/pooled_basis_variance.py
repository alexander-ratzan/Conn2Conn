"""Spec v4 pre-check: does a pooled FC PCA basis serve every condition, and does it keep subjects comparable across states?

Pooled basis: stack the train-split FC of every view (rows = subject x condition), each block centered on its own
train mean (optionally divided by its first singular value, MFA block scaling, or by its total norm), and take the top-K right singular
vectors. Compared with each condition's own train PCA (the task_bidirectional_bench basis).

Reports, per seed, basis variant, view and k:
  1. Variance captured on train and on held-out (val + test) subjects, pooled vs own basis.
  2. Cross-condition identification on held-out subjects (rest -> each other view): top-1 and avg rank from the
     correlation of demeaned codes, in edge space, in the pooled code space, and in per-condition codes (whose
     coordinates are not comparable across conditions; a check of the shared-space argument).

    python scripts/experiments/joint_flow_matching/checks/pooled_basis_variance.py [--seeds 0 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from data.dataset_utils import FC_TASK_CONDITIONS  # noqa: E402
from data.hcp_dataset import HCP_Base  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "pooled_basis_variance"
KS = (16, 32, 64, 128, 256)
K_ID = 64
VARIANTS = {
    # name: (views pooled into the basis, block scaling: None, first singular value, or total (Frobenius) norm)
    "rest+tasks": (("rest", *FC_TASK_CONDITIONS), None),
    "rest+tasks_scaled": (("rest", *FC_TASK_CONDITIONS), "sv1"),
    "rest+tasks_frob": (("rest", *FC_TASK_CONDITIONS), "frob"),
    "S1S2+tasks_scaled": (("rest_S1", "rest_S2", *FC_TASK_CONDITIONS), "sv1"),
}


def top_basis(X, k):
    """Top-k right singular vectors of X (rows << cols) via the row Gram matrix. Returns (E, k) and singular values."""
    evals, U = np.linalg.eigh(X @ X.T)
    order = np.argsort(evals)[::-1][:k]
    s = np.sqrt(np.clip(evals[order], 0, None))
    return (X.T @ U[:, order]) / s, s


def captured(X, W):
    """Share of the (already centered) X's total variance captured by orthonormal columns of W, for each prefix k."""
    total = np.square(X).sum()
    proj = np.square(X @ W).sum(0)
    return {k: float(proj[:k].sum() / total) for k in KS if k <= W.shape[1]}


def identification(A, B):
    """Rows are subjects. Correlation of row-demeaned A_i with B_j; top-1 and avg rank of each A_i's own B_i."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    R = A @ B.T / A.shape[1]
    n = len(R)
    top1 = float(np.mean(R.argmax(1) == np.arange(n)))
    rank = float(np.mean([(R[i] < R[i, i]).sum() / (n - 1) for i in range(n)]))
    return top1, rank


def run_seed(seed):
    b = HCP_Base(parcellation="Glasser", hemi="both", source="FC", target="SC", data_load_mode="precomputed",
                 shuffle_seed=seed, fc_conditions=["rest", *FC_TASK_CONDITIONS], expose_fc_sessions=True)
    fc = {c: b.fc_condition_upper_triangles[c] for c in ("rest", *FC_TASK_CONDITIONS)}
    fc["rest_S1"] = b.fc_session1_upper_triangles
    fc["rest_S2"] = b.fc_session2_upper_triangles
    fc["SC"] = b.sc_upper_triangles
    tr = np.asarray(b.trainvaltest_partition_indices["train"])
    ho = np.concatenate([b.trainvaltest_partition_indices["val"], b.trainvaltest_partition_indices["test"]])
    print(f"seed {seed}: cohort {len(b.sc_upper_triangles)}, train {len(tr)}, held-out {len(ho)}", flush=True)

    mean = {v: fc[v][tr].astype(np.float64).mean(0) for v in fc}
    Xtr = {v: fc[v][tr].astype(np.float64) - mean[v] for v in fc}
    Xho = {v: fc[v][ho].astype(np.float64) - mean[v] for v in fc}
    own, s1 = {}, {}
    for v in fc:
        own[v], s = top_basis(Xtr[v], max(KS))
        s1[v] = s[0]

    var_rows, id_rows = [], []
    for v in fc:
        for split, X in (("train", Xtr[v]), ("heldout", Xho[v])):
            for k, val in captured(X, own[v]).items():
                var_rows.append(dict(seed=seed, basis="own", view=v, split=split, k=k, captured=val))

    for name, (views, scaling) in VARIANTS.items():
        div = {"sv1": s1, "frob": {v: np.linalg.norm(Xtr[v]) for v in views}, None: {v: 1.0 for v in views}}[scaling]
        blocks = [Xtr[v] / div[v] for v in views]
        W, _ = top_basis(np.vstack(blocks), max(KS))
        for v in (*views, *(("rest",) if "rest" not in views else ())):
            for split, X in (("train", Xtr[v]), ("heldout", Xho[v])):
                for k, val in captured(X, W).items():
                    var_rows.append(dict(seed=seed, basis=name, view=v, split=split, k=k, captured=val))
        ref = "rest" if "rest" in views else "rest_S1"
        for v in views:
            if v == ref:
                continue
            for space, A, B in (
                ("edges", Xho[ref], Xho[v]),
                (f"pooled_k{K_ID}", Xho[ref] @ W[:, :K_ID], Xho[v] @ W[:, :K_ID]),
                (f"own_k{K_ID}", Xho[ref] @ own[ref][:, :K_ID], Xho[v] @ own[v][:, :K_ID]),
            ):
                if space == "edges" and name in ("rest+tasks_scaled", "rest+tasks_frob"):  # same as rest+tasks
                    continue
                top1, rank = identification(A, B)
                id_rows.append(dict(seed=seed, basis=name, source=ref, target=v, space=space, top1=top1, avg_rank=rank))
    return var_rows, id_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = ap.parse_args()
    var_rows, id_rows = [], []
    for seed in args.seeds:
        v, i = run_seed(seed)
        var_rows += v
        id_rows += i
    OUT_DIR.mkdir(exist_ok=True)
    var = pd.DataFrame(var_rows)
    ids = pd.DataFrame(id_rows)
    var.to_csv(OUT_DIR / "variance_captured.csv", index=False)
    ids.to_csv(OUT_DIR / "cross_condition_identification.csv", index=False)

    pd.set_option("display.width", 200)
    ho = var[var.split == "heldout"].groupby(["basis", "view", "k"]).captured.mean().unstack("k").round(3)
    print("\nHeld-out variance captured (mean over seeds)\n", ho)
    own = ho.loc["own"]
    for name in VARIANTS:
        ratio = (ho.loc[name] / own.loc[ho.loc[name].index]).round(3)
        print(f"\nPooled / own held-out ratio: {name}\n", ratio)
    print("\nCross-condition identification, held-out (mean over seeds)\n",
          ids.groupby(["basis", "space", "target"])[["top1", "avg_rank"]].mean().round(3).unstack("space"))


if __name__ == "__main__":
    main()
