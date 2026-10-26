"""Spec v4 E2.0b: PCA spectra of the E2 code bases (how many components exist, how many reach a variance share).

For each basis (each view's own train PCA, and the pooled block-scaled FC basis over `rest` + 7 tasks), reports the
maximum number of components (train rank), the k reaching 50 / 80 / 90 / 95 / 99 % of train variance, and the
held-out (val + test) variance captured at those k and at fixed k. Same cohort, splits and centering as
pooled_basis_variance.py.

    python scripts/experiments/joint_flow_matching/checks/basis_spectrum.py [--seeds 0 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled_basis_variance import FC_TASK_CONDITIONS, HCP_Base  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "pooled_basis_variance"
SHARES = (0.5, 0.8, 0.9, 0.95, 0.99)
FIXED_K = (64, 128, 256, 512)
FC_VIEWS = ("rest", *FC_TASK_CONDITIONS)


def full_basis(X):
    """All right singular vectors with nonzero singular value, via the row Gram matrix. Returns (E, r), eigenvalues."""
    evals, U = np.linalg.eigh(X @ X.T)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    r = int((evals > evals[0] * 1e-10).sum())
    return (X.T @ U[:, order[:r]]) / np.sqrt(evals[:r]), evals[:r]


def summarize(seed, name, W, evals, Xho_by_view):
    cum = np.cumsum(evals) / evals.sum()
    k_at = {s: int(np.searchsorted(cum, s) + 1) for s in SHARES}
    rows = []
    for view, Xho in Xho_by_view.items():
        proj = np.cumsum(np.square(Xho @ W).sum(0)) / np.square(Xho).sum()
        row = dict(seed=seed, basis=name, view=view, rank=W.shape[1])
        for s, k in k_at.items():
            row[f"k_train{int(s * 100)}"] = k
            row[f"heldout_at_k{int(s * 100)}"] = float(proj[k - 1])
        for k in FIXED_K:
            row[f"train_k{k}"] = float(cum[min(k, len(cum)) - 1])
            row[f"heldout_k{k}"] = float(proj[min(k, len(proj)) - 1])
        row["heldout_full"] = float(proj[-1])
        rows.append(row)
    return rows


def run_seed(seed):
    b = HCP_Base(parcellation="Glasser", hemi="both", source="FC", target="SC", data_load_mode="precomputed",
                 shuffle_seed=seed, fc_conditions=["rest", *FC_TASK_CONDITIONS])
    data = {c: b.fc_condition_upper_triangles[c] for c in FC_VIEWS}
    data["SC"] = b.sc_upper_triangles
    tr = np.asarray(b.trainvaltest_partition_indices["train"])
    ho = np.concatenate([b.trainvaltest_partition_indices["val"], b.trainvaltest_partition_indices["test"]])
    Xtr, Xho = {}, {}
    for v, a in data.items():
        m = a[tr].astype(np.float64).mean(0)
        Xtr[v] = a[tr].astype(np.float64) - m
        Xho[v] = a[ho].astype(np.float64) - m
    rows, s1 = [], {}
    for v in data:
        W, evals = full_basis(Xtr[v])
        s1[v] = np.sqrt(evals[0])
        rows += summarize(seed, "own", W, evals, {v: Xho[v]})
        print(f"seed {seed}: own {v} rank {W.shape[1]}", flush=True)
    W, evals = full_basis(np.vstack([Xtr[v] / s1[v] for v in FC_VIEWS]))
    rows += summarize(seed, "pooled_scaled", W, evals, {v: Xho[v] / s1[v] for v in FC_VIEWS})
    print(f"seed {seed}: pooled rank {W.shape[1]}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = ap.parse_args()
    df = pd.DataFrame([r for s in args.seeds for r in run_seed(s)])
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "basis_spectrum.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.groupby(["basis", "view"]).mean(numeric_only=True).drop(columns="seed").round(3).T)


if __name__ == "__main__":
    main()
