"""How similar are a subject's FC deviations from the condition mean across conditions? (Glasser, all FC conditions)

For subject i and condition a, the deviation is d_i^a = FC_i^a - mean_a (edge vector; mean over the subjects used).
  within-subject similarity   r_i(a, b) = corr over edges of d_i^a and d_i^b        -> mean and SD over subjects
  between-subject reference   corr(d_i^a, d_j^b), i != j                             -> mean over subject pairs
  self - other gap            how much of the overlap is individual rather than generic
  identification              average rank of the own subject among all subjects (chance 0.5)
Subjects: present in every condition; means over those subjects. Correlation
(not covariance) so that conditions with noisier FC do not dominate. No split-half reliability or
disattenuation: task runs differ in phase-encoding direction, stimuli and block order, so LR / RL halves are not
parallel measurements (2026-10-28). Raw-connectome similarity (with the condition
means left in) is in FC_matrix_analysis.ipynb (similarity_*_Glasser.png).

    python scripts/notebooks/EDA/FC_deviation_similarity.py      # -> scripts/notebooks/EDA/figures/fc_conditions/deviation_similarity/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
FC_ROOT = Path("/scratch/asr655/neuroinformatics/Conn2Conn_data/fc")
OUT = Path(__file__).resolve().parent / "figures" / "fc_conditions" / "deviation_similarity"   # next to the EDA notebooks
P = "Glasser"
CONDS = ["rest", "emotion", "gambling", "language", "motor", "relational", "social", "wm"]
LABEL = {c: c.capitalize() if c != "wm" else "WM" for c in CONDS}


def folder(cond):
    return FC_ROOT / (f"parc-{P}_hemi-both" + ("" if cond == "rest" else f"_task-{cond}"))


def load(path, ids):
    s = np.load(path / "subject_ids.npy")
    X = np.load(path / "upper_triangles.npy", mmap_mode="r")
    return np.asarray(X[np.searchsorted(s, ids)], dtype=np.float32)


def zrows(D):
    """Deviation rows standardised across edges (Pearson r = mean product of two standardised rows)."""
    D = D - D.mean(1, keepdims=True)
    return D / D.std(1, keepdims=True)


def main():
    paths = [folder(c) for c in CONDS]
    ids = None
    for p in paths:
        s = set(np.load(p / "subject_ids.npy").tolist())
        ids = s if ids is None else ids & s
    ids = np.array(sorted(ids))
    n = len(ids)
    print(f"{n} subjects with every condition", flush=True)

    Z = {}
    for c in CONDS:
        X = load(folder(c), ids)
        Z[c] = zrows(X - X.mean(0))
        print(f"loaded {c}", flush=True)
    E = Z[CONDS[0]].shape[1]
    rows = []
    for a in CONDS:
        for b in CONDS:
            C = Z[a] @ Z[b].T / E                                   # (n, n): corr(d_i^a, d_j^b)
            own = np.diag(C).astype(np.float64)
            off = (C.sum(1) - np.diag(C)) / (n - 1)
            rank = (C < own[:, None]).sum(1) / (n - 1)               # own vs others (identification, avg rank)
            rows.append(dict(a=a, b=b, self_mean=own.mean(), self_sd=own.std(ddof=1), other_mean=off.mean(),
                             gap=(own - off).mean(), avg_rank=rank.mean(), top1=float((C.argmax(1) == np.arange(n)).mean())))
    pairs = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT / f"deviation_similarity_pairs_{P}.csv", index=False, float_format="%.5g")
    figures(pairs, n)


def figures(pairs, n):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402  (FIG_RC, house style)

    k = len(CONDS)
    M = lambda col: pairs.pivot(index="a", columns="b", values=col).loc[CONDS, CONDS].to_numpy()
    self_m, self_sd, other_m, rank = M("self_mean"), M("self_sd"), M("other_mean"), M("avg_rank")
    off = ~np.eye(k, dtype=bool)

    def heat(ax, A, cmap, vmin, vmax, text, title, diag=None):
        A = A.copy()
        if diag is not None:
            A[np.eye(k, dtype=bool)] = diag
        ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax)
        for (i, j), v in np.ndenumerate(A):
            if np.isfinite(v):
                x = (v - vmin) / (vmax - vmin)
                ax.text(j, i, text(i, j, v), ha="center", va="center", fontsize=10.5,
                        color="white" if (x < 0.45 if cmap == "viridis" else abs(v) > 0.55 * vmax) else "black",
                        fontweight="bold" if i == j else "normal")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(range(k), [LABEL[c] for c in CONDS], rotation=40, ha="right", fontsize=13)
        ax.set_yticks(range(k), [LABEL[c] for c in CONDS], fontsize=13)
        ax.set_title(title, fontsize=15, pad=10)

    with plt.rc_context({**mb.FIG_RC, "font.size": 15}):
        fig, axes = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw=dict(wspace=0.3))
        offv = self_m[off]
        heat(axes[0], np.where(off, self_m, np.nan), "viridis", 0, 1.08 * np.nanmax(offv),
             lambda i, j, v: f"{v:.2f}\n±{self_sd[i, j]:.2f}", "Within subject: corr(d_i^a, d_i^b)\nmean ± SD over subjects")
        heat(axes[1], np.where(off, rank, np.nan), "viridis", 0.5, 1.0, lambda i, j, v: f"{v:.2f}",
             f"Own subject among all n: average rank (chance 0.5)\nbetween-subject mean r {np.nanmean(other_m[off]):+.4f}")
        for ax, lab in zip(axes, ("within-subject r", "average rank")):
            fig.colorbar(ax.images[-1], ax=ax, shrink=0.78, pad=0.03).set_label(lab, fontsize=14)
        fig.suptitle(f"Glasser FC deviations from each condition's mean: how much does a subject's pattern carry across "
                     f"conditions? (n = {n})", fontsize=17, y=0.99)
        fig.savefig(OUT / f"deviation_similarity_{P}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
