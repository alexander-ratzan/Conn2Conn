"""How similar are a subject's FC deviations from the condition mean across conditions? (Glasser, all FC conditions)

For subject i and condition a, the deviation is d_i^a = FC_i^a - mean_a (edge vector; mean over the subjects used).
  within-subject similarity   r_i(a, b) = corr over edges of d_i^a and d_i^b        -> mean and SD over subjects
  between-subject reference   corr(d_i^a, d_j^b), i != j                             -> mean over subject pairs
  self - other gap            how much of the overlap is individual rather than generic
  reliability (diagonal)      tasks: the LR run's deviation vs the RL run's; rest: session 1 vs session 2;
                              Spearman-Brown to full length (2 halves): rel = 2 r / (1 + r)
  disattenuated similarity    r(a, b) / sqrt(rel_a rel_b): ~1 = the same individual pattern, only noisier
Subjects: present in every combined condition and every half (runs / sessions); means over those subjects. Correlation
(not covariance) so that conditions with noisier FC do not dominate. Raw-connectome similarity (with the condition
means left in) is in FC_matrix_analysis.ipynb (similarity_*_Glasser.png).

    python scripts/notebooks/EDA/fc_deviation_similarity.py      # -> scripts/notebooks/EDA/figures/fc_conditions/deviation_similarity/
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


def folder(cond, half=None):
    base = f"parc-{P}_hemi-both" + ("" if cond == "rest" else f"_task-{cond}")
    if half is None:
        return FC_ROOT / base
    if cond == "rest":
        return FC_ROOT / f"parc-{P}_hemi-both_session{half}"
    return FC_ROOT / f"{base}_dir-{'LR' if half == 1 else 'RL'}_run-1"


def load(path, ids):
    s = np.load(path / "subject_ids.npy")
    X = np.load(path / "upper_triangles.npy", mmap_mode="r")
    return np.asarray(X[np.searchsorted(s, ids)], dtype=np.float32)


def zrows(D):
    """Deviation rows standardised across edges (Pearson r = mean product of two standardised rows)."""
    D = D - D.mean(1, keepdims=True)
    return D / D.std(1, keepdims=True)


def main():
    paths = [folder(c) for c in CONDS] + [folder(c, h) for c in CONDS for h in (1, 2)]
    ids = None
    for p in paths:
        s = set(np.load(p / "subject_ids.npy").tolist())
        ids = s if ids is None else ids & s
    ids = np.array(sorted(ids))
    n = len(ids)
    print(f"{n} subjects with every condition and both halves", flush=True)

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

    rel = []
    for c in CONDS:
        A, B = load(folder(c, 1), ids), load(folder(c, 2), ids)
        za, zb = zrows(A - A.mean(0)), zrows(B - B.mean(0))
        r = (za * zb).mean(1).astype(np.float64)
        rel.append(dict(condition=c, half_r_mean=r.mean(), half_r_sd=r.std(ddof=1), rel_full=2 * r.mean() / (1 + r.mean()),
                        halves="session 1 vs 2" if c == "rest" else "LR run vs RL run"))
        print(f"reliability {c}: split-half r {r.mean():.3f}", flush=True)
    rel = pd.DataFrame(rel).set_index("condition")
    pairs["disattenuated"] = [r.self_mean / np.sqrt(rel.rel_full[r.a] * rel.rel_full[r.b]) if r.a != r.b else np.nan
                              for r in pairs.itertuples()]
    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT / f"deviation_similarity_pairs_{P}.csv", index=False, float_format="%.5g")
    rel.to_csv(OUT / f"deviation_reliability_{P}.csv", float_format="%.5g")
    figures(pairs, rel, n)


def figures(pairs, rel, n):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402  (FIG_RC, house style)

    k = len(CONDS)
    M = lambda col: pairs.pivot(index="a", columns="b", values=col).loc[CONDS, CONDS].to_numpy()
    self_m, self_sd, other_m, rank, dis = M("self_mean"), M("self_sd"), M("other_mean"), M("avg_rank"), M("disattenuated")
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
        fig, axes = plt.subplots(1, 3, figsize=(27, 9), gridspec_kw=dict(wspace=0.32))
        half = rel.half_r_mean.reindex(CONDS).to_numpy()
        hi = max(np.nanmax(self_m[off]), np.nanmax(half))
        lo = min(0.0, np.nanmin(other_m))
        pad = 0.08 * (hi - lo)
        heat(axes[0], self_m, "viridis", lo - pad, hi + pad,
             lambda i, j, v: f"{v:.2f}\n±{self_sd[i, j]:.2f}" if i != j else f"{v:.2f}",
             "Within subject: corr(d_i^a, d_i^b)\nmean ± SD over subjects · diagonal: split-half r", diag=half)
        heat(axes[1], rank, "viridis", 0.5, 1.0, lambda i, j, v: f"{v:.2f}",
             f"Own subject among all n: average rank of\ncorr(d_i^a, d_i^b) vs corr(d_i^a, d_j^b) (chance 0.5)\n"
             f"between-subject mean r {np.nanmean(other_m):+.4f}")
        dmax = 1.1
        heat(axes[2], dis, "viridis", 0, dmax, lambda i, j, v: f"{v:.2f}",
             "Disattenuated: r(a, b) / √(rel_a · rel_b)\n1 = same individual pattern, only noisier\ndiagonal: reliability (full length)",
             diag=rel.rel_full.reindex(CONDS).to_numpy())
        for ax, lab in zip(axes, ("within-subject r", "average rank", "disattenuated r")):
            fig.colorbar(ax.images[-1], ax=ax, shrink=0.78, pad=0.03).set_label(lab, fontsize=14)
        fig.suptitle(f"Glasser FC deviations from each condition's mean: how much does a subject's pattern carry across "
                     f"conditions? (n = {n})", fontsize=18, y=0.98)
        fig.savefig(OUT / f"deviation_similarity_{P}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 6.5))
        x = np.arange(k)
        cross = np.array([np.nanmean(np.where(off[i], self_m[i], np.nan)) for i in range(k)])
        w = 0.27
        ax.bar(x - w, rel.half_r_mean.reindex(CONDS), w, color="#CFCECE", edgecolor="black", label="split-half r (one run / session)")
        ax.bar(x, rel.rel_full.reindex(CONDS), w, color="#0F4D92", edgecolor="black", label="reliability, full length (Spearman–Brown)")
        ax.bar(x + w, cross, w, color="#8BCF8B", edgecolor="black", label="mean within-subject r with the other 7 conditions")
        ax.set_xticks(x, [LABEL[c] for c in CONDS], fontsize=14)
        ax.set_ylabel("correlation of deviations (edges)")
        ax.legend(frameon=False, fontsize=13, loc="upper right")
        ax.set_title("Reliability of each condition's individual deviation vs how much of it transfers to other conditions",
                     fontsize=15)
        fig.tight_layout()
        fig.savefig(OUT / f"deviation_reliability_{P}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
