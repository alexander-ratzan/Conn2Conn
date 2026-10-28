"""Spec v4 E4.0 analysis: baseline capacity across all cross-condition cells (test split, seeds 0-4).

Tables (tables/): baselines_cells.csv (seed mean ± SE per model x cell x metric), baselines_settings.csv (means over
each setting's cells), baselines_g0.csv, baselines_summary.md.
Figures (figures/): fig_baselines_<metric>.png (heatmaps: rows = source, columns = target, one panel per baseline),
fig_baselines_differences.png, fig_baselines_distribution.png (B3 energy score, coverage), fig_baselines_settings.png.

    python scripts/experiments/cross_condition_translation/analyze_baselines.py
"""
import json
import sys

import numpy as np
import pandas as pd

from e4_common import FC, HERE, OTHER7, REPO_ROOT, SC_ONLY

RES, TAB, FIG = HERE / "results", HERE / "tables", HERE / "figures"
ROWS = FC + [OTHER7, SC_ONLY]
ROW_LABEL = {**{v: v for v in FC}, OTHER7: "other 7", SC_ONLY: "SC only"}
POINT = {"demeaned_pearson": "test demeaned r", "avg_rank": "test average rank (chance 0.5)", "top1_acc": "test top-1"}
PANELS = [("B0", "B0 identity (copy source edges)", False), ("B0-codes", "B0 identity in 256 codes", False), ("B1", "B1 ridge", False), ("B2", "B2 multiview CCA", False),
          ("B1+SC", "B1 ridge + SC", True), ("B2+SC", "B2 multiview CCA + SC", True)]
SETTING = {**{v: "single source" for v in FC}, OTHER7: "other 7", SC_ONLY: "SC only"}


def load():
    rows, dist, g0 = [], [], []
    for f in sorted(RES.glob("baselines_seed*.json")):
        r = json.loads(f.read_text())
        rows += r["rows"]
        dist += r["dist"]
        g0.append(dict(seed=r["seed"], **r["g0"]))
    if (RES / "identity_codes.json").exists():           # B0-codes: identity in code space (run_identity_codes.py)
        rows += json.loads((RES / "identity_codes.json").read_text())["rows"]
    rows, dist = pd.DataFrame(rows), pd.DataFrame(dist)
    # SC-only cells are shown in the "+SC" panel of their family (B1's are already labelled B1+SC)
    rows.loc[(rows.source == SC_ONLY) & (rows.model == "B2"), "model"] = "B2+SC"
    return rows, dist, pd.DataFrame(g0)


def matrix(df, model, metric):
    q = df[(df.model == model)].groupby(["source", "target"])[metric].mean()
    M = np.full((len(ROWS), len(FC)), np.nan)
    for i, s in enumerate(ROWS):
        for j, t in enumerate(FC):
            if (s, t) in q.index:
                M[i, j] = q[(s, t)]
    return M


def heat(ax, M, cmap, vmin, vmax, fmt="{:.2f}", title=""):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (v - vmin) / (vmax - vmin + 1e-12) > 0.6 and cmap != "RdBu" else "black")
    ax.set_xticks(range(len(FC)), FC, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(ROWS)), [ROW_LABEL[r] for r in ROWS], fontsize=9)
    ax.axhline(len(FC) - 0.5, color="black", lw=1)
    ax.set_title(title, fontsize=11)
    return im


def main():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402  (FIG_RC)

    rows, dist, g0 = load()
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    g0.to_csv(TAB / "baselines_g0.csv", index=False)
    test = rows[rows.split == "test"]
    seeds = test.seed.nunique()
    per_seed = test.groupby(["model", "source", "target", "seed"])[list(POINT)].mean().reset_index()
    cell = per_seed.groupby(["model", "source", "target"])[list(POINT)].agg(["mean", "sem"])
    cell.columns = [f"{a}_{b}" for a, b in cell.columns]
    cell.reset_index().to_csv(TAB / "baselines_cells.csv", index=False, float_format="%.5g")
    dtest = dist[dist.split == "test"]
    dcell = dtest.groupby(["model", "source", "target"])[["energy_score", "coverage90", "spread_error_r", "pred_spread", "sq_error"]].mean()
    dcell.reset_index().to_csv(TAB / "baselines_distribution_cells.csv", index=False, float_format="%.5g")

    per_seed["setting"] = per_seed.source.map(SETTING)
    sett = per_seed.groupby(["model", "setting", "seed"])[list(POINT)].mean().groupby(["model", "setting"]).agg(["mean", "sem"])
    sett.columns = [f"{a}_{b}" for a, b in sett.columns]
    sett = sett.reset_index()
    ceil = per_seed[per_seed.model == "ceiling"].groupby("seed")[list(POINT)].mean().mean()
    ret = per_seed[per_seed.model == "retest"].groupby("seed")[list(POINT)].mean().mean()
    sett.to_csv(TAB / "baselines_settings.csv", index=False, float_format="%.5g")

    with plt.rc_context(mb.FIG_RC):
        for metric, lab in POINT.items():
            vals = np.concatenate([matrix(test, m, metric).ravel() for m, _, _ in PANELS])
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 6.2), sharey=True)
            for ax, (m, title, _) in zip(axes, PANELS):
                im = heat(ax, matrix(test, m, metric), "viridis", vmin, vmax, title=title)
            fig.colorbar(im, ax=axes, shrink=0.75, pad=0.01).set_label(lab)
            cm = per_seed[per_seed.model == "ceiling"].groupby("target")[metric].mean()
            fig.suptitle(f"E4.0 baselines: {lab} per cell (rows = source, columns = target; mean of {seeds} seeds). "
                         f"Ceilings: own 256-code reconstruction {ceil[metric]:.2f} (per target "
                         f"{cm.min():.2f}–{cm.max():.2f}); rest session 1 ↔ 2 copy {ret[metric]:.2f}", fontsize=12)
            fig.savefig(FIG / f"fig_baselines_{metric}.png", dpi=250, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        diffs = [("B1", "B0-codes", "ridge − identity in codes"), ("B2", "B1", "multiview CCA − ridge"),
                 ("B1+SC", "B1", "ridge: + SC − without"), ("B2+SC", "B2", "multiview CCA: + SC − without")]
        fig, axes = plt.subplots(2, len(diffs), figsize=(5.2 * len(diffs), 12), sharey=True)
        for r, metric in enumerate(("demeaned_pearson", "avg_rank")):
            Ms = [matrix(test, a, metric) - matrix(test, b, metric) for a, b, _ in diffs]
            lim = np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in Ms])), 98)
            for ax, M, (_, _, title) in zip(axes[r], Ms, diffs):
                im = heat(ax, M, "RdBu", -lim, lim, fmt="{:+.3f}", title=f"{title}\n({POINT[metric]})")
            fig.colorbar(im, ax=axes[r], shrink=0.75, pad=0.01).set_label(f"Δ {POINT[metric]} (blue = first better)")
        fig.suptitle("E4.0 baselines: differences per cell (test, mean of seeds)", fontsize=13)
        fig.savefig(FIG / "fig_baselines_differences.png", dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 12), sharey=True)
        for c, m in enumerate(("B3", "B3+SC")):
            for r, (metric, cmap, lab) in enumerate((("energy_score", "viridis_r", "energy score (lower = better)"),
                                                     ("coverage90", "viridis", "90 % coverage (0.9 = calibrated)"))):
                q = dtest[dtest.model == m]
                M = matrix(q.assign(model=m), m, metric)
                allv = np.concatenate([matrix(dtest[dtest.model == x], x, metric).ravel() for x in ("B3", "B3+SC")])
                im = heat(axes[r, c], M, cmap, np.nanmin(allv), np.nanmax(allv),
                          fmt="{:.2f}", title=f"{m} linear-Gaussian · {lab}")
        fig.suptitle("E4.0 B3 (ridge mean + per-cell residual covariance): distributional scores, code space (test)", fontsize=12)
        fig.savefig(FIG / "fig_baselines_distribution.png", dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
        models = ["B0", "B0-codes", "B1", "B2", "B1+SC", "B2+SC"]
        cols = ["#BDBDBD", "#8C8C8C", "#3775BA", "#42949E", "#0F4D92", "#1B5E5E"]
        settings = ["single source", "other 7", "SC only"]
        w = 0.8 / len(models)
        for ax, metric in zip(axes, ("demeaned_pearson", "avg_rank")):
            for j, (m, c) in enumerate(zip(models, cols)):
                for i, st in enumerate(settings):
                    q = sett[(sett.model == m) & (sett.setting == st)]
                    if len(q):
                        ax.bar(i + (j - (len(models) - 1) / 2) * w, q[f"{metric}_mean"].iloc[0], w, yerr=q[f"{metric}_sem"].iloc[0], color=c,
                               capsize=2, label=m if i == 0 or (m.endswith("SC") and st == "SC only") else None)
            ax.axhline(ceil[metric], color="black", ls="--", lw=1.2, label="own reconstruction (ceiling)")
            ax.axhline(ret[metric], color="black", ls=":", lw=1.2, label="rest session 1 ↔ 2 copy")
            ax.set_xticks(range(len(settings)), settings)
            ax.set_ylabel(POINT[metric])
        h, l = axes[0].get_legend_handles_labels()
        uniq = dict(zip(l, h))
        axes[1].legend(uniq.values(), uniq.keys(), fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle("E4.0 baselines: mean over each setting's cells (test, mean ± SE over seeds)", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG / "fig_baselines_settings.png", dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # summary
    f = lambda x: f"{x:.4f}"
    md = ["# E4.0 baselines — summary (test, mean ± SE over seeds)", "",
          f"G0 (B1 / B2 without SC reproduce E2.0 L1 / L4 at k = 256): {'PASS' if g0['pass'].all() else 'FAIL'} "
          f"(max |Δ| L1 {g0['L1_max_abs_diff'].max():.1e}, L4 {g0['L4_max_abs_diff'].max():.1e}; {len(g0)} seeds).", "",
          f"Ceilings: own 256-code reconstruction demeaned r {f(ceil['demeaned_pearson'])}, average rank {f(ceil['avg_rank'])}; "
          f"rest session 1 ↔ 2 copy demeaned r {f(ret['demeaned_pearson'])}, average rank {f(ret['avg_rank'])}.", "",
          "| model | setting | demeaned r | average rank | top-1 |", "|---|---|---|---|---|"]
    for r in sett.sort_values(["setting", "model"]).itertuples(index=False):
        d = r._asdict()
        md.append(f"| {r.model} | {r.setting} | " + " | ".join(f"{f(d[f'{k}_mean'])} ± {f(d[f'{k}_sem'])}" for k in POINT) + " |")
    md += ["", "Distribution (B3, code space; means over cells):", "", "| model | setting | energy score | 90 % coverage |", "|---|---|---|---|"]
    dtest = dtest.assign(setting=dtest.source.map(SETTING))
    for (m, st), g in dtest.groupby(["model", "setting"]):
        md.append(f"| {m} | {st} | {f(g.energy_score.mean())} | {g.coverage90.mean():.3f} |")
    b1 = matrix(test, "B1", "demeaned_pearson")[:len(FC)]
    b0 = matrix(test, "B0", "demeaned_pearson")[:len(FC)]
    b0c = matrix(test, "B0-codes", "demeaned_pearson")[:len(FC)]
    md += ["", f"Single-source cells where ridge beats identity on demeaned r: {int(np.nansum(b1 > b0))} / {int(np.isfinite(b1).sum())} "
           f"(edges); {int(np.nansum(b1 > b0c))} / {int(np.isfinite(b1).sum())} (identity in codes).",
           "", "Best single source per target (B1, demeaned r):", ""]
    for j, t in enumerate(FC):
        col = b1[:, j]
        i = int(np.nanargmax(col))
        md.append(f"- {t}: {FC[i]} ({col[i]:.3f})")
    (TAB / "baselines_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
