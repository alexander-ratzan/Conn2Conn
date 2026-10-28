"""Spec v4 E4.0 analysis: baseline capacity across all cross-condition cells (test split, seeds 0-4), for one code size.

Reads results/k<k>/ (baselines_seed*.json, identity_codes.json); writes tables/k<k>/ and figures/k<k>/:
  tables: baselines_cells.csv (seed mean ± SE per model x cell x metric), baselines_distribution_cells.csv,
          baselines_settings.csv (means over each setting's cells), baselines_g0.csv, baselines_summary.md
  figures: fig_baselines_<metric>.png (heatmaps: rows = source, columns = target; panels: copy, copy in codes, ridge,
           ridge + SC, CCA, CCA + SC), fig_baselines_differences.png (each panel = model − its reference, named in
           the title; red = gain), fig_baselines_distribution.png (B3), fig_baselines_settings.png

    python scripts/experiments/cross_condition_translation/analyze_baselines.py [--k 512]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from e4_common import CFG, FC, HERE, OTHER7, REPO_ROOT, SC_ONLY, kdir

ROWS = FC + [OTHER7, SC_ONLY]
ROW_LABEL = {**{v: v for v in FC}, OTHER7: "other 7", SC_ONLY: "SC only"}
POINT = {"demeaned_pearson": "test demeaned r", "avg_rank": "test average rank (chance 0.5)", "top1_acc": "test top-1"}
PANELS = [("B0", "B0 copy source (edges)"), ("B0-codes", "B0 copy source (codes)"), ("B1", "B1 ridge"),
          ("B1+SC", "B1 ridge + SC"), ("B2", "B2 multiview CCA"), ("B2+SC", "B2 multiview CCA + SC")]
DIFFS = [("B1", "B0-codes", "ridge − copy (codes)"), ("B1+SC", "B1", "ridge + SC − ridge"),
         ("B2", "B0-codes", "multiview CCA − copy (codes)"), ("B2+SC", "B2", "multiview CCA + SC − multiview CCA")]
SETTING = {**{v: "single source" for v in FC}, OTHER7: "other 7", SC_ONLY: "SC only"}
PAD = 0.08            # colour range headroom (fraction of the data range) at both ends of sequential scales
DIV_PAD = 1.15        # diverging scales: limit = 1.15 x the largest |difference|


def load(res):
    rows, dist, g0 = [], [], []
    for f in [res / f"baselines_seed{s}.json" for s in CFG["seeds"]]:
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows += r["rows"]
        dist += r["dist"]
        g0.append(dict(seed=r["seed"], **r["g0"]))
    if (res / "identity_codes.json").exists():           # B0-codes: copy in code space (run_identity_codes.py)
        rows += json.loads((res / "identity_codes.json").read_text())["rows"]
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


def seq_limits(vals):
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = PAD * (hi - lo or 1.0)
    return lo - pad, hi + pad


def heat(ax, M, cmap, vmin, vmax, fmt="{:.2f}", title="", diverging=False, rows=ROWS, ylabels=True):
    """Annotated heatmap (house style: no spines, readable cell text, other-7 / SC-only rows set off by a line)."""
    M = M[:len(rows)]
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if not np.isfinite(v):
            continue
        if diverging:
            light = abs(v) > 0.55 * vmax
        else:
            x = (v - vmin) / (vmax - vmin)
            light = x < 0.45 if not cmap.endswith("_r") else x > 0.55     # viridis: dark at the low end
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=10, color="white" if light else "black")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(range(len(FC)), FC, rotation=40, ha="right", fontsize=12)
    ax.set_yticks(range(len(rows)), [ROW_LABEL[r] for r in rows] if ylabels else [""] * len(rows), fontsize=12)
    ax.axhline(len(FC) - 0.5, color="black", lw=1.5)
    ax.set_title(title, fontsize=14, pad=8)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=CFG["k"])
    K = ap.parse_args().k
    res, tab, fig_dir = kdir(HERE / "results", K), kdir(HERE / "tables", K), kdir(HERE / "figures", K)
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402  (FIG_RC)

    rows, dist, g0 = load(res)
    tab.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    g0.to_csv(tab / "baselines_g0.csv", index=False)
    test = rows[rows.split == "test"]
    seeds = test.seed.nunique()
    per_seed = test.groupby(["model", "source", "target", "seed"])[list(POINT)].mean().reset_index()
    cell = per_seed.groupby(["model", "source", "target"])[list(POINT)].agg(["mean", "sem"])
    cell.columns = [f"{a}_{b}" for a, b in cell.columns]
    cell.reset_index().to_csv(tab / "baselines_cells.csv", index=False, float_format="%.5g")
    dtest = dist[dist.split == "test"]
    dcell = dtest.groupby(["model", "source", "target"])[["energy_score", "coverage90", "spread_error_r", "pred_spread", "sq_error"]].mean()
    dcell.reset_index().to_csv(tab / "baselines_distribution_cells.csv", index=False, float_format="%.5g")

    per_seed["setting"] = per_seed.source.map(SETTING)
    sett = per_seed.groupby(["model", "setting", "seed"])[list(POINT)].mean().groupby(["model", "setting"]).agg(["mean", "sem"])
    sett.columns = [f"{a}_{b}" for a, b in sett.columns]
    sett = sett.reset_index()
    ceil = per_seed[per_seed.model == "ceiling"].groupby("seed")[list(POINT)].mean().mean()
    ret = per_seed[per_seed.model == "retest"].groupby("seed")[list(POINT)].mean().mean()
    sett.to_csv(tab / "baselines_settings.csv", index=False, float_format="%.5g")

    rc = {**mb.FIG_RC, "font.size": 15}
    grid = [["B0", "B1", "B2"], ["B0-codes", "B1+SC", "B2+SC"]]       # columns = family; rows = variant
    with plt.rc_context(rc):
        for metric, lab in POINT.items():
            vmin, vmax = seq_limits(np.concatenate([matrix(test, m, metric).ravel() for m, _ in PANELS]))
            fig, axes = plt.subplots(2, 3, figsize=(20, 15), gridspec_kw=dict(hspace=0.32, wspace=0.08))
            names = dict(PANELS)
            for r in range(2):
                for c in range(3):
                    m = grid[r][c]
                    im = heat(axes[r, c], matrix(test, m, metric), "viridis", vmin, vmax, title=names[m], ylabels=c == 0,
                             rows=ROWS if r == 1 else ROWS[:-1])     # no-SC row: no SC-only cells
            fig.colorbar(im, ax=axes, shrink=0.6, pad=0.015).set_label(lab, fontsize=15)
            cm = per_seed[per_seed.model == "ceiling"].groupby("target")[metric].mean()
            fig.suptitle(f"E4.0 baselines (k = {K}): {lab.replace('test ', '')} per cell — rows = source, columns = target",
                         fontsize=18, y=0.965)
            fig.text(0.44, 0.925, f"test, mean of {seeds} seeds · ceilings: own {K}-code reconstruction {ceil[metric]:.2f} "
                     f"(per target {cm.min():.2f}–{cm.max():.2f}); rest session 1 ↔ 2 copy {ret[metric]:.2f}",
                     ha="center", fontsize=13, color="#4D4D4D")
            fig.savefig(fig_dir / f"fig_baselines_{metric}.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        fig, axes = plt.subplots(2, len(DIFFS), figsize=(24, 15), gridspec_kw=dict(hspace=0.35, wspace=0.08))
        for r, metric in enumerate(("demeaned_pearson", "avg_rank")):
            Ms = [100 * (matrix(test, a, metric) - matrix(test, b, metric)) for a, b, _ in DIFFS]
            lim = DIV_PAD * np.nanmax(np.abs(np.concatenate([m.ravel() for m in Ms])))
            for c, (ax, M, (a, b, title)) in enumerate(zip(axes[r], Ms, DIFFS)):
                im = heat(ax, M, "RdBu_r", -lim, lim, fmt="{:+.1f}", diverging=True, rows=ROWS[:-1], ylabels=c == 0, title=title)
            name = POINT[metric].replace("test ", "").replace(" (chance 0.5)", "")
            fig.colorbar(im, ax=axes[r], shrink=0.8, pad=0.012).set_label(f"Δ {name} × 100", fontsize=15)
            axes[r, 0].annotate(name, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center",
                                fontsize=17, fontweight="bold")
        fig.suptitle(f"E4.0 baselines (k = {K}): model − its reference, per cell (× 100; red = better than the reference, blue = worse)",
                     fontsize=18, y=0.96)
        fig.savefig(fig_dir / "fig_baselines_differences.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(15, 15), gridspec_kw=dict(hspace=0.3, wspace=0.08))
        for r, (metric, cmap, lab, fmt) in enumerate((("energy_score", "viridis_r", "energy score (lower = better)", "{:.3f}"),
                                                      ("coverage90", "viridis", "90 % coverage (0.90 = calibrated)", "{:.2f}"))):
            allv = np.concatenate([matrix(dtest[dtest.model == x], x, metric).ravel() for x in ("B3", "B3+SC")])
            vmin, vmax = seq_limits(allv)
            for c, m in enumerate(("B3", "B3+SC")):
                im = heat(axes[r, c], matrix(dtest[dtest.model == m], m, metric), cmap, vmin, vmax, fmt=fmt, ylabels=c == 0,
                          title=f"{'B3 linear-Gaussian' if m == 'B3' else 'B3 linear-Gaussian + SC'}")
            fig.colorbar(im, ax=axes[r], shrink=0.8, pad=0.015).set_label(lab, fontsize=15)
        fig.suptitle(f"E4.0 distributional baseline (k = {K}): ridge mean + per-cell residual covariance, scored in code space (test)",
                     fontsize=17, y=0.96)
        fig.savefig(fig_dir / "fig_baselines_distribution.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        models = [m for m, _ in PANELS]
        cols = ["#CFCECE", "#8C8C8C", "#7FB2E5", "#0F4D92", "#8BCF8B", "#42949E"]   # house palette: neutrals, blues, greens
        labels = dict(PANELS)
        settings = ["single source", "other 7", "SC only"]
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        w = 0.8 / len(models)
        for ax, metric in zip(axes, ("demeaned_pearson", "avg_rank")):
            for j, (m, c) in enumerate(zip(models, cols)):
                for i, st in enumerate(settings):
                    q = sett[(sett.model == m) & (sett.setting == st)]
                    if len(q):
                        ax.bar(i + (j - (len(models) - 1) / 2) * w, q[f"{metric}_mean"].iloc[0], w, yerr=q[f"{metric}_sem"].iloc[0],
                               color=c, edgecolor="black", linewidth=1.0, capsize=3, label=labels[m])
            ax.axhline(ceil[metric], color="black", ls="--", lw=1.8, label=f"own {K}-code reconstruction (ceiling)")
            ax.axhline(ret[metric], color="black", ls=":", lw=1.8, label="rest session 1 ↔ 2 copy")
            ax.set_xticks(range(len(settings)), settings, fontsize=16)
            ax.set_ylabel(POINT[metric], fontsize=16)
            if metric == "avg_rank":
                ax.set_ylim(0.5, 1.02)
        h, l = axes[0].get_legend_handles_labels()
        uniq = dict(zip(l, h))
        axes[1].legend(uniq.values(), uniq.keys(), fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.suptitle(f"E4.0 baselines (k = {K}): mean over each setting's cells (test, mean ± SE over seeds)", fontsize=18)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_baselines_settings.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # summary
    f = lambda x: f"{x:.4f}"
    md = [f"# E4.0 baselines — summary (k = {K}; test, mean ± SE over seeds)", "",
          f"G0 (B1 / B2 without SC reproduce E2.0 L1 / L4 at k = {K}): {'PASS' if g0['pass'].all() else 'FAIL'} "
          f"(max |Δ| L1 {g0['L1_max_abs_diff'].max():.1e}, L4 {g0['L4_max_abs_diff'].max():.1e}; {len(g0)} seeds).", "",
          f"Ceilings: own {K}-code reconstruction demeaned r {f(ceil['demeaned_pearson'])}, average rank {f(ceil['avg_rank'])}; "
          f"rest session 1 ↔ 2 copy demeaned r {f(ret['demeaned_pearson'])}, average rank {f(ret['avg_rank'])}.", "",
          "| model | setting | demeaned r | average rank | top-1 |", "|---|---|---|---|---|"]
    order = {m: i for i, (m, _) in enumerate(PANELS + [("ceiling", ""), ("retest", "")])}
    for r in sorted(sett.itertuples(index=False), key=lambda r: (r.setting, order.get(r.model, 99))):
        d = r._asdict()
        md.append(f"| {r.model} | {r.setting} | " + " | ".join(f"{f(d[f'{k}_mean'])} ± {f(d[f'{k}_sem'])}" for k in POINT) + " |")
    md += ["", "Distribution (B3, code space; means over cells):", "", "| model | setting | energy score | 90 % coverage |", "|---|---|---|---|"]
    dtest = dtest.assign(setting=dtest.source.map(SETTING))
    for (m, st), g in dtest.groupby(["model", "setting"]):
        md.append(f"| {m} | {st} | {f(g.energy_score.mean())} | {g.coverage90.mean():.3f} |")
    b1 = matrix(test, "B1", "demeaned_pearson")[:len(FC)]
    b0 = matrix(test, "B0", "demeaned_pearson")[:len(FC)]
    b0c = matrix(test, "B0-codes", "demeaned_pearson")[:len(FC)]
    sel = [json.loads((res / f"baselines_seed{s}.json").read_text())["hyperparams"]["B2"] for s in CFG["seeds"]
           if (res / f"baselines_seed{s}.json").exists()]
    md += ["", "B2 selected (d, lambda) per seed: " + ", ".join(f"({x['d']}, {x['lambda_rel']:g})" for x in sel) + ".",
           "", f"Single-source cells where ridge beats the copy on demeaned r: {int(np.nansum(b1 > b0))} / {int(np.isfinite(b1).sum())} "
           f"(edges); {int(np.nansum(b1 > b0c))} / {int(np.isfinite(b1).sum())} (codes).",
           "", "Best single source per target (B1, demeaned r):", ""]
    for j, t in enumerate(FC):
        col = b1[:, j]
        i = int(np.nanargmax(col))
        md.append(f"- {t}: {FC[i]} ({col[i]:.3f})")
    (tab / "baselines_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
