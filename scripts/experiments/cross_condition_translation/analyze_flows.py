"""Spec v4 E4.4 analysis: the flows against the baselines (test split, seeds 0-4, code size k), hypotheses H1-H6.

Baselines: E4.0c records (results/e40c/k<k>/, B1 / B2 selected on val average rank, D9; B0, B0-codes; B3 distribution);
the pooled-ridge controls B1-pooled / B1-pooled-tgt (results/pooled_ridge/k<k>/, run_pooled_ridge.py) when present.
Flows: results/flows/k<k>/<model>_seed<S>.json (run_flows.py final), every flow model found there.
Distribution metrics of the flows at T = 1 (the trained generative model; records' dist_T1), the selected-T ones as
"<model> @T*". Settings: single source (56 cells, no SC), single source + SC (56), SC only (8). Per seed, a setting's value is the
mean over its cells; comparisons are paired by seed; support rule as in v4 (seed-mean difference above 2 SE, same sign
in >= 4 of 5 seeds).

  H1  flow - B1 and flow - B2 (demeaned r, average rank, top-1)          H2  B3 - flow (energy score; > 0 = flow better)
  H1c linear flows - B1-pooled / B1-pooled-tgt (control: the gain from sharing the map, fit in closed form)
  H3  FM-nl - each linear flow (point and energy); linear flow - B1 (point) and B3 - linear flow (energy)
  H4  spread-error r of each flow vs 0; 90 % coverage                   H5  (+SC) - (no SC) for B1, B2 and the flows
  H6  share of the 56 single-source cells beating B0 (edges) and B0-codes, demeaned r and average rank

    python scripts/experiments/cross_condition_translation/analyze_flows.py [--k 256]
        -> tables/flows/k<k>/{settings.csv, hypotheses.csv, h6_vs_copy.csv, cells.csv, summary.md}
           figures/flows/k<k>/{fig_flows_settings.png, fig_flows_cells_<metric>.png, fig_flows_differences.png}
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from analyze_baselines import ROWS, heat, matrix, seq_limits
from e4_common import CFG, FC, HERE, REPO_ROOT, SC_ONLY

FLOW_LABEL = {"fm_nl": "FM-nl", "fm_lin": "FM-lin", "fm_shared": "FM-lin shared", "fm_shared_tgt": "FM-lin per-target"}
LINEAR_FLOWS = ("FM-lin", "FM-lin shared", "FM-lin per-target")
POINT = {"demeaned_pearson": "demeaned r", "avg_rank": "average rank", "top1_acc": "top-1"}
DIST = {"energy_score": "energy score", "coverage90": "90 % coverage", "spread_error_r": "spread-error r"}
SETTINGS = ("single source", "single source + SC", "SC only")


def setting(source, with_sc):
    if source == SC_ONLY:
        return "SC only"
    if source in FC:
        return "single source + SC" if with_sc else "single source"
    return None


def load(k):
    rows, dist, flows = [], [], []
    for s in CFG["seeds"]:
        r = json.loads((HERE / "results" / "e40c" / f"k{k}" / f"seed{s}.json").read_text())
        rows += r["rows"]
        dist += r["dist"]
        pf = HERE / "results" / "pooled_ridge" / f"k{k}" / f"seed{s}.json"
        if pf.exists():
            rows += json.loads(pf.read_text())["rows"]
    for f in sorted((HERE / "results" / "flows" / f"k{k}").glob("*_seed*.json")):
        r = json.loads(f.read_text())
        lab = FLOW_LABEL[r["model"]]
        flows.append(lab)
        # distribution: at T = 1 (dist_T1, run_flows.py dist) when present; the selected T (chosen for point accuracy,
        # it narrows the samples) is kept as "<model> @T*"
        sel_lab = lab + " @T*" if "dist_T1" in r else lab
        for x, out, l in ((r["rows"], rows, lab), (r["dist"], dist, sel_lab), (r.get("dist_T1", []), dist, lab)):
            for row in x:
                out.append(dict(row, seed=r["seed"], model=l + ("+SC" if row["with_sc"] else "")))
    rows, dist = pd.DataFrame(rows), pd.DataFrame(dist)
    for df in (rows, dist):
        df["setting"] = [setting(s, w) for s, w in zip(df.source, df.with_sc)]
        sc_only = (df.source == SC_ONLY) & ~df.model.str.endswith("+SC")
        df.loc[sc_only, "model"] = df.loc[sc_only, "model"] + "+SC"       # B2 on SC only is labelled B2 in E4.0c
    rows = rows[rows.setting.notna() & (rows.split == "test")]
    dist = dist[dist.setting.notna() & (dist.split == "test")]
    return rows, dist, list(dict.fromkeys(flows))


def base(model):
    return model[:-3] if model.endswith("+SC") else model


def per_seed(df, metric):
    """(model base, setting, seed) -> mean over the setting's cells."""
    q = df.assign(base=df.model.map(base)).groupby(["base", "setting", "seed"])[metric].mean()
    return q


def compare(q, a, b, sett, sign=1):
    """Paired over seeds: sign * (a - b) in setting `sett`."""
    try:
        d = sign * (q.loc[(a, sett)] - q.loc[(b, sett)]).dropna()
    except KeyError:
        return None
    if len(d) < 2:
        return None
    m, se, pos = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int((d > 0).sum())
    return dict(mean=m, se=se, n_pos=pos, n=len(d), supported=bool(m > 2 * se and pos >= len(d) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=256)
    a = ap.parse_args()
    rows, dist, flows = load(a.k)
    if not flows:
        sys.exit(f"no flow records in results/flows/k{a.k}")
    tab, figdir = HERE / "tables" / "flows" / f"k{a.k}", HERE / "figures" / "flows" / f"k{a.k}"
    tab.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    # settings table: seed mean ± SE of the per-seed setting means
    sett = []
    for metric in list(POINT) + list(DIST):
        df = rows if metric in POINT else dist
        q = per_seed(df, metric)
        for (b, s), v in q.groupby(level=[0, 1]):
            sett.append(dict(model=b, setting=s, metric=metric, mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))), n=len(v)))
    sett = pd.DataFrame(sett)
    sett.to_csv(tab / "settings.csv", index=False, float_format="%.5g")

    # hypotheses
    H = []
    qp = {m: per_seed(rows, m) for m in POINT}
    qd = {m: per_seed(dist, m) for m in DIST}

    def add(h, comp, metric, sett_, r, note=""):
        if r is not None:
            H.append(dict(H=h, comparison=comp, metric=metric, setting=sett_, note=note, **r))

    for f in flows:
        for ref in ("B1", "B2"):
            for m in POINT:
                for s in SETTINGS:
                    add("H1", f"{f} − {ref}", m, s, compare(qp[m], f, ref, s))
        if f in LINEAR_FLOWS:
            for ref in ("B1-pooled", "B1-pooled-tgt"):
                for m in POINT:
                    add("H1c", f"{f} − {ref}", m, "single source", compare(qp[m], f, ref, "single source"),
                        "> 0: flow beats the closed-form fit with the same sharing")
        for s in SETTINGS:
            add("H2", f"B3 − {f}", "energy_score", s, compare(qd["energy_score"], f, "B3", s, sign=-1), "> 0: flow better")
    for f in flows:                                                      # H4: vs 0 (one-sample over seeds)
        for s in SETTINGS:
            try:
                d = qd["spread_error_r"].loc[(f, s)]
            except KeyError:
                continue
            m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            H.append(dict(H="H4", comparison=f"{f} spread-error r vs 0", metric="spread_error_r", setting=s, mean=m, se=se,
                          n_pos=int((d > 0).sum()), n=len(d), supported=bool(m > 2 * se and (d > 0).sum() >= len(d) - 1), note=""))
    if "FM-nl" in flows:
        for lf in [f for f in flows if f in LINEAR_FLOWS]:
            for s in SETTINGS:
                for m in POINT:
                    add("H3", f"FM-nl − {lf}", m, s, compare(qp[m], "FM-nl", lf, s), "nonlinearity")
                add("H3", f"{lf} − FM-nl", "energy_score", s, compare(qd["energy_score"], "FM-nl", lf, s, sign=-1), "> 0: FM-nl better")
    for lf in [f for f in flows if f in LINEAR_FLOWS]:
        for s in SETTINGS:
            for m in POINT:
                add("H3", f"{lf} − B1", m, s, compare(qp[m], lf, "B1", s), "fitting by flow matching vs closed form")
            add("H3", f"B3 − {lf}", "energy_score", s, compare(qd["energy_score"], lf, "B3", s, sign=-1), "fitting by flow matching vs closed form")
    for b in ["B1", "B2"] + flows:
        for m in POINT:
            r = None
            try:
                d = (qp[m].loc[(b, "single source + SC")] - qp[m].loc[(b, "single source")]).dropna()
                mm, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
                r = dict(mean=mm, se=se, n_pos=int((d > 0).sum()), n=len(d), supported=bool(mm > 2 * se and (d > 0).sum() >= len(d) - 1))
            except KeyError:
                pass
            add("H5", f"{b}: (+SC) − (no SC)", m, "single source", r)
    hyp = pd.DataFrame(H)
    hyp.to_csv(tab / "hypotheses.csv", index=False, float_format="%.5g")

    # H6: share of single-source cells beating the copy (per seed, then mean ± SE)
    single = rows[rows.setting == "single source"]
    cell = single.groupby(["model", "seed", "source", "target"])[list(POINT)].mean()
    h6 = []
    for mdl in ["B1", "B1-pooled", "B1-pooled-tgt", "B2"] + flows:
        for ref in ("B0", "B0-codes"):
            for m in ("demeaned_pearson", "avg_rank"):
                try:
                    w = (cell.loc[mdl][m] > cell.loc[ref][m]).groupby(level=0).mean()
                except KeyError:
                    continue
                h6.append(dict(model=mdl, reference=ref, metric=m, share_mean=float(w.mean()), share_se=float(w.std(ddof=1) / np.sqrt(len(w)))))
    h6 = pd.DataFrame(h6)
    h6.to_csv(tab / "h6_vs_copy.csv", index=False, float_format="%.4g")

    cells = (rows.groupby(["model", "source", "target"])[list(POINT)].agg(["mean", "sem"]))
    cells.columns = [f"{x}_{y}" for x, y in cells.columns]
    cells.reset_index().to_csv(tab / "cells.csv", index=False, float_format="%.5g")

    # summary
    order = ["B0", "B0-codes", "B1", "B1-pooled", "B1-pooled-tgt", "B2", "B3"] + flows + [f + " @T*" for f in flows]
    f3 = lambda x: f"{x:.4f}"
    md = [f"# E4.4 — flows vs baselines (test, seeds 0–4, k = {a.k})", "", "Setting means (mean ± SE over seeds):", "",
          "| setting | model | demeaned r | average rank | top-1 | energy score | coverage 90 | spread-error r |", "|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        for mdl in order:
            g = sett[(sett.model == mdl) & (sett.setting == s)].set_index("metric")
            if g.empty:
                continue
            md.append(f"| {s} | {mdl} | " + " | ".join(f"{f3(g.loc[m, 'mean'])} ± {f3(g.loc[m, 'se'])}" if m in g.index else "–"
                                                     for m in list(POINT) + list(DIST)) + " |")
    md += ["", "Hypotheses (support rule: seed-mean Δ > 2 SE, same sign in ≥ 4 / 5 seeds):", "",
           "| H | comparison | metric | setting | Δ ± SE | seeds > 0 | supported | note |", "|---|---|---|---|---|---|---|---|"]
    for r in hyp.itertuples(index=False):
        md.append(f"| {r.H} | {r.comparison} | {r.metric} | {r.setting} | {r.mean:+.4f} ± {r.se:.4f} | {r.n_pos} / {r.n} | {r.supported} | {r.note} |")
    md += ["", "H6 — share of the 56 single-source cells beating the copy (mean ± SE over seeds):", "",
           "| model | vs | demeaned r | average rank |", "|---|---|---|---|"]
    for (mdl, ref), g in h6.groupby(["model", "reference"], sort=False):
        g = g.set_index("metric")
        md.append(f"| {mdl} | {ref} | " + " | ".join(f"{g.loc[m, 'share_mean']:.2f} ± {g.loc[m, 'share_se']:.2f}" if m in g.index else "–"
                                                    for m in ("demeaned_pearson", "avg_rank")) + " |")
    (tab / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    figures(rows, sett, flows, figdir, a.k)


def figures(rows, sett, flows, figdir, k):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402

    colours = {"B0": "#CFCECE", "B0-codes": "#8C8C8C", "B1": "#7FB2E5", "B1-pooled": "#0F4D92", "B1-pooled-tgt": "#3A6EA5", "B2": "#42949E", "FM-nl": "#D6604D", "FM-lin": "#F4A582",
               "FM-lin shared": "#B2182B", "FM-lin per-target": "#FDDBC7"}
    names = {"B0": "copy (edges)", "B0-codes": "copy (codes)", "B1": "ridge", "B1-pooled": "ridge, one map for all pairs",
             "B1-pooled-tgt": "ridge, one map per target", "B2": "multiview CCA"}
    present = set(sett.model)
    models = [m for m in ["B0", "B0-codes", "B1", "B1-pooled", "B1-pooled-tgt", "B2"] + flows if m in present]
    with plt.rc_context({**mb.FIG_RC, "font.size": 15}):
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        for ax, m in zip(axes, POINT):
            w = 0.8 / len(models)
            for j, mdl in enumerate(models):
                for i, s in enumerate(SETTINGS):
                    g = sett[(sett.model == mdl) & (sett.setting == s) & (sett.metric == m)]
                    if len(g):
                        ax.bar(i + (j - (len(models) - 1) / 2) * w, g["mean"].iloc[0], w, yerr=g["se"].iloc[0], color=colours.get(mdl, "#CFCECE"),
                               edgecolor="black", lw=1, capsize=3, label=names.get(mdl, mdl) if i == 0 else None)
            ax.set_xticks(range(len(SETTINGS)), SETTINGS, fontsize=14)
            ax.set_ylabel(f"test {POINT[m]}", fontsize=16)
            if m == "avg_rank":
                ax.set_ylim(0.5, None)
        h, l = axes[0].get_legend_handles_labels()
        axes[-1].legend(h, l, fontsize=13, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"E4.4: flows vs linear baselines by setting (test, seeds 0–4, k = {k})", fontsize=18)
        fig.tight_layout()
        fig.savefig(figdir / "fig_flows_settings.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        rows = rows.copy()
        rows.loc[rows.source == SC_ONLY, "model"] = rows.loc[rows.source == SC_ONLY, "model"].map(base)  # SC-only row in the no-SC panel
        single = rows[rows.setting.isin(["single source", "SC only"])]
        panels = [(p, t) for p, t in (("B1", "B1 ridge"), ("B1-pooled", "B1 pooled (one map)"), ("B2", "B2 multiview CCA"))
                  if p in present] + [(f, f) for f in flows]
        rws = FC + [SC_ONLY]
        for m in POINT:
            Ms = [matrix(single, p, m) for p, _ in panels]
            Ms = [np.vstack([M[:len(FC)], M[ROWS.index(SC_ONLY)][None]]) for M in Ms]
            vmin, vmax = seq_limits(np.concatenate([M.ravel() for M in Ms]))
            fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 7.5))
            for i, (ax, (p, t), M) in enumerate(zip(np.atleast_1d(axes), panels, Ms)):
                im = heat(ax, M, "viridis", vmin, vmax, title=t, rows=rws, ylabels=i == 0)
            fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02).set_label(f"test {POINT[m]}", fontsize=14)
            fig.suptitle(f"E4.4: test {POINT[m]} per cell (rows = source, columns = target; no SC except the SC-only row)", fontsize=17)
            fig.savefig(figdir / f"fig_flows_cells_{m}.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        diffs = [(f, ref) for f in flows for ref in ("B1", "B2") + (("B1-pooled",) if f in LINEAR_FLOWS and "B1-pooled" in present else ())]
        fig, axes = plt.subplots(len(POINT) - 1, len(diffs), figsize=(6.4 * len(diffs), 7.2 * (len(POINT) - 1)), squeeze=False)
        for r_, m in enumerate(("demeaned_pearson", "avg_rank")):
            Ds = [100 * (np.vstack([(matrix(single, f, m) - matrix(single, ref, m))[:len(FC)],
                                    (matrix(single, f, m) - matrix(single, ref, m))[ROWS.index(SC_ONLY)][None]])) for f, ref in diffs]
            lim = 1.15 * np.nanmax(np.abs(np.concatenate([D.ravel() for D in Ds]))) or 1
            for c, ((f, ref), D) in enumerate(zip(diffs, Ds)):
                im = heat(axes[r_, c], D, "RdBu_r", -lim, lim, fmt="{:+.1f}", title=f"{f} − {names.get(ref, ref)}", diverging=True,
                          rows=rws, ylabels=c == 0)
            fig.colorbar(im, ax=axes[r_], shrink=0.8, pad=0.02).set_label(f"Δ test {POINT[m]} × 100 (red = flow better)", fontsize=14)
        fig.suptitle("E4.4: flow − linear baseline per cell", fontsize=18)
        fig.savefig(figdir / "fig_flows_differences.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("wrote", figdir)


if __name__ == "__main__":
    main()
