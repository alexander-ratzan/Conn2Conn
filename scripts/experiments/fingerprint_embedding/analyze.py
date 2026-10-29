"""Spec v4 E6 analysis: tables and figures from results/seed*.json (test split, mean ± SE over seeds).

SC comparators from E4.0c (cross_condition_translation/results/e40c/k256): B1+SC and B2 on the SC-only cells, i.e.
SC -> each condition by ridge and by the 9-view multiview CCA (encode SC, decode the condition).

    python scripts/experiments/fingerprint_embedding/analyze.py
        -> tables/{retest, confounds_r2, confounds_ident, behaviour, sc_targets}.csv, tables/summary.md, figures/fig_*.png
"""
import json
import sys

import numpy as np
import pandas as pd

from fe_common import E4, FC, HERE, REPO_ROOT, SEEDS

SE = lambda x: x.std(ddof=1) / np.sqrt(len(x))


def load():
    rows, w = [], {}
    for s in SEEDS:
        f = HERE / "results" / f"seed{s}.json"
        if f.exists():
            r = json.loads(f.read_text())
            rows += [dict(x, seed=s) for x in r["rows"]]
            w[s] = (np.array(r["w"]), r["closed_form_max_abs_diff"])
    return pd.DataFrame(rows), w


def e40c_sc():
    out = []
    for s in SEEDS:
        r = json.loads((E4 / "results" / "e40c" / "k256" / f"seed{s}.json").read_text())
        for x in r["rows"]:
            if x["source"] == "SC" and x["split"] == "test" and x["model"] in ("B1+SC", "B2"):
                out.append(dict(seed=s, route="SC → condition, ridge (E4 B1)" if x["model"] == "B1+SC" else "SC → condition, multiview CCA (E4 B2)",
                                against=x["target"], avg_rank=x["avg_rank"], demeaned_pearson=x["demeaned_pearson"], top1_acc=x["top1_acc"]))
    return pd.DataFrame(out)


def agg(df, keys, metrics):
    g = df.groupby(keys + ["seed"])[metrics].mean().groupby(keys)
    m, e = g.mean(), g.agg(SE)
    out = m.copy()
    for c in metrics:
        out[f"{c}_se"] = e[c]
    return out.reset_index()


def main():
    df, w = load()
    test = df[df.split == "test"]
    tab, fig = HERE / "tables", HERE / "figures"
    tab.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)
    P = ["avg_rank", "top1_acc", "demeaned_pearson"]

    rt = agg(test[test.check == "retest"], ["model"], P)
    r2 = agg(test[test.check == "confound_r2"], ["representation", "of"], ["r2"])
    ci = agg(test[test.check == "confound_ident"], ["representation", "residualised"], ["avg_rank", "top1"])
    bh = agg(test[test.check == "behaviour"], ["target", "features"], ["r"])
    sc = test[test.check == "sc_target"].copy()
    sc["route"] = "SC → " + sc.target + " (ridge)"
    own = agg(sc[sc.against == "own target (codes)"], ["route"], ["r2", "avg_rank", "top1_acc"])
    cond = pd.concat([sc[sc.against != "own target (codes)"][["seed", "route", "against"] + P], e40c_sc()])
    vs_rest = agg(cond[cond.against == "rest"], ["route"], P)
    vs_all = agg(cond.groupby(["seed", "route"])[P].mean().reset_index(), ["route"], P)
    for name, t in (("retest", rt), ("confounds_r2", r2), ("confounds_ident", ci), ("behaviour", bh), ("sc_targets_own", own),
                    ("sc_targets_vs_rest", vs_rest), ("sc_targets_vs_all8", vs_all)):
        t.to_csv(tab / f"{name}.csv", index=False, float_format="%.5g")

    f4 = lambda m, e: f"{m:.3f} ± {e:.3f}"
    md = ["# E6 — fingerprint embedding checks (test, seeds 0–4, mean ± SE)", ""]
    cf = [v[1] for v in w.values()]
    md += [f"Closed form B = (8/7)W − I/7 vs the fitted map: max |diff| {max(cf):.1e}. Directions with fingerprint share w > 0.5: "
           + ", ".join(f"seed {s} {int((v[0] > 0.5).sum())}" for s, v in w.items()) + f" of {len(next(iter(w.values()))[0])}.", ""]
    md += ["## 1. Rest session 1 ↔ session 2 (different days)", "", "| model | average rank | top-1 | demeaned r |", "|---|---|---|---|"]
    md += [f"| {r.model} | " + " | ".join(f4(getattr(r, c), getattr(r, f'{c}_se')) for c in P) + " |" for r in rt.itertuples()]
    md += ["", "## 2a. Confound proxies → representation (test R²)", "", "| representation | of | R² |", "|---|---|---|"]
    md += [f"| {r.representation} | {r.of} | {f4(r.r2, r.r2_se)} |" for r in r2.itertuples()]
    md += ["", "## 2b. Cross-condition identification (56 pairs, code space), confounds removed or not", "",
           "| representation | confounds removed | average rank | top-1 |", "|---|---|---|---|"]
    md += [f"| {r.representation} | {r.residualised} | {f4(r.avg_rank, r.avg_rank_se)} | {f4(r.top1, r.top1_se)} |" for r in ci.itertuples()]
    md += ["", "## 3. Behaviour (test Pearson r of ridge predictions)", "", "| target | " + " | ".join(bh.features.unique()) + " |",
           "|---|" + "---|" * bh.features.nunique()]
    for t, g in bh.groupby("target", sort=False):
        g = g.set_index("features")
        md.append(f"| {t} | " + " | ".join(f4(g.loc[f, 'r'], g.loc[f, 'r_se']) for f in bh.features.unique()) + " |")
    md += ["", "## 4. SC → targets", "", "Against the own training target (codes):", "", "| route | R² | average rank | top-1 |", "|---|---|---|---|"]
    md += [f"| {r.route} | {f4(r.r2, r.r2_se)} | {f4(r.avg_rank, r.avg_rank_se)} | {f4(r.top1_acc, r.top1_acc_se)} |" for r in own.itertuples()]
    for lab, t in (("measured rest FC", vs_rest), ("measured FC, mean over the 8 conditions", vs_all)):
        md += ["", f"Against {lab} (decoded edges):", "", "| route | average rank | top-1 | demeaned r |", "|---|---|---|---|"]
        md += [f"| {r.route} | " + " | ".join(f4(getattr(r, c), getattr(r, f'{c}_se')) for c in P) + " |" for r in t.itertuples()]
    (tab / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    figures(rt, ci, bh, vs_rest, vs_all, w, fig)


def figures(rt, ci, bh, vs_rest, vs_all, w, fig):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402

    def bars(ax, labels, vals, errs, colours, ylabel, ylim=None):
        x = np.arange(len(labels))
        ax.bar(x, vals, 0.7, yerr=errs, color=colours, edgecolor="black", lw=1, capsize=3)
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=15)
        if ylim:
            ax.set_ylim(*ylim)

    grey, blue, red = "#CFCECE", "#7FB2E5", "#B2182B"
    with plt.rc_context({**mb.FIG_RC, "font.size": 14}):
        order = ["copy (edges)", "copy (codes)", "embedding"]
        r = rt.set_index("model").loc[order]
        f_, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        for ax, (c, lab, lim) in zip(axes, [("avg_rank", "test average rank", (0.5, 1)), ("top1_acc", "test top-1", None),
                                             ("demeaned_pearson", "test demeaned r", None)]):
            bars(ax, order, r[c], r[f"{c}_se"], [grey, "#8C8C8C", red], lab, lim)
        f_.suptitle("Rest session 1 ↔ session 2 (different days): the map is fit across conditions only", fontsize=16)
        f_.tight_layout()
        f_.savefig(fig / "fig_retest.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        f_, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, (c, lab) in zip(axes, [("avg_rank", "average rank"), ("top1", "top-1")]):
            labels, vals, errs, cols = [], [], [], []
            for rep, col in (("codes", blue), ("embedding", red)):
                for res in (False, True):
                    q = ci[(ci.representation == rep) & (ci.residualised == res)].iloc[0]
                    labels.append(f"{rep}{', confounds removed' if res else ''}")
                    vals.append(q[c]), errs.append(q[f"{c}_se"]), cols.append(col if not res else "white")
            bars(ax, labels, vals, errs, cols, f"cross-condition {lab} (56 pairs)", (0.5, 1) if c == "avg_rank" else None)
        f_.suptitle("Does identification rely on motion / head-size proxies?", fontsize=16)
        f_.tight_layout()
        f_.savefig(fig / "fig_confounds.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        feats = list(bh.features.unique())
        cols = [grey, "#8C8C8C", red, blue, "#0F4D92"][:len(feats)]
        targets = list(bh.target.unique())
        f_, ax = plt.subplots(figsize=(14, 5.5))
        wd = 0.8 / len(feats)
        for j, (ft, c) in enumerate(zip(feats, cols)):
            q = bh[bh.features == ft].set_index("target").loc[targets]
            ax.bar(np.arange(len(targets)) + (j - (len(feats) - 1) / 2) * wd, q.r, wd, yerr=q.r_se, color=c, edgecolor="black",
                   lw=1, capsize=3, label=ft)
        ax.set_xticks(range(len(targets)), targets, fontsize=13)
        ax.set_ylabel("test r (prediction vs score)", fontsize=15)
        ax.axhline(0, color="black", lw=1)
        ax.legend(frameon=False, fontsize=12, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.set_title("Predicting HCP cognition from rest FC vs the fingerprint embedding (ridge)", fontsize=16)
        f_.savefig(fig / "fig_behaviour.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        short = {"SC → condition, ridge (E4 B1)": "ridge → condition", "SC → rest FC (ridge)": "ridge → rest FC",
                 "SC → condition, multiview CCA (E4 B2)": "CCA → condition", "SC → fingerprint (ridge)": "ridge → fingerprint",
                 "SC → fingerprint embedding (ridge)": "ridge → fp embedding"}
        routes = [r_ for r_ in short if r_ in set(vs_rest.route)]
        rc = dict(zip(routes, [grey, "#8C8C8C", "#42949E", blue, "#B2182B"]))
        f_, axes = plt.subplots(2, 3, figsize=(20, 11))
        for i, (t, lab) in enumerate(((vs_rest, "measured rest FC"), (vs_all, "measured FC, mean of 8 conditions"))):
            for ax, (c, ml, lim) in zip(axes[i], [("avg_rank", "average rank", (0.5, None)), ("top1_acc", "top-1", None),
                                                  ("demeaned_pearson", "demeaned r", None)]):
                q = t.set_index("route").loc[routes]
                bars(ax, [short[x] for x in routes], q[c], q[f"{c}_se"], [rc[x] for x in routes], f"test {ml}\nvs {lab}", lim)
        f_.suptitle("SC → FC: which training target makes SC most predictive of measured FC? (all routes start from SC)", fontsize=17)
        f_.tight_layout()
        f_.savefig(fig / "fig_sc_targets.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        f_, ax = plt.subplots(figsize=(8, 5))
        for s, (v, _) in w.items():
            ax.plot(np.arange(1, len(v) + 1), v, lw=1.5, label=f"seed {s}")
        ax.set_xlabel("direction (sorted)", fontsize=15)
        ax.set_ylabel("fingerprint share w", fontsize=15)
        ax.set_title("Per-direction fingerprint share of the pooled map", fontsize=15)
        ax.legend(frameon=False, fontsize=11)
        f_.savefig(fig / "fig_w_spectrum.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)
    print("wrote", fig)


if __name__ == "__main__":
    main()
