"""Spec v4 E6 analysis: tables and figures from results/seed*.json (test split, mean ± SE over seeds).

SC → rest FC routes (run_checks.py check 4; input: a test subject's SC only; output: predicted rest FC edges):
  1 Rest codes              ridge SC -> the subject's rest FC (block-PCA codes); block-PCA decoder
  2 CCA, SC as a view       CCA (MAXVAR) over SC + 8 FC conditions; SC's own encoder into the shared latent; CCA decoder
  3 CCA consensus (E1)      CCA over the 8 FC conditions; target = mean of the subject's 8 encodings; ridge; CCA decoder
  4 Mean fingerprint        target = mean of the subject's 8 conditions' block-PCA codes; ridge; block-PCA decoder
  5 Fingerprint embedding   target = mean fingerprint x pooled map B (reliability-weighted); ridge; block-PCA decoder
Stage 1: pairing with the route's own target (identification, R²); stage 2: predicted rest FC against measured rest FC,
with each route's ceiling (the true target decoded the same way). E4.0c's SC-only cells and route 3 at E1's fixed
latent are kept in the tables.

    python scripts/experiments/fingerprint_embedding/analyze.py
        -> tables/*.csv, tables/summary.md, figures/fig_*.png
"""
import json
import sys

import numpy as np
import pandas as pd

from fe_common import E4, FC, HERE, REPO_ROOT, SEEDS

SE = lambda x: x.std(ddof=1) / np.sqrt(len(x))
P = ["avg_rank", "top1_acc", "demeaned_pearson"]
ROUTES = {  # run_checks target name -> (label, group, colour, explanation)
    "rest codes": ("1  Rest codes", "A", "#8C8C8C", "SC → the subject's rest FC (block-PCA codes), ridge"),
    "CCA latent": ("2  CCA, SC as a view", "B", "#2A7F87", "CCA over SC + 8 FC conditions; SC's own encoder into the latent (no regression)"),
    "CCA consensus": ("3  CCA consensus", "B", "#7CC4C9", "CCA over the 8 FC conditions; target = mean of the subject's 8 encodings (E1), ridge"),
    "fingerprint": ("4  Mean fingerprint", "C", "#7FB2E5", "target = mean of the subject's 8 conditions' block-PCA codes, ridge"),
    "fingerprint embedding": ("5  Fingerprint embedding", "C", "#B2182B", "target = mean fingerprint × pooled map (reliability-weighted), ridge"),
}
EXTRA = {"CCA consensus (E1 setting)": "3b CCA consensus, E1's latent (d 32, λ 0.1)"}
GROUPS = {"A": "single condition", "B": "CCA latents", "C": "block-PCA fingerprints"}
COND = {v: v.capitalize() if v != "wm" else "WM" for v in FC}


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
                out.append(dict(seed=s, route="E4.0c B1+SC (= route 1)" if x["model"] == "B1+SC" else "E4.0c B2 (CCA chosen on FC → FC)",
                                against=x["target"], **{m: x[m] for m in P}))
    return pd.DataFrame(out)


def agg(df, keys, metrics):
    g = df.groupby(keys + ["seed"])[metrics].mean().groupby(keys)
    m, e = g.mean(), g.agg(SE)
    out = m.copy()
    for c in metrics:
        out[f"{c}_se"] = e[c]
    return out.reset_index()


def label(t):
    return ROUTES[t][0] if t in ROUTES else EXTRA.get(t, t)


def main():
    df, w = load()
    test = df[df.split == "test"]
    tab, fig = HERE / "tables", HERE / "figures"
    tab.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)

    rt = agg(test[test.check == "retest"], ["model"], P)
    r2 = agg(test[test.check == "confound_r2"], ["representation", "of"], ["r2"])
    ci = agg(test[test.check == "confound_ident"], ["representation", "residualised"], ["avg_rank", "top1"])
    bh = agg(test[test.check == "behaviour"], ["target", "features"], ["r"])
    sc = test[test.check == "sc_target"].assign(route=lambda d: d.target.map(label))
    ce = test[test.check == "sc_ceiling"].assign(route=lambda d: d.target.map(label))
    own = agg(sc[sc.against == "own target (codes)"], ["route"], ["r2", "avg_rank", "top1_acc"])
    cond = pd.concat([sc[sc.against != "own target (codes)"][["seed", "route", "against"] + P], e40c_sc()])
    per_cond = agg(cond, ["route", "against"], P)
    vs_rest = per_cond[per_cond.against == "rest"].drop(columns="against")
    vs_all = agg(cond.groupby(["seed", "route"])[P].mean().reset_index(), ["route"], P)
    ceil = agg(ce, ["route", "against"], P)
    ceil_rest = ceil[ceil.against == "rest"].drop(columns="against")
    for name, t in (("retest", rt), ("confounds_r2", r2), ("confounds_ident", ci), ("behaviour", bh), ("sc_routes_stage1_own_target", own),
                    ("sc_routes_stage2_vs_rest", vs_rest), ("sc_routes_vs_each_condition", per_cond), ("sc_routes_vs_mean_of_8", vs_all),
                    ("sc_routes_ceilings", ceil)):
        t.to_csv(tab / f"{name}.csv", index=False, float_format="%.5g")
    for old in ("sc_targets_own", "sc_targets_vs_rest", "sc_targets_vs_all8"):
        (tab / f"{old}.csv").unlink(missing_ok=True)
    summary(rt, r2, ci, bh, own, vs_rest, vs_all, ceil_rest, w, tab)
    figures(rt, r2, ci, bh, own, vs_rest, per_cond, ceil_rest, w, fig)


def summary(rt, r2, ci, bh, own, vs_rest, vs_all, ceil_rest, w, tab):
    f4 = lambda m, e: f"{m:.3f} ± {e:.3f}"
    md = ["# E6 — fingerprint embedding checks (test, seeds 0–4, mean ± SE)", ""]
    cf = [v[1] for v in w.values()]
    md += [f"Closed form B = (8/7)W − I/7 vs the fitted map: max |diff| {max(cf):.1e}. Directions with fingerprint share w > 0.5: "
           + ", ".join(f"seed {s} {int((v[0] > 0.5).sum())}" for s, v in w.items()) + f" of {len(next(iter(w.values()))[0])}.", ""]
    md += ["## SC → rest FC routes", "", "| route | group | what SC is trained to predict |", "|---|---|---|"]
    md += [f"| {lab} | {GROUPS[g]} | {ex} |" for lab, g, _, ex in ROUTES.values()]
    md += ["", "Stage 1, pairing with the route's own target (not comparable across routes):", "",
           "| route | R² | average rank | top-1 |", "|---|---|---|---|"]
    md += [f"| {r.route} | {f4(r.r2, r.r2_se)} | {f4(r.avg_rank, r.avg_rank_se)} | {f4(r.top1_acc, r.top1_acc_se)} |" for r in own.itertuples()]
    c = ceil_rest.set_index("route")
    md += ["", "Stage 2, predicted vs measured rest FC (ceiling: the true target decoded the same way):", "",
           "| route | average rank | top-1 | demeaned r | ceiling average rank | ceiling demeaned r |", "|---|---|---|---|---|---|"]
    for r in vs_rest.itertuples():
        cl = (f"{c.loc[r.route, 'avg_rank']:.3f} | {c.loc[r.route, 'demeaned_pearson']:.3f}" if r.route in c.index else "– | –")
        md.append(f"| {r.route} | " + " | ".join(f4(getattr(r, m), getattr(r, f'{m}_se')) for m in P) + f" | {cl} |")
    md += ["", "Against measured FC, mean over the 8 conditions:", "", "| route | average rank | top-1 | demeaned r |", "|---|---|---|---|"]
    md += [f"| {r.route} | " + " | ".join(f4(getattr(r, m), getattr(r, f'{m}_se')) for m in P) + " |" for r in vs_all.itertuples()]
    md += ["", "## Rest session 1 ↔ session 2 (different days)", "", "| prediction | average rank | top-1 | demeaned r |", "|---|---|---|---|"]
    md += [f"| {r.model} | " + " | ".join(f4(getattr(r, m), getattr(r, f'{m}_se')) for m in P) + " |" for r in rt.itertuples()]
    md += ["", "## Confound proxies", "", "| representation | of | R² explained by the proxies |", "|---|---|---|"]
    md += [f"| {r.representation} | {r.of} | {f4(r.r2, r.r2_se)} |" for r in r2.itertuples()]
    md += ["", "| representation | proxies removed | cross-condition average rank | top-1 |", "|---|---|---|---|"]
    md += [f"| {r.representation} | {r.residualised} | {f4(r.avg_rank, r.avg_rank_se)} | {f4(r.top1, r.top1_se)} |" for r in ci.itertuples()]
    md += ["", "## Behaviour (test r)", "", "| target | " + " | ".join(bh.features.unique()) + " |", "|---|" + "---|" * bh.features.nunique()]
    for t, g in bh.groupby("target", sort=False):
        g = g.set_index("features")
        md.append(f"| {t} | " + " | ".join(f4(g.loc[f, 'r'], g.loc[f, 'r_se']) for f in bh.features.unique()) + " |")
    (tab / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def figures(rt, r2, ci, bh, own, vs_rest, per_cond, ceil_rest, w, fig):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch
    import run as mb  # noqa: E402

    routes = [v[0] for v in ROUTES.values()]
    col = {v[0]: v[2] for v in ROUTES.values()}
    common = ("All routes: input = a test subject's SC only; output = predicted rest FC edges. Task FC enters only through the "
              "training targets (train subjects). Settings chosen on val average rank against measured rest FC.")

    def bars(ax, names, vals, errs, colours, ylabel, ylim=None, ticks=None):
        x = np.arange(len(names))
        ax.bar(x, vals, 0.72, yerr=errs, color=colours, edgecolor="black", lw=1, capsize=3)
        ax.set_xticks(x, ticks or names, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=14)
        if ylim:
            ax.set_ylim(*ylim)
        return x

    short = lambda r_: r_.split("  ")[0]        # route number on the axis; the key explains it
    with plt.rc_context({**mb.FIG_RC, "font.size": 13}):
        # 1. schematic
        f_, ax = plt.subplots(figsize=(17, 8.5))
        ax.set_xlim(0, 17)
        ax.set_ylim(-0.8, 6.4)
        ax.axis("off")

        def box(x, y, wd, txt, fc):
            ax.add_patch(FancyBboxPatch((x, y - 0.36), wd, 0.72, boxstyle="round,pad=0.05", fc=fc, ec="black", lw=1.2))
            ax.text(x + wd / 2, y, txt, ha="center", va="center", fontsize=12)

        def arrow(x0, x1, y0, y1):
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))

        spaces = {"rest codes": ("ridge", "rest FC codes (block PCA)", "block-PCA decoder"),
                  "CCA latent": ("CCA encoder of SC", "shared CCA latent (SC + 8 FC)", "CCA rest decoder"),
                  "CCA consensus": ("ridge", "consensus: mean of 8 CCA encodings", "CCA rest decoder"),
                  "fingerprint": ("ridge", "mean of 8 conditions (block PCA)", "block-PCA decoder"),
                  "fingerprint embedding": ("ridge", "mean of 8 × pooled map", "block-PCA decoder")}
        box(0.1, 3, 1.8, "test subject's\nSC", "#EEEEEE")
        box(14.7, 3, 2.2, "predicted\nrest FC edges", "#EEEEEE")
        for i, (t, (lab, g, c, _)) in enumerate(ROUTES.items()):
            y = 5.6 - i * 1.3
            how, space, dec = spaces[t]
            ax.text(2.6, y + 0.48, f"{lab}   ({GROUPS[g]})", fontsize=12, fontweight="bold", color=c)
            arrow(1.95, 2.55, 3, y)
            box(2.6, y, 2.2, how, "white")
            arrow(4.85, 5.35, y, y)
            box(5.4, y, 4.6, space, c + "40")
            arrow(10.05, 10.55, y, y)
            box(10.6, y, 3.2, dec, "white")
            arrow(13.85, 14.65, y, 3)
        ax.text(8.5, -0.65, "Target spaces of routes 2–5 are built from the train subjects' rest + task FC (training only); "
                "at test time every route sees SC alone.", ha="center", fontsize=12, style="italic")
        ax.set_title("E6: five routes from SC to rest FC", fontsize=17)
        f_.savefig(fig / "fig_routes_schematic.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        # 2. main: stage 1 and stage 2
        f_ = plt.figure(figsize=(21, 13))
        gs = f_.add_gridspec(2, 3, hspace=0.55, wspace=0.3)
        o = own.set_index("route").reindex(routes)
        for j, (c, lab, lim) in enumerate([("avg_rank", "average rank (chance 0.5)", (0.5, 1)), ("top1_acc", "top-1", None)]):
            ax = f_.add_subplot(gs[0, j])
            bars(ax, routes, o[c], o[f"{c}_se"], [col[r_] for r_ in routes], f"SC's prediction vs own target:\n{lab}", lim,
                 [short(r_) for r_ in routes])
            ax.set_xlabel("route", fontsize=13)
        f_.text(0.5, 0.925, "Stage 1 — how strongly SC pairs with each route's own target (identification among test subjects; "
                "each route has its own target, so not a fair ranking)", ha="center", fontsize=14, fontweight="bold")
        ax = f_.add_subplot(gs[0, 2])
        ax.axis("off")
        ax.legend(handles=[Patch(fc=col[r_], ec="black", label=r_) for r_ in routes], loc="upper left", frameon=False, fontsize=13,
                  title="routes", title_fontsize=13)
        ax.text(0, 0.36, "\n".join(f"{lab.split('  ')[0]}: {ex}" for lab, _, _, ex in ROUTES.values()), fontsize=10, va="top",
                transform=ax.transAxes)
        v = vs_rest.set_index("route").reindex(routes)
        cl = ceil_rest.set_index("route").reindex(routes)
        for j, (c, lab, lim) in enumerate([("avg_rank", "average rank (chance 0.5)", (0.5, 1)), ("top1_acc", "top-1", None),
                                           ("demeaned_pearson", "demeaned r", None)]):
            ax = f_.add_subplot(gs[1, j])
            x = bars(ax, routes, v[c], v[f"{c}_se"], [col[r_] for r_ in routes], f"predicted vs measured rest FC:\n{lab}", lim,
                     [short(r_) for r_ in routes])
            ax.hlines(cl[c], x - 0.36, x + 0.36, color="black", lw=2, ls="--", label="ceiling: true target decoded")
            ax.set_xlabel("route", fontsize=13)
            if c == "top1_acc":
                ax.set_yscale("log")
                ax.set_ylabel(f"predicted vs measured rest FC:\n{lab} (log scale)", fontsize=14)
            if j == 2:
                ax.legend(frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(1, 1.15))
        f_.text(0.5, 0.47, "Stage 2 — the common readout: decoded rest FC against the test subjects' measured rest FC",
                ha="center", fontsize=14, fontweight="bold")
        f_.text(0.5, 0.03, common, ha="center", fontsize=11.5, style="italic")
        f_.suptitle("SC → rest FC through five target spaces (test, 5 seeds)", fontsize=18, y=0.97)
        f_.savefig(fig / "fig_sc_routes.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        # 3. supplement: against each condition
        f_, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
        wd = 0.8 / len(routes)
        for ax, (c, lab) in zip(axes, [("avg_rank", "average rank"), ("top1_acc", "top-1"), ("demeaned_pearson", "demeaned r")]):
            for j, r_ in enumerate(routes):
                q = per_cond[per_cond.route == r_].set_index("against").reindex(FC)
                ax.bar(np.arange(len(FC)) + (j - (len(routes) - 1) / 2) * wd, q[c], wd, yerr=q[f"{c}_se"], color=col[r_],
                       edgecolor="black", lw=0.8, capsize=2, label=r_)
            ax.set_ylabel(f"test {lab}", fontsize=14)
            if c == "avg_rank":
                ax.set_ylim(0.5, None)
        axes[-1].set_xticks(range(len(FC)), [COND[v] for v in FC], fontsize=13)
        axes[0].legend(frameon=False, fontsize=12, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
        f_.suptitle("Supplement: the same SC predictions decoded to each condition and scored against its measured FC", fontsize=16, y=1.04)
        f_.text(0.5, -0.01, common, ha="center", fontsize=11.5, style="italic")
        f_.tight_layout()
        f_.savefig(fig / "fig_sc_routes_conditions.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        # 4. retest
        names = {"copy (edges)": "copy the session's\nFC edges", "copy (codes)": "copy the session's\nFC codes (256)",
                 "embedding": "the session's codes\n× pooled map"}
        order = list(names)
        r = rt.set_index("model").loc[order]
        f_, axes = plt.subplots(1, 3, figsize=(17, 5.8))
        for ax, (c, lab, lim) in zip(axes, [("avg_rank", "average rank", (0.9, 1)), ("top1_acc", "top-1", None),
                                             ("demeaned_pearson", "demeaned r", None)]):
            bars(ax, order, r[c], r[f"{c}_se"], ["#CFCECE", "#8C8C8C", "#B2182B"], f"test {lab}", lim, [names[o_] for o_ in order])
        f_.suptitle("Does the embedding transfer across days? Predict rest session 2 FC from session 1 (and back)", fontsize=16)
        f_.text(0.5, -0.06, "The pooled map is fit on train subjects across conditions (combined rest + 7 tasks); the separate sessions are never "
                "used to fit it.", ha="center", fontsize=11.5, style="italic")
        f_.tight_layout()
        f_.savefig(fig / "fig_retest.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        # 5. confounds
        f_, axes = plt.subplots(1, 3, figsize=(19, 6.2))
        rn = {("codes", "rest"): "rest codes", ("embedding", "rest"): "rest codes ×\npooled map",
              ("codes", "fingerprint"): "mean fingerprint", ("embedding", "fingerprint"): "mean fingerprint ×\npooled map"}
        q = r2.set_index(["representation", "of"]).loc[list(rn)]
        bars(axes[0], list(rn.values()), q.r2, q.r2_se, ["#8C8C8C", "#B2182B", "#7FB2E5", "#0F4D92"], "test R² explained by the proxies")
        axes[0].set_title("How much of each representation\nthe proxies explain", fontsize=14)
        for ax, (c, lab, lim) in zip(axes[1:], [("avg_rank", "average rank", (0.8, 1)), ("top1", "top-1", None)]):
            labs, vals, errs, cols = [], [], [], []
            for rep, cc in (("codes", "#8C8C8C"), ("embedding", "#B2182B")):
                for res in (False, True):
                    g = ci[(ci.representation == rep) & (ci.residualised == res)].iloc[0]
                    labs.append(("codes" if rep == "codes" else "codes ×\npooled map") + ("\nproxies removed" if res else ""))
                    vals.append(g[c])
                    errs.append(g[f"{c}_se"])
                    cols.append(cc if not res else cc + "66")
            bars(ax, labs, vals, errs, cols, f"cross-condition {lab}\n(56 condition pairs)", lim)
            ax.set_title("Identification with and without the proxies", fontsize=14)
        f_.suptitle("Is the embedding's identifiability driven by motion or head size?", fontsize=16)
        f_.text(0.5, -0.06, "Proxies (no framewise displacement in our data): DVARS and global-signal SD of the rest timeseries, mean rest FC, "
                "intracranial volume, total parcel volume.", ha="center", fontsize=11.5, style="italic")
        f_.tight_layout()
        f_.savefig(fig / "fig_confounds.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        # 6. behaviour
        feat = {"rest edges": "rest FC edges", "rest codes": "rest codes (256)", "rest embedding": "rest codes × pooled map",
                "fingerprint": "mean fingerprint", "fingerprint embedding": "mean fingerprint × pooled map"}
        fc_ = ["#CFCECE", "#8C8C8C", "#B2182B", "#7FB2E5", "#0F4D92"]
        tnames = {"CogTotalComp_AgeAdj": "total cognition", "CogFluidComp_AgeAdj": "fluid cognition",
                  "CogCrystalComp_AgeAdj": "crystallised cognition", "PMAT24_A_CR": "PMAT (fluid reasoning)"}
        f_, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
        for ax, (t, tl) in zip(axes, tnames.items()):
            q = bh[bh.target == t].set_index("features").loc[list(feat)]
            bars(ax, list(feat), q.r, q.r_se, fc_, "test r (prediction vs score)" if t == "CogTotalComp_AgeAdj" else "", None,
                 ["1", "2", "3", "4", "5"])
            ax.set_title(tl, fontsize=14)
        axes[-1].legend(handles=[Patch(fc=c, ec="black", label=f"{i + 1}  {l}") for i, (l, c) in enumerate(zip(feat.values(), fc_))],
                        frameon=False, fontsize=12, loc="center left", bbox_to_anchor=(1.02, 0.5), title="features (ridge)")
        f_.suptitle("Does the embedding predict cognition better than rest FC? (HCP, age-adjusted composites)", fontsize=16)
        f_.tight_layout()
        f_.savefig(fig / "fig_behaviour.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)

        # 7. w spectrum
        f_, ax = plt.subplots(figsize=(9, 5.5))
        for s, (v, _) in w.items():
            ax.plot(np.arange(1, len(v) + 1), v, lw=1.5, label=f"seed {s}")
        ax.axhline(0.5, color="black", ls="--", lw=1)
        n = int(np.mean([(v[0] > 0.5).sum() for v in w.values()]))
        ax.annotate(f"≈ {n} of {len(next(iter(w.values()))[0])} directions\nmostly fingerprint (w > 0.5)", (n, 0.5), (n + 40, 0.75),
                    arrowprops=dict(arrowstyle="->"), fontsize=12)
        ax.set_xlabel("code direction (sorted by w)", fontsize=14)
        ax.set_ylabel("fingerprint share w", fontsize=14)
        ax.set_title("How much of each direction the pooled map keeps\n(w = fingerprint share of the direction's variance)", fontsize=14)
        ax.legend(frameon=False, fontsize=11)
        f_.savefig(fig / "fig_w_spectrum.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f_)
    (fig / "fig_sc_targets.png").unlink(missing_ok=True)
    print("wrote", fig)


if __name__ == "__main__":
    main()
