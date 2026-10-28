"""windowed_fc_gain G7 analysis: does training on part samples help? (closed form + flows, both routes, 5 seeds)

Per model, route, arm and seed (test split): mean over the 8 conditions of decoded demeaned r, average rank and top-1
(point), and the distribution metrics (energy distance, set identification, spread ratio). Paired by seed against the
same model's `none` arm and against the route's linear baselines (ridge for point metrics, linear-Gaussian `own` for
distribution metrics, both `none`). Support rule (E2.1): seed-mean difference above 2 SE and the same sign in >= 4 of 5
seeds. G7 passes if an augmented arm of some model has a supported gain over both its own `none` and the linear
baseline on a primary metric (demeaned r, average rank, energy) without a supported loss on the other two.
Regression check: each flow's `none` arm vs its E2.1 record.

    python scripts/experiments/windowed_fc_gain/analyze_g7.py     # -> tables/g7_*.csv, tables/g7_summary.md,
        figures/fig_g7_deltas.png (vs full targets only), fig_g7_absolute.png, fig_g7_conditions.png (per condition)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
RES, TAB, FIG = HERE / "results" / "g7", HERE / "tables", HERE / "figures"
E21 = REPO_ROOT / "scripts" / "experiments" / "joint_flow_matching" / "results" / "Glasser" / "records" / "e21"
POINT = ["demeaned_pearson", "avg_rank", "top1_acc"]
DIST = ["energy", "set_avg_rank", "set_top1", "spread_ratio"]
PRIMARY = {"demeaned_pearson": 1, "avg_rank": 1, "energy": -1}       # sign: +1 higher is better
ARMS = ["none", "aug_halves", "token_halves", "parts_halves", "aug_quarters", "token_quarters", "parts_quarters"]
E21_NAME = {("flow", "cca"): "flow_seed{s}.json", ("flow_linear", "cca"): "flow_linear_seed{s}_long.json",
            ("flow_free", "cca"): "flow_free_seed{s}_long.json", ("flow_linear", "pca"): "flow_linear_seed{s}_pca_longT025.json",
            ("flow_free", "pca"): "flow_free_seed{s}_pca_longT025.json"}


def point_means(rows, split="test"):
    d = pd.DataFrame(rows)
    d = d[d.split == split]
    return {k: float(d[k].mean()) for k in POINT}


def collect():
    recs = []
    for f in sorted(RES.glob("closed_seed*.json")):
        r = json.loads(f.read_text())
        rows = pd.DataFrame(r["rows"])
        dist = pd.DataFrame(r["dist"])
        for (route, arm), g in rows[rows.split == "test"].groupby(["route", "arm"]):
            pm = {k: float(g[k].mean()) for k in POINT}
            recs.append(dict(model="ridge", route=route, arm=arm, seed=r["seed"], **pm))
            for name, h in dist[(dist.route == route) & (dist.arm == arm) & (dist.split == "test")].groupby("model"):
                recs.append(dict(model=name, route=route, arm=arm, seed=r["seed"], **pm,
                                 **{k: float(h[k].iloc[0]) for k in DIST}))
    for f in sorted(RES.glob("flow*_seed*.json")):
        r = json.loads(f.read_text())
        for arm, a in r["arms"].items():
            recs.append(dict(model=r["model"], route=r["route"], arm=arm, seed=r["seed"], **point_means(a["rows"]),
                             **{k: a["dist"]["test"][k] for k in DIST}, epochs=a["epochs_run"], best_epoch=a["best_epoch"],
                             seconds=a["seconds"], n_units=a["n_units"]))
    return pd.DataFrame(recs)


def paired(df, ref_of):
    """Seed-paired differences of every (model, route, arm) against ref_of(model, route) -> (model, arm) reference."""
    out = []
    for (model, route, arm), g in df.groupby(["model", "route", "arm"]):
        rm, ra = ref_of(model, route)
        if (rm, ra) == (model, arm):
            continue
        ref = df[(df.model == rm) & (df.route == route) & (df.arm == ra)].set_index("seed")
        g = g.set_index("seed")
        seeds = g.index.intersection(ref.index)
        for k in POINT + DIST:
            if k not in g or k not in ref or g[k].isna().all() or ref[k].isna().all():
                continue
            dlt = (g.loc[seeds, k] - ref.loc[seeds, k]).to_numpy(dtype=float)
            if len(dlt) < 2:
                continue
            m, se = dlt.mean(), dlt.std(ddof=1) / np.sqrt(len(dlt))
            out.append(dict(model=model, route=route, arm=arm, ref=f"{rm}/{ra}", metric=k, delta=m, se=se, n=len(dlt),
                            n_pos=int((dlt > 0).sum()), supported=bool(abs(m) > 2 * se and max((dlt > 0).sum(), (dlt < 0).sum()) >= 4)))
    return pd.DataFrame(out)


def regression_check(df):
    rows = []
    for (model, route), name in E21_NAME.items():
        for s in range(5):
            f = E21 / name.format(s=s)
            g = df[(df.model == model) & (df.route == route) & (df.arm == "none") & (df.seed == s)]
            if not f.exists() or g.empty:
                continue
            r = json.loads(f.read_text())
            e = point_means([x for x in r["rows"] if x["model"] == model])
            rows.append(dict(model=model, route=route, seed=s, e21_record=f.name,
                             dr_e21=e["demeaned_pearson"], dr_g7=float(g.demeaned_pearson.iloc[0]),
                             energy_e21=r["dist"]["test"]["energy"], energy_g7=float(g.energy.iloc[0])))
    return pd.DataFrame(rows)


ARM_LABEL = {"none": "full targets only (= E2.1)",
             "aug_halves": "full + 2 halves", "token_halves": "full + 2 halves, sample-type token", "parts_halves": "2 halves only",
             "aug_quarters": "full + 4 quarters", "token_quarters": "full + 4 quarters, sample-type token",
             "parts_quarters": "4 quarters only"}
MODEL_LABEL = {"ridge": "ridge", "gauss_own": "linear-Gaussian\n(pooled cov.)", "gauss_fullcov": "linear-Gaussian\n(full-target cov.)",
               "flow": "residual flow\n(ridge-anchored)", "flow_linear": "linear flow\n(from noise)",
               "flow_free": "nonlinear flow\n(from noise)"}
ROUTE_LABEL = {"cca": "CCA latent (32-d)", "pca": "block FC PCA codes (256)"}
ROUTE_SHORT = {"cca": "CCA", "pca": "block PCA"}
CONDITIONS = ["rest", "emotion", "gambling", "language", "motor", "relational", "social", "wm"]


def arm_style(arm):
    """Colour by part level (halves blue, quarters red), variant by fill: solid = full + parts, hatched = with the
    sample-type token, light = parts only; full targets only grey."""
    if arm == "none":
        return dict(facecolor="#BDBDBD", edgecolor="#6E6E6E")
    base = "#3775BA" if arm.endswith("halves") else "#B64342"
    if arm.startswith("parts"):
        return dict(facecolor=base, alpha=0.4, edgecolor=base)
    if arm.startswith("token"):
        return dict(facecolor="white", edgecolor=base, hatch="////")
    return dict(facecolor=base, edgecolor=base)


def arm_legend(ax, arms, **kw):
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(label=ARM_LABEL[a], linewidth=1.2, **arm_style(a)) for a in arms]
    return ax.legend(handles=handles, **kw)


def collect_conditions():
    """Per (model, route, arm, seed, condition) test demeaned r."""
    out = []
    for f in sorted(RES.glob("closed_seed*.json")):
        r = json.loads(f.read_text())
        out += [dict(model="ridge", route=x["route"], arm=x["arm"], seed=r["seed"], condition=x["condition"],
                     demeaned_pearson=x["demeaned_pearson"]) for x in r["rows"] if x["split"] == "test"]
    for f in sorted(RES.glob("flow*_seed*.json")):
        r = json.loads(f.read_text())
        for arm, a in r["arms"].items():
            out += [dict(model=r["model"], route=r["route"], arm=arm, seed=r["seed"], condition=x["condition"],
                         demeaned_pearson=x["demeaned_pearson"]) for x in a["rows"] if x["split"] == "test"]
    return pd.DataFrame(out)


def figures(df, own, plt, rc):
    arms = [a for a in ARMS if a != "none"]
    with plt.rc_context(rc):
        # 1. change vs none
        fams = [(m, r) for r in ("cca", "pca") for m in ("ridge", "gauss_own", "gauss_fullcov", "flow", "flow_linear", "flow_free")
                if ((own.model == m) & (own.route == r)).any()]
        mets = [("demeaned_pearson", "Δ demeaned r\n(higher = better)"), ("avg_rank", "Δ average rank\n(higher = better)"),
                ("energy", "Δ energy distance\n(lower = better)")]
        fig, axes = plt.subplots(len(mets), 1, figsize=(max(12, 1.45 * len(fams)), 3.5 * len(mets)), sharex=True)
        w = 0.8 / len(arms)
        for ax, (k, lab) in zip(axes, mets):
            for j, arm in enumerate(arms):
                for i, (m, r) in enumerate(fams):
                    o = own[(own.model == m) & (own.route == r) & (own.arm == arm) & (own.metric == k)]
                    if len(o):
                        ax.bar(i + (j - (len(arms) - 1) / 2) * w, o.delta.iloc[0], w, yerr=o.se.iloc[0], capsize=2,
                               linewidth=1.0, error_kw=dict(lw=1), **arm_style(arm))
            ax.axhline(0, color="0.3", lw=1)
            ax.set_ylabel(lab, fontsize=12)
            for i in range(1, len(fams)):
                if fams[i][1] != fams[i - 1][1]:
                    ax.axvline(i - 0.5, color="0.6", lw=1, ls=":")
        axes[-1].set_xticks(range(len(fams)), [f"{MODEL_LABEL[m]}\n[{ROUTE_SHORT[r]}]" for m, r in fams], fontsize=9.5)
        arm_legend(axes[0], arms, ncol=2, fontsize=10.5, loc="lower left", bbox_to_anchor=(0, 1.02),
                   title="Training targets per subject (all scored on full val / test targets)", title_fontsize=11)
        fig.suptitle("G7: change vs the same model trained on full targets only (test, mean ± SE over 5 seeds)", fontsize=13, y=1.0)
        fig.tight_layout()
        fig.savefig(FIG / "fig_g7_deltas.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # 2. absolute values, linear baseline as reference
        panels = [("demeaned_pearson", "test demeaned r\n(mean of 8 conditions)", ["ridge", "flow", "flow_linear", "flow_free"], "ridge"),
                  ("avg_rank", "test average rank\n(chance 0.5)", ["ridge", "flow", "flow_linear", "flow_free"], "ridge"),
                  ("energy", "energy distance to the\nsubject's 8 encodings (lower = better)",
                   ["gauss_own", "gauss_fullcov", "flow", "flow_linear", "flow_free"], "gauss_own")]
        fig, axes = plt.subplots(2, 3, figsize=(22, 10.5))
        w = 0.85 / len(ARMS)
        for ri, route in enumerate(("cca", "pca")):
            for ci, (k, lab, models, ref) in enumerate(panels):
                ax = axes[ri, ci]
                models = [m for m in models if ((df.model == m) & (df.route == route)).any()]
                vals = []
                for i, m in enumerate(models):
                    for j, arm in enumerate(ARMS):
                        g = df[(df.model == m) & (df.route == route) & (df.arm == arm)][k].dropna()
                        if g.empty:
                            continue
                        x = i + (j - (len(ARMS) - 1) / 2) * w
                        ax.bar(x, g.mean(), w, linewidth=1.0, **arm_style(arm))
                        ax.scatter(np.full(len(g), x), g, s=6, color="black", zorder=3)
                        vals += list(g)
                b = df[(df.model == ref) & (df.route == route) & (df.arm == "none")][k].mean()
                ax.axhline(b, color="black", ls="--", lw=1.2, label=f"linear baseline ({MODEL_LABEL[ref].split(chr(10))[0]}, full targets)")
                lo, hi = min(vals + [b]), max(vals + [b])
                pad = (hi - lo) * 0.12 or 0.01
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_xticks(range(len(models)), [MODEL_LABEL[m] for m in models], fontsize=10)
                ax.set_ylabel(lab, fontsize=11)
                if ci == 0:
                    ax.set_title(ROUTE_LABEL[route], loc="left", fontsize=13, fontweight="bold")
                ax.legend(fontsize=9, loc="best")
        arm_legend(axes[0, 0], ARMS, ncol=4, fontsize=10.5, loc="lower left", bbox_to_anchor=(0, 1.12),
                   title="Training targets per subject (dots: seeds; y-axes zoomed)", title_fontsize=11)
        axes[0, 0].add_artist(axes[0, 0].get_legend())
        fig.suptitle("G7: SC → FC latent models trained with part samples, absolute test values", fontsize=14, y=1.06)
        fig.tight_layout()
        fig.savefig(FIG / "fig_g7_absolute.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # 3. per-condition change in demeaned r (heatmap)
        cd = collect_conditions()
        if cd.empty:
            return
        base = cd[cd.arm == "none"].set_index(["model", "route", "seed", "condition"]).demeaned_pearson
        cd = cd[cd.arm != "none"].copy()
        cd["delta"] = cd.demeaned_pearson.to_numpy() - base.reindex(pd.MultiIndex.from_frame(cd[["model", "route", "seed", "condition"]])).to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(17, 12), sharey=True, gridspec_kw=dict(wspace=0.04))
        vmax = float(np.nanpercentile(np.abs(cd.groupby(["model", "route", "arm", "condition"]).delta.mean()), 98))
        keys = [(m, arm) for m in ("ridge", "flow", "flow_linear", "flow_free") for arm in arms]
        labels = [f"{MODEL_LABEL[m].replace(chr(10), ' ')} · {ARM_LABEL[arm]}" for m, arm in keys]
        for ax, route in zip(axes, ("cca", "pca")):
            M = np.full((len(keys), len(CONDITIONS)), np.nan)
            for i, (m, arm) in enumerate(keys):
                g = cd[(cd.model == m) & (cd.route == route) & (cd.arm == arm)]
                if len(g):
                    M[i] = g.groupby("condition").delta.mean().reindex(CONDITIONS).to_numpy()
            ax.imshow(np.isnan(M)[..., None] * np.array([0.92, 0.92, 0.92]) + ~np.isnan(M)[..., None], aspect="auto")
            im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
            for (i, j), v in np.ndenumerate(M):
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v * 1000:+.1f}", ha="center", va="center", fontsize=7.5,
                        color="0.5" if np.isnan(v) else ("white" if abs(v) > 0.6 * vmax else "black"))
            ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=30, ha="right", fontsize=11)
            for i in range(1, len(keys)):
                if keys[i][0] != keys[i - 1][0]:
                    ax.axhline(i - 0.5, color="black", lw=1.2)
            ax.set_title(ROUTE_LABEL[route], fontsize=13, fontweight="bold")
        axes[0].set_yticks(range(len(labels)), labels, fontsize=9)
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.015)
        cb.set_label("Δ test demeaned r vs full targets only (blue = better); cells × 1000", fontsize=11)
        fig.suptitle("G7: per-condition change in demeaned r (mean over 5 seeds)", fontsize=14)
        fig.savefig(FIG / "fig_g7_conditions.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402  (FIG_RC)

    df = collect()
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df.to_csv(TAB / "g7_seed_records.csv", index=False, float_format="%.6g")
    agg = df.groupby(["model", "route", "arm"])[POINT + DIST].agg(["mean", "sem"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["arm"] = pd.Categorical(agg.arm, ARMS)
    agg = agg.sort_values(["route", "model", "arm"])
    agg.to_csv(TAB / "g7_arms.csv", index=False, float_format="%.5g")
    own = paired(df, lambda m, r: (m, "none"))
    lin = paired(df, lambda m, r: ("ridge", "none"))
    lin_d = paired(df, lambda m, r: ("gauss_own", "none"))
    own.to_csv(TAB / "g7_paired_vs_none.csv", index=False, float_format="%.5g")
    reg = regression_check(df)
    reg.to_csv(TAB / "g7_regression_check.csv", index=False, float_format="%.6g")

    # G7 verdict per (model, route, arm)
    verdict = []
    for (model, route, arm), g in own.groupby(["model", "route", "arm"]):
        if arm == "none":
            continue
        row = dict(model=model, route=route, arm=arm)
        gains, losses = [], []
        for k, sign in PRIMARY.items():
            o = g[g.metric == k]
            ref = (lin if k in POINT else lin_d)
            b = ref[(ref.model == model) & (ref.route == route) & (ref.arm == arm) & (ref.metric == k)]
            if o.empty:
                continue
            od, os_ = float(o.delta.iloc[0]) * sign, bool(o.supported.iloc[0])
            bd = float(b.delta.iloc[0]) * sign if len(b) else np.nan
            bs = bool(b.supported.iloc[0]) if len(b) else False
            row[f"{k} vs none"] = od                     # sign-adjusted: + = better
            row[f"{k} vs linear"] = bd
            if od > 0 and os_ and (bd > 0 and bs):
                gains.append(k)
            if od < 0 and os_:
                losses.append(k)
        row["supported gains (vs none and linear)"] = ", ".join(gains)
        row["supported losses vs none"] = ", ".join(losses)
        row["G7"] = bool(gains) and not losses
        verdict.append(row)
    verdict = pd.DataFrame(verdict)
    verdict.to_csv(TAB / "g7_verdicts.csv", index=False, float_format="%.5g")

    f = lambda x: "–" if pd.isna(x) else f"{x:.4f}"
    md = ["# windowed_fc_gain G7 — does training on part samples help?", "",
          "Test split, mean over the 8 conditions, mean ± SE over seeds (n = seeds). Energy: lower is better.", "",
          "| route | model | arm | n | demeaned r | avg rank | top-1 | energy | set avg rank | spread ratio |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    n = df.groupby(["model", "route", "arm"]).seed.nunique()
    for r in agg.itertuples(index=False):
        d = r._asdict()
        md.append(f"| {r.route} | {r.model} | {r.arm} | {n.loc[(r.model, r.route, r.arm)]} | "
                  + " | ".join(f"{f(d[f'{k}_mean'])} ± {f(d[f'{k}_sem'])}" if not pd.isna(d[f'{k}_mean']) else "–"
                               for k in ["demeaned_pearson", "avg_rank", "top1_acc", "energy", "set_avg_rank", "spread_ratio"]) + " |")
    md += ["", "## G7 verdicts (supported = seed-mean beyond 2 SE, same sign in ≥ 4 / 5 seeds; signs: + = better)", "",
           "| " + " | ".join(verdict.columns) + " |", "|" + "---|" * len(verdict.columns)]
    md += ["| " + " | ".join(f(x) if isinstance(x, float) else str(x) for x in row) + " |" for row in verdict.itertuples(index=False)]
    if len(reg):
        rg = reg.groupby(["model", "route"])[["dr_e21", "dr_g7", "energy_e21", "energy_g7"]].mean().reset_index()
        md += ["", "## Regression check: `none` arm vs E2.1 records (test, mean over seeds)", "",
               "| model | route | demeaned r E2.1 | demeaned r G7 | energy E2.1 | energy G7 |", "|---|---|---|---|---|---|"]
        md += [f"| {r.model} | {r.route} | {r.dr_e21:.4f} | {r.dr_g7:.4f} | {r.energy_e21:.4f} | {r.energy_g7:.4f} |" for r in rg.itertuples()]
    (TAB / "g7_summary.md").write_text("\n".join(md) + "\n")

    figures(df, own, plt, mb.FIG_RC)
    print((TAB / "g7_summary.md").read_text())


if __name__ == "__main__":
    main()
