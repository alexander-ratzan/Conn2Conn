"""windowed_fc_gain G7 analysis: does training on part samples help? (closed form + flows, both routes, 5 seeds)

Per model, route, arm and seed (test split): mean over the 8 conditions of decoded demeaned r, average rank and top-1
(point), and the distribution metrics (energy distance, set identification, spread ratio). Paired by seed against the
same model's `none` arm and against the route's linear baselines (ridge for point metrics, linear-Gaussian `own` for
distribution metrics, both `none`). Support rule (E2.1): seed-mean difference above 2 SE and the same sign in >= 4 of 5
seeds. G7 passes if an augmented arm of some model has a supported gain over both its own `none` and the linear
baseline on a primary metric (demeaned r, average rank, energy) without a supported loss on the other two.
Regression check: each flow's `none` arm vs its E2.1 record.

    python scripts/experiments/windowed_fc_gain/analyze_g7.py     # -> tables/g7_*.csv, tables/g7_summary.md, figures/fig_g7_*.png
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
            row[f"{k} vs none"] = od * sign
            row[f"{k} vs linear"] = bd * sign if np.isfinite(bd) else np.nan
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

    with plt.rc_context(mb.FIG_RC):
        fams = [(m, r) for r in ("cca", "pca") for m in ("ridge", "gauss_own", "gauss_fullcov", "flow", "flow_linear", "flow_free")
                if ((own.model == m) & (own.route == r)).any()]
        mets = [("demeaned_pearson", "Δ demeaned r"), ("avg_rank", "Δ average rank"), ("energy", "Δ energy (lower = better)")]
        fig, axes = plt.subplots(len(mets), 1, figsize=(max(10, 1.3 * len(fams)), 3.4 * len(mets)), sharex=True)
        arms = [a for a in ARMS if a != "none"]
        cols = ["#0F4D92", "#3775BA", "#8BCF8B", "#B64342", "#E3962B", "#9A4D8E"]
        w = 0.8 / len(arms)
        for ax, (k, lab) in zip(axes, mets):
            for j, arm in enumerate(arms):
                xs, ys, es = [], [], []
                for i, (m, r) in enumerate(fams):
                    o = own[(own.model == m) & (own.route == r) & (own.arm == arm) & (own.metric == k)]
                    if len(o):
                        xs.append(i + (j - (len(arms) - 1) / 2) * w); ys.append(o.delta.iloc[0]); es.append(o.se.iloc[0])
                ax.bar(xs, ys, w, yerr=es, color=cols[j], label=arm, capsize=2)
            ax.axhline(0, color="0.3", lw=1)
            ax.set_ylabel(lab, fontsize=12)
        axes[-1].set_xticks(range(len(fams)), [f"{m}\n{r}" for m, r in fams], fontsize=11)
        axes[0].legend(ncol=3, fontsize=10, loc="upper left", bbox_to_anchor=(0, 1.35))
        fig.suptitle("Change vs the same model trained on full targets only (test, mean ± SE over seeds)", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG / "fig_g7_deltas.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print((TAB / "g7_summary.md").read_text())


if __name__ == "__main__":
    main()
