"""Spec v4 E2.1 addendum: can un-anchored flows (linear or nonlinear velocity, from noise, no ridge) learn SC -> latent?

Compares, on test over 5 seeds: ridge (per condition), the linear-Gaussian, the ridge-anchored residual flow (E2.1)
and the two un-anchored flows, each with its full null. Writes results/Glasser/e21/tables/addendum.{csv,md} and
figures/fig_addendum_free_flows.png.

    python scripts/experiments/joint_flow_matching/e21_addendum_analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402

REC = HERE / "results" / "Glasser" / "records" / "e21"
OUT = HERE / "results" / "Glasser" / "e21"
LABEL = {"ridge_cond": "ridge", "gauss": "linear-\nGaussian", "flow": "flow,\nridge-anchored",
         "flow_linear": "flow from noise,\nlinear", "flow_free": "flow from noise,\nnonlinear"}
COL = {"ridge_cond": "#42949E", "gauss": "#CFCECE", "flow": "#8BCF8B", "flow_linear": "#3775BA", "flow_free": "#0F4D92"}


def collect():
    rows = []
    for p in sorted(REC.glob("*_seed*.json")):
        r = json.loads(p.read_text())
        if r["model"] not in ("refs", "flow", "flow_free", "flow_linear"):
            continue
        pts = pd.DataFrame([x for x in r["rows"] if x["split"] == "test" and "input" not in x and x.get("steps", 16) == 16])
        for model, g in pts.groupby("model"):
            rows.append(dict(seed=r["seed"], model=model, dr=g.demeaned_pearson.mean(), rank=g.avg_rank.mean(),
                             top1=g.top1_acc.mean()))
        dist_name = "gauss" if r["model"] == "refs" else r["model"]
        for key, name in (("dist", dist_name), ("dist_shuffled", f"{r['model']}_shuffled")):
            d = r.get(key, {}).get("test")
            if d:
                rows.append(dict(seed=r["seed"], model=name, energy=d["energy"], set_rank=d["set_avg_rank"],
                                 set_top1=d["set_top1"], spread_r=d["spread_corr"]))
    df = pd.DataFrame(rows).groupby(["model", "seed"]).first().reset_index()
    return df


def main():
    df = collect()
    tab = df.groupby("model")[["dr", "rank", "top1", "energy", "set_rank", "set_top1", "spread_r"]].agg(["mean", "sem"])
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "tables" / "addendum.csv")
    order = ["ridge_cond", "gauss", "flow", "flow_linear", "flow_free", "flow_linear_shuffled", "flow_free_shuffled", "flow_shuffled"]
    md = ["# E2.1 addendum: un-anchored flows (test, 5 seeds; point metrics are means over the 8 conditions)", "",
          "| model | demeaned r | average rank | energy (lower is better) | set average rank | set top-1 | spread r |",
          "|---|---|---|---|---|---|---|"]
    m = df.groupby("model").mean(numeric_only=True)
    for k in order:
        if k in m.index:
            r = m.loc[k]
            f = lambda v: "—" if pd.isna(v) else f"{v:.4f}"
            md.append(f"| {k} | {f(r.get('dr'))} | {f(r.get('rank'))} | {f(r.get('energy'))} | {f(r.get('set_rank'))} | "
                      f"{f(r.get('set_top1'))} | {f(r.get('spread_r'))} |")
    piv = df.pivot_table(index="seed", columns="model", values=["dr", "energy"])
    for k in ("flow_linear", "flow_free"):
        if ("dr", k) in piv.columns:
            d = piv["dr"][k] - piv["dr"]["ridge_cond"]
            e = piv["energy"]["gauss"] - piv["energy"][k]
            md.append(f"\n{k}: demeaned r − ridge {d.mean():+.4f} ± {d.sem():.4f} ({int((d > 0).sum())} / {len(d)} seeds > 0); "
                      f"Gaussian − {k} energy {e.mean():+.4f} ± {e.sem():.4f} ({int((e > 0).sum())} / {len(e)} seeds flow better)")
    (OUT / "tables" / "addendum.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({**mb.FIG_RC, "font.size": 13, "axes.linewidth": 2.0}):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
        for ax, (metric, models, ylab) in zip(axes, (
                ("dr", ["ridge_cond", "flow", "flow_linear", "flow_free"], "test demeaned r, SC → task FC\n(mean of 8 conditions)"),
                ("energy", ["gauss", "flow", "flow_linear", "flow_free"], "energy distance to the subject's 8 encodings\n(lower is better)"),
                ("set_rank", ["gauss", "flow", "flow_linear", "flow_free"], "set-to-set identification\n(average rank; chance 0.5)"))):
            g = df.groupby("model")[metric]
            ms = [k for k in models if k in g.mean().index]
            ax.bar(range(len(ms)), [g.mean()[k] for k in ms], yerr=[g.sem()[k] for k in ms], color=[COL[k] for k in ms],
                   edgecolor="black", lw=1.2, capsize=3)
            for i, k in enumerate(ms):
                ax.scatter(np.full(df[df.model == k][metric].notna().sum(), i), df[df.model == k][metric].dropna(),
                           s=14, color="black", zorder=3)
            ax.set_xticks(range(len(ms)), [LABEL[k] for k in ms], fontsize=11)
            ax.set_ylabel(ylab)
            lo = np.nanmin([g.mean()[k] - 3 * g.sem()[k] for k in ms])
            hi = np.nanmax([g.mean()[k] + 3 * g.sem()[k] for k in ms])
            ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        fig.suptitle("E2.1 addendum: can a flow learn SC → latent without a ridge anchor? (test, 5 seeds; y-axes zoomed)",
                     fontweight="bold")
        fig.savefig(OUT / "figures" / "fig_addendum_free_flows.png", dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
