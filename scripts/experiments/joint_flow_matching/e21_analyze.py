"""Spec v4 E2.1e: records -> hypothesis tests (support rule), tables and figures.

Test split, 5 seeds. Point metrics are averaged over the 8 conditions within a seed before the rule is applied.
References: v3:E0 SC -> FC rows (task_bidirectional_bench), E1's SC -> consensus latent ridge and the latent ceiling.

    python scripts/experiments/joint_flow_matching/e21_analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402  (FIG_RC)

REC = HERE / "results" / "Glasser" / "records" / "e21"
OUT = HERE / "results" / "Glasser" / "e21"
E1 = REPO_ROOT / "scripts" / "experiments" / "cca_latent_bench" / "results" / "Glasser" / "records"
BENCH = REPO_ROOT / "scripts" / "experiments" / "task_bidirectional_bench" / "results" / "Glasser" / "sc2fc" / "tables" / "seed_records.csv"
FC = ["rest", "emotion", "gambling", "language", "motor", "relational", "social", "wm"]
PAL = {"red": "#B64342", "neutral": "#CFCECE", "teal": "#42949E", "blue": "#0F4D92", "green": "#8BCF8B",
       "violet": "#9A4D8E", "gold": "#D9A441"}
LABEL = {"v3E0": "SC PCA → task FC (per-task FC PCA) · PCA-PLS learnable [v3:E0]",
         "ridge_cons": "SC PCA → consensus latent → task FC · ridge [E1]",
         "ridge_cond": "SC PCA → task latent → task FC · ridge",
         "corrector": "SC PCA → task latent → task FC · ridge + MLP corrector",
         "corrector_id": "… · ridge + MLP corrector, identity loss",
         "bridge": "… · ridge-start flow (16 steps)",
         "flow": "… · conditional flow from noise (mean of samples)"}


def load():
    recs = [json.loads(p.read_text()) for p in sorted(REC.glob("*_seed*.json"))]
    point, dist, l2s, hps, retest = [], [], [], [], []
    for r in recs:
        for row in r.get("rows", []):
            row = dict(row, seed=r["seed"])
            (l2s if "input" in row else point).append(row)
        for key, name in (("dist", "flow" if r["model"] == "flow" else "gauss"), ("dist_shuffled", "flow_shuffled"),
                          ("dist_spread_null", "flow_spread_null")):
            for split, d in r.get(key, {}).items():
                dist.append(dict({k: v for k, v in d.items() if not isinstance(v, dict)}, model=name, split=split,
                                 seed=r["seed"]))
        for key, name in (("retest", "flow"), ("retest_spread_null", "flow_spread_null")):
            for split, d in r.get(key, {}).items():
                retest.append(dict(d, model=name, split=split, seed=r["seed"]))
        if "hp" in r:
            hps.append(dict(model=r["model"], seed=r["seed"], minutes=r["seconds_total"] / 60, **r["hp"]))
    point = pd.DataFrame(point)
    point["steps"] = point.get("steps", pd.Series(dtype=float)).fillna(16)
    e1 = pd.concat([pd.read_csv(p) for p in sorted(E1.glob("seed[0-9].csv"))])
    e1 = e1[(e1.direction == "sc2fc") & (e1.split == "test")]
    extra = []
    for name, sel in (("ridge_cons", (e1.input == "via consensus latent") & (e1.readout == "ridge")),
                      ("ceiling", e1.input == "ceiling: true consensus latent")):
        for _, r in e1[sel].iterrows():
            extra.append(dict(model=name, condition=r.condition, split="test", seed=r.seed, steps=16,
                              demeaned_pearson=r.demeaned_pearson, avg_rank=r.avg_rank, top1_acc=r.top1_acc))
    b = pd.read_csv(BENCH)
    for _, r in b.iterrows():
        extra.append(dict(model="v3E0", condition=r.model, split="test", seed=r.seed, steps=16,
                          demeaned_pearson=r.test_demeaned_pearson, avg_rank=r.test_avg_rank, top1_acc=r.test_top1_acc))
    point = pd.concat([point, pd.DataFrame(extra)], ignore_index=True)
    return point, pd.DataFrame(dist), pd.DataFrame(l2s), pd.DataFrame(hps), recs, pd.DataFrame(retest)


def rule(delta):
    d = np.asarray(delta, dtype=float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    pos = int((d > 0).sum())
    return dict(mean=m, se=se, n_pos=pos, n=len(d), supported=bool(m > 2 * se and pos >= len(d) - 1))


def seed_mean(point, model, metric, steps=16):
    q = point[(point.model == model) & (point.split == "test") & (point.steps == steps)]
    return q.groupby("seed")[metric].mean()


def hypotheses(point, dist, l2s, retest):
    rows = []
    def add(h, label, delta, metric, note=""):
        rows.append(dict(H=h, comparison=label, metric=metric, note=note, **rule(delta)))
    add("H1", "ridge + MLP corrector − ridge (SC → task latent → task FC)",
        seed_mean(point, "corrector", "demeaned_pearson") - seed_mean(point, "ridge_cond", "demeaned_pearson"), "demeaned r")
    d_rank = seed_mean(point, "corrector_id", "avg_rank") - seed_mean(point, "corrector", "avg_rank")
    d_dr = seed_mean(point, "corrector_id", "demeaned_pearson") - seed_mean(point, "corrector", "demeaned_pearson")
    r = rule(d_dr)
    add("H2", "corrector + identity loss − corrector", d_rank, "average rank",
        f"demeaned r Δ {r['mean']:+.4f} ± {r['se']:.4f}; non-inferior (> −0.005): {bool(r['mean'] - 2 * r['se'] > -0.005)}")
    dt = dist[dist.split == "test"].pivot_table(index="seed", columns="model", values=["energy", "set_avg_rank", "spread_corr"])
    add("H3 (primary)", "linear-Gaussian − flow (energy distance to each subject's 8 encodings; > 0 = flow better)",
        dt["energy"]["gauss"] - dt["energy"]["flow"], "energy distance")
    add("H4", "flow − linear-Gaussian (set-to-set identification)", dt["set_avg_rank"]["flow"] - dt["set_avg_rank"]["gauss"],
        "set average rank", f"shuffled-SC flow set average rank {dt['set_avg_rank']['flow_shuffled'].mean():.3f} (chance 0.5)")
    add("H5", "flow spread correlation with each subject's observed spread (vs 0)", dt["spread_corr"]["flow"], "correlation")
    if "flow_spread_null" in dt["spread_corr"]:
        add("H5 (diagnostic)", "flow − spread-only null (SC shuffled for the residual only): spread correlation",
            dt["spread_corr"]["flow"] - dt["spread_corr"]["flow_spread_null"], "correlation")
    rt = retest[retest.split == "test"] if len(retest) else retest
    if len(rt):
        f = rt[rt.model == "flow"].set_index("seed")
        add("H5 (diagnostic)", "rest test–retest: flow's predicted rest spread vs each subject's session 1 − session 2 difference",
            f.retest_corr, "correlation", f"spread ratio flow {f.ratio_flow.mean():.2f}, Gaussian {f.ratio_gauss.mean():.2f} (1 = calibrated)")
    d6 = seed_mean(point, "flow", "demeaned_pearson") - seed_mean(point, "corrector", "demeaned_pearson")
    r6 = rule(d6)
    rows.append(dict(H="H6", comparison="flow sample mean − corrector (point accuracy)", metric="demeaned r",
                     note=f"non-inferior (mean − 2 SE > −0.005): {bool(r6['mean'] - 2 * r6['se'] > -0.005)}", **r6))
    t = l2s[l2s.split == "test"]
    add("H7", "latent → SC: ridge + MLP corrector − ridge (mean over consensus and 7 task inputs)",
        t[t.model == "latent2sc"].groupby("seed").demeaned_pearson.mean() - t[t.model == "ridge_latent2sc"].groupby("seed").demeaned_pearson.mean(),
        "SC demeaned r")
    return pd.DataFrame(rows)


def fig_conditions(point, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["v3E0", "ridge_cons", "ridge_cond", "corrector", "corrector_id", "bridge", "flow"]
    colors = [PAL["red"], PAL["neutral"], PAL["teal"], PAL["blue"], PAL["violet"], PAL["gold"], PAL["green"]]
    q = point[(point.split == "test") & (point.steps == 16)]
    with plt.rc_context({**mb.FIG_RC, "font.size": 14, "axes.linewidth": 2.0}):
        fig, ax = plt.subplots(figsize=(17, 7), layout="constrained")
        x, w = np.arange(len(FC)), 0.11
        for j, (m, c) in enumerate(zip(models, colors)):
            g = q[q.model == m].groupby("condition").demeaned_pearson
            ax.bar(x + (j - 3) * w, [g.mean().get(v, np.nan) for v in FC], w, yerr=[g.sem().get(v, np.nan) for v in FC],
                   color=c, edgecolor="black", lw=0.9, capsize=1.5, label=LABEL[m])
        ceil = q[q.model == "ceiling"].groupby("condition").demeaned_pearson.mean()
        ax.scatter(x + 3.6 * w, [ceil.get(v) for v in FC], marker="_", s=300, color="black", lw=2.5,
                   label="ceiling: measured FC consensus latent → task FC (no SC) [E1]")
        ax.set_xticks(x, FC); ax.set_xlabel("target FC condition")
        ax.set_ylabel("Test demeaned r (predicted vs measured FC)")
        ax.set_title("E2.1 SC → FC per condition through the latent: linear, nonlinear and flow models; mean ± SE, 5 seeds",
                     fontweight="bold", fontsize=15)
        ax.legend(frameon=False, fontsize=11, loc="upper right")
        fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)


def fig_distribution(dist, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = dist[dist.split == "test"]
    names = {"gauss": "linear-Gaussian\n(ridge mean + shared covariance)", "flow": "conditional flow\n(no condition token)",
             "flow_shuffled": "conditional flow,\nSC shuffled (full null)",
             "flow_spread_null": "conditional flow,\nSC shuffled for spread only"}
    cols = {"gauss": PAL["neutral"], "flow": PAL["green"], "flow_shuffled": PAL["red"], "flow_spread_null": PAL["gold"]}
    with plt.rc_context({**mb.FIG_RC, "font.size": 14, "axes.linewidth": 2.0}):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
        for ax, metric, ylab in zip(axes, ("energy", "set_avg_rank", "spread_corr"),
                                    ("energy distance to the subject's 8 encodings\n(lower is better)",
                                     "set-to-set identification\n(average rank; chance 0.5)",
                                     "correlation of predicted with observed\nper-subject spread")):
            ms = [m for m in names if m in set(q.model)]
            g = q.groupby("model")[metric]
            ax.bar(range(len(ms)), [g.mean()[m] for m in ms], yerr=[g.sem()[m] for m in ms],
                   color=[cols[m] for m in ms], edgecolor="black", lw=1.2, capsize=3)
            for i, m in enumerate(ms):
                ax.scatter(np.full(5, i), q[q.model == m][metric], s=14, color="black", zorder=3)
            ax.set_xticks(range(len(ms)), [names[m] for m in ms], fontsize=11)
            ax.set_ylabel(ylab)
        fig.suptitle("E2.1 SC → distribution of a subject's 8 condition encodings (test, 5 seeds)", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)


def fig_spread_and_clouds(recs, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flow = [r for r in recs if r["model"] == "flow" and r["seed"] == 0]
    if not flow:
        return
    d = flow[0]["dist"]["test"]
    with plt.rc_context({**mb.FIG_RC, "font.size": 13, "axes.linewidth": 2.0}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
        ps, os_ = np.array(d["per_subject"]["pred_spread"]), np.array(d["per_subject"]["obs_spread"])
        axes[0].scatter(os_, ps, s=14, color=PAL["blue"])
        axes[0].set_xlabel("observed spread of the subject's 8 encodings")
        axes[0].set_ylabel("predicted spread (flow, no token)")
        axes[0].set_title(f"per-subject spread, seed 0 (r = {np.corrcoef(ps, os_)[0, 1]:.2f})")
        G, O = np.array(d["example"]["generated"]), np.array(d["example"]["observed"])
        colors = plt.cm.tab10(np.arange(G.shape[1]))
        for i, c in enumerate(colors):
            axes[1].scatter(G[:, i, 0], G[:, i, 1], s=10, color=c, alpha=0.35)
            axes[1].scatter(O[:, i, 0], O[:, i, 1], s=60, color=c, edgecolor="black", marker="o")
        axes[1].set_xlabel("latent dimension 1"); axes[1].set_ylabel("latent dimension 2")
        axes[1].set_title("6 test subjects: generated (small) vs observed 8 encodings (large)")
        fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)


def fig_bridge_steps(point, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = point[(point.split == "test")]
    with plt.rc_context({**mb.FIG_RC, "font.size": 13, "axes.linewidth": 2.0}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for ax, metric in zip(axes, ("demeaned_pearson", "avg_rank")):
            b = q[q.model == "bridge"].groupby(["steps", "seed"])[metric].mean().groupby("steps")
            ax.errorbar(b.mean().index, b.mean(), yerr=b.sem(), marker="o", color=PAL["gold"], label="ridge-start flow")
            for m, c in (("ridge_cond", PAL["teal"]), ("corrector", PAL["blue"])):
                ax.axhline(seed_mean(point, m, metric).mean(), color=c, ls="--", label=LABEL[m].split("· ")[-1])
            ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64], ["1 (Euler)", "4", "16", "64"])
            ax.set_xlabel("integration steps"); ax.set_ylabel(f"test {metric.replace('_', ' ')} (mean of 8 conditions)")
        axes[0].legend(frameon=False, fontsize=10)
        fig.suptitle("E2.1 ridge-start flow: does integration add over one step?", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)


def main():
    point, dist, l2s, hps, recs, retest = load()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    hyp = hypotheses(point, dist, l2s, retest)
    if len(retest):
        retest.to_csv(OUT / "tables" / "retest.csv", index=False)
    hyp.to_csv(OUT / "tables" / "hypotheses.csv", index=False)
    q = point[point.split == "test"]
    q.groupby(["model", "steps", "condition"])[["demeaned_pearson", "avg_rank", "top1_acc"]].agg(["mean", "sem"]).to_csv(OUT / "tables" / "point.csv")
    dist[dist.split == "test"].groupby("model").agg(["mean", "sem"]).to_csv(OUT / "tables" / "distribution.csv")
    l2s[l2s.split == "test"].groupby(["model", "input"])[["demeaned_pearson", "avg_rank", "top1_acc"]].agg(["mean", "sem"]).to_csv(OUT / "tables" / "latent2sc.csv")
    hps.to_csv(OUT / "tables" / "selected_hyperparameters.csv", index=False)
    fig_conditions(point, OUT / "figures" / "fig_sc2fc_conditions.png")
    fig_distribution(dist, OUT / "figures" / "fig_distribution.png")
    fig_spread_and_clouds(recs, OUT / "figures" / "fig_spread_and_clouds.png")
    fig_bridge_steps(point, OUT / "figures" / "fig_bridge_steps.png")
    md = ["# E2.1 summary (test, 5 seeds; point metrics averaged over the 8 conditions within a seed)", "",
          "| H | comparison | metric | Δ ± SE | seeds > 0 | supported | note |", "|---|---|---|---|---|---|---|"]
    for _, r in hyp.iterrows():
        md.append(f"| {r.H} | {r.comparison} | {r.metric} | {r['mean']:+.4f} ± {r.se:.4f} | {r.n_pos} / {r.n} | "
                  f"{r.supported} | {r.note} |")
    (OUT / "tables" / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
