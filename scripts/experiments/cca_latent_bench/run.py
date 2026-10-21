"""Spec v4 E2 results: per-seed records -> hypothesis tests (support rule), tables and figures.

Headline readout is ridge for every hypothesis (pre-specified); PLS rows are reported alongside. The rest-PCA-code
baseline uses, per seed, the k with the best val demeaned r. References: v3:E0 seed_records (fc2sc, sc2fc).

    python scripts/experiments/cca_latent_bench/run.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402  (FIG_RC: shared figure style)

CFG = yaml.safe_load((HERE / "config.yml").read_text())
OUT = HERE / "results" / "Glasser"
REC = REPO_ROOT / CFG["records_dir"]
FC, TASKS = CFG["fc_views"], CFG["tasks"]
MET = ["demeaned_pearson", "avg_rank", "top1_acc"]
PAL = {"blue": "#0F4D92", "blue2": "#3775BA", "green": "#8BCF8B", "red": "#B64342", "teal": "#42949E",
       "violet": "#9A4D8E", "neutral": "#CFCECE"}


def load():
    rec = pd.concat([pd.read_csv(p) for p in sorted(REC.glob("seed[0-9].csv"))], ignore_index=True)
    lat = pd.concat([pd.read_csv(p) for p in sorted(REC.glob("seed[0-9]_latent.csv"))], ignore_index=True)
    r2 = pd.concat([pd.read_csv(p) for p in sorted(REC.glob("seed[0-9]_latent_r2.csv"))], ignore_index=True)
    bench = {d: pd.read_csv(REPO_ROOT / CFG["bench"] / d / "tables" / "seed_records.csv") for d in ("fc2sc", "sc2fc")}
    return rec, lat, r2, bench


def rule(delta):
    """Support rule (spec v4 E1.A): seed-mean above 2 SE and same sign in >= 4 of 5 seeds."""
    d = np.asarray(delta, dtype=float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    pos = int((d > 0).sum())
    return dict(mean=m, se=se, n_pos=pos, n=len(d), supported=bool(m > 2 * se and pos >= len(d) - 1))


def per_seed(rec, metric="demeaned_pearson", **sel):
    q = rec[(rec.split == "test")]
    for k, v in sel.items():
        q = q[q[k] == v]
    return q.set_index("seed")[metric].sort_index()


def pca_baseline(rec, metric="demeaned_pearson", readout="ridge"):
    """Rest PCA codes: per seed, the k with the best val demeaned r."""
    q = rec[(rec.direction == "fc2sc") & rec.input.str.startswith("rest PCA codes") & (rec.readout == readout)]
    out = {}
    for s, g in q.groupby("seed"):
        best = g[g.split == "val"].sort_values("demeaned_pearson").iloc[-1].input
        out[s] = g[(g.split == "test") & (g.input == best)][metric].iloc[0]
    return pd.Series(out).sort_index()


def bench_series(bench, direction, condition, metric="demeaned_pearson"):
    b = bench[direction]
    return b[b.model == condition].set_index("seed")[f"test_{metric}"].sort_index()


def hypotheses(rec, bench):
    f = dict(direction="fc2sc", latent="sc_blind", readout="ridge")
    rest_enc = per_seed(rec, input="rest encoding", **f)
    cons8 = per_seed(rec, input="8-condition consensus", **f)
    rows = []
    def add(hid, label, delta, metric="demeaned_pearson", note=""):
        rows.append(dict(H=hid, comparison=label, metric=metric, note=note, **rule(delta)))
    for m in MET:
        add("H1", "rest CCA encoding − rest PCA codes (FC → SC)", per_seed(rec, m, input="rest encoding", **f) - pca_baseline(rec, m), m)
    add("H2", "8-condition consensus − rest encoding (FC → SC)", cons8 - rest_enc)
    add("H2 (reported)", "7-task consensus − rest encoding", per_seed(rec, input="7-task consensus", **f) - rest_enc)
    add("H2 (reported)", "concatenated encodings − consensus", per_seed(rec, input="concatenated encodings", **f) - cons8)
    d3 = cons8 - bench_series(bench, "fc2sc", "rest")
    r3 = rule(d3)
    add("H3", "8-condition consensus − v3:E0 rest (PCA-PLS learnable)", d3,
        note=f"non-inferior (margin 0.01): {bool(r3['mean'] - 2 * r3['se'] > -0.01)}")
    via, direct, ben = [], [], []
    for m in FC:
        g = dict(direction="sc2fc", condition=m, readout="ridge")
        via.append(per_seed(rec, input="via consensus latent", **g))
        direct.append(per_seed(rec, input="direct codes", **g))
        ben.append(bench_series(bench, "sc2fc", m))
    via, direct, ben = (pd.concat(x, axis=1).mean(1) for x in (via, direct, ben))
    add("H4", "SC → consensus latent → each FC condition − v3:E0 SC → FC (mean over 8 conditions)", via - ben)
    add("H4", "SC → consensus latent → condition − direct SC → condition ridge (mean over 8)", via - direct)
    return pd.DataFrame(rows)


def summarize(rec):
    q = rec[rec.split == "test"].copy()
    q["input"] = q.input.where(~q.input.str.startswith("rest PCA codes"), q.input)
    g = q.groupby(["direction", "latent", "input", "condition", "readout"])[MET].agg(["mean", "sem"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def md_table(df, cols, fmt="{:.3f}"):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c])
                                       for c in cols) + " |")
    return "\n".join(lines)


def fig_fc2sc(rec, bench, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["rest PCA codes (val-selected k)", "rest encoding", "7-task consensus", "8-condition consensus",
             "concatenated encodings", *[f"{t} encoding" for t in TASKS]]
    ticks = ["rest FC PCA → SC", "rest FC → latent → SC", "7 task FCs → consensus latent → SC",
             "8 FCs → consensus latent → SC", "8 FCs → 8 latents (concat.) → SC",
             *[f"{t} FC → latent → SC" for t in TASKS]]
    vals = {}
    for ro in ("ridge", "pls"):
        f = dict(direction="fc2sc", latent="sc_blind", readout=ro)
        vals[ro] = [pca_baseline(rec, readout=ro)] + [per_seed(rec, input=i, **f) for i in order[1:]]
    with plt.rc_context({**mb.FIG_RC, "font.size": 15, "axes.linewidth": 2.0}):
        fig, ax = plt.subplots(figsize=(15, 6.5), layout="constrained")
        x = np.arange(len(order))
        for j, (ro, c) in enumerate((("ridge", PAL["blue"]), ("pls", PAL["teal"]))):
            means = [v.mean() for v in vals[ro]]
            ses = [v.sem() for v in vals[ro]]
            ax.bar(x + (j - 0.5) * 0.38, means, 0.38, yerr=ses, color=c, edgecolor="black", lw=1.2, capsize=3,
                   label=f"readout: {'PLS' if ro == 'pls' else 'ridge'}")
            for xi, v in zip(x, vals[ro]):
                ax.scatter(np.full(len(v), xi + (j - 0.5) * 0.38), v, s=14, color="black", zorder=3)
        b = bench_series(bench, "fc2sc", "rest")
        ax.axhline(b.mean(), color=PAL["red"], lw=2, ls="--",
                   label=f"rest FC PCA → SC · PCA-PLS learnable [v3:E0] ({b.mean():.3f})")
        for xi, name in zip(x, order):     # the same condition's own v3:E0 row on each single-condition bar
            cond = name.split(" ")[0]
            if name.endswith("encoding") and cond in FC:
                ax.plot([xi - 0.4, xi + 0.4], [bench_series(bench, "fc2sc", cond).mean()] * 2, color=PAL["red"], lw=3,
                        solid_capstyle="butt", label="same FC condition, own FC PCA → SC · PCA-PLS learnable [v3:E0]" if cond == "rest" else None)
        ax.set_xticks(x, ticks, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Test demeaned r (predicted vs measured SC)")
        ax.set_title("E2 FC → SC: which FC encoding predicts SC best (latent = FC-only CCA, never sees SC); "
                     "mean ± SE, 5 seeds", fontweight="bold", fontsize=15)
        ax.set_ylim(0, 0.2)
        ax.legend(frameon=False, fontsize=12, loc="upper center", ncol=2)
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_sc2fc(rec, bench, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [("SC PCA → task FC (per-task FC PCA) · PCA-PLS learnable [v3:E0]", lambda m: bench_series(bench, "sc2fc", m), PAL["red"]),
              ("SC PCA → task FC (block FC PCA) · ridge", lambda m: per_seed(rec, input="direct codes", direction="sc2fc", condition=m, readout="ridge"), PAL["neutral"]),
              ("SC PCA → task latent → task FC · ridge", lambda m: per_seed(rec, input="via own encoding", direction="sc2fc", condition=m, readout="ridge"), PAL["teal"]),
              ("SC PCA → consensus latent → task FC · ridge", lambda m: per_seed(rec, input="via consensus latent", direction="sc2fc", condition=m, readout="ridge"), PAL["blue"])]
    with plt.rc_context({**mb.FIG_RC, "font.size": 15, "axes.linewidth": 2.0}):
        fig, ax = plt.subplots(figsize=(15, 6.5), layout="constrained")
        x = np.arange(len(FC))
        w = 0.2
        for j, (lab, fn, c) in enumerate(series):
            v = [fn(m) for m in FC]
            ax.bar(x + (j - 1.5) * w, [s.mean() for s in v], w, yerr=[s.sem() for s in v], color=c, edgecolor="black",
                   lw=1.1, capsize=2, label=lab)
        ceil = [per_seed(rec, input="ceiling: true consensus latent", direction="sc2fc", condition=m).mean() for m in FC]
        ax.scatter(x + 1.5 * w, ceil, marker="_", s=400, color="black", lw=2.5, zorder=4,
                   label="ceiling: measured FC consensus latent → task FC (no SC)")
        ax.set_xticks(x, FC)
        ax.set_xlabel("target FC condition")
        ax.set_ylabel("Test demeaned r (predicted vs measured FC)")
        ax.set_title("E2 SC → FC: predicting each condition's FC from SC (latent = FC-only CCA); mean ± SE, 5 seeds",
                     fontweight="bold", fontsize=15)
        ax.legend(frameon=False, fontsize=12, ncol=1, loc="upper right")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_latent_r2(r2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({**mb.FIG_RC, "font.size": 15, "axes.linewidth": 2.0}):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for s, g in r2.groupby("seed"):
            ax.plot(g.dim, g.test_r2, color=PAL["blue"], alpha=0.5, lw=1.5, label="one line per seed" if s == r2.seed.min() else None)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("consensus latent dimension (1 = leading CCA component)")
        ax.set_ylabel("test R²")
        ax.legend(frameon=False)
        ax.set_title("SC PCA → consensus latent · ridge: test R² per dimension", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_latent_selection(lat, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = lat.groupby(["k_fc", "d", "lambda_rel"]).val_cross_avg_rank.mean().reset_index()
    with plt.rc_context({**mb.FIG_RC, "font.size": 14, "axes.linewidth": 2.0}):
        fig, axes = plt.subplots(1, len(CFG["latent"]["k_fc"]), figsize=(15, 4.5), sharey=True, layout="constrained")
        for ax, k in zip(axes, CFG["latent"]["k_fc"]):
            for lam, c in zip(sorted(g.lambda_rel.unique()), [PAL["red"], PAL["teal"], PAL["blue"], PAL["violet"]]):
                q = g[(g.k_fc == k) & (g.lambda_rel == lam)]
                ax.plot(q.d, q.val_cross_avg_rank, marker="o", color=c, label=f"CCA ridge λ = {lam:g}")
            ax.set_xscale("log", base=2)
            ds = sorted(g.d[g.k_fc == k].unique())
            ax.set_xticks(ds, [str(int(d)) for d in ds])
            ax.minorticks_off()
            ax.set_title(f"{k} block FC PCA codes per condition")
            ax.set_xlabel("CCA latent dimension d")
        axes[0].set_ylabel("val cross-condition identification\n(avg rank, 56 condition pairs)")
        axes[-1].legend(frameon=False)
        fig.suptitle("Choosing the FC-only CCA latent without SC: does a subject's encoding from one FC condition find "
                     "them in another? (mean over 5 seeds)", fontweight="bold", fontsize=14)
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def main():
    rec, lat, r2, bench = load()
    seeds = sorted(rec.seed.unique())
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    hyp = hypotheses(rec, bench)
    hyp.to_csv(OUT / "tables" / "hypotheses.csv", index=False)
    summ = summarize(rec)
    summ[summ.direction == "fc2sc"].to_csv(OUT / "tables" / "fc2sc.csv", index=False)
    summ[summ.direction == "sc2fc"].to_csv(OUT / "tables" / "sc2fc.csv", index=False)
    sel_cols = [c for c in lat.columns if c.startswith("selected_")]
    lat[lat[sel_cols].fillna(False).astype(bool).any(axis=1)].to_csv(OUT / "tables" / "latent_selection.csv", index=False)
    r2.to_csv(OUT / "tables" / "latent_r2.csv", index=False)
    fig_fc2sc(rec, bench, OUT / "figures" / "fig_fc2sc_inputs.png")
    fig_sc2fc(rec, bench, OUT / "figures" / "fig_sc2fc_conditions.png")
    fig_latent_r2(r2, OUT / "figures" / "fig_latent_r2.png")
    fig_latent_selection(lat, OUT / "figures" / "fig_latent_selection.png")
    hyp_md = hyp.assign(**{"Δ ± SE": hyp["mean"].map("{:+.4f}".format) + " ± " + hyp["se"].map("{:.4f}".format),
                           "seeds > 0": hyp.n_pos.astype(str) + " / " + hyp.n.astype(str)})
    text = [f"# E2 summary (seeds {', '.join(map(str, seeds))}; test split; headline readout ridge)\n",
            md_table(hyp_md, ["H", "comparison", "metric", "Δ ± SE", "seeds > 0", "supported", "note"]), ""]
    (OUT / "tables" / "summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
