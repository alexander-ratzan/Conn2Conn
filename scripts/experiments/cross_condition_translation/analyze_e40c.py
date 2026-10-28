"""Spec v4 E4.0c analysis and the D10 code-size decision.

Score per code size (validation only): mean val average rank over the 56 single-source cells (without SC), for B1 and
B2 (both selected on val average rank, D9), averaged over seeds; the size's score is the better of the two. Rule: the
smallest k in {256, 512} within 0.005 of the better of the two; full rank flagged if it beats the choice by > 0.01.
Also: test summaries per k, and the consistency check (B1 with E4.0's GCV choice reproduces E4.0's test rows).

    python scripts/experiments/cross_condition_translation/analyze_e40c.py
        -> tables/e40c/{decision.json, summary.md, settings.csv}, figures/e40c/fig_e40c_code_size.png
"""
import json
import sys

import numpy as np
import pandas as pd

from e4_common import FC, HERE, OTHER7, REPO_ROOT, SC_ONLY

KS = ["256", "512", "full"]
TOL, FLAG = 0.005, 0.01
SETTING = {**{v: "single source" for v in FC}, OTHER7: "other 7", SC_ONLY: "SC only"}
METRICS = ["demeaned_pearson", "avg_rank", "top1_acc"]


def load():
    rows, hps, checks = [], {}, []
    for k in KS:
        for f in sorted((HERE / "results" / "e40c" / f"k{k}").glob("seed*.json")):
            r = json.loads(f.read_text())
            rows += r["rows"]
            hps[(k, r["seed"])] = r["hyperparams"]
            checks += [dict(k=k, seed=r["seed"], **c) for c in r["gcv_check"]]
    return pd.DataFrame(rows), hps, pd.DataFrame(checks)


def main():
    rows, hps, checks = load()
    rows["setting"] = rows.source.map(SETTING)
    rows.loc[(rows.source == SC_ONLY), "model"] = rows.loc[(rows.source == SC_ONLY), "model"].str.replace("+SC", "", regex=False) + "+SC"
    tab, fig = HERE / "tables" / "e40c", HERE / "figures" / "e40c"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    single = rows[(rows.setting == "single source") & (~rows.with_sc) & rows.model.isin(["B1", "B2"])]
    val = single[single.split == "val"].groupby(["k", "model", "seed"]).avg_rank.mean().groupby(["k", "model"]).mean().unstack()
    score = val.max(axis=1)
    cand = {k: float(score[k]) for k in ("256", "512") if k in score}
    best = max(cand.values())
    chosen = min((int(k) for k, v in cand.items() if v >= best - TOL))
    flag = "full" in score and float(score["full"]) > cand[str(chosen)] + FLAG
    decision = dict(rule="smallest k in {256, 512} within 0.005 of the better (val average rank, 56 single-source cells, "
                         "best of B1 / B2, mean over seeds); full rank diagnostic (> 0.01 better: flag)",
                    scores={k: float(v) for k, v in score.items()}, by_model={k: val.loc[k].to_dict() for k in val.index},
                    chosen_k=chosen, full_rank_flag=bool(flag))
    (tab / "decision.json").write_text(json.dumps(decision, indent=1))

    test = rows[rows.split == "test"]
    sett = (test.groupby(["k", "model", "setting", "seed"])[METRICS].mean().groupby(["k", "model", "setting"])
            .agg(["mean", "sem"]))
    sett.columns = [f"{a}_{b}" for a, b in sett.columns]
    sett = sett.reset_index()
    sett.to_csv(tab / "settings.csv", index=False, float_format="%.5g")

    ref = {}
    for k in ("256", "512"):
        f = HERE / "tables" / f"k{k}" / "baselines_cells.csv"
        if f.exists():
            ref[k] = pd.read_csv(f).query("model == 'B1'").set_index(["source", "target"])
    # E4.0's table holds seed means; compare seed means of the GCV check
    cm = checks.groupby(["k", "source", "target"]).demeaned_pearson.mean()
    md_diff = max((abs(v - ref[k].loc[(s, t), "demeaned_pearson_mean"]) for (k, s, t), v in cm.items()
                   if k in ref and (s, t) in ref[k].index), default=float("nan"))

    f = lambda x: f"{x:.4f}"
    md = ["# E4.0c — baselines selected on val average rank (D9), k = 256 / 512 / full rank; D10 decision", "",
          f"**Decision (D10):** k = **{chosen}**. Val score (mean average rank, 56 single-source cells, best of B1 / B2): "
          + ", ".join(f"k {k} {v:.4f}" for k, v in score.items()) + "."
          + (" **Flag:** full rank beats the chosen size by more than 0.01 (signal beyond the shared basis)." if flag else ""), "",
          f"Consistency: B1 with E4.0's GCV choice reproduces E4.0's test demeaned r (max |Δ| of seed means {md_diff:.1e}).", "",
          "Selected B2 (d, lambda) per seed: " + "; ".join(f"k {k}: " + ", ".join(f"({hps[(k, s)]['B2']['d']}, {hps[(k, s)]['B2']['lambda_rel']:g})"
                                                                                 for s in range(5) if (k, s) in hps) for k in KS) + ".", "",
          "Test (mean ± SE over seeds):", "", "| k | model | setting | demeaned r | average rank | top-1 |", "|---|---|---|---|---|---|"]
    order = {"B0": 0, "B0-codes": 1, "B1": 2, "B1+SC": 3, "B2": 4, "B2+SC": 5}
    for r in sorted(sett.itertuples(index=False), key=lambda r: (r.setting, KS.index(r.k), order.get(r.model, 9))):
        d = r._asdict()
        md.append(f"| {r.k} | {r.model} | {r.setting} | " + " | ".join(f"{f(d[f'{m}_mean'])} ± {f(d[f'{m}_sem'])}" for m in METRICS) + " |")
    (tab / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run as mb  # noqa: E402
    models = [("B0", "copy (edges)", "#CFCECE"), ("B0-codes", "copy (codes)", "#8C8C8C"), ("B1", "ridge", "#7FB2E5"),
              ("B1+SC", "ridge + SC", "#0F4D92"), ("B2", "multiview CCA", "#8BCF8B"), ("B2+SC", "multiview CCA + SC", "#42949E")]
    with plt.rc_context({**mb.FIG_RC, "font.size": 15}):
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        for ax, m in zip(axes, ("avg_rank", "demeaned_pearson")):
            q = sett[sett.setting == "single source"]
            w = 0.8 / len(models)
            for j, (mod, lab, c) in enumerate(models):
                for i, k in enumerate(KS):
                    r = q[(q.k == k) & (q.model == mod)]
                    if len(r):
                        ax.bar(i + (j - (len(models) - 1) / 2) * w, r[f"{m}_mean"].iloc[0], w, yerr=r[f"{m}_sem"].iloc[0],
                               color=c, edgecolor="black", lw=1, capsize=3, label=lab if i == 0 or mod in ("B1", "B2") else None)
            ax.set_xticks(range(len(KS)), [f"k = {k}" for k in KS], fontsize=16)
            ax.set_ylabel(f"test {'average rank (chance 0.5)' if m == 'avg_rank' else 'demeaned r'}", fontsize=16)
            if m == "avg_rank":
                ax.set_ylim(0.6, 0.9)
        h, l = axes[0].get_legend_handles_labels()
        uniq = dict(zip(l, h))
        axes[1].legend(list(uniq.values()), list(uniq.keys()), fontsize=13, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"E4.0c: single-source translation by code size (selection on val average rank); D10 choice k = {chosen}", fontsize=17)
        fig.tight_layout()
        fig.savefig(fig / "fig_e40c_code_size.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
