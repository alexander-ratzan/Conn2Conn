"""Spec v3 E3: composite-loss benchmark vs each model's own E2.2 MSE result (same splits, paired by seed).

Reads composite_loss/<direction>/<selection>/{tables/seed_records.csv, records.json} (run.py with
MB_CAMPAIGN_CONFIG=config_composite_loss.yml) and mse/<direction>/tables/seed_records.csv. Writes, per direction,
composite_loss/<direction>/comparison/:
  delta_vs_mse.{csv,md}    per model x selection: E2.2 mean, E3 mean, paired delta (mean +- SE, seeds improved) per metric
  selected_weights.md      per model x selection: the selected loss weights per seed and how often each term is on
  tradeoff.png             test demeaned r vs average rank: E2.2 (MSE) -> E3 by val demeaned r -> E3 by val avg rank

    python scripts/experiments/model_benchmark/composite_vs_mse.py            # both directions
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CFG = yaml.safe_load((HERE / "config_composite_loss.yml").read_text())
METRICS = ["demeaned_pearson", "avg_rank", "top1_acc", "pearson"]
LABEL = {"demeaned_pearson": "Demeaned r", "avg_rank": "Average rank", "top1_acc": "Top-1", "pearson": "Pearson r"}
SEL_LABEL = {"by_demeaned_r": "selected on val demeaned r", "by_avg_rank": "selected on val avg rank"}
TERMS = CFG["terms"]


def load(direction):
    mse = pd.read_csv(HERE / "mse" / direction / "tables" / "seed_records.csv")
    out = {}
    for sel in CFG["selections"].values():
        d = HERE / CFG["results_dir"] / direction / sel
        out[sel] = (pd.read_csv(d / "tables" / "seed_records.csv"), json.loads((d / "records.json").read_text()))
    return mse, out


def deltas(direction, mse, e3):
    rows = []
    for model in CFG["directions"][direction]["models"]:
        base = mse[mse.model == model].set_index("seed")
        for sel, (df, _) in e3.items():
            comp = df[df.model == model].set_index("seed")
            seeds = sorted(set(base.index) & set(comp.index))
            row = {"model": model, "label": CFG["models"][model]["label"], "selection": sel, "n_seeds": len(seeds)}
            for m in METRICS:
                d = comp.loc[seeds, f"test_{m}"] - base.loc[seeds, f"test_{m}"]
                row |= {f"{m}_mse": base.loc[seeds, f"test_{m}"].mean(), f"{m}_e3": comp.loc[seeds, f"test_{m}"].mean(),
                        f"{m}_delta": d.mean(), f"{m}_delta_se": d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0,
                        f"{m}_n_up": int((d > 0).sum())}
            rows.append(row)
    return pd.DataFrame(rows)


def delta_md(df):
    lines = ["| Model | Selection | " + " | ".join(f"{LABEL[m]}: E2.2 → E3 (Δ ± SE, seeds up)" for m in METRICS[:3]) + " |",
             "|---|---|" + "---|" * 3]
    for r in df.to_dict("records"):
        cells = [f"{r[f'{m}_mse']:.4f} → {r[f'{m}_e3']:.4f} ({r[f'{m}_delta']:+.4f} ± {r[f'{m}_delta_se']:.4f}, "
                 f"{r[f'{m}_n_up']}/{r['n_seeds']})" for m in METRICS[:3]]
        lines.append(f"| {r['label']} | {SEL_LABEL[r['selection']]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def weights_md(direction, e3):
    lines = ["| Model | Selection | " + " | ".join(f"`{t}` per seed (0-4)" for t in TERMS) + " | Seeds with any term on |",
             "|---|---|" + "---|" * len(TERMS) + "---|"]
    for model in CFG["directions"][direction]["models"]:
        for sel, (_, recs) in e3.items():
            rs = sorted((r for r in recs if r["model"] == model), key=lambda r: int(r["seed"]))
            w = {t: [float((r["config"].get("trainer") or {}).get(f"loss_weight_{t}", 0.0)) for r in rs] for t in TERMS}
            any_on = sum(any(w[t][i] > 0 for t in TERMS) for i in range(len(rs)))
            lines.append(f"| {CFG['models'][model]['label']} | {SEL_LABEL[sel]} | "
                         + " | ".join(", ".join(f"{v:g}" for v in w[t]) for t in TERMS) + f" | {any_on}/{len(rs)} |")
    return "\n".join(lines) + "\n"


def tradeoff_fig(direction, df, path):
    import importlib.util

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = importlib.util.spec_from_file_location("model_benchmark_run", HERE / "run.py")
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)
    colors = ["#0F4D92", "#C2410C", "#15803D", "#7C3AED"]
    with plt.rc_context({**mb.FIG_RC, "font.size": 13}):
        fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
        for color, model in zip(colors, CFG["directions"][direction]["models"]):
            sub = df[df.model == model].set_index("selection")
            pts = [(sub.iloc[0]["demeaned_pearson_mse"], sub.iloc[0]["avg_rank_mse"], "o", "E2.2 MSE")]
            pts += [(sub.loc[s, "demeaned_pearson_e3"], sub.loc[s, "avg_rank_e3"], mk, s)
                    for s, mk in (("by_demeaned_r", "s"), ("by_avg_rank", "^")) if s in sub.index]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.2, alpha=0.6)
            for x, y, mk, _ in pts:
                ax.scatter(x, y, marker=mk, s=70, color=color, edgecolor="black", lw=0.6, zorder=3)
            ax.annotate(CFG["models"][model]["label"], (pts[0][0], pts[0][1]), textcoords="offset points",
                        xytext=(6, -12), fontsize=10, color=color)
        for mk, lab in (("o", "E2.2 (MSE only)"), ("s", "E3, selected on val demeaned r"),
                        ("^", "E3, selected on val avg rank")):
            ax.scatter([], [], marker=mk, color="white", edgecolor="black", label=lab)
        ax.legend(frameon=False, fontsize=10, loc="best")
        ax.set_xlabel("Test demeaned r")
        ax.set_ylabel("Test average rank")
        arrow = "SC → FC" if direction == "sc2fc" else "FC → SC"
        ax.set_title(f"{arrow}: composite loss vs MSE (mean over seeds 0–4)", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def main():
    for direction in ("sc2fc", "fc2sc"):
        mse, e3 = load(direction)
        df = deltas(direction, mse, e3)
        out = HERE / CFG["results_dir"] / direction / "comparison"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "delta_vs_mse.csv", index=False, float_format="%.6g")
        (out / "delta_vs_mse.md").write_text(delta_md(df))
        (out / "selected_weights.md").write_text(weights_md(direction, e3))
        tradeoff_fig(direction, df, out / "tradeoff.png")
        print(f"{direction}: wrote {out}")


if __name__ == "__main__":
    main()
