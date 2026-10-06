"""Spec v3 E0 results: scrape the task logs (job name e0_taskfc_<condition>), write per-seed and summary tables with
paired differences against rest, and draw the four metric bar charts (shared model_benchmark bar code).

    python scripts/experiments/task_fc_to_sc/run.py [--allow-partial]
"""
import argparse
import glob
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402  (model_benchmark/run.py: parse_log, tables, figures)

METRICS = ("pearson", "demeaned_pearson", "avg_rank", "top1_acc", "mse")


def mb_cfg(cfg):
    """The slice of a model_benchmark config that its tables / figures read, with conditions as 'models'."""
    return {"campaign": cfg["campaign"], "seeds": cfg["seeds"], "metrics": cfg["metrics"], "types": cfg["types"],
            "reuse": {}, "directions": {cfg["direction"]: {"models": cfg["conditions"]}},
            "models": {c: {"label": cfg["labels"][c], "type": "rest" if c.startswith("rest") else "task"}
                       for c in cfg["conditions"]}}


def scrape(cfg):
    records = []
    for cond in cfg["conditions"]:
        latest = {}
        # exact <job>_<task> suffix so "rest" does not also match rest_S1 logs
        pat = re.compile(rf"{re.escape(cfg['campaign'])}_{re.escape(cond)}_\d+_\d+\.out")
        paths = glob.glob(str(Path(cfg.get("log_dir") or mb.LOG_DIR) / f"{cfg['campaign']}_{cond}_*_*.out"))
        for path in sorted((p for p in paths if pat.fullmatch(os.path.basename(p))), key=os.path.getmtime):
            got = mb.parse_log(path)
            if got:
                latest[got[0]] = (path, got[1])
        for seed, (path, s) in sorted(latest.items()):
            if seed not in cfg["seeds"]:
                continue
            m = s.get("metrics") or {}
            run = s.get("run") or {}
            want = f"scripts/experiments/task_fc_to_sc/configs/{cond}.yml"
            if run.get("config_path") != want or (run.get("source"), run.get("target")) != ("FC", "SC"):
                raise SystemExit(f"{path}: ran {run.get('config_path')} {run.get('source')}->{run.get('target')}, "
                                 f"expected {want} FC->SC")
            records.append({"model": cond, "seed": seed, "log": os.path.basename(path),
                            "selected_val": (s.get("selected_by") or {}).get("value"),
                            "config": (s.get("best_trial") or {}).get("config"), "val": m.get("val") or {},
                            "test": m.get("test") or {}})
    return records


def paired_vs(seed_df, ref):
    base = seed_df[seed_df["model"] == ref].set_index("seed")
    rows = []
    for cond, g in seed_df.groupby("model", sort=False):
        g = g.set_index("seed")
        shared = sorted(set(g.index) & set(base.index))
        r = {"condition": cond, "label": g["label"].iloc[0], "reference": ref, "n_shared_seeds": len(shared)}
        for m in ("pearson", "demeaned_pearson", "avg_rank", "top1_acc"):
            d = (g.loc[shared, f"test_{m}"] - base.loc[shared, f"test_{m}"]).dropna()
            r[f"d_{m}_mean"] = d.mean() if len(d) else np.nan
            r[f"d_{m}_se"] = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("d_demeaned_pearson_mean", ascending=False)


def md(df, cols, fmt=4):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(f"{r[c]:.{fmt}f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
                   for r in df.to_dict("records"))
    return head + body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true", help="draw even if some condition x seed cells are missing")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    mcfg = mb_cfg(cfg)
    records = scrape(cfg)
    have = {(r["model"], r["seed"]) for r in records}
    missing = [(c, s) for c in cfg["conditions"] for s in cfg["seeds"] if (c, s) not in have]
    print(f"{len(records)} condition x seed results; missing {len(missing)}: {missing[:12]}")
    if not records or (missing and not args.allow_partial):
        raise SystemExit("incomplete (use --allow-partial for a partial render)")
    seed_df, summary = mb.tables(mcfg, records)
    out = HERE / cfg["output_dir"]
    (out / "tables").mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out / "tables" / "seed_records.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "tables" / "summary.csv", index=False, float_format="%.6g")
    pd.DataFrame([{"condition": r["model"], "seed": r["seed"], "log": r["log"], "selected_val": r["selected_val"],
                   "config": r["config"]} for r in records]).to_csv(out / "tables" / "runs.csv", index=False)
    pv = paired_vs(seed_df, cfg["reference"])
    pv.to_csv(out / "tables" / "paired_vs_rest.csv", index=False, float_format="%.6g")
    s = summary.sort_values("demeaned_pearson_mean", ascending=False)
    (out / "tables" / "summary.md").write_text(
        md(s, ["label", "n_seeds", "pearson_mean", "demeaned_pearson_mean", "avg_rank_mean", "top1_acc_mean"])
        + "\nPaired vs " + cfg["reference"] + " (mean over shared seeds):\n\n"
        + md(pv, ["label", "n_shared_seeds", "d_pearson_mean", "d_demeaned_pearson_mean", "d_avg_rank_mean", "d_top1_acc_mean"]))
    for col in ("role", "objective"):
        summary[col] = summary[col].fillna("").astype(str)
    # E0 has no native-objective / null / extra-input / gated bars: keep only the type swatches and replace the footer
    _handles = mb._legend_handles
    mb._legend_handles = lambda c, g, **kw: [h for h in _handles(c, g, **kw)
                                             if h.get_label() not in ("native objective (*)", "PCA null")]
    mb._figure_note = lambda *_: ("Bars: mean ± SE over 5 seeds (dots = seeds). CrossModal PCA-PLS learnable, MSE, FC → SC, "
                                  "Glasser, test split of the matched 917-subject cohort; only the source FC differs.")
    written = mb.figures(mcfg, summary, seed_df, out / "figures", cfg["direction"])
    print("wrote", out / "tables", "and", [str(out / "figures" / w) for w in written])


if __name__ == "__main__":
    main()
