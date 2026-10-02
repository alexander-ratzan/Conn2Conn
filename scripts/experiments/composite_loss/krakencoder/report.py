"""
Krakencoder loss-grid report: per-cell summary tables and PNG figures from the collected tables.

    python scripts/experiments/composite_loss/krakencoder/grid_runner.py collect --set grid   # (and --set noise)
    python scripts/experiments/composite_loss/krakencoder/report.py [--set grid]

Inputs:  tables/<set>_seed_records.csv, tables/<set>_epoch_history.csv (and tables/noise_seed_records.csv if present)
Outputs: tables/summary.{csv,md}  per cell x direction: test metrics mean +- SD over seeds, and the paired-by-seed
                                   difference from mse_only (mean +- SE)
         tables/noise.md           retrain variability: init-seed SD (fixed split) vs split-seed SD, kraken_default
         per direction (grid set), E1 instance schema: {sc2fc,fc2sc}/tables/{seed_records.csv, combo_summary.csv,
         epoch_history.csv.gz} and {sc2fc,fc2sc}/figures/ from ../report.py: tradeoff_scatter.png,
         tradeoff_interactive.html, term_trajectories.png, val_trajectories.png, dose_response.png
Missing cells or seeds are skipped (the report also runs on the pilot set).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {"blue_main": "#0F4D92", "green_3": "#8BCF8B", "red_strong": "#B64342", "teal": "#42949E",
           "violet": "#9A4D8E", "neutral": "#CFCECE"}
TERM_COLORS = {"varmatch": PALETTE["teal"], "correye": PALETTE["blue_main"], "neidist": PALETTE["red_strong"],
               "all": PALETTE["violet"]}
TERM_LABELS = {"varmatch": "var (variance match)", "correye": "correye (~ correye_dm)", "neidist": "neidist",
               "all": "all three"}
METRICS = {"test_demeaned_pearson": "test demeaned r", "test_avg_rank": "test avg rank", "test_top1_acc": "test top-1"}
DIRECTIONS = ("SC->FC", "FC->SC")


def style():
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.6, "legend.frameon": False, "font.family": "DejaVu Sans"})


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def mean_sd(v):
    v = [x for x in v if x is not None and not math.isnan(x)]
    if not v:
        return math.nan, math.nan, 0
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else math.nan, len(v)


def summary(seed_rows: list[dict]) -> list[dict]:
    by = defaultdict(dict)  # (cell, direction) -> seed -> row
    meta = {}
    for r in seed_rows:
        by[(r["combo_id"], r["direction"])][int(r["seed"])] = r
        meta[r["combo_id"]] = r
    out = []
    for (cell, d), seeds in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = {"combo_id": cell, "block": meta[cell]["block"], "direction": d, "n_seeds": len(seeds),
               **{f"w_{t}": meta[cell][f"w_{t}"] for t in ("varmatch", "correye", "neidist")}}
        base = by.get(("mse_only", d), {})
        for m in METRICS:
            mu, sd, _ = mean_sd([float(s[m]) for s in seeds.values()])
            row[f"{m}_mean"], row[f"{m}_sd"] = mu, sd
            diffs = [float(seeds[s][m]) - float(base[s][m]) for s in seeds if s in base]
            dm, dsd, n = mean_sd(diffs)
            row[f"{m}_vs_mse_only"] = dm
            row[f"{m}_vs_mse_only_se"] = dsd / math.sqrt(n) if n > 1 else math.nan
        out.append(row)
    return out


def fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_summary(rows: list[dict], tables: Path) -> None:
    if not rows:
        return
    with open(tables / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = []
    for d in DIRECTIONS:
        sub = [r for r in rows if r["direction"] == d]
        if not sub:
            continue
        lines += [f"### {d}", "", "| cell | block | n | test demeaned r | Δ vs mse_only | test avg rank | Δ vs mse_only | test top-1 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(sub, key=lambda r: -r["test_demeaned_pearson_mean"]):
            lines.append(
                f"| {r['combo_id']} | {r['block']} | {r['n_seeds']} | {fmt(r['test_demeaned_pearson_mean'])} ± "
                f"{fmt(r['test_demeaned_pearson_sd'])} | {fmt(r['test_demeaned_pearson_vs_mse_only'])} ± "
                f"{fmt(r['test_demeaned_pearson_vs_mse_only_se'])} | {fmt(r['test_avg_rank_mean'], 3)} ± "
                f"{fmt(r['test_avg_rank_sd'], 3)} | {fmt(r['test_avg_rank_vs_mse_only'], 3)} ± "
                f"{fmt(r['test_avg_rank_vs_mse_only_se'], 3)} | {fmt(r['test_top1_acc_mean'], 3)} |")
        lines.append("")
    (tables / "summary.md").write_text("\n".join(lines))


def write_noise(grid_rows: list[dict], noise_rows: list[dict], tables: Path) -> None:
    kd = [r for r in grid_rows if r["combo_id"] == "kraken_default"]
    if not noise_rows:
        return
    lines = ["| direction | metric | init-seed SD (split 0, n) | split-seed SD (init 0, n) |", "|---|---|---|---|"]
    for d in DIRECTIONS:
        init = [r for r in noise_rows + kd if r["direction"] == d and int(r["seed"]) == 0]
        split = [r for r in kd if r["direction"] == d]
        for m, label in METRICS.items():
            _, sd_i, n_i = mean_sd([float(r[m]) for r in init])
            _, sd_s, n_s = mean_sd([float(r[m]) for r in split])
            lines.append(f"| {d} | {label} | {fmt(sd_i)} ({n_i}) | {fmt(sd_s)} ({n_s}) |")
    (tables / "noise.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", default="grid")
    args = ap.parse_args()
    tables, figures = HERE / "tables", HERE / "figures"
    figures.mkdir(exist_ok=True)
    seed_rows = read(tables / f"{args.set_name}_seed_records.csv")
    history = read(tables / f"{args.set_name}_epoch_history.csv")
    if not seed_rows:
        raise SystemExit(f"no {tables}/{args.set_name}_seed_records.csv; run grid_runner.py collect --set {args.set_name}")
    style()
    rows = summary(seed_rows)
    write_summary(rows, tables)
    write_noise(seed_rows, read(tables / "noise_seed_records.csv"), tables)
    print(f"wrote {tables}/summary.(csv|md), noise.md ({len(rows)} cell x direction rows)")
    if args.set_name == "grid":
        e1_export(args.set_name)   # per-direction E1-schema tables + figures (shared ../report.py functions)



# --------------------------------------------------------------------------------------------- per-direction export
# Bidirectional layout (spec v2 E3 template): krakencoder/<direction>/{tables,figures}/ in the E1 instance schema, one
# folder per direction, written from the shared fits (one Krakencoder fit serves both directions, so the direction
# folders hold views, not runs). Figures use the shared ../report.py functions (imported, not copied).
# TRANSITION: until compare.py reads <model>/<direction>/ (Phase B), the flat tables/seed_records.csv (both directions)
# is also written; drop it when Phase B merges.
E1_NAMES = {"val_demeaned_pearson": "val_demeaned_r"}
DIRECTIONS_DIR = {"SC->FC": "sc2fc", "FC->SC": "fc2sc"}
FLAT_SEED_RECORDS_FOR_COMPARE = False   # Phase B: compare.py reads <model>/<direction>/


def _load_e1_report():
    import importlib.util
    spec = importlib.util.spec_from_file_location("composite_loss_report", HERE.parent / "report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e1_tables(set_name: str = "grid"):
    import sys
    import pandas as pd
    import yaml
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    tables = HERE / "tables"
    seeds = pd.read_csv(tables / f"{set_name}_seed_records.csv")
    hist = pd.read_csv(tables / f"{set_name}_epoch_history.csv").rename(columns=E1_NAMES)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import grid_runner
    cells = {c["id"]: c for c in grid_runner.cells(cfg)}
    sig = {cid: grid_runner.loss_string(cfg, c) for cid, c in cells.items()}
    for df in (seeds, hist):
        df["stage"] = "stage2"
        df["instance"] = "krakencoder/" + df["direction"].map(DIRECTIONS_DIR)
        df["loss_signature"] = df["combo_id"].map(sig)
        df["w_correye_dm"] = 0.0      # Krakencoder's native correye is kept as w_correye (≈ demeaned, D5)
    seeds = seeds.rename(columns=E1_NAMES)
    seeds["val_demeaned_r_last"] = seeds["val_demeaned_r"]
    return cfg, cells, seeds, hist


def e1_export(set_name: str = "grid") -> None:
    import pandas as pd
    rep = _load_e1_report()
    cfg, cells, seeds, hist = e1_tables(set_name)
    combos = [{**c, "block": c.get("block", "")} for c in cells.values()]
    orig_save = rep._save

    def save_relabel(fig, path):  # E1 axis text assumes E1's scaled weights; Krakencoder levels are paper-anchored
        for ax in fig.axes:
            if "scaled; MSE = 1" in ax.get_xlabel():
                ax.set_xlabel("Grid level × anchor (paper = 1; symlog)")
            if Path(path).name.startswith("tradeoff_scatter") and ax.get_legend() is not None:
                ax.get_legend().set_loc("lower left")   # Krakencoder's points fill E1's default legend corner
        orig_save(fig, path)

    rep._save = save_relabel
    try:
        for d, name in DIRECTIONS_DIR.items():
            out = HERE / name
            tables, figures = out / "tables", out / "figures"
            tables.mkdir(parents=True, exist_ok=True)
            figures.mkdir(parents=True, exist_ok=True)
            all_rec = seeds[seeds["direction"] == d]
            all_rec.to_csv(tables / "seed_records.csv", index=False, float_format="%.6g")
            ep_all = hist[hist["direction"] == d]
            ep_all.to_csv(tables / "epoch_history.csv.gz", index=False, float_format="%.6g")
            rec = all_rec[all_rec["random_seed"] == 0].drop(columns=["direction"]).copy()
            ep = ep_all[ep_all["random_seed"] == 0].copy()
            # grid fits are scored every recipe.checkpoint_every epochs; the reused pilot fits more often: keep shared ones
            ep = ep[ep["epoch"] % int(cfg["recipe"]["checkpoint_every"]) == 0]
            rec_in = rec.drop(columns=["block"] + [c for c in rec if c.startswith("w_")])
            _, summary = rep.build_tables({"grid": {"combos": combos}}, rec_in, None)
            summary.insert(0, "direction", d)
            summary.to_csv(tables / "combo_summary.csv", index=False, float_format="%.6g")
            title = f"Krakencoder {d}: composite-loss trade-off (grid v1 + paper default + level 2)"
            rep.fig_tradeoff(summary, figures / "tradeoff_scatter.png")
            rep.fig_interactive(summary, rec, figures / "tradeoff_interactive.html", title)
            rep.fig_term_trajectories(ep, combos, figures / "term_trajectories.png")
            rep.fig_val_trajectories(ep, combos, figures / "val_trajectories.png")
            rep.fig_dose_response(summary, figures / "dose_response.png")
            print(f"{d}: wrote {out.relative_to(HERE.parent)}/tables + figures")
    finally:
        rep._save = orig_save
    if FLAT_SEED_RECORDS_FOR_COMPARE:   # TRANSITION (see above)
        seeds.to_csv(HERE / "tables" / "seed_records.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
