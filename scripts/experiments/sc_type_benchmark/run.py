"""
SC-type benchmark: scrape (or load) best-trial W&B results, then write tables and figures.

    python scripts/experiments/sc_type_benchmark/run.py              # from the tracked records.json
    python scripts/experiments/sc_type_benchmark/run.py --rescrape   # refresh records.json from W&B

Everything configurable lives in config.yml next to this file. records.json is the tracked
snapshot of the most recent scrape; tables/, figures/ and manifest.json are written next to it
(`output_dir`, relative to this folder) and can be regenerated from it at any time. Figures are
PNG (300 dpi); every plotted value is recomputable from tables/seed_records.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

from scripts.results_utils import (  # noqa: E402
    build_experiment_records,
    build_sc_type_summary_table,
    build_status_table,
    records_to_df,
    save_records_cache,
)
from scripts.results_utils import runner  # noqa: E402


def grid(cfg: dict) -> dict:
    """Scrape grid; also the cache-coverage contract checked on load."""
    return {
        "models": list(cfg["models"]),
        "sources": list(cfg["sources"]),
        "seeds": list(cfg["seeds"]),
        "selection_metric": cfg["selection"]["metric"],
        "selection_mode": cfg["selection"]["mode"],
        "direct_prod_models": list(cfg.get("direct_prod_models") or []),
    }


def scrape(cfg: dict, cache_path: Path) -> list:
    print("Scraping W&B...", flush=True)
    records = build_experiment_records(
        models=cfg["models"],
        sources=cfg["sources"],
        seeds=cfg["seeds"],
        count_trials=bool(cfg.get("count_trials", True)),
        verbose=True,
        selection_metric=cfg["selection"]["metric"],
        selection_mode=cfg["selection"]["mode"],
        direct_prod_models=cfg.get("direct_prod_models") or None,
    )
    save_records_cache(records, cache_path, metadata=runner.scrape_metadata(grid(cfg)))
    print(f"Saved {len(records)} records to {runner.rel(cache_path)}")
    return records


def load(cfg: dict, cache_path: Path) -> list:
    want = grid(cfg)
    records, _ = runner.load_cached(cache_path, want, exact=("selection_metric", "selection_mode"))
    models, sources, seeds = set(want["models"]), set(want["sources"]), set(want["seeds"])
    return [r for r in records if r.model_name in models and r.source in sources and r.seed in seeds]


def write_tables(cfg: dict, records: list, out_dir: Path) -> list:
    written = runner.write_table(build_status_table(records), out_dir, "status_table", index=True)

    seed_df = records_to_df(records).sort_values(["model", "source", "seed"]).reset_index(drop=True)
    written += runner.write_table(seed_df, out_dir, "seed_records")

    trials = (
        seed_df[seed_df["status"] == "complete"]
        .pivot_table(index=["model", "source"], columns="seed", values="n_tune_trials", aggfunc="first")
    )
    written += runner.write_table(trials, out_dir, "trial_counts", index=True)

    table_cfg = cfg.get("table", {})
    summary = build_sc_type_summary_table(
        records=records,
        models=cfg["models"],
        sources=cfg["sources"],
        metrics=table_cfg.get("metrics"),
        metric_labels=table_cfg.get("metric_labels"),
        seeds=cfg["seeds"],
    )
    written += runner.write_table(summary, out_dir, "summary_table", markdown=True)
    return written


def main():
    args = runner.arg_parser(__doc__, HERE).parse_args()
    cfg = runner.load_config(args.config)
    cfg.setdefault("plot_models", cfg["models"])
    cache_path = args.cache.resolve()
    records = scrape(cfg, cache_path) if args.rescrape else load(cfg, cache_path)

    n_complete = sum(r.status == "complete" for r in records)
    print(f"Cells: {len(records)} total, {n_complete} complete, {len(records) - n_complete} missing")

    out_dir = (args.out or HERE / cfg.get("output_dir", ".")).resolve()
    written = write_tables(cfg, records, out_dir / "tables")
    written += runner.write_figures(
        cfg, out_dir / "figures",
        resolve=lambda spec: (spec, {"records": records, "models": cfg["plot_models"], "seeds": cfg["seeds"]}),
    )
    runner.write_manifest(out_dir, args.config, cache_path, written)


if __name__ == "__main__":
    main()
