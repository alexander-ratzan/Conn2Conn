"""
Covariate-projector + deep-model benchmark: scrape (or load) best-trial W&B results for a set of
labelled conditions (config `rows`), then write summary tables and figures per row group.

    python scripts/experiments/cov_projector_benchmark/run.py              # from the tracked records.json
    python scripts/experiments/cov_projector_benchmark/run.py --rescrape   # refresh records.json from W&B

Everything configurable lives in config.yml next to this file. records.json is the tracked
snapshot of the most recent scrape; tables/, figures/ and manifest.json are written next to it.
Every plotted value is recomputable from tables/row_seed_metrics.csv.
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
    build_cov_dl_seed_df,
    build_cov_dl_summary_table,
    build_covtype_status_table,
    build_experiment_records,
    build_experiment_records_covtype,
    build_status_table,
    records_to_df,
    save_records_cache,
)
from scripts.results_utils import runner  # noqa: E402

ROW_KEYS = {"model", "source", "cov_type", "display_model", "sc_source_label", "cov_source_label", "plot_label"}


def validate_rows(cfg: dict) -> None:
    seen = {}
    for key, row in cfg["rows"].items():
        unknown = sorted(set(row) - ROW_KEYS)
        if unknown or "model" not in row or "source" not in row:
            sys.exit(f"rows.{key}: needs model + source; unknown keys {unknown}; allowed {sorted(ROW_KEYS)}")
        cond = (row["model"], row["source"], row.get("cov_type"))
        if cond in seen:  # duplicates would double-count seeds in summaries
            sys.exit(f"rows.{key} repeats the condition of rows.{seen[cond]}: {cond}")
        seen[cond] = key


def row_group(cfg: dict, name: str) -> list:
    """Resolve a row group into row specs (entries are row keys or {row: key, ...overrides})."""
    if name not in cfg["row_groups"]:
        sys.exit(f"Unknown row group {name!r}; defined: {sorted(cfg['row_groups'])}")
    specs = []
    for entry in cfg["row_groups"][name]:
        overrides = dict(entry) if isinstance(entry, dict) else {"row": entry}
        key = overrides.pop("row")
        if key not in cfg["rows"]:
            sys.exit(f"row_groups.{name}: unknown row {key!r}")
        specs.append({**cfg["rows"][key], **overrides})
    return specs


def grid(cfg: dict) -> dict:
    """Scrape grid derived from `rows`; also the cache-coverage contract checked on load."""
    rows = cfg["rows"].values()
    standard = [r for r in rows if r.get("cov_type") is None]
    projector = [r for r in rows if r.get("cov_type") is not None]
    return {
        "standard_models": sorted({r["model"] for r in standard}),
        "standard_sources": sorted({r["source"] for r in standard}),
        "projector_models": sorted({r["model"] for r in projector}),
        "projector_sources": sorted({r["source"] for r in projector}),
        "cov_types": sorted({r["cov_type"] for r in projector}),
        "seeds": list(cfg["seeds"]),
        "selection_metric": cfg["selection"]["metric"],
        "selection_mode": cfg["selection"]["mode"],
        "direct_prod_models": sorted(cfg.get("direct_prod_models") or []),
    }


def scrape(cfg: dict, cache_path: Path) -> list:
    g = grid(cfg)
    common = dict(seeds=g["seeds"], count_trials=bool(cfg.get("count_trials", True)), verbose=True,
                  selection_metric=g["selection_metric"], selection_mode=g["selection_mode"])
    print("Scraping W&B: standard models...", flush=True)
    records = build_experiment_records(models=g["standard_models"], sources=g["standard_sources"],
                                       direct_prod_models=g["direct_prod_models"] or None, **common)
    if g["projector_models"]:
        print("Scraping W&B: projector covariate conditions...", flush=True)
        records += build_experiment_records_covtype(models=g["projector_models"], sources=g["projector_sources"],
                                                    cov_types=g["cov_types"], **common)
    save_records_cache(records, cache_path, metadata=runner.scrape_metadata(g))
    print(f"Saved {len(records)} records to {runner.rel(cache_path)}")
    return records


def load(cfg: dict, cache_path: Path) -> list:
    g = grid(cfg)
    records, _ = runner.load_cached(cache_path, g, exact=("selection_metric", "selection_mode", "direct_prod_models"))
    keep_std = {(m, s) for m in g["standard_models"] for s in g["standard_sources"]}
    keep_proj = {(m, s, c) for m in g["projector_models"] for s in g["projector_sources"] for c in g["cov_types"]}
    return [r for r in records if r.seed in g["seeds"] and (
        (r.cov_type is None and (r.model_name, r.source) in keep_std)
        or (r.model_name, r.source, r.cov_type) in keep_proj)]


def main():
    args = runner.arg_parser(__doc__, HERE).parse_args()
    cfg = runner.load_config(args.config)
    validate_rows(cfg)
    cache_path = args.cache.resolve()
    records = scrape(cfg, cache_path) if args.rescrape else load(cfg, cache_path)

    n_complete = sum(r.status == "complete" for r in records)
    print(f"Cells: {len(records)} total, {n_complete} complete, {len(records) - n_complete} missing")

    out_dir = (args.out or HERE / cfg.get("output_dir", ".")).resolve()
    tables_dir = out_dir / "tables"
    table_cfg = cfg.get("tables", {})
    metrics = table_cfg.get("metrics")

    standard = [r for r in records if r.cov_type is None]
    projector = [r for r in records if r.cov_type is not None]
    written = runner.write_table(build_status_table(standard), tables_dir, "status_standard", index=True)
    if projector:
        written += runner.write_table(build_covtype_status_table(projector), tables_dir, "status_projector", index=True)
    written += runner.write_table(
        records_to_df(records).sort_values(["model", "source", "cov_type", "seed"], na_position="first"),
        tables_dir, "seed_records",
    )

    # one row per (defined condition, seed): the data behind every summary table and figure
    all_rows = [dict(row) for row in cfg["rows"].values()]
    seed_df = build_cov_dl_seed_df(records=records, row_specs=all_rows, metrics=metrics, seeds=cfg["seeds"])
    written += runner.write_table(seed_df, tables_dir, "row_seed_metrics")

    for summary in table_cfg.get("summaries", []):
        table = build_cov_dl_summary_table(
            seed_df=seed_df, row_specs=row_group(cfg, summary["rows"]), metrics=metrics,
            metric_labels=table_cfg.get("metric_labels"),
        )
        written += runner.write_table(table, tables_dir, summary["name"], markdown=True)

    def resolve(spec):
        spec = dict(spec)
        group = spec.pop("rows", None)
        if group is None:
            sys.exit(f"Figure {spec.get('type')} needs a `rows` group")
        return spec, {"seed_df": seed_df, "row_specs": row_group(cfg, group)}

    written += runner.write_figures(cfg, out_dir / "figures", resolve=resolve)
    runner.write_manifest(out_dir, args.config, cache_path, written)


if __name__ == "__main__":
    main()
