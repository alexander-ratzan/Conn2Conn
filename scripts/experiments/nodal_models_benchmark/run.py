"""
Nodal models benchmark (spec v2 E0): scrape (or load) NodalMLP / NodalGNN tune trials and best-trial runs,
add reference rows from the tracked benchmark snapshots, then write tables and figures.

    python scripts/experiments/nodal_models_benchmark/run.py              # from records.json + trials.json
    python scripts/experiments/nodal_models_benchmark/run.py --rescrape   # refresh both from W&B

records.json holds one best-trial RunRecord per (variant, seed) (cov_type = variant name); trials.json holds the
tune trials (params + val metric) used for hyperparameter importance. Both are tracked snapshots; everything in
tables/ and figures/ is regenerated from them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.results_utils import (  # noqa: E402
    build_cov_dl_seed_df,
    build_cov_dl_summary_table,
    fetch_best_trial_runs,
    load_records_cache,
    parse_run_record,
    records_to_df,
    save_records_cache,
)
from scripts.results_utils import runner  # noqa: E402
from scripts.results_utils.optuna_importance import (  # noqa: E402
    fetch_tune_trial_runs,
    get_importance,
    rows_to_frozen_trials,
    study_from_trials,
    yaml_search_space_to_distributions,
)


# ---------------------------------------------------------------------------
# Variant membership
# ---------------------------------------------------------------------------

def _get(cfg: dict, key: str):
    """Config value from a flat (tune trial) or nested (best-trial) W&B config."""
    for k in (key, f"model.{key}", f"data.{key}", f"trainer.{key}"):
        if k in cfg:
            return cfg[k]
    for section in ("model", "data", "trainer"):
        if isinstance(cfg.get(section), dict) and key in cfg[section]:
            return cfg[section][key]
    return None


def _seed(cfg: dict):
    v = _get(cfg, "shuffle_seed")
    return int(v) if v is not None else None


def _allowed_values(variant: dict) -> dict:
    ycfg = yaml.safe_load((REPO_ROOT / variant["config"]).read_text())
    ss, default_model = ycfg.get("search_space", {}), ycfg["default"]["model"]
    allowed = {}
    for key in variant.get("options_from_config", []):
        allowed[key] = set(ss[key]["values"]) if key in ss else {default_model.get(key)}
    return allowed


def _matches(variant: dict, allowed: dict, cfg: dict, created: str) -> bool:
    if variant.get("created_after") and created < variant["created_after"]:
        return False
    if variant.get("created_before") and created >= variant["created_before"]:
        return False
    if any(_get(cfg, k) not in vals for k, vals in allowed.items()):
        return False
    if any(_get(cfg, k) != v for k, v in (variant.get("match") or {}).items()):
        return False
    return all(_get(cfg, k) is None for k in variant.get("require_missing", []))


def grid(cfg: dict) -> dict:
    return {
        "variants": [v["name"] for v in cfg["variants"]],
        "seeds": list(cfg["seeds"]),
        "selection_metric": cfg["selection"]["metric"],
        "selection_mode": cfg["selection"]["mode"],
        "variant_filters": {v["name"]: {k: v[k] for k in sorted(v) if k != "name"} for v in cfg["variants"]},
    }


# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------

def scrape(cfg: dict, records_path: Path, trials_path: Path) -> tuple[list, list]:
    metric = cfg["selection"]["metric"]
    seeds = set(cfg["seeds"])
    trials, best = [], []
    by_model: dict = {}
    for v in cfg["variants"]:
        by_model.setdefault(v["model"], []).append(v)
    for model, variants in by_model.items():
        allowed = {v["name"]: _allowed_values(v) for v in variants}
        starts = [v.get("created_after") for v in variants if v.get("importance")]
        since = min(starts) if starts and all(starts) else None
        print(f"Scraping W&B tune trials for {model} (since {since})...", flush=True)
        for r in (fetch_tune_trial_runs(model, since=since) if starts else []):
            c, s, created = dict(r.config), dict(r.summary), str(r.created_at)
            for v in variants:
                if v.get("importance") and _matches(v, allowed[v["name"]], c, created) and _seed(c) in seeds:
                    ss = yaml.safe_load((REPO_ROOT / v["config"]).read_text())["search_space"]
                    keys = set(ss) | set(cfg.get("param_aliases", {}))
                    trials.append({
                        "variant": v["name"], "seed": _seed(c), "value": s.get(metric), "run_id": r.id,
                        "run_name": r.name, "created": created, "loss_signature": c.get("loss_signature"),
                        "config": {k: c[k] for k in sorted(keys) if k in c},
                    })
        print(f"Scraping W&B best-trial runs for {model}...", flush=True)
        cells: dict = {}
        for r in fetch_best_trial_runs(model):
            c, created = dict(r.config), str(r.created_at)
            for v in variants:
                if _matches(v, allowed[v["name"]], c, created) and _seed(c) in seeds:
                    cells.setdefault((v["name"], _seed(c)), []).append(r)
        for (name, seed), runs in sorted(cells.items()):
            sign = 1 if cfg["selection"]["mode"] == "max" else -1
            runs = [r for r in runs if dict(r.summary).get(metric) is not None]
            if not runs:
                continue
            winner = max(runs, key=lambda r: (sign * float(dict(r.summary)[metric]), str(r.created_at)))
            rec = parse_run_record(winner, count_trials=False, model_name=model)
            rec.cov_type = name
            best.append(rec)
            print(f"  {name} seed {seed}: {winner.id} ({len(runs)} candidate runs)", flush=True)
    meta = runner.scrape_metadata(grid(cfg))
    save_records_cache(best, records_path, metadata=meta)
    trials_path.write_text(json.dumps({"metadata": meta, "trials": trials}, indent=1))
    print(f"Saved {len(best)} best-trial records and {len(trials)} tune trials")
    return best, trials


def load(cfg: dict, records_path: Path, trials_path: Path) -> tuple[list, list]:
    records, _ = runner.load_cached(records_path, grid(cfg),
                                    exact=("selection_metric", "selection_mode", "variant_filters"))
    if not trials_path.exists():
        sys.exit(f"No trials cache at {trials_path}. Run with --rescrape.")
    trials = json.loads(trials_path.read_text())["trials"]
    return records, trials


def reference_records(cfg: dict) -> list:
    seeds = set(cfg["seeds"])
    out = []
    for ref in cfg.get("references", []):
        recs, _ = load_records_cache(REPO_ROOT / ref["snapshot"])
        keep = [r for r in recs if r.model_name == ref["model"] and r.source == ref["source"]
                and r.seed in seeds and r.status == "complete" and r.cov_type in (None, "", "-")]
        print(f"  reference {ref['name']}: {len(keep)} seed(s) from {ref['snapshot']}")
        out += keep
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def importance_table(cfg: dict, trials: list) -> pd.DataFrame:
    rows = []
    for v in cfg["variants"]:
        if not v.get("importance"):
            continue
        ss = yaml.safe_load((REPO_ROOT / v["config"]).read_text())["search_space"]
        dists = yaml_search_space_to_distributions(ss)
        vt = [t for t in trials if t["variant"] == v["name"] and t["value"] is not None]
        if cfg.get("importance_center") == "seed":
            # Each split sets its own score level (trials on one seed often tie); centre so fANOVA sees
            # within-seed hyperparameter effects rather than between-split variance.
            med = pd.DataFrame(vt).groupby("seed")["value"].median().to_dict() if vt else {}
            vt = [{**t, "value": t["value"] - med[t["seed"]]} for t in vt]
        frozen, diag = rows_to_frozen_trials(vt, dists, verbose=False, aliases=cfg.get("param_aliases"))
        if len(frozen) < 2:
            continue
        imp = get_importance(study_from_trials(frozen, direction="maximize"), seed=cfg.get("importance_seed", 0))
        for param, value in imp.items():
            rows.append({"variant": v["name"], "param": param, "importance": float(value),
                         "n_trials": len(frozen), "n_dropped": diag["n_total"] - diag["n_kept"]})
    return pd.DataFrame(rows)


def trial_summary(cfg: dict, trials_df: pd.DataFrame) -> pd.DataFrame:
    if trials_df.empty:
        return trials_df
    g = trials_df.groupby("variant")
    out = pd.DataFrame({
        "n_trials": g.size(),
        "seeds": g["seed"].apply(lambda s: ",".join(str(x) for x in sorted(set(s)))),
        "best_val": g["value"].max(),
        "median_val": g["value"].median(),
        "with_loss_signature": g["loss_signature"].apply(lambda s: int(s.notna().sum())),
    }).reset_index()
    order = [v["name"] for v in cfg["variants"]]
    return out.sort_values("variant", key=lambda s: s.map({n: i for i, n in enumerate(order)}))


def row_group(cfg: dict, name: str) -> list:
    return [dict(cfg["rows"][k]) for k in cfg["row_groups"][name]]


def main():
    args = runner.arg_parser(__doc__, HERE).parse_args()
    cfg = runner.load_config(args.config)
    records_path = args.cache.resolve()
    trials_path = records_path.with_name("trials.json")
    best, trials = scrape(cfg, records_path, trials_path) if args.rescrape else load(cfg, records_path, trials_path)
    records = best + reference_records(cfg)
    print(f"Records: {len(best)} variant best-trials + {len(records) - len(best)} reference; {len(trials)} tune trials")

    out_dir = (args.out or HERE / cfg.get("output_dir", ".")).resolve()
    tables_dir = out_dir / "tables"
    metrics = cfg.get("tables", {}).get("metrics")

    trials_df = pd.DataFrame(trials)
    written = runner.write_table(trial_summary(cfg, trials_df), tables_dir, "trial_summary", markdown=True)
    imp = importance_table(cfg, trials)
    written += runner.write_table(imp, tables_dir, "importance")
    if not imp.empty:
        wide = imp.pivot_table(index="param", columns="variant", values="importance").round(3).reset_index()
        written += runner.write_table(wide, tables_dir, "importance_wide", markdown=True)
    written += runner.write_table(
        records_to_df(records).sort_values(["model", "cov_type", "seed"], na_position="first"), tables_dir, "seed_records")
    seed_df = build_cov_dl_seed_df(records=records, row_specs=[dict(r) for r in cfg["rows"].values()],
                                   metrics=metrics, seeds=cfg["seeds"])
    written += runner.write_table(seed_df, tables_dir, "row_seed_metrics")
    for summary in cfg.get("tables", {}).get("summaries", []):
        table = build_cov_dl_summary_table(seed_df=seed_df, row_specs=row_group(cfg, summary["rows"]),
                                           metrics=metrics, metric_labels=cfg["tables"].get("metric_labels"))
        written += runner.write_table(table, tables_dir, summary["name"], markdown=True)

    def resolve(spec):
        spec = dict(spec)
        if spec["type"] == "importance_heatmap":
            return spec, {"importance_df": imp}
        if spec["type"] == "trial_distribution":
            return spec, {"trials_df": trials_df}
        return spec, {"seed_df": seed_df, "row_specs": row_group(cfg, spec.pop("rows"))}

    written += runner.write_figures(cfg, out_dir / "figures", resolve=resolve)
    runner.write_manifest(out_dir, args.config, records_path, written)


if __name__ == "__main__":
    main()
