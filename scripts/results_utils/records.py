"""
records.py
==========
Pull Conn2Conn experiment results from W&B into per-cell RunRecords.

Architecture
------------
Two W&B run types are produced per (model, source, seed) experiment:

  1. Tune-trial runs  — one per Ray Tune trial, logged by
     TrialFamilyWandbLoggerCallback. Config is a FLAT dict; keys include
     "data.source" and "data.shuffle_seed" (dot-notation strings, not nested).
     Tags: [model_name, "tune", "ray_tune_id:{id}"].

  2. Best-trial run   — one per experiment, logged by report_best_tune_trial
     via _run_closed_form_single / _run_learned_single in "prod" mode.
     Config is a NESTED dict {"data": {...}, "model": {...}, "trainer": {...}}
     with top-level "ray_tune_id" / "ray_trial_id" added via config.update().
     Tags: [model_name, "prod", "best_trial_report",
            "source_trial:{trial_id}", "ray_tune_id:{id}"].
     W&B summary carries the logged metrics:
       val_demeaned_r, val_mse, val_pearson_r,
       train_demeaned_r, train_mse, train_pearson_r,
       eval_test/{metric_name}   (e.g. eval_test/demeaned_pearson)

The best-trial runs are the primary target of this module.

Also holds the shared constants for the results_utils package: repo/results paths,
W&B project/entity, and metric/model display vocabulary used by tables.py and
plots.py.

Public API
----------
  RunRecord                     — per-(model, source, seed) container
  wandb_api()                   → wandb.Api
  fetch_best_trial_runs(model)  → list of wandb Run objects
  fetch_direct_prod_runs(model) → list of wandb Run objects
  parse_run_record(run)         → RunRecord
  build_experiment_records(...) → list[RunRecord]
  build_experiment_records_covtype(...) → list[RunRecord] keyed by cov_type
  records_to_df(records)        → flat DataFrame
  save_records_cache(...)       -> persist scraped seed-level records
  load_records_cache(...)       -> load cached seed-level records
  enrich_records_with_local(records) -> merge local metrics_final.json
  load_local_artifact_df(records)    -> DataFrame of local Ray artifacts
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (W&B values mirror main.py)
# ---------------------------------------------------------------------------

WANDB_PROJECT = "conn2conn"
WANDB_ENTITY = "alexander-ratzan-new-york-university"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
RAY_RESULTS_DIR = RESULTS_ROOT / "ray_results"
RAY_CHECKPOINTS_DIR = RESULTS_ROOT / "ray_checkpoints"
LOCAL_RESULTS_DIR = RESULTS_ROOT / "local_results"

# W&B summary keys logged by the best-trial run (both model types).
_VAL_METRIC_KEYS = [
    "val_demeaned_r",
    "val_mse",
    "val_pearson_r",
    "train_demeaned_r",
    "train_mse",
    "train_pearson_r",
]
_TEST_PREFIX = "eval_test/"
_DISPLAY_METRIC_LABELS = {
    "pearson": "Correlation",
    "demeaned_pearson": "Demeaned Corr.",
    "mse": "Mean Squared Error",
    "avg_rank": "Average Rank",
    "top1_acc": "Top-1 Accuracy",
}
_DISPLAY_MODEL_LABELS = {
    "Krakencoder_precomputed": "Krakencoder",
}
_MODEL_PLOT_COLORS = {
    "CrossModalPCA": "#4C78A8",
    "CrossModal_PLS_SVD": "#1F77B4",
    "CrossModal_PCA_PLS": "#F28E2B",
    "CrossModal_PCA_PLS_learnable": "#D67C1C",
    "CrossModal_PCA_PLS_CovProjector": "#F4A259",
    "Krakencoder_precomputed": "#9467BD",
    "Sarwar2020MLP": "#B07AA1",
    "Chen2024GCN": "#59A14F",
    "NodalGNN": "#17BECF",
}

_METRIC_FEASIBLE_BOUNDS = {
    "pearson": (-1.0, 1.0),
    "demeaned_pearson": (-1.0, 1.0),
    "top1_acc": (0.0, 1.0),
    "avg_rank": (0.0, 1.0),
    "mse": (0.0, None),
}


def _display_model_label(model_name: str) -> str:
    return _DISPLAY_MODEL_LABELS.get(model_name, model_name)


def _model_plot_color(model_name: str) -> str:
    return _MODEL_PLOT_COLORS.get(model_name, "#4C78A8")


_METRIC_OPT_DIRECTION = {
    "pearson": "max",
    "demeaned_pearson": "max",
    "top1_acc": "max",
    "avg_rank": "max",
    "mse": "min",
}


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """One entry per (model_name, source, seed) cell."""
    model_name: str
    source: str
    seed: int
    # Optional subgroup label (e.g., cov_type for CovProjector experiments)
    cov_type: Optional[str] = None
    # "complete" → best-trial W&B run found and parsed
    # "missing"  → no W&B run exists for this cell
    status: str = "missing"

    # How many tune-sweep trials ran for this experiment (0 if missing).
    n_tune_trials: int = 0

    # W&B run metadata
    wandb_run_id: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_created_at: Optional[str] = None

    # Ray Tune linkage (from W&B config top-level keys added by report_best_tune_trial)
    ray_tune_id: Optional[str] = None
    ray_trial_id: Optional[str] = None

    # Full W&B config dict (nested: data / model / trainer + top-level metadata)
    config: dict = field(default_factory=dict)

    # Metrics from W&B summary
    val_metrics: dict = field(default_factory=dict)   # val_demeaned_r, etc.
    test_metrics: dict = field(default_factory=dict)  # after stripping eval_test/

    # Path to local ray_results/{model}/{trial_id}/ directory (may be None)
    local_artifact_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def wandb_api():
    import wandb
    return wandb.Api()


def _source_from_config(cfg: dict) -> str:
    """
    Extract the source modality from a W&B run config.

    Best-trial runs log a NESTED config {"data": {"source": ...}, ...}.
    Tune-trial runs log a FLAT config with key "data.source".
    Fall back to bare "source" key for ad-hoc runs.
    """
    # Nested (best-trial runs)
    data = cfg.get("data")
    if isinstance(data, dict) and "source" in data:
        return str(data["source"])
    # Flat with dot-notation (tune-trial runs / legacy)
    for key in ("data.source", "source"):
        val = cfg.get(key)
        if val is not None:
            return str(val)
    return "unknown"


def _seed_from_config(cfg: dict) -> int:
    """
    Extract shuffle_seed from a W&B run config (nested or flat).
    Returns -1 if not found.
    """
    # Nested
    data = cfg.get("data")
    if isinstance(data, dict) and "shuffle_seed" in data:
        try:
            return int(data["shuffle_seed"])
        except (TypeError, ValueError):
            pass
    # Flat
    for key in ("data.shuffle_seed", "shuffle_seed"):
        val = cfg.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return -1


def _normalize_cov_sources(cov_sources) -> tuple:
    if cov_sources is None:
        return tuple()
    if isinstance(cov_sources, str):
        cov_sources = cov_sources.strip()
        if cov_sources.startswith("[") and cov_sources.endswith("]"):
            try:
                parsed = ast.literal_eval(cov_sources)
                if isinstance(parsed, (list, tuple)):
                    cov_sources = parsed
                else:
                    cov_sources = [cov_sources]
            except (SyntaxError, ValueError):
                cov_sources = [x.strip() for x in cov_sources.strip("[]").split(",") if x.strip()]
        else:
            cov_sources = [x.strip() for x in cov_sources.split("+") if x.strip()]
    elif not isinstance(cov_sources, (list, tuple)):
        cov_sources = [cov_sources]
    return tuple(sorted(str(x) for x in cov_sources if x is not None))


def _cov_sources_from_config(cfg: dict) -> tuple:
    model = cfg.get("model")
    if isinstance(model, dict):
        cov_sources = model.get("cov_sources")
        if cov_sources is not None:
            return _normalize_cov_sources(cov_sources)

    for key in ("model.cov_sources", "cov_sources", "model.cov_sources_str"):
        if key in cfg:
            out = _normalize_cov_sources(cfg.get(key))
            if out:
                return out
    return tuple()


def cov_type_from_config(cfg: dict) -> Optional[str]:
    """Infer projector covariate condition label from W&B config."""
    cov_sources = _cov_sources_from_config(cfg)
    mapping = {
        ("fs_all",): "fs_all",
        ("fs_volumes",): "fs_volumes",
        ("age", "race_eth", "sex"): "demo",
        ("age", "fs_all", "race_eth", "sex"): "fs_all_demo",
    }
    if cov_sources in mapping:
        return mapping[cov_sources]
    if not cov_sources:
        return None
    return "+".join(cov_sources)


def _normalize_cov_type_value(cov_type):
    if cov_type in (None, "", "-", "unknown"):
        return None
    return cov_type


def cov_sources_str_from_config(cfg: dict) -> str:
    cov_sources = _cov_sources_from_config(cfg)
    return "+".join(cov_sources) if cov_sources else ""


def _extract_val_metrics(summary: dict) -> dict:
    return {k: float(summary[k]) for k in _VAL_METRIC_KEYS if k in summary}


def _extract_test_metrics(summary: dict) -> dict:
    """Strip 'eval_test/' prefix; e.g. 'eval_test/demeaned_pearson' → 'demeaned_pearson'."""
    result = {}
    for k, v in summary.items():
        if k.startswith(_TEST_PREFIX):
            try:
                result[k[len(_TEST_PREFIX):]] = float(v)
            except (TypeError, ValueError):
                pass
    return result


def _metric_for_selection(summary: dict, metric: str) -> Optional[float]:
    """Read a selection metric from W&B summary, returning None if unavailable."""
    val = summary.get(metric)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _select_best_run(
    runs_for_key: list,
    selection_metric: str = "val_demeaned_r",
    selection_mode: str = "max",
):
    """
    Choose the winning W&B run among duplicates for the same experiment cell.

    Priority:
    1. valid selection metric beats missing
    2. better metric wins (max or min)
    3. newer created_at wins
    """
    if selection_mode not in {"max", "min"}:
        raise ValueError(f"selection_mode must be 'max' or 'min', got {selection_mode!r}")

    def _sort_key(run):
        summary = dict(getattr(run, "summary", {}) or {})
        metric_val = _metric_for_selection(summary, selection_metric)
        has_metric = metric_val is not None
        metric_sort = metric_val if metric_val is not None else (-np.inf if selection_mode == "max" else np.inf)
        created_at = getattr(run, "created_at", "") or ""
        if selection_mode == "max":
            return (has_metric, metric_sort, created_at)
        return (has_metric, -metric_sort, created_at)

    return max(runs_for_key, key=_sort_key)


def _extract_ray_tune_id_from_tags(tags: list) -> Optional[str]:
    for t in tags:
        m = re.match(r"ray_tune_id:(\d+)", str(t))
        if m:
            return m.group(1)
    return None



def _source_seed_from_local_checkpoint(
    ray_tune_id: Optional[str],
    ray_trial_id: Optional[str],
    model_name: Optional[str],
) -> tuple:
    """
    Fallback: read source and seed from the local tune-trial config.json.

    Each tune trial writes its full flat config (including data.source and
    data.shuffle_seed) to:
        ray_checkpoints/{model}_tune_{ray_tune_id}/{ray_trial_id}/final/config.json

    This file is created by _write_final_artifact() inside train_func in main.py.
    It is the only reliable source for pre-fix W&B runs that lack source/seed in
    their W&B config.

    Returns ("unknown", -1) if the file cannot be found or parsed.
    """
    if not (ray_tune_id and ray_trial_id and model_name):
        return "unknown", -1

    run_dir_name = f"{model_name}_tune_{ray_tune_id}"
    config_path = RAY_CHECKPOINTS_DIR / run_dir_name / ray_trial_id / "final" / "config.json"
    if not config_path.exists():
        return "unknown", -1

    try:
        cfg_local = json.loads(config_path.read_text())
        # _flat_to_nested stores data keys under cfg["data"]
        data = cfg_local.get("data", {})
        source = str(data.get("source", "unknown"))
        seed_raw = data.get("shuffle_seed", -1)
        return source, int(seed_raw)
    except Exception:
        return "unknown", -1


# ---------------------------------------------------------------------------
# Per-run fetchers
# ---------------------------------------------------------------------------

def fetch_best_trial_runs(
    model_name: str,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
) -> list:
    """
    Fetch all W&B runs that are best-trial prod reports for `model_name`.

    These runs are produced by report_best_tune_trial() when called with
    report_to_wandb=True (i.e. --report_best_after_tune flag in sbatch).
    They are tagged with both 'best_trial_report' and model_name.
    """
    api = wandb_api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"$and": [
            {"tags": "best_trial_report"},
            {"tags": model_name},
        ]},
        order="-created_at",
    )
    return list(runs)


def fetch_direct_prod_runs(
    model_name: str,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
) -> list:
    """
    Fetch direct prod runs for `model_name` that were not logged as
    best-trial reports.

    This supports lightweight notebook / prod logging workflows where a model
    is evaluated directly over seeds without the report_best_tune_trial path.
    """
    api = wandb_api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"$and": [
            {"tags": "prod"},
            {"tags": model_name},
        ]},
        order="-created_at",
    )
    return [run for run in runs if "best_trial_report" not in list(getattr(run, "tags", []) or [])]


def count_tune_trials_for_run(
    run,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
) -> int:
    """
    Count how many tune-trial W&B runs belong to the same Ray Tune sweep as `run`.

    Every tune trial is tagged "ray_tune_id:{id}" by TrialFamilyWandbLoggerCallback
    (via the `tags` arg in run_tune). Best-trial runs usually store `ray_tune_id` in
    config, but we also fall back to extracting it from run tags for robustness.
    We filter by both the ray_tune_id tag and the "tune" tag to isolate per-trial
    runs (excluding the best-trial report itself).

    NOTE: The group-based approach is incorrect because each trial is assigned its own
    unique group "{model_name}_tune_{trial_family}", so no two trials share a group.
    """
    cfg = getattr(run, "config", {}) or {}
    tags = list(getattr(run, "tags", []) or [])
    ray_tune_id = cfg.get("ray_tune_id") or _extract_ray_tune_id_from_tags(tags)
    if not ray_tune_id:
        return 0

    api = wandb_api()
    trial_runs = api.runs(
        f"{entity}/{project}",
        filters={"$and": [{"tags": f"ray_tune_id:{ray_tune_id}"}, {"tags": "tune"}]},
    )
    return len(list(trial_runs))


def parse_run_record(
    wandb_run,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    count_trials: bool = True,
) -> RunRecord:
    """
    Parse a single W&B best-trial run into a RunRecord.

    The W&B run config for best-trial runs is structured as:
        {
            "data":    {"source": str, "shuffle_seed": int, "target": str, ...},
            "model":   {...hyperparams...},
            "trainer": {...},
            "ray_tune_id":  str  (top-level, added via wandb.config.update),
            "ray_trial_id": str  (top-level),
        }

    Parameters
    ----------
    count_trials : bool
        Issue an extra W&B API call to count how many tune trials ran.
        Set False for faster bulk parsing.
    """
    cfg = dict(wandb_run.config)
    summary = dict(wandb_run.summary)
    tags = list(wandb_run.tags)

    # ray_tune_id / ray_trial_id are at the top level of the config (added
    # via wandb.config.update(wandb_metadata) in report_best_tune_trial).
    ray_tune_id = cfg.get("ray_tune_id") or _extract_ray_tune_id_from_tags(tags)
    ray_trial_id = cfg.get("ray_trial_id")

    # Model name from tags (the tag that isn't a known system tag).
    _SYSTEM_TAGS = {"best_trial_report", "prod", "tune"}
    _SKIP_PREFIXES = ("ray_tune_id:", "source_trial:")
    model_name_from_tag = next(
        (
            t for t in tags
            if t not in _SYSTEM_TAGS
            and not any(t.startswith(p) for p in _SKIP_PREFIXES)
        ),
        None,
    )

    # Extract source and seed.
    # Priority:
    #   1. W&B config top-level scalars "source" / "shuffle_seed"  (runs after main.py fix)
    #   2. Nested cfg["data"]["source"] / cfg["data"]["shuffle_seed"]  (if data section present)
    #   3. Flat dot-notation keys "data.source" / "data.shuffle_seed"
    #   4. Local checkpoint config.json  (fallback for pre-fix runs)
    source = _source_from_config(cfg)
    seed = _seed_from_config(cfg)
    if source == "unknown" or seed == -1:
        local_src, local_seed = _source_seed_from_local_checkpoint(
            ray_tune_id, ray_trial_id, model_name_from_tag
        )
        if source == "unknown" and local_src != "unknown":
            source = local_src
        if seed == -1 and local_seed != -1:
            seed = local_seed

    n_tune_trials = 0
    if count_trials:
        n_tune_trials = count_tune_trials_for_run(wandb_run, project=project, entity=entity)

    # Resolve local artifact path
    local_path = None
    if ray_trial_id and model_name_from_tag:
        candidate = RAY_RESULTS_DIR / model_name_from_tag / ray_trial_id
        if candidate.exists():
            local_path = str(candidate)

    return RunRecord(
        model_name=model_name_from_tag or "unknown",
        source=source,
        seed=seed,
        cov_type=cov_type_from_config(cfg),
        status="complete",
        n_tune_trials=n_tune_trials,
        wandb_run_id=wandb_run.id,
        wandb_run_name=wandb_run.name,
        wandb_group=getattr(wandb_run, "group", None),
        wandb_created_at=getattr(wandb_run, "created_at", None),
        ray_tune_id=ray_tune_id,
        ray_trial_id=ray_trial_id,
        config=cfg,
        val_metrics=_extract_val_metrics(summary),
        test_metrics=_extract_test_metrics(summary),
        local_artifact_path=local_path,
    )


# ---------------------------------------------------------------------------
# Experiment-level aggregator
# ---------------------------------------------------------------------------

def build_experiment_records(
    models: list,
    sources: list,
    seeds: list,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    count_trials: bool = True,
    verbose: bool = True,
    selection_metric: str = "val_demeaned_r",
    selection_mode: str = "max",
    direct_prod_models: Optional[list] = None,
) -> list:
    """
    Fetch all best-trial runs for each model, cross-join against the expected
    (model × source × seed) matrix, and return a complete list of RunRecords.

    Cells with no matching W&B run are filled with status="missing".
    When multiple runs exist for the same cell (e.g. reruns), the run with the
    best validation metric is kept.

    Parameters
    ----------
    models : list[str]
        Model names — must match W&B tags set by main.py.
    sources : list[str]
        Source modalities to include (e.g. ["SC", "SC_r2t", "SC+SC_r2t"]).
    seeds : list[int]
        Shuffle seeds to include.
    count_trials : bool
        Issue an extra W&B API call per run to count tune-sweep trials.
    verbose : bool
        Print a status line for each cell.
    direct_prod_models : list[str] | None
        Models to fetch from direct `prod` runs instead of
        `best_trial_report` runs.
    """
    complete: dict = {}  # (model, source, seed) → RunRecord
    direct_prod_models = set(direct_prod_models or [])

    for model_name in models:
        if verbose:
            print(f"Fetching W&B runs for {model_name}…", flush=True)

        use_direct_prod = model_name in direct_prod_models
        wandb_runs = (
            fetch_direct_prod_runs(model_name, project=project, entity=entity)
            if use_direct_prod
            else fetch_best_trial_runs(model_name, project=project, entity=entity)
        )

        if verbose and not wandb_runs:
            run_type = "direct 'prod'" if use_direct_prod else "'best_trial_report'"
            print(f"  (no {run_type} runs found for {model_name})", flush=True)

        # Group by (model, source, seed), keeping only runs whose source/seed
        # match what we asked for.
        by_key: dict = {}
        for run in wandb_runs:
            cfg = dict(run.config)
            src = _source_from_config(cfg)
            seed = _seed_from_config(cfg)

            if verbose and (src == "unknown" or seed == -1):
                print(
                    f"  ⚠ run {run.name} ({run.id}): "
                    f"could not parse source='{src}' seed={seed} — skipping",
                    flush=True,
                )
                continue

            if src not in sources or seed not in seeds:
                continue

            key = (model_name, src, seed)
            by_key.setdefault(key, []).append(run)

        for key, runs_for_key in by_key.items():
            winner = _select_best_run(
                runs_for_key,
                selection_metric=selection_metric,
                selection_mode=selection_mode,
            )
            record = parse_run_record(
                winner, project=project, entity=entity, count_trials=count_trials
            )
            complete[key] = record
            if verbose:
                print(
                    f"  ✓ {key[0]} | src={key[1]} | seed={key[2]} "
                    f"→ {record.n_tune_trials} tune trials  "
                    f"(run: {record.wandb_run_name}; selected by {selection_metric})",
                    flush=True,
                )

    # Fill every expected cell; mark absent ones as missing.
    all_records: list = []
    for model_name in models:
        for source in sources:
            for seed in seeds:
                key = (model_name, source, seed)
                if key in complete:
                    all_records.append(complete[key])
                else:
                    if verbose:
                        print(
                            f"  ✗ MISSING  {model_name} | src={source} | seed={seed}",
                            flush=True,
                        )
                    all_records.append(RunRecord(
                        model_name=model_name,
                        source=source,
                        seed=seed,
                        status="missing",
                    ))

    return all_records


def build_experiment_records_covtype(
    models: list,
    sources: list,
    seeds: list,
    cov_types: list,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    count_trials: bool = True,
    verbose: bool = True,
    selection_metric: str = "val_demeaned_r",
    selection_mode: str = "max",
) -> list:
    """
    CovType-aware variant of build_experiment_records.

    Keys records by (model, source, cov_type, seed) so multiple covariate
    conditions for the same source/seed are not collapsed.
    """
    complete: dict = {}  # (model, source, cov_type, seed) -> RunRecord

    for model_name in models:
        if verbose:
            print(f"Fetching W&B runs for {model_name}…", flush=True)

        wandb_runs = fetch_best_trial_runs(model_name, project=project, entity=entity)

        if verbose and not wandb_runs:
            print(f"  (no 'best_trial_report' runs found for {model_name})", flush=True)

        by_key: dict = {}
        for run in wandb_runs:
            cfg = dict(run.config)
            src = _source_from_config(cfg)
            seed = _seed_from_config(cfg)
            cov_type = cov_type_from_config(cfg)

            if verbose and (src == "unknown" or seed == -1):
                print(
                    f"  ⚠ run {run.name} ({run.id}): "
                    f"could not parse source='{src}' seed={seed} — skipping",
                    flush=True,
                )
                continue

            if src not in sources or seed not in seeds or cov_type not in cov_types:
                continue

            key = (model_name, src, cov_type, seed)
            by_key.setdefault(key, []).append(run)

        for key, runs_for_key in by_key.items():
            latest = _select_best_run(
                runs_for_key,
                selection_metric=selection_metric,
                selection_mode=selection_mode,
            )
            record = parse_run_record(
                latest, project=project, entity=entity, count_trials=count_trials
            )
            record.cov_type = key[2]
            complete[key] = record
            if verbose:
                print(
                    f"  ✓ {key[0]} | src={key[1]} | cov={key[2]} | seed={key[3]} "
                    f"→ {record.n_tune_trials} tune trials  "
                    f"(run: {record.wandb_run_name})",
                    flush=True,
                )

    all_records: list = []
    for model_name in models:
        for source in sources:
            for cov_type in cov_types:
                for seed in seeds:
                    key = (model_name, source, cov_type, seed)
                    if key in complete:
                        all_records.append(complete[key])
                    else:
                        if verbose:
                            print(
                                f"  ✗ MISSING  {model_name} | src={source} | cov={cov_type} | seed={seed}",
                                flush=True,
                            )
                        all_records.append(
                            RunRecord(
                                model_name=model_name,
                                source=source,
                                seed=seed,
                                cov_type=cov_type,
                                status="missing",
                            )
                        )

    return all_records


def records_to_df(records: list) -> pd.DataFrame:
    """
    Convert RunRecords to a flat DataFrame.
    Test metrics are prefixed with 'test_'; val metrics keep their names.
    """
    rows = []
    for r in records:
        row = {
            "model":          r.model_name,
            "source":         r.source,
            "cov_type":       _normalize_cov_type_value(r.cov_type),
            "seed":           r.seed,
            "status":         r.status,
            "n_tune_trials":  r.n_tune_trials,
            "wandb_run_id":   r.wandb_run_id,
            "wandb_run_name": r.wandb_run_name,
            "wandb_group":    r.wandb_group,
            "ray_tune_id":    r.ray_tune_id,
            "ray_trial_id":   r.ray_trial_id,
            "local_artifact": r.local_artifact_path,
        }
        row.update(r.val_metrics)
        for k, v in r.test_metrics.items():
            row[f"test_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def save_records_cache(records: list, cache_path: str | Path, metadata: Optional[dict] = None) -> Path:
    """
    Save seed-level scraped records to a JSON cache file.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload_records = []
    for r in records:
        row = asdict(r)
        row["cov_type"] = _normalize_cov_type_value(row.get("cov_type"))
        payload_records.append(row)
    payload = {
        "metadata": metadata or {},
        "records": payload_records,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return cache_path


def load_records_cache(cache_path: str | Path) -> tuple[list, dict]:
    """
    Load seed-level scraped records from a JSON cache file.

    Returns
    -------
    (records, metadata)
    """
    cache_path = Path(cache_path)
    payload = json.loads(cache_path.read_text())
    records = []
    for row in payload.get("records", []):
        row = dict(row)
        row["cov_type"] = _normalize_cov_type_value(row.get("cov_type"))
        records.append(RunRecord(**row))
    metadata = payload.get("metadata", {})
    return records, metadata



# ---------------------------------------------------------------------------
# Local artifact integration (wired in later)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def enrich_records_with_local(records: list) -> list:
    """
    For complete records that have a local artifact path, load
    metrics_final.json and merge richer metrics into test_metrics.

    W&B summary scalars take precedence for base eval_test/* keys;
    local metrics fill in identifiability / PCA fields not logged to W&B.
    """

    def _flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            full_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, prefix=f"{full_key}_"))
            else:
                out[full_key] = v
        return out

    enriched = []
    for r in records:
        if r.status == "complete" and r.local_artifact_path:
            metrics_path = Path(r.local_artifact_path) / "metrics_final.json"
            full = _load_json(metrics_path)
            if full:
                flat_test = _flatten(full.get("test", {}))
                # W&B values take precedence; local fills the rest
                r.test_metrics = {**flat_test, **r.test_metrics}
        enriched.append(r)
    return enriched


def load_local_artifact_df(records: list) -> pd.DataFrame:
    """
    Load full metrics_final.json for every complete record with a local
    artifact path, returning a DataFrame with one row per run.
    """
    def _flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            full_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, prefix=f"{full_key}_"))
            else:
                out[full_key] = v
        return out

    rows = []
    for r in records:
        if r.status != "complete" or not r.local_artifact_path:
            continue
        base = Path(r.local_artifact_path)
        full = _load_json(base / "metrics_final.json")
        cfg = _load_json(base / "config.json")
        row = {
            "model":        r.model_name,
            "source":       r.source,
            "seed":         r.seed,
            "ray_trial_id": r.ray_trial_id,
            "artifact_dir": r.local_artifact_path,
        }
        row.update(_flatten(cfg.get("model",   {}), prefix="model_"))
        row.update(_flatten(cfg.get("trainer", {}), prefix="trainer_"))
        row.update(_flatten(full.get("train",  {}), prefix="train_"))
        row.update(_flatten(full.get("val",    {}), prefix="val_"))
        row.update(_flatten(full.get("test",   {}), prefix="test_"))
        rows.append(row)
    return pd.DataFrame(rows)
