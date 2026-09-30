"""
optuna_importance.py
====================
Compute hparam importance for a Conn2Conn model by reconstructing an Optuna
study from existing W&B tune-trial runs (no reruns needed).

Pulls tune trials tagged with the model class name, optionally filters by
config keys (e.g. decoder_type=dot to narrow to one probe variant), maps each
trial's flat config to an `optuna.FrozenTrial` against the YAML's declared
search space, replays them into a fresh study, and prints the importance
ranking via fANOVA (or mean-decrease-impurity).

CLI usage
---------
    python -m scripts.results_utils.optuna_importance --model NodalMLP \
        --config NodalMLP_dot --match decoder_type=dot

Programmatic usage
------------------
    from scripts.results_utils.optuna_importance import (
        load_search_space, yaml_search_space_to_distributions,
        fetch_tune_trial_runs, filter_by_config_match,
        runs_to_frozen_trials, study_from_trials, get_importance,
    )
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

import numpy as np
import yaml

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from .records import WANDB_ENTITY, WANDB_PROJECT, query_runs, wandb_api
from .records import REPO_ROOT as _REPO_ROOT

REPO_ROOT = str(_REPO_ROOT)  # .../Conn2Conn


# ---------------------------------------------------------------------------
# Search space  →  Optuna distributions
# ---------------------------------------------------------------------------

def load_search_space(config_stem: str, repo_root: str = REPO_ROOT) -> dict:
    """Read the `search_space` block from `models/configs/<stem>.yml`."""
    path = os.path.join(repo_root, "models", "configs", f"{config_stem}.yml")
    with open(path) as fp:
        cfg = yaml.safe_load(fp)
    return cfg.get("search_space", {}) or {}


def yaml_search_space_to_distributions(ss: dict) -> dict:
    """Convert YAML search_space -> {param_name: optuna distribution}.

    Categorical values that are lists (e.g. decoder_dims) are coerced to tuples
    so they're hashable for CategoricalDistribution.
    """
    out: dict = {}
    for name, spec in ss.items():
        ttype = spec.get("type")
        if ttype == "choice":
            choices = []
            for v in spec["values"]:
                if isinstance(v, list):
                    v = tuple(v)
                choices.append(v)
            out[name] = CategoricalDistribution(choices)
        elif ttype in ("uniform", "loguniform", "qloguniform"):
            out[name] = FloatDistribution(
                low=float(spec["lower"]),
                high=float(spec["upper"]),
                log=ttype != "uniform",
            )
        elif ttype == "randint":
            out[name] = IntDistribution(low=int(spec["lower"]), high=int(spec["upper"]))
        else:
            print(
                f"  warning: unknown search_space type {ttype!r} for {name!r}; skipping",
                file=sys.stderr,
            )
    return out


# ---------------------------------------------------------------------------
# W&B fetch / filter
# ---------------------------------------------------------------------------

def fetch_tune_trial_runs(
    model_class_name: str,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    extra_tags: Optional[Iterable[str]] = None,
    since: Optional[str] = None,
) -> list:
    """Fetch all tune-trial runs (tagged 'tune' AND model class name).

    Tune trials log a flat config; best-trial reports log nested. We only want
    flat tune trials here for hparam importance, since they sample diverse
    points across the search space.

    `since` is an ISO-format timestamp (e.g. '2026-04-27' or '2026-04-27T18:00:00').
    Runs created before this time are filtered out — useful after a YAML schema
    change to exclude pre-refactor runs whose configs no longer match the new
    search-space keys.
    """
    api = wandb_api()
    tag_filters = [{"tags": "tune"}, {"tags": model_class_name}]
    for t in (extra_tags or []):
        tag_filters.append({"tags": t})
    filters = {"$and": tag_filters}
    if since is not None:
        filters["$and"].append({"createdAt": {"$gt": since}})
    runs = query_runs(
        api,
        f"{entity}/{project}",
        filters=filters,
        order="-created_at",
    )
    return list(runs)


def _coerce(value: str):
    """Best-effort scalar coercion for --match cli values."""
    v = value.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def filter_by_config_match(runs, match: dict) -> list:
    """Keep runs whose flat (or nested model.*) config matches all k=v in `match`."""
    out = []
    for r in runs:
        cfg = dict(r.config)
        ok = True
        for k, v in match.items():
            cfg_v = cfg.get(k)
            if cfg_v is None and isinstance(cfg.get("model"), dict):
                cfg_v = cfg["model"].get(k)
            if cfg_v != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Trial reconstruction
# ---------------------------------------------------------------------------

def _summary_dict(run) -> dict:
    """Robustly pull a run's summary as a plain dict."""
    s = run.summary
    if hasattr(s, "_json_dict"):
        return dict(s._json_dict)
    try:
        return dict(s)
    except Exception:
        return {}


def runs_to_frozen_trials(
    runs,
    distributions: dict,
    metric_key: str = "val_demeaned_r",
    verbose: bool = True,
    aliases: Optional[dict] = None,
) -> tuple:
    """Convert W&B runs to `optuna.FrozenTrial`s (see `rows_to_frozen_trials`)."""
    rows = []
    for r in runs:
        summary = _summary_dict(r)
        rows.append({"config": dict(r.config), "value": summary.get(metric_key),
                     "run_id": r.id, "run_name": r.name})
    return rows_to_frozen_trials(rows, distributions, verbose=verbose, aliases=aliases)


def rows_to_frozen_trials(
    rows,
    distributions: dict,
    verbose: bool = True,
    aliases: Optional[dict] = None,
) -> tuple:
    """Convert trial rows {config, value, run_id, run_name} to `optuna.FrozenTrial`s.

    A row is dropped if (a) the value is missing/non-finite, or (b) any of the
    distribution keys is missing from its config, or (c) a categorical value
    is outside the declared choices, or (d) a numeric value is outside range.
    `aliases` maps an old config key to its current name (e.g. {"reg": "l2_reg"}),
    applied when the current key is absent.

    Returns (trials, diagnostics) where diagnostics is a dict with skip reasons
    keyed by '<reason>:<param>' so the caller can see which keys are at fault.
    """
    trials = []
    diagnostics = {
        "n_total": len(rows),
        "n_kept": 0,
        "no_metric": 0,
        "missing_param_by_key": {},
        "value_oob_by_key": {},
    }
    for row in rows:
        cfg = dict(row["config"])
        for old_key, new_key in (aliases or {}).items():
            if new_key not in cfg and old_key in cfg:
                cfg[new_key] = cfg[old_key]
        val = row.get("value")
        if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
            diagnostics["no_metric"] += 1
            continue
        params = {}
        ok = True
        for k, dist in distributions.items():
            if k not in cfg:
                diagnostics["missing_param_by_key"][k] = diagnostics["missing_param_by_key"].get(k, 0) + 1
                ok = False
                break
            v = cfg[k]
            if isinstance(v, list):
                v = tuple(v)
            if isinstance(dist, CategoricalDistribution):
                if v not in dist.choices:
                    diagnostics["value_oob_by_key"][k] = diagnostics["value_oob_by_key"].get(k, 0) + 1
                    ok = False
                    break
            elif isinstance(dist, FloatDistribution):
                if not (dist.low <= float(v) <= dist.high):
                    diagnostics["value_oob_by_key"][k] = diagnostics["value_oob_by_key"].get(k, 0) + 1
                    ok = False
                    break
                v = float(v)
            elif isinstance(dist, IntDistribution):
                if not (dist.low <= int(v) <= dist.high):
                    diagnostics["value_oob_by_key"][k] = diagnostics["value_oob_by_key"].get(k, 0) + 1
                    ok = False
                    break
                v = int(v)
            params[k] = v
        if not ok:
            continue
        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=float(val),
        )
        trial.user_attrs["wandb_run_id"] = row.get("run_id")
        trial.user_attrs["wandb_run_name"] = row.get("run_name")
        trials.append(trial)
    diagnostics["n_kept"] = len(trials)

    if verbose and diagnostics["n_kept"] < diagnostics["n_total"]:
        print(f"  diagnostics: {diagnostics['n_kept']}/{diagnostics['n_total']} runs kept")
        if diagnostics["no_metric"]:
            print(f"    skipped (no/inf metric)         : {diagnostics['no_metric']}")
        for k, n in sorted(diagnostics["missing_param_by_key"].items(), key=lambda kv: -kv[1]):
            print(f"    skipped (missing param {k!r:24s}): {n}")
        for k, n in sorted(diagnostics["value_oob_by_key"].items(), key=lambda kv: -kv[1]):
            print(f"    skipped (oob param {k!r:28s}): {n}")
        if diagnostics["n_kept"] == 0 and diagnostics["missing_param_by_key"]:
            top_missing = max(diagnostics["missing_param_by_key"].items(), key=lambda kv: kv[1])[0]
            print(
                f"  HINT: 0 runs matched and {top_missing!r} is missing from every fetched run's "
                f"config. This usually means the runs were created before the YAML's current "
                f"search_space schema. Use --since YYYY-MM-DD to exclude pre-refactor runs."
            )
    return trials, diagnostics


def study_from_trials(trials, direction: str = "maximize") -> optuna.Study:
    study = optuna.create_study(direction=direction)
    for t in trials:
        study.add_trial(t)
    return study


def get_importance(study: optuna.Study, evaluator: str = "fanova", seed: Optional[int] = None) -> dict:
    """Parameter importances; pass `seed` for a reproducible fANOVA / MDI forest."""
    if evaluator == "fanova":
        ev = optuna.importance.FanovaImportanceEvaluator(seed=seed)
    elif evaluator == "mdi":
        ev = optuna.importance.MeanDecreaseImpurityImportanceEvaluator(seed=seed)
    else:
        raise ValueError(f"unknown evaluator {evaluator!r}; choose fanova|mdi")
    return optuna.importance.get_param_importances(study, evaluator=ev)


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def importance_for_variant(
    model_class_name: str,
    config_stem: str,
    match: Optional[dict] = None,
    metric_key: str = "val_demeaned_r",
    direction: str = "maximize",
    evaluator: str = "fanova",
    extra_tags: Optional[Iterable[str]] = None,
    since: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """End-to-end: fetch runs, reconstruct trials, return importance + diagnostics.

    Returns dict with keys: 'importance', 'study', 'n_runs_fetched',
    'n_trials_reconstructed', 'best_value', 'best_params', 'diagnostics'.

    Pass `since` (e.g. '2026-04-27') to exclude pre-refactor runs whose configs
    don't carry the YAML's current search-space keys.
    """
    ss = load_search_space(config_stem)
    dists = yaml_search_space_to_distributions(ss)
    runs = fetch_tune_trial_runs(model_class_name, extra_tags=extra_tags, since=since)
    n_fetched = len(runs)
    if match:
        runs = filter_by_config_match(runs, match)
    n_after_match = len(runs)
    trials, diagnostics = runs_to_frozen_trials(runs, dists, metric_key=metric_key, verbose=verbose)
    study = study_from_trials(trials, direction=direction)
    diagnostics["n_after_match"] = n_after_match
    diagnostics["n_runs_fetched"] = n_fetched
    if not study.trials:
        return {
            "importance": {},
            "study": study,
            "n_runs_fetched": n_fetched,
            "n_trials_reconstructed": 0,
            "best_value": None,
            "best_params": None,
            "diagnostics": diagnostics,
        }
    importance = get_importance(study, evaluator=evaluator)
    return {
        "importance": importance,
        "study": study,
        "n_runs_fetched": n_fetched,
        "n_trials_reconstructed": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    p.add_argument("--model", required=True, help="W&B-tagged model class name (e.g. NodalMLP)")
    p.add_argument("--config", default=None, help="YAML stem for search space (default: --model)")
    p.add_argument("--metric", default="val_demeaned_r")
    p.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    p.add_argument("--match", default="", help="comma-sep k=v config filters (e.g. 'decoder_type=dot,encoder_type=pca')")
    p.add_argument("--evaluator", default="fanova", choices=["fanova", "mdi"])
    p.add_argument("--since", default=None,
                   help="ISO-format cutoff for run createdAt (e.g. 2026-04-27). "
                        "Exclude pre-schema-refactor runs whose configs lack the YAML's current keys.")
    args = p.parse_args()

    config_stem = args.config or args.model
    match = None
    if args.match:
        match = {k: _coerce(v) for k, v in (kv.split("=", 1) for kv in args.match.split(","))}

    print(f"model={args.model!r}  config_stem={config_stem!r}  match={match}  metric={args.metric!r}  since={args.since!r}")
    result = importance_for_variant(
        model_class_name=args.model,
        config_stem=config_stem,
        match=match,
        metric_key=args.metric,
        direction=args.direction,
        evaluator=args.evaluator,
        since=args.since,
    )
    print(f"  W&B runs fetched : {result['n_runs_fetched']}")
    print(f"  trials replayed  : {result['n_trials_reconstructed']}")
    if not result["importance"]:
        print("\nNo valid trials — nothing to score.")
        sys.exit(1)
    print(f"  best value       : {result['best_value']:.5f}")
    print(f"  best params      : {result['best_params']}")
    print(f"\nHparam importance ({args.evaluator}):")
    print("-" * 60)
    for k, v in result["importance"].items():
        print(f"  {k:30s}  {v:.4f}")
    print()
    if result["n_trials_reconstructed"] < 25:
        print(
            f"  NOTE: importance estimates with only "
            f"{result['n_trials_reconstructed']} trials are noisy — "
            "treat ranks as directional, not exact.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
