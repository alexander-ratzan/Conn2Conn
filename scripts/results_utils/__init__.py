"""
Conn2Conn results utilities: scrape W&B / Ray runs, build tables, plot figures.

  records        W&B runs → RunRecords (+ shared paths / constants)
  tables         RunRecords → summary DataFrames
  plots          DataFrames → matplotlib figures
  local_results  notebook-written results/local_results/ artifacts

`optuna_importance` (hparam importance CLI) is not re-exported here so that
importing the package does not require optuna:
    python -m scripts.results_utils.optuna_importance --help
"""

from .records import (
    LOCAL_RESULTS_DIR,
    RAY_CHECKPOINTS_DIR,
    RAY_RESULTS_DIR,
    REPO_ROOT,
    RESULTS_ROOT,
    WANDB_ENTITY,
    WANDB_PROJECT,
    RunRecord,
    build_experiment_records,
    build_experiment_records_covtype,
    count_tune_trials_for_run,
    cov_sources_str_from_config,
    cov_type_from_config,
    enrich_records_with_local,
    fetch_best_trial_runs,
    fetch_direct_prod_runs,
    load_local_artifact_df,
    load_records_cache,
    parse_run_record,
    records_to_df,
    save_records_cache,
    wandb_api,
)
from .tables import (
    build_cov_dl_seed_df,
    build_cov_dl_summary_table,
    build_covtype_metric_table,
    build_covtype_status_table,
    build_metric_table,
    build_sc_type_summary_table,
    build_status_table,
)
from .plots import (
    plot_cov_dl_global_metric_panels,
    plot_cov_dl_metric_bars,
    plot_model_metric_scatter,
    plot_source_metric_bars,
)
from .local_results import (
    best_local_results,
    load_local_results,
    plot_metric_bar,
    plot_metric_scatter,
)

# Exclude submodule names so `from scripts.results_utils import *` in notebooks
# never clobbers user variables such as `records`.
from types import ModuleType as _ModuleType

__all__ = [
    name for name, obj in list(globals().items())
    if not name.startswith("_") and not isinstance(obj, _ModuleType)
]


def reload():
    """Reload submodules in dependency order, then this package (notebook iteration)."""
    import importlib
    import sys

    for name in ("records", "tables", "plots", "local_results"):
        importlib.reload(sys.modules[f"{__name__}.{name}"])
    return importlib.reload(sys.modules[__name__])
