"""
Main script for Conn2Conn: fit and evaluate models.
Use the Sim class to set up an experiment (data, config) and run single or tuning jobs.
Callable from notebook or via CLI/sbatch (python main.py --mode dev|prod).
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb
import yaml
import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
try:
    from ray.tune.search.optuna import OptunaSearch
except ImportError:
    OptunaSearch = None
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air import session
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger


from data.hcp_dataset import HCP_Base, HCP_Partition
from data.dataset_utils import DEFAULT_CONN2CONN_CACHE_ROOT
from models.models import predict_from_loader
from models.loss import train_model, get_target_train_mean
from models.eval import Evaluator
from models.lightning_module import CrossModalLightningModule
from models.config import (
    load_config,
    get_default_config,
    get_search_space,
    build_model,
    resolve_source_dependent_config,
    TRAINER_KEYS,
    DATA_KEYS,
    FLAT_METADATA_KEYS,
)

# Ensure project root is on path when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Wandb: project and entity to log to. Authenticate once with `wandb login` or set WANDB_API_KEY.
WANDB_PROJECT = "conn2conn"
WANDB_ENTITY = "alexander-ratzan-new-york-university"
RESULTS_ROOT = os.path.join(_SCRIPT_DIR, "results")
RAY_CHECKPOINTS_DIR = os.path.join(RESULTS_ROOT, "ray_checkpoints")
RAY_RESULTS_DIR = os.path.join(RESULTS_ROOT, "ray_results")
RAY_TMP_DIR = os.path.join(RESULTS_ROOT, "ray_tmp")

_WORKER_CACHE = {}


def _extract_ray_tune_id(name_or_path: str) -> Optional[str]:
    if not name_or_path:
        return None
    m = re.search(r"_tune_(\d+)", name_or_path)
    return m.group(1) if m else None


class TrialFamilyWandbLoggerCallback(WandbLoggerCallback):
    """Minimal WandB callback wrapper to attach Ray Tune IDs and group by trial family."""

    def __init__(self, *args, extra_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_config = extra_config or {}

    def log_trial_start(self, trial):
        config = trial.config.copy()
        config.pop("callbacks", None)
        config.update(self._extra_config)

        exclude_results = self._exclude_results.copy()
        exclude_results += self.excludes
        if not self.log_config:
            exclude_results += ["config"]

        trial_id = trial.trial_id if trial else None
        trial_family = trial_id.split("_")[0] if trial_id and "_" in trial_id else trial_id
        ray_tune_id = _extract_ray_tune_id(trial.experiment_dir_name if trial else "")

        if ray_tune_id:
            config["ray_tune_id"] = ray_tune_id
        if trial_id:
            config["ray_trial_id"] = trial_id

        config = {
            key: value for key, value in config.items() if key not in self.excludes
        }

        base_group = self.group or (trial.experiment_dir_name if trial else None)
        wandb_group = f"{base_group}_{trial_family}" if (base_group and trial_family) else base_group
        trial_name = f"_tune_trainable_{trial_id}" if trial_id else (str(trial) if trial else None)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=trial_name,
            resume=False,
            reinit=True,
            allow_val_change=True,
            group=wandb_group,
            project=self.project,
            config=config,
        )
        wandb_init_kwargs.update(self.kwargs)
        self._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)


def _flat_to_nested(flat: dict, model_name: str) -> dict:
    """Split flat config (e.g. from Tune) into nested {data, model, trainer}."""
    trainer = {k: flat[k] for k in TRAINER_KEYS if k in flat}
    data = {k: flat[k] for k in DATA_KEYS if k in flat}
    _model_exclude = TRAINER_KEYS | FLAT_METADATA_KEYS | {"name"}
    model = {"name": model_name, **{k: flat[k] for k in flat if k not in _model_exclude}}
    for key in DATA_KEYS:
        model.pop(key, None)
    for key in list(model):
        if key.startswith("data."):
            model.pop(key, None)
    return {"data": data, "model": model, "trainer": trainer}


def _to_serializable(obj):
    """Convert nested objects with numpy/tensors to JSON-serializable python types."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


class Sim:
    """
    Experiment runner: holds config, dataset, and loaders; runs single or tuning jobs.
    """

    def __init__(
        self,
        model_name: str,
        config_path: str = None,
        config_overrides: dict = None,
        parcellation: str = "Glasser",
        hemi: str = "both",
        source: str = "SC",
        target: str = "FC",
        shuffle_seed: int = 0,
        data_load_mode: str = None,
        precompute_cache_root: str = None,
        write_manual_cache: bool = None,
        batch_size: int = 128,
    ):
        """
        model_name: name of the model (must have models/configs/<model_name>.yml).
        config_path: optional path to a YAML config file (overrides model_name lookup).
        config_overrides: optional dict merged onto default config (nested: { "model": {...}, "trainer": {...} }).
        Remaining args: data options for HCP_Base and DataLoader batch_size.
        """
        self.model_name = model_name
        self.config_path = config_path
        self.parcellation = parcellation
        self.hemi = hemi
        self.source = source
        self.target = target
        self.shuffle_seed = shuffle_seed
        full = load_config(model_name, path=config_path)
        self._raw_config = full
        self.learned = full.get("learned", True)

        self.config = get_default_config(model_name, path=config_path)
        self.config.setdefault("data", {})
        self.config["data"].update(
            {
                "parcellation": parcellation,
                "hemi": hemi,
                "source": source,
                "target": target,
                "shuffle_seed": shuffle_seed,
            }
        )
        if config_overrides:
            for section in ("data", "model", "trainer"):
                if section in config_overrides and section in self.config:
                    self.config[section].update(config_overrides[section])
                elif section in config_overrides:
                    self.config[section] = deepcopy(config_overrides[section])

        # Always trust the invocation (sbatch/CLI) for data identity to ensure correct tracking.
        # Config files should not override these runtime selections.
        self.config.setdefault("data", {})
        self.config["data"].update(
            {
                "source": source,
                "target": target,
                "parcellation": parcellation,
                "hemi": hemi,
                "shuffle_seed": shuffle_seed,
            }
        )
        if data_load_mode is not None:
            self.config["data"]["data_load_mode"] = data_load_mode
        if precompute_cache_root is not None:
            self.config["data"]["precompute_cache_root"] = precompute_cache_root
        if write_manual_cache is not None:
            self.config["data"]["write_manual_cache"] = bool(write_manual_cache)

        # Convenience defaults for cached-data runs.
        # If the caller selects precomputed mode, automatically point to the
        # canonical cache root and disable manual cache writes.
        effective_load_mode = self.config["data"].get("data_load_mode")
        if effective_load_mode == "precomputed":
            self.config["data"].setdefault("precompute_cache_root", DEFAULT_CONN2CONN_CACHE_ROOT)
            self.config["data"]["write_manual_cache"] = False
        self.config = resolve_source_dependent_config(self.config)
        data_cfg = deepcopy(self.config.get("data", {}))
        self.config["data"] = data_cfg

        batch_size = self.config.get("trainer", {}).get("batch_size", batch_size)
        self.parcellation = data_cfg["parcellation"]
        self.hemi = data_cfg["hemi"]
        self.source = data_cfg["source"]
        self.target = data_cfg["target"]
        self.shuffle_seed = data_cfg["shuffle_seed"]

        model_cfg = self.config.get("model", {}) or {}
        self._cov_sources_explicit = "cov_sources" in model_cfg
        cov_sources = list(model_cfg.get("cov_sources") or []) if self._cov_sources_explicit else []
        self._cov_sources = cov_sources
        hcp_base_kwargs = {
            "parcellation": self.parcellation,
            "hemi": self.hemi,
            "shuffle_seed": self.shuffle_seed,
            "source": self.source,
            "target": self.target,
            "cov_sources": cov_sources,
            "expose_node_features": (self.model_name == "NodalGNN"),
        }
        for _k in (
            "HCP_dir",
            "sc_metric_type",
            "sc_apply_log1p",
            "volume_feature_type",
            "centroid_feature_type",
            "data_load_mode",
            "precompute_cache_root",
            "write_manual_cache",
        ):
            if _k in data_cfg:
                hcp_base_kwargs[_k] = data_cfg[_k]
        self.base = HCP_Base(**hcp_base_kwargs)

        # Loggable covariate metadata for wandb filtering.
        self.config.setdefault("model", {})
        self.config["model"]["cov_sources_str"] = (
            "+".join(cov_sources) if self._cov_sources_explicit else ""
        )
        self.config["model"]["cov_dims"] = (
            getattr(self.base, "cov_dims", {}) if self._cov_sources_explicit else {}
        )
        self.train_ds = HCP_Partition(self.base, "train")
        self.val_ds = HCP_Partition(self.base, "val")
        self.test_ds = HCP_Partition(self.base, "test")
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

    def run_single(
        self,
        mode: str = "dev",
        save_checkpoint: bool = False,
        run_dir: str = None,
        store_eval_md: bool = False,
    ):
        """
        Run a single configuration (one model, one training run). No Ray; runs in this process.

        mode: "dev" = no wandb, live plot, verbose=False. "prod" = log to wandb, verbose eval.
        For one run with wandb from a notebook or script, use run_single(mode="prod").
        save_checkpoint: if True, write checkpoint and eval report to run_dir (default results/run_<timestamp>).
        store_eval_md: if True, write the eval markdown report to run_dir even without a checkpoint.
        Returns dict with model, evaluators, test_metrics, and for learned models train_result.
        """
        if run_dir is None and (save_checkpoint or store_eval_md):
            run_dir = os.path.join("results", f"run_{int(time.time())}")
        
        if mode == "prod" and wandb is not None and not self.learned: # need to do this specifically for 'closed form' models
            wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, config=self.config, reinit=True, tags=[self.model_name, "prod"])
        
        try:
            if self.learned:
                return self._run_learned_single(
                    mode,
                    save_checkpoint,
                    run_dir,
                    store_eval_md=store_eval_md,
                )
            else: 
                return self._run_closed_form_single(
                    mode,
                    save_checkpoint,
                    run_dir,
                    store_eval_md=store_eval_md,
                )
        finally:
            if mode == "prod" and wandb is not None: # fallback to finish any wandb run - can consider replacing with WandbLogger to manage init and finish
                wandb.finish()

    def _merge_config(self, config_override: dict = None) -> dict:
        cfg = deepcopy(self.config)
        if not config_override:
            return resolve_source_dependent_config(cfg)
        for section in ("data", "model", "trainer"):
            if section in config_override and isinstance(config_override[section], dict):
                cfg.setdefault(section, {})
                cfg[section].update(config_override[section])
        return resolve_source_dependent_config(cfg)

    def _evaluate_model(
        self,
        model,
        mode: str = "dev",
        test_report_path: str = None,
        model_name: str = None,
    ):
        if getattr(model, "is_precomputed", False):
            # Bypass DataLoader iteration; slice precomputed arrays by split indices directly.
            train_preds, train_targets = model.predict_split("train")
            val_preds,   val_targets   = model.predict_split("val")
            test_preds,  test_targets  = model.predict_split("test")
        else:
            train_preds, train_targets = predict_from_loader(model, self.train_loader)
            val_preds, val_targets = predict_from_loader(model, self.val_loader)
            test_preds, test_targets = predict_from_loader(model, self.test_loader)
        train_partition = HCP_Partition(self.base, "train")
        val_partition = HCP_Partition(self.base, "val")
        test_partition = HCP_Partition(self.base, "test")
        train_eval = Evaluator(train_preds, train_targets, train_partition, self.base)
        val_eval = Evaluator(val_preds, val_targets, val_partition, self.base)
        test_eval = Evaluator(test_preds, test_targets, test_partition, self.base)
        test_metrics = test_eval.analyze_results(
            verbose=(mode == "prod"),
            filepath=test_report_path,
            model_name=model_name,
        )
        return {
            "evaluators": {"train": train_eval, "val": val_eval, "test": test_eval},
            "metrics": {
                "train": train_eval._metrics,
                "val": val_eval._metrics,
                "test": (test_metrics.get("base_metrics") if isinstance(test_metrics, dict) else test_metrics),
            },
            "test_metrics": test_metrics,
        }

    def _run_learned_single(
        self,
        mode: str,
        save_checkpoint: bool,
        run_dir: str = None,
        config_override: dict = None,
        checkpoint_path: str = None,
        train_from_scratch: bool = True,
        test_report_basename: str = "eval_test",
        wandb_name: str = None,
        wandb_group: str = None,
        wandb_tags: list = None,
        wandb_metadata: dict = None,
        store_eval_md: bool = False,
    ):
        cfg = self._merge_config(config_override)
        model_cfg = cfg["model"].copy()
        model_cfg.pop("name")
        trainer_cfg = cfg.get("trainer", {})
        lr = trainer_cfg.get("lr", 1e-4)
        loss_type = trainer_cfg.get("loss_type", "mse")
        loss_alpha = trainer_cfg.get("loss_alpha", 0.5)
        loss_beta = trainer_cfg.get("loss_beta", 1.0)
        loss_corr_target = trainer_cfg.get("loss_corr_target", 0.4)
        loss_corr_weight = trainer_cfg.get("loss_corr_weight", 1e-3)
        max_epochs = trainer_cfg.get("max_epochs", 100)
        log_every = trainer_cfg.get("log_every", 5)

        model = build_model(self.base, self.model_name, model_cfg)
        pl_logger = None
        train_result = None
        
        if mode == "prod" and WandbLogger is not None:
            tags = [self.model_name, "prod"]
            if wandb_tags:
                tags.extend(wandb_tags)
            pl_logger = WandbLogger(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                save_dir=os.path.join(_SCRIPT_DIR, "results"),
                tags=tags,
                name=wandb_name or f"{self.model_name}_prod",
                group=wandb_group,
            )
            # log full config once
            pl_logger.log_hyperparams(cfg)
            if wandb_metadata:
                pl_logger.experiment.config.update(_to_serializable(wandb_metadata), allow_val_change=True)

        if train_from_scratch:
            train_result = train_model(
                model,
                self.train_loader,
                self.val_loader,
                base=self.base,
                log_every=log_every,
                lr=lr,
                loss_type=loss_type,
                loss_alpha=loss_alpha,
                loss_beta=loss_beta,
                loss_corr_target=loss_corr_target,
                loss_corr_weight=loss_corr_weight,
                max_epochs=max_epochs,
                logger=False if mode == "dev" else (pl_logger is None),
                pl_logger=pl_logger,
                enable_progress_bar=(mode == "dev"),
            )
            if mode == "dev":
                train_result.plot()
            model = train_result.pl_module.model
        elif checkpoint_path:
            ckpt_state = torch.load(checkpoint_path, map_location="cpu")
            pl_module = CrossModalLightningModule(
                model=model,
                base=self.base,
                lr=lr,
                loss_type=loss_type,
                loss_alpha=loss_alpha,
                loss_beta=loss_beta,
                loss_corr_target=loss_corr_target,
                loss_corr_weight=loss_corr_weight,
            )
            pl_module.load_state_dict(ckpt_state["state_dict"], strict=False)
            model = pl_module.model

        test_filepath = None
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            if save_checkpoint or (store_eval_md and (not train_from_scratch and checkpoint_path)):
                test_filepath = os.path.join(run_dir, test_report_basename)
        eval_out = self._evaluate_model(
            model,
            mode=mode,
            test_report_path=test_filepath,
            model_name=self.model_name,
        )
        test_metrics = eval_out["test_metrics"]

        if mode == "prod" and pl_logger is not None:
            metrics_to_log = {}
            for k, v in (test_metrics.get("base_metrics") or {}).items():
                if isinstance(v, (int, float, np.floating)):
                    metrics_to_log[f"eval_test/{k}"] = float(v)
            if metrics_to_log:
                # WandbLogger uses .experiment as the wandb run
                pl_logger.experiment.log(metrics_to_log)        
        
        if save_checkpoint and run_dir and train_result is not None:
            train_result.trainer.save_checkpoint(os.path.join(run_dir, "checkpoint.ckpt"))

        return {
            "train_result": train_result,
            "model": model,
            "pl_module": train_result.pl_module if train_result is not None else None,
            "evaluators": eval_out["evaluators"],
            "metrics": eval_out["metrics"],
            "test_metrics": test_metrics,
        }

    def _run_closed_form_single(
        self,
        mode: str,
        save_checkpoint: bool,
        run_dir: str = None,
        config_override: dict = None,
        test_report_basename: str = "eval_test",
        wandb_name: str = None,
        wandb_group: str = None,
        wandb_tags: list = None,
        wandb_metadata: dict = None,
        store_eval_md: bool = False,
    ):
        cfg = self._merge_config(config_override)
        model_cfg = cfg["model"].copy()
        model_cfg.pop("name")
        model = build_model(self.base, self.model_name, model_cfg)
        test_filepath = None
        if run_dir and (save_checkpoint or store_eval_md):
            os.makedirs(run_dir, exist_ok=True)
            test_filepath = os.path.join(run_dir, test_report_basename)
        eval_out = self._evaluate_model(
            model,
            mode=mode,
            test_report_path=test_filepath,
            model_name=self.model_name,
        )
        test_metrics = eval_out["test_metrics"]

        wandb_started_here = False
        if mode == "prod" and wandb is not None and wandb.run is None:
            tags = [self.model_name, "prod"]
            if wandb_tags:
                tags.extend(wandb_tags)
            wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                config=cfg,
                reinit=True,
                tags=tags,
                name=wandb_name or f"{self.model_name}_prod",
                group=wandb_group,
            )
            if wandb_metadata:
                wandb.config.update(_to_serializable(wandb_metadata), allow_val_change=True)
            wandb_started_here = True

        if mode == "prod" and wandb is not None and wandb.run is not None:
            # Log train and val metrics
            train_metrics = eval_out["metrics"]["train"]
            val_metrics = eval_out["metrics"]["val"]
            wandb.log({
                "train_mse": train_metrics.get("mse", np.nan),
                "train_demeaned_r": train_metrics.get("demeaned_pearson", np.nan),
                "train_pearson_r": train_metrics.get("pearson", np.nan),
                "val_mse": val_metrics.get("mse", np.nan),
                "val_demeaned_r": val_metrics.get("demeaned_pearson", np.nan),
                "val_pearson_r": val_metrics.get("pearson", np.nan),
            })
            # Log test metrics
            for k, v in (test_metrics.get("base_metrics") or {}).items():
                if isinstance(v, (int, float, np.floating)):
                    wandb.log({f"eval_test/{k}": float(v)})
        if wandb_started_here:
            wandb.finish()
        
        return {
            "model": model,
            "evaluators": eval_out["evaluators"],
            "metrics": eval_out["metrics"],
            "test_metrics": test_metrics,
        }

    
    def run_tune(
        self,
        num_samples: int = 10,
        save_checkpoint: bool = False,
        metric: str = "val_demeaned_r",
        mode: str = "max",
        max_epochs: int = None,
        cpus_per_trial: float = 2.0,
        gpus_per_trial: float = 1.0,
        max_concurrent_trials: int = None,
        search_alg: str = "random",
        persist_final_artifacts: bool = False,
    ):
        """
        Run Ray Tune for this model. Returns ResultGrid.

        For a single run with wandb logging (no Ray), use run_single(mode="prod") instead.
        """
        if tune is None:
            raise RuntimeError("Ray Tune is not installed. pip install 'ray[tune]'")
        
        os.makedirs(RAY_CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(RAY_RESULTS_DIR, exist_ok=True)
        
        if not ray.is_initialized():
            if "RAY_worker_register_timeout_seconds" not in os.environ:
                os.environ["RAY_worker_register_timeout_seconds"] = "120"
            ray_init_kwargs = {"ignore_reinit_error": True}
            if os.environ.get("RAY_INCLUDE_DASHBOARD", "").strip() != "1":
                ray_init_kwargs["include_dashboard"] = False
            ray_tmpdir = os.environ.get("RAY_TMPDIR")
            if ray_tmpdir:
                ray_tmpdir = os.path.abspath(ray_tmpdir)
            else:
                job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
                long_dir = os.path.join(RAY_TMP_DIR, f"ray_{job_id}")
                os.makedirs(long_dir, exist_ok=True)
                short_path = f"/tmp/ray_{job_id}"
                try:
                    if os.path.lexists(short_path) and not os.path.islink(short_path):
                        short_path = f"/tmp/ray_{os.getpid()}_{job_id}"
                    if not os.path.lexists(short_path):
                        os.symlink(long_dir, short_path)
                    ray_tmpdir = short_path
                except OSError:
                    ray_tmpdir = long_dir
            os.makedirs(ray_tmpdir, exist_ok=True)
            ray_init_kwargs["_temp_dir"] = ray_tmpdir
            ray.init(**ray_init_kwargs)
            time.sleep(10)
        resources = ray.cluster_resources()
        print(f"Ray cluster resources: {resources}", flush=True)
        
        # Avoid capturing the full Sim instance (which can include large in-memory data) inside the Ray trainable closure.
        model_name = self.model_name
        learned = self.learned
        cov_sources_explicit = bool(self._cov_sources_explicit)
        param_space = get_search_space(self.model_name, path=self.config_path)
        if not param_space:
            raise ValueError(f"No search_space for model: {self.model_name} in config YAML.")
        tuned_keys = set(param_space.keys())
        tuned_data_keys = tuned_keys & DATA_KEYS
        
        if tuned_data_keys:
            raise ValueError(
                "Ray Tune search_space contains data identity keys "
                f"{sorted(tuned_data_keys)}. For fixed-base tuning, pass data choices via "
                "--source/--target/--parcellation/--hemi/--shuffle_seed and remove these keys from search_space."
            )
        if "batch_size" in tuned_keys:
            raise ValueError(
                "Ray Tune search_space contains 'batch_size', but this run reuses one fixed set of DataLoaders. "
                "Remove batch_size from search_space for this fixed-base tuning flow."
            )
        print("Tune: fixed data (HCP_Base built in worker, reused across trials).", flush=True)

        # Config setup 
        default = deepcopy(self.config)
        default_flat = {"name": self.model_name}
        for k, v in default.get("data", {}).items():
            default_flat[k] = v
        for k, v in default.get("model", {}).items():
            if k != "name":
                default_flat[k] = v
        for k, v in default.get("trainer", {}).items():
            default_flat[k] = v


        # Expose scalar summary keys for dict-valued hparams so W&B table/filter works cleanly.
        # cov_projectors and cov_fusion are tune.choice over full dicts; a string tag lets you
        # group/filter trials without unpacking nested objects in the W&B UI.
        if cov_sources_explicit:
            _cov_proj_default = default.get("model", {}).get("cov_projectors") or {}
            default_flat["cov_projectors_tag"] = "|".join(
                f"{src}:{cfg.get('out_dim', '?')}" for src, cfg in sorted(_cov_proj_default.items())
            )
            _cov_fusion_default = default.get("model", {}).get("cov_fusion") or {}
            default_flat["cov_fusion_tag"] = (
                _cov_fusion_default.get("type", "linear")
                + (f"_{'x'.join(str(d) for d in _cov_fusion_default['hidden_dims'])}"
                if _cov_fusion_default.get("hidden_dims") else "")
                + ("_ln" if _cov_fusion_default.get("layer_norm") else "")
            )
        else:
            default_flat["cov_projectors_tag"] = ""
            default_flat["cov_fusion_tag"] = ""

        fixed_data_cfg = deepcopy(default.get("data", {}))
        fixed_cov_sources = (
            list(((default.get("model", {}) or {}).get("cov_sources") or []))
            if cov_sources_explicit else []
        )
        fixed_batch_size = default.get("trainer", {}).get("batch_size", 128)
        print(f"Fixed Tune batch_size: {fixed_batch_size}", flush=True)
       
        # Determine ASHA max_t from config/search-space first, then fallback.
        # This keeps scheduler budget aligned with trainer.max_epochs choices.
        scheduler_max_t = max_epochs
        if not scheduler_max_t:
            search_max_epochs = (
                param_space.get("max_epochs", {}).get("values")
                if isinstance(param_space.get("max_epochs"), dict)
                else None
            )
            if search_max_epochs:
                scheduler_max_t = max(search_max_epochs)
            else:
                scheduler_max_t = default.get("trainer", {}).get("max_epochs", 100)
        scheduler_max_t = int(scheduler_max_t)
        if scheduler_max_t <= 0:
            raise ValueError(f"Invalid scheduler max_t: {scheduler_max_t}")
            
        # Run name and Ray dataset and loader initialization
        run_name = f"{self.model_name}_tune_{int(time.time())}"
        ray_tune_id = _extract_ray_tune_id(run_name)
        run_root = os.path.join(RAY_CHECKPOINTS_DIR, run_name)
        
        
        def _trial_artifact_dir(trial_id: str) -> str:
            return os.path.join(run_root, trial_id, "final")

        def _write_final_artifact(trial_id: str, config_nested: dict, learned_trial: bool, trainer=None, final_metrics=None):
            artifact_dir = _trial_artifact_dir(trial_id)
            os.makedirs(artifact_dir, exist_ok=True)
            config_path = os.path.join(artifact_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_to_serializable(config_nested), f, indent=2)

            model_path = None
            if learned_trial and trainer is not None:
                model_path = os.path.join(artifact_dir, "model.ckpt")
                trainer.save_checkpoint(model_path)

            if final_metrics:
                with open(os.path.join(artifact_dir, "metrics_final.json"), "w") as f:
                    json.dump(_to_serializable(final_metrics), f, indent=2)

            manifest = {
                "trial_id": trial_id,
                "learned": learned_trial,
                "config_path": config_path,
                "model_path": model_path,
                "metrics_path": os.path.join(artifact_dir, "metrics_final.json") if final_metrics else None,
            }
            with open(os.path.join(artifact_dir, "artifact_manifest.json"), "w") as f:
                json.dump(_to_serializable(manifest), f, indent=2)
            return manifest

        def train_func(default_flat, param_space, tune_config):
            allowed_keys = set(param_space) | TRAINER_KEYS
            config_flat = {**default_flat, **{k: v for k, v in tune_config.items() if k in allowed_keys}}
            config_flat["name"] = model_name
            config_flat = resolve_source_dependent_config(config_flat)
            config = _flat_to_nested(config_flat, model_name)
            # Recompute scalar W&B filter tags from the sampled dict-valued hparams.
            if cov_sources_explicit:
                _sampled_proj = config_flat.get("cov_projectors") or {}
                if isinstance(_sampled_proj, dict):
                    config_flat["cov_projectors_tag"] = "|".join(
                        f"{src}:{cfg.get('out_dim', '?')}" for src, cfg in sorted(_sampled_proj.items())
                    )
                _sampled_fusion = config_flat.get("cov_fusion") or {}
                if isinstance(_sampled_fusion, dict):
                    config_flat["cov_fusion_tag"] = (
                        _sampled_fusion.get("type", "linear")
                        + (f"_{'x'.join(str(d) for d in _sampled_fusion['hidden_dims'])}"
                        if _sampled_fusion.get("hidden_dims") else "")
                        + ("_ln" if _sampled_fusion.get("layer_norm") else "")
                    )
            else:
                config_flat["cov_projectors_tag"] = ""
                config_flat["cov_fusion_tag"] = ""
            trial_id = tune.get_context().get_trial_id()
            if "base" not in _WORKER_CACHE:
                print(f"[tune] {trial_id}: building HCP_Base and DataLoaders in worker", flush=True)
                _worker_hcp_kwargs = {
                    "parcellation": fixed_data_cfg.get("parcellation", "Glasser"),
                    "hemi": fixed_data_cfg.get("hemi", "both"),
                    "shuffle_seed": fixed_data_cfg.get("shuffle_seed", 0),
                    "source": fixed_data_cfg.get("source", "SC"),
                    "target": fixed_data_cfg.get("target", "FC"),
                    "cov_sources": fixed_cov_sources,
                    "expose_node_features": (model_name == "NodalGNN"),
                }
                for _k in (
                    "HCP_dir",
                    "sc_metric_type",
                    "sc_apply_log1p",
                    "volume_feature_type",
                    "centroid_feature_type",
                    "data_load_mode",
                    "precompute_cache_root",
                    "write_manual_cache",
                ):
                    if _k in fixed_data_cfg:
                        _worker_hcp_kwargs[_k] = fixed_data_cfg[_k]
                _WORKER_CACHE["base"] = HCP_Base(**_worker_hcp_kwargs)
                b = _WORKER_CACHE["base"]
                _WORKER_CACHE["train_ds"] = HCP_Partition(b, "train")
                _WORKER_CACHE["val_ds"] = HCP_Partition(b, "val")
                _WORKER_CACHE["test_ds"] = HCP_Partition(b, "test")
                _WORKER_CACHE["train_loader"] = DataLoader(_WORKER_CACHE["train_ds"], batch_size=fixed_batch_size, shuffle=True)
                _WORKER_CACHE["val_loader"] = DataLoader(_WORKER_CACHE["val_ds"], batch_size=fixed_batch_size, shuffle=False)
                _WORKER_CACHE["test_loader"] = DataLoader(_WORKER_CACHE["test_ds"], batch_size=fixed_batch_size, shuffle=False)
            base = _WORKER_CACHE["base"]
            train_ds = _WORKER_CACHE["train_ds"]
            val_ds = _WORKER_CACHE["val_ds"]
            test_ds = _WORKER_CACHE["test_ds"]
            train_loader = _WORKER_CACHE["train_loader"]
            val_loader = _WORKER_CACHE["val_loader"]
            test_loader = _WORKER_CACHE["test_loader"]

            report_dict = {}
            should_write_final_artifact = save_checkpoint or persist_final_artifacts

            if learned:
                model_cfg = config["model"].copy()
                model_cfg.pop("name")
                trainer_cfg = config.get("trainer", {})
                epochs = trainer_cfg.get("max_epochs", max_epochs or 100)
                model = build_model(base, model_name, model_cfg)
                pl_module = CrossModalLightningModule(
                    model=model,
                    base=base,
                    lr=trainer_cfg.get("lr", 1e-4),
                    loss_type=trainer_cfg.get("loss_type", "mse"),
                    loss_alpha=trainer_cfg.get("loss_alpha", 0.5),
                    loss_beta=trainer_cfg.get("loss_beta", 1.0),
                    loss_corr_target=trainer_cfg.get("loss_corr_target", 0.4),
                    loss_corr_weight=trainer_cfg.get("loss_corr_weight", 1e-3),
                )
                tune_metrics = {
                    "train_loss": "train_loss",
                    "train_demeaned_r": "train_demeaned_r",
                    "train_pearson_r": "train_pearson_r",
                    "val_loss": "val_loss",
                    "val_demeaned_r": "val_demeaned_r",
                    "val_pearson_r": "val_pearson_r",
                }
                
                callbacks = [TuneReportCallback(metrics=tune_metrics, on="validation_end"),
                            TuneReportCheckpointCallback(metrics=tune_metrics, filename="checkpoint",on="fit_end")] # train_end, fit_end
                
                trainer = pl.Trainer(max_epochs=epochs, logger=False, callbacks=callbacks, enable_progress_bar=False)
                trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
                final_report = {}
                for metric_name in tune_metrics:
                    val = trainer.callback_metrics.get(metric_name)
                    if val is not None:
                        final_report[metric_name] = float(val.item()) if hasattr(val, "item") else float(val)

                # Learned-model checkpointing is handled by TuneReportCheckpointCallback.
                # Keep manual final artifact writing for closed-form models only.
            else:
                model_cfg = config["model"].copy()
                model_cfg.pop("name")
                model = build_model(base, model_name, model_cfg)
                
                train_preds, train_targets = predict_from_loader(model, train_loader)
                val_preds, val_targets = predict_from_loader(model, val_loader)
                test_preds, test_targets = predict_from_loader(model, test_loader)
                train_partition = train_ds
                val_partition = val_ds
                test_partition = test_ds
                
                train_eval = Evaluator(train_preds, train_targets, train_partition, base)
                val_eval = Evaluator(val_preds, val_targets, val_partition, base)

                train_metrics = train_eval._metrics
                val_metrics = val_eval._metrics
                
                # populate report_dict
                report_dict = {
                    "train_mse": train_metrics.get("mse", np.nan),
                    "train_pearson_r": train_metrics.get("pearson", np.nan),
                    "train_demeaned_r": train_metrics.get("demeaned_pearson", np.nan),
                    "val_mse": val_metrics.get("mse", np.nan),
                    "val_pearson_r": val_metrics.get("pearson", np.nan),
                    "val_demeaned_r": val_metrics.get("demeaned_pearson", np.nan),
                }

                checkpoint = None
                if should_write_final_artifact:
                    trial_dir = None
                    try:
                        trial_dir = tune.get_context().get_trial_dir()
                    except Exception:
                        trial_dir = None
                    checkpoint_dir = tempfile.mkdtemp(
                        prefix="closed_form_ckpt_",
                        dir=trial_dir if trial_dir and os.path.isdir(trial_dir) else None,
                    )
                    with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
                        json.dump(_to_serializable(config), f, indent=2)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)

                if should_write_final_artifact:
                    _write_final_artifact(trial_id, config, learned_trial=False, final_metrics=report_dict)
                session.report(report_dict, checkpoint=checkpoint)

        # Ray Tune requires the trainable to have a single positional parameter named 'config'.
        def _tune_trainable(config):
            return train_func(default_flat, param_space, config)

        resources = {"cpu": float(cpus_per_trial)}
        if gpus_per_trial and float(gpus_per_trial) > 0:
            resources["gpu"] = float(gpus_per_trial)
        print(f"Tune trial resources: {resources}", flush=True)

        train_with_resources = tune.with_resources(_tune_trainable, resources)
        callbacks = []
        
        if WandbLoggerCallback is not None:
            callbacks.append(
                TrialFamilyWandbLoggerCallback(
                    project=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    group=f"{self.model_name}_tune",
                    tags=[self.model_name, "tune"] + ([f"ray_tune_id:{ray_tune_id}"] if ray_tune_id else []),
                    log_config=True,
                    extra_config={
                        "data.source": self.source,
                        "data.target": self.target,
                        "data.parcellation": self.parcellation,
                        "data.hemi": self.hemi,
                        "data.shuffle_seed": self.shuffle_seed,
                    },
                )
            )
        
        reporter = CLIReporter()
        reporter.add_metric_column("val_demeaned_r")

        print(f"Tune run: {run_name}  num_samples={num_samples}  storage={RAY_CHECKPOINTS_DIR}", flush=True)
        run_config = tune.RunConfig(
            name=run_name,
            storage_path=RAY_CHECKPOINTS_DIR,
            callbacks=callbacks,
            progress_reporter=reporter,
        )

        try:
            import cloudpickle
            payload = cloudpickle.dumps(_tune_trainable)
            size_mb = len(payload) / (1024 * 1024)
            print(f"Tune trainable serialized size: {size_mb:.1f} MB", flush=True)
            if size_mb > 1800:
                raise RuntimeError(f"Trainable closure too large ({size_mb:.0f} MB); will exceed Ray 2 GB limit.")
        except ImportError:
            pass
        _search_alg = None
        if search_alg == "optuna":
            if OptunaSearch is None:
                raise RuntimeError("optuna is not installed. pip install optuna")
            _search_alg = OptunaSearch(metric=metric, mode=mode)
            print(f"Tune search algorithm: OptunaSearch (Bayesian optimization)", flush=True)
        else:
            print(f"Tune search algorithm: random", flush=True)

        print("Tune starting...", flush=True)
        tuner = tune.Tuner(
            train_with_resources,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                metric=metric,
                mode=mode,
                max_concurrent_trials=max_concurrent_trials,
                reuse_actors=True,
                search_alg=_search_alg,
                scheduler=ASHAScheduler(max_t=scheduler_max_t, grace_period=scheduler_max_t / 3, reduction_factor=2) if learned else None,
            ),
            run_config=run_config,
        )
        result_grid = tuner.fit()
        n = len(result_grid)
        print(f"Tune finished: {n} trial(s).", flush=True)
        return result_grid

    def report_best_tune_trial(
        self,
        results,
        metric: str = "val_demeaned_r",
        mode: str = "max",
        report_to_wandb: bool = False,
        store_eval_md: bool = False,
    ):
        """
        Select best Tune trial by validation metric and report comprehensive metrics.
        For learned models, selection is based purely on the chosen metric; a fresh
        prod run of the best config retrains once and saves a final checkpoint and
        eval report under RAY_RESULTS_DIR. For closed-form models, the analytic
        solution is recomputed from base + best config.
        """
        if self.learned:
            all_results = list(results)
            scored_results = []
            for res in all_results:
                metrics = getattr(res, "metrics", {}) or {}
                metric_value = metrics.get(metric)
                if metric_value is None:
                    continue
                try:
                    metric_value = float(metric_value)
                except (TypeError, ValueError):
                    continue
                if np.isnan(metric_value):
                    continue
                scored_results.append((res, metric_value))

            if not scored_results:
                raise RuntimeError(
                    f"No learned trials have a finite value for metric '{metric}'. "
                    "Cannot report best Tune trial."
                )

            reverse = (mode == "max")
            best = sorted(scored_results, key=lambda x: x[1], reverse=reverse)[0][0]
            print(
                f"Best-trial selection (learned) considered {len(scored_results)} trial(s) "
                f"with finite '{metric}'.",
                flush=True,
            )
        else:
            best = results.get_best_result(metric=metric, mode=mode)
        best_config_flat = dict(best.config)
        best_config = _flat_to_nested(best_config_flat, self.model_name)

        best_path = getattr(best, "path", None)
        best_metrics = getattr(best, "metrics", {}) or {}
        # Final run should use the same number of epochs the best trial actually ran
        # (e.g. if ASHA stopped at 100, refit for 100, not config max_epochs).
        if self.learned and "trainer" in best_config:
            actual_epochs = best_metrics.get("training_iteration")
            if actual_epochs is not None:
                best_config["trainer"]["max_epochs"] = int(actual_epochs)
        best_trial_id = best_metrics.get("trial_id")
        if not best_trial_id and best_path:
            m = re.search(r"([0-9a-f]{5}_\d{5})", best_path)
            best_trial_id = m.group(1) if m else None
        if not best_trial_id:
            best_trial_id = f"trial_{int(time.time())}"
        best_trial_family = best_trial_id.split("_")[0] if "_" in best_trial_id else best_trial_id
        ray_tune_id = _extract_ray_tune_id(best_path or "")
        best_run_name = f"{best_trial_family}_best"
        best_group = f"{self.model_name}_tune_{best_trial_family}"
        report_dir = os.path.join(RAY_RESULTS_DIR, self.model_name, best_trial_id)
        checkpoint_dir = os.path.join(RAY_CHECKPOINTS_DIR, "best", self.model_name, best_trial_id)

        best_checkpoint_path = None
        run_mode = "prod" if report_to_wandb else "dev"
        run_parent = os.path.dirname(best_path) if best_path else None
        canonical_artifact_dir = (
            os.path.join(run_parent, best_trial_id, "final")
            if run_parent else None
        )
        manifest = None
        if canonical_artifact_dir and os.path.isfile(os.path.join(canonical_artifact_dir, "artifact_manifest.json")):
            with open(os.path.join(canonical_artifact_dir, "artifact_manifest.json")) as f:
                manifest = json.load(f)

        if self.learned:
            # Retrain the best learned-config once in a clean prod run on the driver,
            # and save the final checkpoint alongside the eval report under RAY_RESULTS_DIR.
            os.makedirs(report_dir, exist_ok=True)
            run_out = self._run_learned_single(
                mode=run_mode,
                save_checkpoint=True,
                run_dir=report_dir,
                config_override=best_config,
                checkpoint_path=None,
                train_from_scratch=True,
                test_report_basename="test_results",
                wandb_name=best_run_name,
                wandb_group=best_group,
                wandb_tags=[
                    "best_trial_report",
                    f"source_trial:{best_trial_id}",
                ] + ([f"ray_tune_id:{ray_tune_id}"] if ray_tune_id else []),
                wandb_metadata={
                    "ray_tune_id": ray_tune_id,
                    "ray_trial_id": best_trial_id,
                    "source": self.source,
                    "shuffle_seed": self.shuffle_seed,
                    "target": self.target,
                },
                store_eval_md=store_eval_md,
            )
            final_ckpt = os.path.join(report_dir, "checkpoint.ckpt")
            best_checkpoint_path = final_ckpt if os.path.isfile(final_ckpt) else None
            # For learned models, treat the report_dir as the effective checkpoint directory.
            checkpoint_dir = report_dir
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            config_path = os.path.join(checkpoint_dir, "best_model_config.json")
            config_to_save = best_config
            if best.checkpoint is not None:
                ckpt_dir = best.checkpoint.to_directory()
                ckpt_config_path = os.path.join(ckpt_dir, "model_config.json")
                if os.path.isfile(ckpt_config_path):
                    with open(ckpt_config_path) as f:
                        config_to_save = json.load(f)
            with open(config_path, "w") as f:
                json.dump(_to_serializable(config_to_save), f, indent=2)
            best_checkpoint_path = config_path
            run_out = self._run_closed_form_single(
                mode=run_mode,
                save_checkpoint=False,
                run_dir=report_dir,
                config_override=best_config,
                test_report_basename="test_results",
                wandb_name=best_run_name,
                wandb_group=best_group,
                wandb_tags=[
                    "best_trial_report",
                    f"source_trial:{best_trial_id}",
                ] + ([f"ray_tune_id:{ray_tune_id}"] if ray_tune_id else []),
                wandb_metadata={
                    "ray_tune_id": ray_tune_id,
                    "ray_trial_id": best_trial_id,
                    "source": self.source,
                    "shuffle_seed": self.shuffle_seed,
                    "target": self.target,
                },
                store_eval_md=store_eval_md,
            )

        train_metrics = _to_serializable(run_out["metrics"]["train"])
        val_metrics = _to_serializable(run_out["metrics"]["val"])
        test_metrics = _to_serializable(run_out["metrics"]["test"])
        test_report_path = os.path.join(report_dir, "test_results")

        summary = {
            "selected_by": {
                "metric": metric,
                "mode": mode,
                "value": best.metrics.get(metric),
            },
            "ray_tune_id": ray_tune_id,
            "best_trial": {
                "id": best_trial_id,
                "path": best_path,
                "config": best_config,
                "checkpoint_saved_to": best_checkpoint_path,
                "best_run_name": best_run_name,
                "wandb_group": best_group,
            },
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "paths": {
                "report_dir": report_dir,
                "checkpoint_dir": checkpoint_dir,
                "test_report_md": (f"{test_report_path}.md" if store_eval_md else None),
            },
        }
        summary = _to_serializable(summary)

        if store_eval_md and summary["paths"]["test_report_md"] and os.path.isfile(summary["paths"]["test_report_md"]):
            with open(summary["paths"]["test_report_md"], "r") as f:
                report_body = f.read()
            trial_meta = (
                f"## Ray Tune Linkage\n\n"
                f"- Ray Tune id: `{ray_tune_id}`\n"
                f"- Ray Tune trial id: `{best_trial_id}`\n"
                f"- Ray Tune trial path: `{best_path}`\n"
                f"- Ray Tune trial dir link: [{best_trial_id}]({best_path})\n\n"
            )
            with open(summary["paths"]["test_report_md"], "w") as f:
                f.write(trial_meta + report_body)

        print("Best Tune trial comprehensive summary:", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        if store_eval_md:
            print(f"Saved test report to {summary['paths']['test_report_md']}", flush=True)
        return summary


def _parse_args():
    p = argparse.ArgumentParser(description="Conn2Conn: fit and evaluate models")
    p.add_argument("--mode", choices=["dev", "prod"], default="dev")
    p.add_argument("--model", type=str, default="CrossModal_PCA_PLS_learnable")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config overrides (optional)")
    p.add_argument(
        "--parcellation",
        type=str,
        default="Glasser",
        help="Parcellation to load for FC/SC data.",
    )
    p.add_argument(
        "--hemi",
        choices=["left", "right", "both"],
        default="both",
        help="Hemisphere subset to use.",
    )
    p.add_argument(
        "--source",
        default="SC",
        help="Input/source modality. Use '+' to combine sources, e.g. SC+SC_r2t.",
    )
    p.add_argument(
        "--target",
        default="FC",
        help="Prediction/target modality. Single modality only.",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Shuffle seed for family-aware train/val/test repartitioning. Use 0 for the original split.",
    )
    p.add_argument(
        "--data_load_mode",
        choices=["manual", "precomputed"],
        default=None,
        help="Data loading mode. 'manual' reads raw source files; 'precomputed' reads cached npy arrays.",
    )
    p.add_argument("--save_checkpoint", action="store_true")
    p.add_argument("--use_tune", action="store_true")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument(
        "--search_alg",
        default="random",
        choices=["random", "optuna"],
        help="Hyperparameter search algorithm. 'random' = pure random (default). 'optuna' = Bayesian optimization via Optuna.",
    )
    p.add_argument(
        "--max_concurrent_trials",
        type=int,
        default=None,
        help="Maximum number of Ray Tune trials to run concurrently. Set to 1 for sequential execution.",
    )
    p.add_argument(
        "--tune_cpus_per_trial",
        type=float,
        default=float(os.environ.get("TUNE_CPUS_PER_TRIAL", 2)),
        help="Ray Tune CPUs allocated per trial.",
    )
    p.add_argument(
        "--tune_gpus_per_trial",
        type=float,
        default=float(os.environ.get("TUNE_GPUS_PER_TRIAL", 1)),
        help="Ray Tune GPUs allocated per trial.",
    )
    p.add_argument(
        "--report_best_after_tune",
        action="store_true",
        help="After Tune completes, report comprehensive train/val/test metrics for the best trial only.",
    )
    p.add_argument(
        "--store_eval_md",
        action="store_true",
        help="Store the eval markdown report for the test split for a single run or best Tune trial.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = None
    
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            data = json.load(f) if args.config.endswith(".json") else yaml.safe_load(f)
        overrides = data.get("default", data) if isinstance(data, dict) else None
    
    sim = Sim(
        model_name=args.model,
        config_path=args.config,
        config_overrides=overrides,
        parcellation=args.parcellation,
        hemi=args.hemi,
        source=args.source,
        target=args.target,
        shuffle_seed=args.shuffle_seed,
        data_load_mode=args.data_load_mode,
    )

    if args.use_tune and args.mode == "prod":
        tune_results = sim.run_tune(
            num_samples=args.num_samples,
            save_checkpoint=args.save_checkpoint,
            cpus_per_trial=args.tune_cpus_per_trial,
            gpus_per_trial=args.tune_gpus_per_trial,
            max_concurrent_trials=args.max_concurrent_trials,
            search_alg=args.search_alg,
            persist_final_artifacts=(args.save_checkpoint or args.report_best_after_tune or args.store_eval_md),
        )
        if args.report_best_after_tune or args.store_eval_md:
            sim.report_best_tune_trial(
                tune_results,
                metric="val_demeaned_r",
                mode="max",
                report_to_wandb=args.report_best_after_tune,
                store_eval_md=args.store_eval_md,
            )
    else:
        sim.run_single(
            mode=args.mode,
            save_checkpoint=args.save_checkpoint,
            store_eval_md=args.store_eval_md,
        )
