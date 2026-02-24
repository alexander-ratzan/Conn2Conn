"""
Main script for Conn2Conn: fit and evaluate models.
Use the Sim class to set up an experiment (data, config) and run single or tuning jobs.
Callable from notebook or via CLI/sbatch (python main.py --mode dev|prod).
"""
import argparse
import json
import os
import shutil
import sys
import time
import tempfile
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.hcp_dataset import HCP_Base, HCP_Partition
from models.models import predict_from_loader
from models.loss import train_model, get_target_train_mean
from models.eval import Evaluator
from models.config import (
    load_config,
    get_default_config,
    get_search_space,
    build_model,
    TRAINER_KEYS,
)

import wandb
import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air import session
from ray.train import Checkpoint
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from models.lightning_module import CrossModalLightningModule


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


def _flat_to_nested(flat: dict, model_name: str) -> dict:
    """Split flat config (e.g. from Tune) into nested {model: {...}, trainer: {...}}."""
    trainer = {k: flat[k] for k in TRAINER_KEYS if k in flat}
    model = {"name": model_name, **{k: flat[k] for k in flat if k not in TRAINER_KEYS and k != "name"}}
    return {"model": model, "trainer": trainer}


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
        source: str = "SC",
        target: str = "FC",
        shuffle_seed: int = 0,
        batch_size: int = 128,
    ):
        """
        model_name: name of the model (must have models/configs/<model_name>.yml).
        config_path: optional path to a YAML config file (overrides model_name lookup).
        config_overrides: optional dict merged onto default config (nested: { "model": {...}, "trainer": {...} }).
        Remaining args: data options for HCP_Base and DataLoader batch_size.
        """
        self.model_name = model_name
        full = load_config(model_name, path=config_path)
        self._raw_config = full
        self.learned = full.get("learned", True)

        self.config = get_default_config(model_name, path=config_path)
        if config_overrides:
            for section in ("model", "trainer"):
                if section in config_overrides and section in self.config:
                    self.config[section].update(config_overrides[section])

        batch_size = self.config.get("trainer", {}).get("batch_size", batch_size)

        self.base = HCP_Base(
            parcellation=parcellation,
            shuffle_seed=shuffle_seed,
            source=source,
            target=target,
        )
        train_ds = HCP_Partition(self.base, "train")
        val_ds = HCP_Partition(self.base, "val")
        test_ds = HCP_Partition(self.base, "test")
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def run_single(self, mode: str = "dev", save_checkpoint: bool = False, run_dir: str = None):
        """
        Run a single configuration (one model, one training run). No Ray; runs in this process.

        mode: "dev" = no wandb, live plot, verbose=False. "prod" = log to wandb, verbose eval.
        For one run with wandb from a notebook or script, use run_single(mode="prod").
        save_checkpoint: if True, write checkpoint and eval report to run_dir (default results/run_<timestamp>).
        Returns dict with model, evaluators, test_metrics, and for learned models train_result.
        """
        if run_dir is None and save_checkpoint:
            run_dir = os.path.join("results", f"run_{int(time.time())}")
        
        if mode == "prod" and wandb is not None and not self.learned: # need to do this specifically for 'closed form' models
            wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, config=self.config, reinit=True, tags=[self.model_name, "prod"])
        
        try:
            if self.learned:
                return self._run_learned_single(mode, save_checkpoint, run_dir)
            else: 
                return self._run_closed_form_single(mode, save_checkpoint, run_dir)
        finally:
            if mode == "prod" and wandb is not None: # fallback to finish any wandb run - can consider replacing with WandbLogger to manage init and finish
                wandb.finish()

    def _merge_config(self, config_override: dict = None) -> dict:
        cfg = deepcopy(self.config)
        if not config_override:
            return cfg
        for section in ("model", "trainer"):
            if section in config_override and isinstance(config_override[section], dict):
                cfg.setdefault(section, {})
                cfg[section].update(config_override[section])
        return cfg

    def _evaluate_model(self, model, mode: str = "dev", test_report_path: str = None):
        train_preds, train_targets = predict_from_loader(model, self.train_loader)
        val_preds, val_targets = predict_from_loader(model, self.val_loader)
        test_preds, test_targets = predict_from_loader(model, self.test_loader)
        train_partition = HCP_Partition(self.base, "train")
        val_partition = HCP_Partition(self.base, "val")
        test_partition = HCP_Partition(self.base, "test")
        train_eval = Evaluator(train_preds, train_targets, train_partition, self.base)
        val_eval = Evaluator(val_preds, val_targets, val_partition, self.base)
        test_eval = Evaluator(test_preds, test_targets, test_partition, self.base)
        test_metrics = test_eval.analyze_results(verbose=(mode == "prod"), filepath=test_report_path)
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
        wandb_tags: list = None,
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
                tags=tags,
                name=wandb_name or f"{self.model_name}_prod",
            )
            # log full config once
            pl_logger.log_hyperparams(cfg)

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
                max_epochs=max_epochs,
                logger=False if mode == "dev" else (pl_logger is None),
                pl_logger=pl_logger,
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
            )
            pl_module.load_state_dict(ckpt_state["state_dict"], strict=False)
            model = pl_module.model

        test_filepath = None
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            if save_checkpoint or (store_eval_md and (not train_from_scratch and checkpoint_path)):
                test_filepath = os.path.join(run_dir, test_report_basename)
        eval_out = self._evaluate_model(model, mode=mode, test_report_path=test_filepath)
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
        wandb_tags: list = None,
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
        eval_out = self._evaluate_model(model, mode=mode, test_report_path=test_filepath)
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
            )
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
        gpus_per_trial: float = 0.0,
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
            ray_init_kwargs = {"ignore_reinit_error": True}
            ray_tmpdir = os.environ.get("RAY_TMPDIR")
            if ray_tmpdir:
                ray_tmpdir = os.path.abspath(ray_tmpdir)
            else:
                ray_tmpdir = f"/tmp/ray_{os.environ.get('SLURM_JOB_ID', 'local')}"
            os.makedirs(ray_tmpdir, exist_ok=True)
            ray_init_kwargs["_temp_dir"] = ray_tmpdir
            ray.init(**ray_init_kwargs)
        print(f"Ray cluster resources: {ray.cluster_resources()}", flush=True)
        # Avoid capturing the full Sim instance (which can include large in-memory data)
        # inside the Ray trainable closure.
        model_name = self.model_name
        learned = self.learned
        param_space = get_search_space(self.model_name)
        if not param_space:
            raise ValueError(f"No search_space for model: {self.model_name} in config YAML.")
        default = get_default_config(self.model_name)
        default_flat = {"name": self.model_name}
        for k, v in default.get("model", {}).items():
            if k != "name":
                default_flat[k] = v
        for k, v in default.get("trainer", {}).items():
            default_flat[k] = v
            

        def train_func(default_flat, param_space, tune_config):
            config_flat = {**default_flat, **{k: v for k, v in tune_config.items() if k in param_space or k in TRAINER_KEYS}}
            config_flat["name"] = model_name
            config = _flat_to_nested(config_flat, model_name)
            batch_size = config.get("trainer", {}).get("batch_size", 128)
            base = HCP_Base(parcellation="Glasser", shuffle_seed=0, source="SC", target="FC")
            train_ds = HCP_Partition(base, "train")
            val_ds = HCP_Partition(base, "val")
            test_ds = HCP_Partition(base, "test")
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            report_dict = {}

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
                )
                tune_metrics = {
                    "train_loss": "train_loss",
                    "val_loss": "val_loss",
                    "val_demeaned_r": "val_demeaned_r",
                    "val_pearson_r": "val_pearson_r",
                }
                callbacks = [TuneReportCallback(metrics=tune_metrics, on="validation_end")]
                trainer = pl.Trainer(max_epochs=epochs, logger=False, callbacks=callbacks, enable_progress_bar=False)
                trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
                final_report = {}
                for metric_name in tune_metrics:
                    val = trainer.callback_metrics.get(metric_name)
                    if val is not None:
                        final_report[metric_name] = float(val.item()) if hasattr(val, "item") else float(val)

                if save_checkpoint:
                    try:
                        trial_dir = tune.get_context().get_trial_dir()
                        run_dir = os.path.join("results", os.path.basename(trial_dir))
                    except Exception:
                        trial_dir = None
                        run_dir = os.path.join("results", f"tune_{model_name}_{id(tune_config)}")
                    os.makedirs(run_dir, exist_ok=True)
                    final_ckpt = os.path.join(run_dir, "checkpoint.ckpt")
                    trainer.save_checkpoint(final_ckpt)

                    # Report exactly one checkpoint to Ray at fit end.
                    checkpoint_dir = tempfile.mkdtemp(
                        prefix="final_ckpt_",
                        dir=trial_dir if trial_dir and os.path.isdir(trial_dir) else None,
                    )
                    shutil.copy2(final_ckpt, os.path.join(checkpoint_dir, "checkpoint.ckpt"))
                    session.report(final_report, checkpoint=Checkpoint.from_directory(checkpoint_dir))
            else:
                model_cfg = config["model"].copy()
                model_cfg.pop("name")
                model = build_model(base, model_name, model_cfg)
                
                train_preds, train_targets = predict_from_loader(model, train_loader)
                val_preds, val_targets = predict_from_loader(model, val_loader)
                test_preds, test_targets = predict_from_loader(model, test_loader)
                train_partition = HCP_Partition(base, "train")
                val_partition = HCP_Partition(base, "val")
                test_partition = HCP_Partition(base, "test")
                
                train_eval = Evaluator(train_preds, train_targets, train_partition, base)
                val_eval = Evaluator(val_preds, val_targets, val_partition, base)
                test_eval = Evaluator(test_preds, test_targets, test_partition, base)

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

                session.report(report_dict)

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
                WandbLoggerCallback(
                    project=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    group=f"{self.model_name}_tune",
                    tags=[self.model_name, "tune"],
                    log_config=True,
                )
            )
        
        reporter = CLIReporter()
        reporter.add_metric_column("val_demeaned_r")
        run_name = f"{self.model_name}_tune_{int(time.time())}"
        print(f"Ray checkpoint storage path: {RAY_CHECKPOINTS_DIR}", flush=True)
        run_config = tune.RunConfig(
            name=run_name,
            storage_path=RAY_CHECKPOINTS_DIR,
            callbacks=callbacks,
            progress_reporter=reporter,
        )
        tuner = tune.Tuner(
            train_with_resources,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                metric=metric,
                mode=mode,
                scheduler=ASHAScheduler(max_t=max_epochs or 100, grace_period=1, reduction_factor=2) if learned else None,
            ),
            run_config=run_config,
        )

        return tuner.fit()

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
        Uses best checkpoint/config, writes test eval report via Evaluator.analyze_results,
        and logs train/val/test + selection metadata to W&B.
        """
        best = results.get_best_result(metric=metric, mode=mode)
        best_config_flat = dict(best.config)
        best_config = _flat_to_nested(best_config_flat, self.model_name)

        best_path = getattr(best, "path", None)
        best_trial_id = os.path.basename(best_path) if best_path else f"best_{int(time.time())}"
        report_dir = os.path.join(RAY_RESULTS_DIR, self.model_name, best_trial_id)
        checkpoint_dir = os.path.join(RAY_CHECKPOINTS_DIR, "best", self.model_name, best_trial_id)

        best_checkpoint_path = None
        run_mode = "prod" if report_to_wandb else "dev"
        if self.learned:
            ckpt_dir = best.checkpoint.to_directory() if best.checkpoint is not None else None
            if ckpt_dir is None:
                raise RuntimeError("Best trial has no checkpoint; cannot evaluate learned model on test.")
            ckpt_candidates = list(Path(ckpt_dir).rglob("*.ckpt"))
            if not ckpt_candidates:
                ckpt_candidates = list(Path(ckpt_dir).rglob("checkpoint"))
            if not ckpt_candidates:
                raise FileNotFoundError(f"No checkpoint file found under {ckpt_dir}")
            source_ckpt_path = str(ckpt_candidates[0])
            best_checkpoint_copy = os.path.join(checkpoint_dir, "best_checkpoint.ckpt")
            shutil.copy2(source_ckpt_path, best_checkpoint_copy)
            best_checkpoint_path = best_checkpoint_copy
            run_out = self._run_learned_single(
                mode=run_mode,
                save_checkpoint=False,
                run_dir=report_dir,
                config_override=best_config,
                checkpoint_path=best_checkpoint_path,
                train_from_scratch=False,
                test_report_basename="test_results",
                wandb_name=f"{self.model_name}_best_{best_trial_id}",
                wandb_tags=["best_trial_report", f"source_trial:{best_trial_id}"],
                store_eval_md=store_eval_md,
            )
        else:
            config_path = os.path.join(checkpoint_dir, "best_model_config.json")
            with open(config_path, "w") as f:
                json.dump(_to_serializable(best_config), f, indent=2)
            best_checkpoint_path = config_path
            run_out = self._run_closed_form_single(
                mode=run_mode,
                save_checkpoint=False,
                run_dir=report_dir,
                config_override=best_config,
                test_report_basename="test_results",
                wandb_name=f"{self.model_name}_best_{best_trial_id}",
                wandb_tags=["best_trial_report", f"source_trial:{best_trial_id}"],
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
            "best_trial": {
                "path": best_path,
                "config": best_config,
                "checkpoint_saved_to": best_checkpoint_path,
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
    p.add_argument("--save_checkpoint", action="store_true")
    p.add_argument("--use_tune", action="store_true")
    p.add_argument("--num_samples", type=int, default=10)
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
        help="Store eval markdown report for the best trial (test split) under results/ray_results.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = None
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            data = json.load(f) if args.config.endswith(".json") else yaml.safe_load(f)
        overrides = data.get("default", data) if isinstance(data, dict) else None
    sim = Sim(model_name=args.model, config_overrides=overrides)
    if args.use_tune and args.mode == "prod":
        tune_results = sim.run_tune(
            num_samples=args.num_samples,
            save_checkpoint=args.save_checkpoint,
            cpus_per_trial=args.tune_cpus_per_trial,
            gpus_per_trial=args.tune_gpus_per_trial,
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
        sim.run_single(mode=args.mode, save_checkpoint=args.save_checkpoint)
