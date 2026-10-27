"""Spec v3 E1: train + evaluate CovProjector conditions for one (direction, SC metric, seed) cell.

Loads the data once (all covariate sources), then for each condition trains the frozen E2.2 config with that
condition's cov_sources and writes runs/<sc_metric>/<direction>/<condition>/seed<S>.json. Existing outputs are
skipped, so a requeued task resumes. Mirrors Sim._run_learned_single without W&B.

  python scripts/experiments/covariate_projector/covariate_ablation/run_cell.py --direction fc2sc --sc_metric invnodevol --seed 0 \
      --conditions none full e2_ref
"""
import argparse
import json
import os
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EXP_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from conditions import condition_config, condition_sources, frozen_config, grid_conditions, load_config  # noqa: E402
from main import Sim, _to_serializable  # noqa: E402
from models.registry import build_model  # noqa: E402
from models.train.loss import resolve_loss_config  # noqa: E402
from models.train.trainer import train_model  # noqa: E402

DIRECTIONS = {"fc2sc": ("FC", "SC"), "sc2fc": ("SC", "FC")}


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", REPO_ROOT, "diff", "--quiet", "HEAD", "--", "data", "models",
                                "scripts/experiments/covariate_projector/covariate_ablation"]).returncode != 0
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return None


def _metrics(m):
    return {k: float(v) for k, v in (m or {}).items() if isinstance(v, (int, float, np.floating))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    p.add_argument("--sc_metric", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--conditions", nargs="+", default=["all"])
    a = p.parse_args()

    cfg = load_config()
    sources = condition_sources(cfg)
    conditions = grid_conditions(cfg) if a.conditions == ["all"] else a.conditions
    unknown = [c for c in conditions if c not in sources]
    if unknown:
        raise ValueError(f"unknown conditions {unknown}")
    sc_metric_type = cfg["sc_metrics"][a.sc_metric]
    source, target = DIRECTIONS[a.direction]
    out_root = os.path.join(EXP_DIR, cfg["runs_dir"], a.sc_metric, a.direction)
    todo = [c for c in conditions if not os.path.isfile(os.path.join(out_root, c, f"seed{a.seed}.json"))]
    print(f"[e1] {a.direction} {a.sc_metric} seed {a.seed}: {len(todo)}/{len(conditions)} to run: {todo}", flush=True)
    if not todo:
        return

    frozen, meta = frozen_config(cfg, a.direction, a.seed)
    all_sources = sorted({s for c in sources.values() for s in c})
    init_cfg = {"model": {**frozen["model"], "cov_sources": all_sources}, "trainer": frozen["trainer"],
                "data": {"sc_metric_type": sc_metric_type, "sc_apply_log1p": True}}
    t0 = time.time()
    sim = Sim(model_name=cfg["model"], config_path=os.path.join(REPO_ROOT, cfg["frozen"]["default"]),
              config_overrides=init_cfg, parcellation=cfg["parcellation"], source=source, target=target,
              shuffle_seed=a.seed, data_load_mode="precomputed")
    load_s = time.time() - t0
    cov_dims = sim.base.cov_dims
    assert sim.base.sc_metric_type == sc_metric_type, (sim.base.sc_metric_type, sc_metric_type)
    print(f"[e1] data loaded in {load_s:.0f}s; cov_dims={cov_dims}; n_test={len(sim.test_ds)}", flush=True)

    for cond in todo:
        t1 = time.time()
        run_cfg = condition_config(frozen, sources[cond], cov_dims, cfg.get("fs_projector_min_dim", 2))
        merged = sim._merge_config(run_cfg)
        data_ctx = sim._build_runtime_data_context(merged)
        assert data_ctx["base"] is sim.base, "covariate subset should reuse the loaded base"
        model_cfg = merged["model"].copy()
        model_cfg.pop("name")
        trainer_cfg = merged["trainer"]
        model = build_model(data_ctx["base"], cfg["model"], model_cfg)
        train_result = train_model(
            model, data_ctx["train_loader"], data_ctx["val_loader"], base=data_ctx["base"],
            log_every=trainer_cfg.get("log_every", 5), lr=trainer_cfg.get("lr", 1e-4),
            loss_cfg=resolve_loss_config(trainer_cfg), max_epochs=trainer_cfg.get("max_epochs", 100),
            lr_schedule=trainer_cfg.get("lr_schedule", "none"), logger=False, enable_progress_bar=False,
        )
        model = train_result.pl_module.model
        model.eval()
        ev = sim._evaluate_model(model, mode="dev", model_name=cfg["model"], base=data_ctx["base"],
                                 train_loader=data_ctx["train_loader"], val_loader=data_ctx["val_loader"],
                                 test_loader=data_ctx["test_loader"])
        record = {
            "campaign": cfg["campaign"], "condition": cond, "direction": a.direction, "source": source,
            "target": target, "sc_metric": a.sc_metric, "sc_metric_type": sc_metric_type, "seed": a.seed,
            "cov_sources": sources[cond], "cov_dims": {s: cov_dims[s] for s in sources[cond]},
            "config": _to_serializable(run_cfg), "frozen_from": meta,
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "val": _metrics(ev["metrics"]["val"]), "test": _metrics(ev["metrics"]["test"]),
            "train_s": round(time.time() - t1, 1), "data_load_s": round(load_s, 1),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": _git_commit(),
        }
        out_dir = os.path.join(out_root, cond)
        os.makedirs(out_dir, exist_ok=True)
        tmp = os.path.join(out_dir, f".seed{a.seed}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(record, f, indent=1)
        os.replace(tmp, os.path.join(out_dir, f"seed{a.seed}.json"))
        print(f"[e1] {cond}: test demeaned_r={record['test'].get('demeaned_pearson'):.4f} "
              f"top1={record['test'].get('top1_acc'):.3f} val demeaned_r={record['val'].get('demeaned_pearson'):.4f} "
              f"({record['train_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
