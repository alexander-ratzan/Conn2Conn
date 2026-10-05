"""Real-data verification for the modeling track (spec §8: M5, M6, M7, M8).

Run through verify_modeling_track_array.sh (one task per array index):
  dev_runs    M8: 2-epoch dev run per model family; loss signature, per-term logging, finite metrics.
  cross_tree  M7a + M5: old code (git archive in $OLD_TREES) vs working tree on the real dataset.
  tune        M6: 2-trial Ray Tune run (W&B offline); per-term losses reach the trial results.

Each task writes results/logs/verify_modeling_track_<task>.json and exits non-zero on failure.
Internal: `--worker <spec.json> <out.pt>` builds one model inside whichever tree is on PYTHONPATH.
"""
import glob
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "results" / "logs"
PYG_MODELS = {"Chen2024GCN", "NodalGNN"}  # need torch_geometric, missing from kraken_env (spec M5)


def _report(task, checks):
    failed = [c for c in checks if not c["ok"]]
    out = {"task": task, "n_checks": len(checks), "n_failed": len(failed), "checks": checks}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"verify_modeling_track_{task}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    for c in checks:
        print(("ok  " if c["ok"] else "FAIL"), c["name"], c.get("detail", ""), flush=True)
    print(f"{task}: {len(checks) - len(failed)}/{len(checks)} passed -> {path}", flush=True)
    return 1 if failed else 0


def _has_pyg():
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- dev_runs (M8)
DEV_MODELS = [
    "CrossModal_PCA_PLS_learnable",
    "CrossModal_PCA_PLS_CovProjector",
    "Sarwar2020MLP",
    "CrossModalVAE",
    "LatentAttnMasked",
    "CrossModal_linear_backbone",
    "NodalMLP",
    "MaskedMLPPretrainer",
    "MaskedLatentPretrainer",
    "Chen2024GCN",
    "NodalGNN",
]


def task_dev_runs():
    import numpy as np
    from main import Sim
    from models.registry import load_config

    checks = []
    for name in DEV_MODELS:
        if name in PYG_MODELS and not _has_pyg():
            checks.append({"name": f"{name} dev run", "ok": True, "detail": "SKIPPED: torch_geometric not installed"})
            continue
        t0 = time.time()
        try:
            data_cfg = load_config(name)["default"].get("data", {})
            sim = Sim(model_name=name, source=data_cfg.get("source", "SC"), target=data_cfg.get("target", "FC"),
                      shuffle_seed=0, data_load_mode="precomputed")
            run_out = sim.run_single(mode="dev", config_override={"trainer": {"max_epochs": 2}}, run_eval=True)
            tr = run_out["train_result"]
            hp = dict(tr.pl_module.hparams)
            cm = {k: float(v) for k, v in tr.trainer.callback_metrics.items()}
            base = run_out["test_metrics"]["base_metrics"]
            finite = all(math.isfinite(float(v)) for v in base.values() if isinstance(v, (int, float, np.floating)))
            detail = {"loss_signature": hp.get("loss_signature"), "loss_type": hp.get("loss_type"),
                      "test_pearson": base.get("pearson"), "secs": round(time.time() - t0, 1)}
            ok = finite and hp.get("loss_signature") is not None
            if hp.get("loss_type") == "composite":
                terms = [k[len("val_loss_raw_"):] for k in cm if k.startswith("val_loss_raw_")]
                detail["terms_logged"] = sorted(terms)
                ok = ok and "mse" in terms and "train_loss_raw_mse" in cm
                # total val_loss = sum of weighted terms + reg: checks the logging adds up
                weighted = sum(v for k, v in cm.items() if k.startswith("val_loss_weighted_"))
                recon = weighted + cm.get("val_reg_loss", 0.0)
                detail["val_loss_vs_terms+reg"] = [cm.get("val_loss"), recon]
                ok = ok and abs(cm["val_loss"] - recon) <= 1e-4 * max(1.0, abs(recon))
            checks.append({"name": f"{name} dev run", "ok": bool(ok), "detail": detail})
        except Exception as e:  # report and continue with the next family
            checks.append({"name": f"{name} dev run", "ok": False, "detail": f"{type(e).__name__}: {e}"})
    return _report("dev_runs", checks)


# --------------------------------------------------------------------------- cross_tree (M7a + M5)
def _worker(spec_path, out_path):
    """Runs inside one source tree (its `models` first on PYTHONPATH)."""
    import torch
    from models.registry import build_model, load_config

    spec = json.loads(Path(spec_path).read_text())
    with open(spec["base_pkl"], "rb") as f:
        base = pickle.load(f)
    kwargs = dict(spec.get("kwargs") or {})
    if spec.get("from_yaml"):
        yaml_model = dict(load_config(spec["name"])["default"]["model"])
        yaml_model.pop("name", None)
        kwargs = {**yaml_model, **kwargs}
    kwargs["device"] = "cpu"
    torch.manual_seed(0)
    model = build_model(base, spec["name"], kwargs)
    if spec.get("load"):
        weights = torch.load(spec["load"])
        params = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in weights.items():
                params[k].copy_(v)
    model.eval()
    res = {"state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    reg = model.get_reg_loss()
    res["reg"] = float(reg.detach()) if torch.is_tensor(reg) else float(reg)
    if spec.get("x_pt"):
        x = torch.load(spec["x_pt"])
        with torch.no_grad():
            res["y"] = model(x)
    if spec.get("save_backbone"):
        lin = model.backbone_linear
        torch.save({"W_mid": lin.weight.detach().T.clone(), "mid_bias": lin.bias.detach().clone()}, spec["save_backbone"])
    torch.save(res, out_path)


def _run_in_tree(tree, spec, tmp, tag):
    spec_path, out_path = tmp / f"{tag}.json", tmp / f"{tag}.pt"
    spec_path.write_text(json.dumps(spec))
    env = {**os.environ, "PYTHONPATH": f"{tree}:{REPO_ROOT}"}
    proc = subprocess.run([sys.executable, __file__, "--worker", str(spec_path), str(out_path)],
                          env=env, cwd=str(tree), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError((proc.stderr.strip().splitlines() or ["worker failed"])[-1])
    import torch
    return torch.load(out_path)


def task_cross_tree():
    import torch
    from main import Sim
    from models.architectures.utils import get_model_input

    trees = Path(os.environ["OLD_TREES"])
    pre_m7, pre_m5 = trees / "pre_m7", trees / "pre_m5"
    tmp = Path(os.environ.get("SLURM_TMPDIR") or "/tmp") / "verify_cross_tree"
    tmp.mkdir(parents=True, exist_ok=True)

    # One real dataset (NodalMLP exposes node features / SC matrices; PCA summaries are shared).
    sim = Sim(model_name="NodalMLP", source="SC", target="FC", shuffle_seed=0, data_load_mode="precomputed")
    base_pkl = tmp / "base.pkl"
    with open(base_pkl, "wb") as f:
        pickle.dump(sim.base, f)
    x_pt = tmp / "x_val.pt"
    torch.save(get_model_input(next(iter(sim.val_loader)))[:32].float().cpu(), x_pt)

    checks = []
    # M7a: LatentAttnMasked(residual_mode="none") from the pre-M7 tree == CrossModal_linear_backbone.
    for k, z in [(128, False), (256, False), (128, True)]:
        name = f"M7a real data k={k} zscore={z}"
        try:
            w = tmp / f"bb_{k}_{z}.pt"
            old = _run_in_tree(pre_m7, {"name": "LatentAttnMasked", "base_pkl": str(base_pkl), "x_pt": str(x_pt),
                                        "save_backbone": str(w), "kwargs": {"n_components_pca": k, "n_components_pls": 16,
                                        "residual_mode": "none", "residual_gain_init": 0.0, "zscore_pca_scores": z}},
                               tmp, f"old_{k}_{z}")
            new = _run_in_tree(REPO_ROOT, {"name": "CrossModal_linear_backbone", "base_pkl": str(base_pkl), "x_pt": str(x_pt),
                                           "load": str(w), "kwargs": {"n_components_pca_source": k, "zscore_pca_scores": z}},
                               tmp, f"new_{k}_{z}")
            d = float((old["y"] - new["y"]).abs().max())
            scale = float(old["y"].abs().max())
            checks.append({"name": name, "ok": d <= 1e-5 * max(1.0, scale), "detail": {"max_abs_diff": d, "max_abs_y": scale}})
        except Exception as e:
            checks.append({"name": name, "ok": False, "detail": f"{type(e).__name__}: {e}"})

    # M5: pre-M5 YAML (scalar reg) vs working-tree YAML (l1_reg/l2_reg): same init, same reg loss.
    for model in ["NodalMLP", "NodalMLP_spectral", "LatentAttnMasked", "MaskedMLPPretrainer"]:
        name = f"M5 real data reg {model}"
        cls = "NodalMLP" if model.startswith("NodalMLP") else model
        try:
            spec = {"name": cls, "base_pkl": str(base_pkl), "from_yaml": True}
            if model != cls:  # variant YAML: pass its model section explicitly
                import yaml
                for tree, tag in ((pre_m5, "old"), (REPO_ROOT, "new")):
                    y = yaml.safe_load((tree / "models" / "configs" / f"{model}.yml").read_text())["default"]["model"]
                    y.pop("name", None)
                    spec[f"kwargs_{tag}"] = y
            old = _run_in_tree(pre_m5, {**spec, "from_yaml": model == cls, "kwargs": spec.get("kwargs_old")}, tmp, f"m5_old_{model}")
            new = _run_in_tree(REPO_ROOT, {**spec, "from_yaml": model == cls, "kwargs": spec.get("kwargs_new")}, tmp, f"m5_new_{model}")
            same_state = sorted(old["state"]) == sorted(new["state"]) and all(torch.equal(old["state"][k], new["state"][k]) for k in old["state"])
            checks.append({"name": name, "ok": same_state and old["reg"] == new["reg"] and old["reg"] > 0,
                           "detail": {"reg_old": old["reg"], "reg_new": new["reg"], "same_state": same_state}})
        except Exception as e:
            checks.append({"name": name, "ok": False, "detail": f"{type(e).__name__}: {e}"})
    return _report("cross_tree", checks)


# --------------------------------------------------------------------------- tune (M6)
def task_tune():
    import yaml

    from models.registry import _config_path
    src = yaml.safe_load(Path(_config_path("CrossModal_linear_backbone")).read_text())
    src["search_space"] = {
        "lr": src["search_space"]["lr"],
        "max_epochs": {"type": "choice", "values": [3]},
        "loss_weight_neidist": {"type": "choice", "values": [0.5]},
        "loss_weight_correye": {"type": "choice", "values": [0.25]},
    }
    tmp = Path(os.environ.get("SLURM_TMPDIR") or "/tmp")
    cfg_path = tmp / "CrossModal_linear_backbone_verify.yml"
    cfg_path.write_text(yaml.safe_dump(src, sort_keys=False))
    ckpt_root = REPO_ROOT / "results" / "ray_checkpoints"
    before = set(glob.glob(str(ckpt_root / "CrossModal_linear_backbone_tune_*")))
    cmd = [sys.executable, "main.py", "--mode", "prod", "--model", "CrossModal_linear_backbone", "--config", str(cfg_path),
           "--source", "SC", "--target", "FC", "--shuffle_seed", "0", "--data_load_mode", "precomputed",
           "--use_tune", "--num_samples", "2", "--max_concurrent_trials", "1",
           "--tune_cpus_per_trial", os.environ.get("TUNE_CPUS_PER_TRIAL", "4"),
           "--tune_gpus_per_trial", os.environ.get("TUNE_GPUS_PER_TRIAL", "1")]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ, "WANDB_MODE": "offline"}, capture_output=True, text=True)
    checks = [{"name": "tune run exits 0", "ok": proc.returncode == 0,
               "detail": "" if proc.returncode == 0 else (proc.stderr.strip().splitlines() or ["?"])[-1]}]
    new_dirs = sorted(set(glob.glob(str(ckpt_root / "CrossModal_linear_backbone_tune_*"))) - before)
    checks.append({"name": "new tune directory", "ok": bool(new_dirs), "detail": new_dirs})
    wanted = ["train_loss_raw_mse", "train_loss_raw_correye", "train_loss_raw_neidist",
              "val_loss_raw_mse", "val_loss_raw_neidist", "val_loss_weighted_neidist", "val_loss_ref_neidist"]
    results = glob.glob(os.path.join(new_dirs[-1], "*", "result.json")) if new_dirs else []
    checks.append({"name": "2 trial result files", "ok": len(results) == 2, "detail": results})
    for path in results:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        keys = set().union(*(r.keys() for r in rows)) if rows else set()
        missing = [w for w in wanted if w not in keys]
        checks.append({"name": f"per-term metrics reported ({Path(path).parent.name})", "ok": not missing and len(rows) >= 3,
                       "detail": {"epochs_reported": len(rows), "missing": missing}})
    # Look only where the Ray W&B callback can write (never crawl results/, it is ~100 GB).
    roots = ([os.path.join(new_dirs[-1], "**")] if new_dirs else []) + [str(REPO_ROOT / "wandb"), str(REPO_ROOT / "results" / "wandb")]
    offline = [p for r in roots for p in glob.glob(os.path.join(r, "offline-run-*", "files", "config.yaml"), recursive=True)
               if new_dirs and os.path.getmtime(p) >= os.path.getmtime(new_dirs[-1]) - 60]
    sig = [p for p in offline if "mse+0.25*correye+0.5*neidist" in Path(p).read_text()]
    checks.append({"name": "trial W&B config carries loss_signature (offline run)", "ok": bool(sig),
                   "detail": {"offline_configs_found": len(offline), "with_signature": len(sig)}})
    return _report("tune", checks)


TASKS = {"dev_runs": task_dev_runs, "cross_tree": task_cross_tree, "tune": task_tune}

if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
        sys.exit(0)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    sys.exit(TASKS[sys.argv[1]]())
