"""Collect every local Ray Tune trial of the given models into one table (offline; no W&B API).

Reads results/ray_checkpoints/<Model>_tune_<id>/<trial>/{params.json, result.json} and the trial's local
W&B config (data.source, data.shuffle_seed). One row per trial.

Run (kraken_env, CPU):
    python scripts/experiments/multimodel_scfc/audit/collect_trials.py \
        --models Sarwar2020MLP Chen2024GCN NodalGNN NodalMLP CrossModal_PCA_PLS_learnable CrossModal_PCA_PLS
"""
import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = Path(__file__).resolve().parent


def _wandb_value(cfg, key):
    v = cfg.get(key)
    return v.get("value") if isinstance(v, dict) else v


def collect(models, ckpt_root):
    rows = []
    for model in models:
        for sweep in sorted(glob.glob(os.path.join(ckpt_root, f"{model}_tune_*"))):
            sweep_id = sweep.rsplit("_", 1)[-1]
            if not sweep_id.isdigit():
                continue
            for tdir in glob.glob(os.path.join(sweep, "_tune_trainable_*")):
                try:
                    params = json.load(open(os.path.join(tdir, "params.json")))
                except (OSError, ValueError):
                    continue
                res_path = os.path.join(tdir, "result.json")
                results = [json.loads(line) for line in open(res_path) if line.strip()] if os.path.exists(res_path) else []
                vals = [r["val_demeaned_r"] for r in results if r.get("val_demeaned_r") is not None]
                source = seed = None
                for cfg_path in glob.glob(os.path.join(tdir, "wandb", "*", "files", "config.yaml"))[:1]:
                    cfg = yaml.safe_load(open(cfg_path)) or {}
                    source, seed = _wandb_value(cfg, "data.source"), _wandb_value(cfg, "data.shuffle_seed")
                name = os.path.basename(tdir)
                parts = name.split("_")
                idx = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
                rows.append({
                    "model": model, "sweep": sweep_id, "trial": name[:48], "trial_index": idx,
                    "source": source, "seed": seed, "n_epochs": len(results),
                    "val_last": vals[-1] if vals else None, "val_max": max(vals) if vals else None,
                    "time_s": results[-1].get("time_total_s") if results else None,
                    "keyset": ",".join(sorted(params)),
                    **{f"p.{k}": (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()},
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Only some trials keep a local W&B config; the sweep shares one source.
    df["source"] = df.groupby("sweep")["source"].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else None)
    df["created"] = pd.to_datetime(df["sweep"].astype(int), unit="s")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ckpt-root", default=str(REPO_ROOT / "results" / "ray_checkpoints"))
    ap.add_argument("--out", default=str(AUDIT_DIR / "data" / "trials.csv.gz"))
    args = ap.parse_args()
    df = collect(args.models, args.ckpt_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["model", "source"], dropna=False).size().to_string())
    print(f"{len(df)} trials -> {args.out}")


if __name__ == "__main__":
    main()
