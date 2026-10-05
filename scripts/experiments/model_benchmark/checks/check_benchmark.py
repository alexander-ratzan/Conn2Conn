"""E2.2 build checks (CPU, synthetic logs): configs match their generator, submit plans resolve, the runner parses
best-trial summaries from task logs (latest finished task per seed; unfinished tasks ignored), reuses rows, writes every
table and figure. Synthetic metric values are made up; the figures are for layout only.

    python scripts/experiments/model_benchmark/checks/check_benchmark.py [--keep DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(HERE))
import run  # noqa: E402

fails = 0


def check(label, ok, extra=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), label, extra)


def fake_log(path, model, seed, dm, finished=True):
    rng = np.random.RandomState(seed)
    summary = {"selected_by": {"metric": "val_demeaned_r", "mode": "max", "value": dm + 0.005},
               "ray_tune_id": f"17900{seed}", "best_trial": {"id": "abc", "config": {"model": {"name": model}}},
               "metrics": {"train": {}, "val": {"demeaned_r": dm + 0.005},
                           "test": {"pearson": 0.80 + dm, "demeaned_pearson": dm + rng.normal(0, 0.004),
                                    "avg_rank": 0.55 + 2.5 * dm + rng.normal(0, 0.01),
                                    "top1_acc": max(0.0, 0.4 * dm + rng.normal(0, 0.005)), "mse": 0.0135}}}
    body = f"Starting job 1 (task {seed})\nCampaign=e2_mse Model={model} Direction=sc2fc Seed={seed} (8 samples)\n"
    if finished:
        body += "Best Tune trial comprehensive summary:\n" + json.dumps(summary, indent=2) + "\nJob Over\n"
    Path(path).write_text(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep")
    args = ap.parse_args()
    out = subprocess.run([sys.executable, str(HERE / "build_configs.py"), "--check"], capture_output=True, text=True)
    check("benchmark configs match build_configs.py SPECS", out.returncode == 0, out.stdout.strip())
    for direction in ("sc2fc", "fc2sc"):
        out = subprocess.run([sys.executable, str(HERE / "submit.py"), "--direction", direction, "--stage", "pilot", "--dry-run"],
                             capture_output=True, text=True)
        n = len([l for l in out.stdout.splitlines() if l.startswith("sbatch")])
        check(f"submit.py pilot plan ({direction})", out.returncode == 0 and n > 0, f"{n} arrays")
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    tmp = Path(args.keep or tempfile.mkdtemp())
    logs = tmp / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    level = {"CrossModalPCA": 0.012, "CrossModal_PLS_SVD": 0.07, "CrossModal_PCA_PLS": 0.095,
             "CrossModal_ConditionalGaussian": 0.09, "CrossModal_PCA_PLS_learnable": 0.10, "CrossModal_linear_backbone": 0.103,
             "CrossModal_PCA_PLS_CovProjector": 0.108, "MaskedMLPPretrainer": 0.085, "Sarwar2020MLP": 0.07,
             "Chen2024GCN": 0.025, "NodalGNN": 0.02, "NodalMLP": 0.022}
    for model in cfg["directions"]["sc2fc"]["models"]:
        for seed in cfg["seeds"]:
            fake_log(logs / f"e2_mse_{model}_sc2fc_100_{seed}.out", model, seed, level[model])
    # an older finished attempt and a newer unfinished one for seed 0 of one model: the finished newest wins
    fake_log(logs / "e2_mse_CrossModal_PCA_PLS_sc2fc_050_0.out", "CrossModal_PCA_PLS", 0, 0.5)
    os.utime(logs / "e2_mse_CrossModal_PCA_PLS_sc2fc_050_0.out", (time.time() - 1e5, time.time() - 1e5))
    fake_log(logs / "e2_mse_CrossModal_PCA_PLS_sc2fc_200_0.out", "CrossModal_PCA_PLS", 0, 0.9, finished=False)
    rc = run.build("sc2fc", cfg=cfg, out_root=tmp / "out", log_dir=logs)
    d = tmp / "out" / cfg.get("results_dir", "") / "sc2fc"
    recs = json.loads((d / "records.json").read_text())
    camp = [r for r in recs if r["source"] == "campaign"]
    check("runner: one record per model x seed from logs", rc == 0 and len(camp) == 12 * 5, f"{len(camp)}")
    pca0 = [r for r in camp if r["model"] == "CrossModal_PCA_PLS" and r["seed"] == 0][0]
    check("runner: latest *finished* task per seed wins", abs(pca0["test"]["demeaned_pearson"] - 0.095) < 0.02)
    reused = {r["model"] for r in recs if r["source"].startswith("reuse")}
    check("runner: reused Krakencoder (MSE + paper) + test-retest rows", reused == {"Krakencoder_mse", "Krakencoder_paper", "TestRetest"}, str(reused))
    expected = ["tables/seed_records.csv", "tables/summary.csv", "tables/summary.md", "tables/paired_vs_best_linear.csv",
                "tables/paired_vs_best_linear.md", "figures/scatter_demeaned_vs_rank.png"] + \
               [f"figures/bars_{m}.png" for m in cfg["metrics"]] + ["figures/bars_all_metrics.png",
                                                                   "figures/panel_grouped.png", "figures/panel_performance.png"]
    missing = [e for e in expected if not (d / e).exists()]
    check("runner: every table and figure written", not missing, f"missing={missing}")
    rc2 = run.build("sc2fc", cached=True, cfg=cfg, out_root=tmp / "out", log_dir=logs)
    check("runner: --cached re-render from records.json", rc2 == 0)
    # gate: a gate-failed model is dropped, unless a gate_override includes it (marked ‡)
    import copy
    gcfg = copy.deepcopy(cfg)
    gcfg["directions"]["sc2fc"]["gate_failed"] = {"MaskedMLPPretrainer": "test"}
    gcfg["directions"]["sc2fc"].pop("gate_override", None)
    run.build("sc2fc", cfg=gcfg, out_root=tmp / "gate_drop", log_dir=logs)
    gd = json.loads((tmp / "gate_drop" / gcfg.get("results_dir", "") / "sc2fc" / "records.json").read_text())
    check("runner: gate-failed model dropped", not any(r["model"] == "MaskedMLPPretrainer" for r in gd))
    gcfg["directions"]["sc2fc"]["gate_override"] = {"MaskedMLPPretrainer": "test override"}
    run.build("sc2fc", cfg=gcfg, out_root=tmp / "gate_keep", log_dir=logs)
    gk_dir = tmp / "gate_keep" / gcfg.get("results_dir", "") / "sc2fc"
    gk = json.loads((gk_dir / "records.json").read_text())
    lab = set(__import__("pandas").read_csv(gk_dir / "tables" / "summary.csv").query("model == 'MaskedMLPPretrainer'")["label"])
    check("runner: gate_override keeps it, labelled ‡", sum(r["model"] == "MaskedMLPPretrainer" for r in gk) == 5
          and all(l.endswith("‡") for l in lab) and lab, str(lab))
    print("outputs in", d)
    print("benchmark check failures:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
