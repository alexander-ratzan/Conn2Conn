"""Submit spec v3 E0 tasks: one SLURM array per FC condition (index = seed), job name e0_taskfc_<condition>.

    python scripts/experiments/task_fc_to_sc/submit.py --stage pilot [--dry-run]     # config.yml pilot conditions/seeds
    python scripts/experiments/task_fc_to_sc/submit.py --stage full [--conditions ...] [--dry-run]
    --repo DIR runs the tasks from another checkout (e.g. a worktree before its branch is merged).
"""
import argparse
import os
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pilot", "full"], required=True)
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    conds = args.conditions or (cfg["pilot"]["conditions"] if args.stage == "pilot" else cfg["conditions"])
    seeds = args.seeds or (cfg["pilot"]["seeds"] if args.stage == "pilot" else cfg["seeds"])
    bad = [c for c in conds if c not in cfg["conditions"] or not (HERE / "configs" / f"{c}.yml").exists()]
    if bad:
        raise SystemExit(f"unknown condition or missing configs/<c>.yml (run make_configs.py): {bad}")
    p = cfg["packing"]
    for c in conds:
        cmd = ["sbatch", "--parsable", f"--job-name={cfg['campaign']}_{c}", f"--array={','.join(map(str, seeds))}",
               f"--export=ALL,CONN2CONN_DIR={args.repo}", str(HERE / "launch.sh"), c, str(cfg["num_samples"]),
               cfg["search_alg"], str(p["gpus_per_trial"]), str(p["max_concurrent"])]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        job = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
        print(f"{c}: job {job} seeds {seeds}")


if __name__ == "__main__":
    main()
