"""Submit spec v3 E0 tasks: one SLURM array per step x FC condition (index = seed), job name
<job_prefix>_<condition> (config.yml steps). Usually driven by autopilot.py; stdlib + yaml only (host python).

    python3 scripts/experiments/task_bidirectional_bench/submit.py --step E0.2 --stage pilot [--dry-run]
    python3 scripts/experiments/task_bidirectional_bench/submit.py --step E0.2 --stage full [--conditions ...] [--seeds ...] [--time 1:30:00]
    --repo DIR runs the tasks from another checkout (e.g. a worktree before its branch is merged).
"""
import argparse
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())


def cfg_load():
    return yaml.safe_load((HERE / "config.yml").read_text())


def job_name(cfg, step, cond):
    return f"{cfg['steps'][step]['job_prefix']}_{cond}"


def config_path(cfg, step, cond):
    return HERE / "configs" / cfg["steps"][step]["parcellation"] / f"{cond}.yml"


def submit(cfg, step, conds, seeds, time=None, repo=str(REPO_ROOT), dry_run=False):
    """Submit one array per condition; returns {condition: job id} ('dry-run' when not submitted)."""
    st, p = cfg["steps"][step], cfg["packing"]
    bad = [c for c in conds if c not in cfg["conditions"] or not config_path(cfg, step, c).exists()]
    if bad:
        raise SystemExit(f"unknown condition or missing configs/{st['parcellation']}/<c>.yml (run make_configs.py): {bad}")
    jobs = {}
    for c in conds:
        cmd = ["sbatch", "--parsable", f"--job-name={job_name(cfg, step, c)}", f"--array={','.join(map(str, seeds))}"]
        if time:
            cmd.append(f"--time={time}")
        cmd += [f"--export=ALL,CONN2CONN_DIR={repo}", str(HERE / "launch.sh"), st["parcellation"], st["direction"], c,
                str(cfg["num_samples"]), cfg["search_alg"], str(p["gpus_per_trial"]), str(p["max_concurrent"])]
        if dry_run:
            print(" ".join(cmd))
            jobs[c] = "dry-run"
            continue
        jobs[c] = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
        print(f"{step} {c}: job {jobs[c]} seeds {seeds}")
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True)
    ap.add_argument("--stage", choices=["pilot", "full"], required=True)
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--time")
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = cfg_load()
    if args.step not in cfg["steps"]:
        raise SystemExit(f"unknown step {args.step}; config.yml steps: {list(cfg['steps'])}")
    conds = args.conditions or (cfg["pilot"]["conditions"] if args.stage == "pilot" else cfg["conditions"])
    seeds = args.seeds or (cfg["pilot"]["seeds"] if args.stage == "pilot" else cfg["seeds"])
    time = args.time or (cfg["pilot"]["time"] if args.stage == "pilot" else cfg["steps"][args.step].get("time"))
    submit(cfg, args.step, conds, seeds, time, args.repo, args.dry_run)


if __name__ == "__main__":
    main()
