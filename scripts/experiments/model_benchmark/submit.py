"""Submit E2.2 benchmark tasks from the roster (spec v2 E2.2). One SLURM array per (model, direction), index = seed.

    python scripts/experiments/model_benchmark/submit.py --direction sc2fc --models CrossModal_PCA_PLS --dry-run
    python scripts/experiments/model_benchmark/submit.py --direction sc2fc --stage pilot      # one seed per model + gate
    python scripts/experiments/model_benchmark/submit.py --direction sc2fc --stage full [--models ...]

Job name `e2_mse_<Model>_<direction>` is how run.py finds the campaign's task logs (no W&B tag needed).
`--num-samples` / `--seeds` override the roster (e.g. the latent gate uses seeds 0-1 x 12 trials).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
CONFIGS = REPO_ROOT / "models" / "configs" / "benchmark" / "mse"
# Campaign config: config.yml (E2.2) by default; MB_CAMPAIGN_CONFIG=config_composite_loss.yml selects spec v3 E3.
CAMPAIGN_CONFIG = HERE / os.environ.get("MB_CAMPAIGN_CONFIG", "config.yml")


def cfg_load():
    return yaml.safe_load(CAMPAIGN_CONFIG.read_text())


def configs_dir(cfg=None):
    return REPO_ROOT / (cfg or {}).get("configs_dir", "models/configs/benchmark/mse")


def launcher_export(cfg):
    """sbatch --export for launch_model.sh: campaign name, config dir, best-trial report metrics, Ray actor reuse,
    Lightning per-trial checkpoints (off unless the campaign sets `tune_lightning_checkpoints: true`)."""
    return (f"--export=ALL,CAMPAIGN={cfg['campaign']},CONFIG_DIR={cfg.get('configs_dir', 'models/configs/benchmark/mse')},"
            f"REPORT_METRICS={'+'.join(cfg.get('report_metrics', ['val_demeaned_r']))},"
            f"TUNE_REUSE_ACTORS={'true' if cfg.get('tune_reuse_actors', True) else 'false'},"
            f"TUNE_LIGHTNING_CHECKPOINTS={'true' if cfg.get('tune_lightning_checkpoints', False) else 'false'}")  # '+': --export splits on ','


def job_name(cfg, model, direction, prefix=None):
    """`<campaign>_<Model>_<direction>`: how run.py / autopilot.py find a campaign's task logs. A `prefix` (probe /
    smoke runs) keeps those logs out of the campaign."""
    return f"{prefix or cfg['campaign']}_{model}_{direction}"


def sbatch_resources(job):
    """Per-model sbatch resources from the campaign config (`cpus`, default the launcher's 10)."""
    return [f"--cpus-per-task={job['cpus']}"]


def config_for(model, direction, cfg=None):
    d = configs_dir(cfg)
    alt = d / f"{model}_fc2sc.yml"
    return alt if direction == "fc2sc" and alt.exists() else d / f"{model}.yml"


def plan(cfg, direction, stage, models, seeds_override, samples_override, prefix=None):
    roster = cfg["directions"][direction]["models"]
    models = models or roster
    unknown = [m for m in models if m not in roster]
    if unknown:
        raise SystemExit(f"not in the {direction} roster: {unknown}")
    gate = cfg.get("gate", {})
    jobs = []
    for m in models:
        bench = yaml.safe_load(config_for(m, direction, cfg).read_text())["benchmark"]
        seeds = list(cfg["seeds"])
        samples = bench["num_samples"]
        if stage == "pilot":
            gated = m in gate.get("models", []) or m == gate.get("reference")  # the gate compares on the same seeds
            seeds = list(gate["seeds"]) if gated else seeds[:1]
            if m in gate.get("models", []):
                samples = gate["trials"]
        if seeds_override is not None:
            seeds = seeds_override
        if samples_override is not None:
            samples = samples_override
        info = cfg["models"][m]
        gpus, conc = info["packing"]
        jobs.append(dict(model=m, seeds=seeds, samples=samples, alg=bench["search_alg"], gpus=gpus, conc=conc,
                         cpus=int(info.get("cpus", 10)), time=info["time"], name=job_name(cfg, m, direction, prefix)))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--direction", required=True, choices=["sc2fc", "fc2sc"])
    ap.add_argument("--stage", choices=["pilot", "full"], default="full")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--dependency", help="e.g. afterok:12345")
    ap.add_argument("--job-prefix", help="job-name prefix instead of the campaign (probe / smoke runs; not scraped)")
    ap.add_argument("--time", help="override the config --time (e.g. 0:30:00 for a probe)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = cfg_load()
    jobs = plan(cfg, args.direction, args.stage, args.models, args.seeds, args.num_samples, args.job_prefix)
    for j in jobs:
        cmd = ["sbatch", "--parsable", f"--job-name={j['name']}", f"--time={args.time or j['time']}",
               f"--array={','.join(str(s) for s in j['seeds'])}", *sbatch_resources(j)]
        if args.dependency:
            cmd += [f"--dependency={args.dependency}", "--kill-on-invalid-dep=yes"]
        cmd += [launcher_export(cfg), str(HERE / "launch_model.sh"), j["model"], args.direction, str(j["samples"]), j["alg"],
                str(j["gpus"]), str(j["conc"])]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode:
            print(f"FAILED {j['model']}: {out.stderr.strip()}")
            return 1
        print(f"{j['model']:34s} job {out.stdout.strip()}  seeds {j['seeds']}  {j['samples']} samples ({j['alg']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
