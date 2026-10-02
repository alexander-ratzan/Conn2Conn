"""Submit E2.2 benchmark tasks from the roster (spec v2 E2.2). One SLURM array per (model, direction), index = seed.

    python scripts/experiments/model_benchmark/submit.py --direction sc2fc --models CrossModal_PCA_PLS --dry-run
    python scripts/experiments/model_benchmark/submit.py --direction sc2fc --stage pilot      # one seed per model + gate
    python scripts/experiments/model_benchmark/submit.py --direction sc2fc --stage full [--models ...]

Job name `e2_mse_<Model>_<direction>` is how run.py finds the campaign's task logs (no W&B tag needed).
`--num-samples` / `--seeds` override the roster (e.g. the latent gate uses seeds 0-1 x 12 trials).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
CONFIGS = REPO_ROOT / "models" / "configs" / "benchmark" / "mse"


def job_name(cfg, model, direction):
    return f"{cfg['campaign']}_{model}_{direction}"


def config_for(model, direction):
    alt = CONFIGS / f"{model}_fc2sc.yml"
    return alt if direction == "fc2sc" and alt.exists() else CONFIGS / f"{model}.yml"


def plan(cfg, direction, stage, models, seeds_override, samples_override):
    roster = cfg["directions"][direction]["models"]
    models = models or roster
    unknown = [m for m in models if m not in roster]
    if unknown:
        raise SystemExit(f"not in the {direction} roster: {unknown}")
    gate = cfg.get("gate", {})
    jobs = []
    for m in models:
        bench = yaml.safe_load(config_for(m, direction).read_text())["benchmark"]
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
                         time=info["time"], name=job_name(cfg, m, direction)))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--direction", required=True, choices=["sc2fc", "fc2sc"])
    ap.add_argument("--stage", choices=["pilot", "full"], default="full")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--dependency", help="e.g. afterok:12345")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    jobs = plan(cfg, args.direction, args.stage, args.models, args.seeds, args.num_samples)
    for j in jobs:
        cmd = ["sbatch", "--parsable", f"--job-name={j['name']}", f"--time={j['time']}",
               f"--array={','.join(str(s) for s in j['seeds'])}"]
        if args.dependency:
            cmd += [f"--dependency={args.dependency}", "--kill-on-invalid-dep=yes"]
        cmd += [str(HERE / "launch_model.sh"), j["model"], args.direction, str(j["samples"]), j["alg"],
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
