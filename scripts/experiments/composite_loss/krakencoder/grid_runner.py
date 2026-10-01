"""
Krakencoder instance of the composite-loss grid: expand cells into retrain fits, run them packed, collect tables.

    python grid_runner.py plan    --set pilot|grid|noise              # list fits (tag, seed, loss string, overrides)
    python grid_runner.py run     --set grid --job 3 [--per-job 4]    # fits [3*4, 4*4) concurrently, then score them
    python grid_runner.py collect --set grid                          # -> tables/seed_records.csv, epoch_history.csv
    # quick check: --epochs 20 --checkpoint-every 10 --tag-suffix _smoke (run and collect)

Each fit = `python -m models.architectures.krakencoder.retrain --config <base_config> --seed S --tag T --set ...`
(results/krakencoder/<T>/seed<S>/), then `python -m models.architectures.krakencoder.checkpoint_eval` on it.
Fits whose outputs exist are skipped, so a requeued job resumes.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
RESULTS_ROOT = REPO_ROOT / "results" / "krakencoder"

import yaml  # noqa: E402

TERMS = ("varmatch", "correye", "neidist")


def load_cfg() -> dict:
    return yaml.safe_load((HERE / "config.yml").read_text())


def cells(cfg: dict) -> list[dict]:
    grid = yaml.safe_load((REPO_ROOT / cfg["grid_file"]).read_text())
    out = [{**c, "grid_version": grid["version"]} for c in grid["combos"]]
    out += [{**c, "grid_version": grid["version"]} for c in cfg.get("reference_cells", [])]
    out += [{**c, "grid_version": grid["version"]} for c in cfg.get("extension_cells", [])]
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids):
        sys.exit(f"duplicate cell ids: {ids}")
    return out


def loss_string(cfg: dict, cell: dict) -> str:
    parts = []
    for term in TERMS:
        w = float(cell.get(term, 0.0) or 0.0) * float(cfg.get("term_anchor", {}).get(term, 1.0))
        if w > 0:
            name = cfg["term_map"][term]
            parts.append(name if w == 1 else f"{name}.w{round(w, 4):g}")
    return "+".join(parts + [cfg["fixed_loss"]])


def plan(cfg: dict, set_name: str, args) -> list[dict]:
    spec = cfg["sets"][set_name]
    by_id = {c["id"]: c for c in cells(cfg)}
    chosen = list(by_id) if spec["cells"] == "all" else spec["cells"]
    missing = [c for c in chosen if c not in by_id]
    if missing:
        sys.exit(f"unknown cells in set {set_name}: {missing}")
    recipe = dict(cfg["recipe"])
    if args.epochs:
        recipe["epochs"] = args.epochs
    if args.checkpoint_every:
        recipe["checkpoint_every"] = args.checkpoint_every
    fits = []
    for cid in chosen:
        for seed in spec["seeds"]:
            for rs in spec.get("random_seeds", [None]):
                tag = f"{cfg['tag_prefix']}_{cid}" + (f"_r{rs}" if rs is not None else "") + (args.tag_suffix or "")
                overrides = {**recipe, "losstype": loss_string(cfg, by_id[cid])}
                if rs is not None:
                    overrides["random_seed"] = rs
                fits.append({"cell": by_id[cid], "seed": seed, "random_seed": rs if rs is not None else 0,
                             "tag": tag, "overrides": overrides})
    return fits


def run_dir(fit: dict) -> Path:
    return RESULTS_ROOT / fit["tag"] / f"seed{fit['seed']}"


def retrain_cmd(cfg: dict, fit: dict) -> list[str]:
    cmd = [sys.executable, "-m", "models.architectures.krakencoder.retrain", "--config", cfg["base_config"],
           "--seed", str(fit["seed"]), "--tag", fit["tag"]]
    for k, v in fit["overrides"].items():
        cmd += ["--set", f"{k}={yaml.safe_dump(v, default_flow_style=True).strip()}"]
    return cmd


def run(cfg: dict, fits: list[dict], job: int, per_job: int) -> None:
    mine = fits[job * per_job:(job + 1) * per_job]
    if not mine:
        sys.exit(f"job {job} is empty ({len(fits)} fits, {per_job} per job)")
    t0 = time.time()
    # Shared inputs (flavor files, the seed's split file) once per seed, before fits start writing concurrently.
    for seed in sorted({f["seed"] for f in mine}):
        subprocess.run([sys.executable, "-m", "models.architectures.krakencoder.retrain", "--config", cfg["base_config"],
                        "--seed", str(seed), "--tag", "_inputs_check", "--stage", "inputs"], cwd=REPO_ROOT, check=True)
    procs = []
    for fit in mine:
        d = run_dir(fit)
        if (d / "manifest.json").exists():
            print(f"[grid] {fit['tag']} seed {fit['seed']}: trained, skipping", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        fh = open(d / "runner.log", "a")
        print(f"[grid] start {fit['tag']} seed {fit['seed']}: {fit['overrides']['losstype']}", flush=True)
        procs.append((fit, subprocess.Popen(retrain_cmd(cfg, fit), cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT), fh))
    failed = []
    for fit, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        print(f"[grid] {fit['tag']} seed {fit['seed']}: exit {rc} after {time.time() - t0:.0f} s", flush=True)
        if rc:
            failed.append(fit)
    for fit in mine:
        if fit in failed:
            continue
        d = run_dir(fit)
        if (d / "epoch_history.csv").exists() and (d / "epoch_history.csv").stat().st_mtime >= (d / "manifest.json").stat().st_mtime:
            continue
        env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("EVAL_THREADS", "4"),
               "MKL_NUM_THREADS": os.environ.get("EVAL_THREADS", "4")}
        subprocess.run([sys.executable, "-m", "models.architectures.krakencoder.checkpoint_eval", "--run-dir", str(d),
                        "--parcellation", cfg["eval"]["parcellation"], "--batch-size", str(cfg["eval"]["loss_batch_size"])],
                       cwd=REPO_ROOT, env=env, check=True)
    print(f"[grid] job {job}: {len(mine)} fits, {len(failed)} failed, {time.time() - t0:.0f} s", flush=True)
    if failed:
        sys.exit(1)


def collect(cfg: dict, fits: list[dict], set_name: str, suffix: str) -> None:
    out_dir = HERE / "tables"
    out_dir.mkdir(exist_ok=True)
    history, seeds, missing = [], [], []
    for fit in fits:
        path = run_dir(fit) / "epoch_history.csv"
        if not path.exists():
            missing.append(f"{fit['tag']}/seed{fit['seed']}")
            continue
        cell = fit["cell"]
        anchor = cfg.get("term_anchor", {})
        extra = {"grid_version": cell["grid_version"], "combo_id": cell["id"], "block": cell.get("block", ""),
                 **{f"w_{t}": float(cell.get(t, 0.0) or 0.0) for t in TERMS},
                 **{f"native_w_{t}": float(cell.get(t, 0.0) or 0.0) * float(anchor.get(t, 1.0)) for t in TERMS},
                 "random_seed": fit["random_seed"]}
        rows = list(csv.DictReader(open(path)))
        last = max(int(r["epoch"]) for r in rows)
        for r in rows:
            r = {**extra, **r}
            history.append(r)
            if int(r["epoch"]) == last:
                seeds.append(r)
    stem = f"{set_name}{suffix or ''}"
    for name, rows in (("epoch_history", history), ("seed_records", seeds)):
        if rows:
            with open(out_dir / f"{stem}_{name}.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"[grid] collected {len(seeds)} fits into {out_dir}/{stem}_*.csv; missing {len(missing)}: {missing[:6]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "collect"])
    ap.add_argument("--set", dest="set_name", default="pilot")
    ap.add_argument("--job", type=int, help="run: job index (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--per-job", type=int, help="run: fits per job (default packing.fits_per_job)")
    ap.add_argument("--epochs", type=int, help="override recipe epochs (checks)")
    ap.add_argument("--checkpoint-every", type=int, help="override recipe checkpoint_every (checks)")
    ap.add_argument("--tag-suffix", default="", help="appended to every tag (checks), e.g. _smoke")
    args = ap.parse_args()

    cfg = load_cfg()
    fits = plan(cfg, args.set_name, args)
    per_job = args.per_job or int(cfg["packing"]["fits_per_job"])
    if args.command == "plan":
        for i, fit in enumerate(fits):
            print(f"{i:3d} job {i // per_job:2d}  {fit['tag']:42s} seed {fit['seed']}  rs {fit['random_seed']}  "
                  f"{fit['overrides']['losstype']}")
        print(f"{len(fits)} fits -> {-(-len(fits) // per_job)} jobs of {per_job} (--array=0-{-(-len(fits) // per_job) - 1})")
    elif args.command == "run":
        job = args.job if args.job is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        run(cfg, fits, job, per_job)
    else:
        collect(cfg, fits, args.set_name, args.tag_suffix)


if __name__ == "__main__":
    main()
