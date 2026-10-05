"""E2.2 orchestration (spec v2 E2.2 flight plan): status, latent gate, budget projection and full-run launch, all from
this campaign's task logs (results/logs/e2_mse_<Model>_<direction>_<job>_<seed>.out) and squeue.

    python scripts/experiments/model_benchmark/autopilot.py status --direction sc2fc
    python scripts/experiments/model_benchmark/autopilot.py gate   --direction fc2sc
    python scripts/experiments/model_benchmark/autopilot.py budget                     # both directions vs the cap
    python scripts/experiments/model_benchmark/autopilot.py launch --direction sc2fc [--dry-run]

`launch` submits, per roster model, only the seeds with no finished and no queued task (so the pilot's seed 0 is reused),
skips a gated model that failed its gate, sets each model's --time from its pilot wall time (config.yml `budget`), and
refuses to submit if the projected GPU-h for both directions exceeds the cap. Exit code 2 = a stop condition.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import submit  # noqa: E402  (stdlib + yaml only: this script runs on the host, where squeue / sbatch live)

REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
LOG_DIR = REPO_ROOT / "results" / "logs"
MARKER = "Best Tune trial comprehensive summary:"


def parse_log(path):
    """(seed, best-trial summary) from one task log, or None if unfinished (same rule as run.parse_log)."""
    text = Path(path).read_text(errors="replace")
    seed = re.search(r"Seed=(\d+)", text)
    i = text.rfind(MARKER)
    if not seed or i < 0:
        return None
    try:
        summary, _ = json.JSONDecoder().raw_decode(text[i + len(MARKER):].lstrip())
    except ValueError:
        return None
    return int(seed.group(1)), summary
STOP = 2
DATE_FMT = "%a %b %d %I:%M:%S %p %Z %Y"


def cfg_load():
    return yaml.safe_load((HERE / "config.yml").read_text())


def _when(line):
    m = re.search(r"at (\w{3} \w{3} +\d+ [\d:]+ [AP]M \w+ \d{4})", line)
    if not m:
        return None
    try:
        return datetime.strptime(re.sub(r" +", " ", m.group(1)), DATE_FMT)
    except ValueError:
        return None


def task_logs(cfg, model, direction):
    """[(path, seed, finished, wall_h, n_terminated, n_error)] for every task log of this model x direction."""
    out = []
    for path in sorted(glob.glob(str(LOG_DIR / f"{cfg['campaign']}_{model}_{direction}_*_*.out")), key=os.path.getmtime):
        text = Path(path).read_text(errors="replace")
        seed = re.search(r"Seed=(\d+)", text)
        if not seed:
            continue
        lines = text.splitlines()
        start = next((_when(l) for l in lines if l.startswith("Starting job")), None)
        end = next((_when(l) for l in reversed(lines) if l.startswith("Job Over")), None)
        status = re.findall(r"Trial status: ([^│\n]*)", text)
        last = status[-1] if status else ""
        n_term = int((re.search(r"(\d+) TERMINATED", last) or [0, 0])[1])
        n_err = int((re.search(r"(\d+) ERROR", last) or [0, 0])[1])
        samples = re.search(r"\((\d+) samples", text)
        out.append(dict(path=path, seed=int(seed.group(1)), finished=parse_log(path) is not None,
                        samples=int(samples.group(1)) if samples else None,
                        wall_h=((end - start).total_seconds() / 3600) if (start and end) else None,
                        n_term=n_term, n_err=n_err))
    return out


def queued(cfg, model, direction):
    """Seeds of this model x direction with a task pending or running in squeue."""
    name = f"{cfg['campaign']}_{model}_{direction}"
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-n", name, "-o", "%K"],
                         capture_output=True, text=True).stdout.split()
    return {int(s) for s in out if s.isdigit()}


def full_samples(model, direction):
    return yaml.safe_load(submit.config_for(model, direction).read_text())["benchmark"]["num_samples"]


def model_status(cfg, model, direction):
    """A seed counts as done only when a finished task ran the model's full trial budget (a short gate run does not)."""
    logs = task_logs(cfg, model, direction)
    full = full_samples(model, direction)
    done = sorted({l["seed"] for l in logs if l["finished"] and (l["samples"] is None or l["samples"] >= full)})
    walls = [l["wall_h"] for l in logs if l["finished"] and l["wall_h"]
             and (l["samples"] is None or l["samples"] >= full or not any(x["samples"] and x["samples"] >= full for x in logs))]
    # error rate only from tasks run under the current config (older tasks may predate a search fix, e.g. I2.2)
    since = submit.config_for(model, direction).stat().st_mtime
    recent = [l for l in logs if l["finished"] and os.path.getmtime(l["path"]) > since]
    trials = sum(l["n_term"] + l["n_err"] for l in recent)
    errs = sum(l["n_err"] for l in recent)
    failed = sorted({l["seed"] for l in logs if not l["finished"] and l["wall_h"] is not None} - set(done))
    return dict(done=done, queued=sorted(queued(cfg, model, direction)), failed=failed,
                wall_h=(max(walls) if walls else None), error_frac=(errs / trials if trials else 0.0),
                # attempts at the full budget only: a short gate run is not a failed attempt of that seed
                attempts={s: sum(1 for l in logs if l["seed"] == s and (l["samples"] is None or l["samples"] >= full))
                          for s in cfg["seeds"]})


def gate(cfg, direction):
    """(passed, detail) for the latent gate in this direction; None if its runs are not finished."""
    g = cfg.get("gate") or {}
    res = {}
    for model in g.get("models", []) + [g.get("reference")]:
        vals = {}
        for path in glob.glob(str(LOG_DIR / f"{cfg['campaign']}_{model}_{direction}_*_*.out")):
            got = parse_log(path)
            if model in g.get("models", []):  # gated model: only its short gate runs count (not a later full run)
                n = re.search(r"\((\d+) samples", Path(path).read_text(errors="replace"))
                if not n or int(n.group(1)) != g["trials"]:
                    continue
            if got and got[0] in g["seeds"]:
                vals[got[0]] = (got[1].get("selected_by") or {}).get("value")
        if set(vals) != set(g["seeds"]) or any(v is None for v in vals.values()):
            return None, f"{model}: seeds {sorted(vals)} of {g['seeds']} finished"
        res[model] = sum(vals.values()) / len(vals)
    ref = res[g["reference"]]
    out = {m: (v >= ref - g["margin"]) for m, v in res.items() if m != g["reference"]}
    return out, {m: round(v, 4) for m, v in res.items()} | {"threshold": round(ref - g["margin"], 4)}


def plan_direction(cfg, direction, pilot_walls=None):
    """Per model: missing seeds, per-seed wall estimate (h) and the --time to request."""
    b = cfg["budget"]
    passed, _ = gate(cfg, direction)
    rows = []
    for model in cfg["directions"][direction]["models"]:
        st = model_status(cfg, model, direction)
        missing = [s for s in cfg["seeds"] if s not in st["done"] and s not in st["queued"]]
        if model in (cfg.get("gate") or {}).get("models", []):
            if passed is None:
                rows.append(dict(model=model, missing=[], wall_h=None, note="gate pending"))
                continue
            override = (cfg["directions"][direction].get("gate_override") or {}).get(model)
            if not passed.get(model) and not override:
                rows.append(dict(model=model, missing=[], wall_h=None, note="gate failed: negative result"))
                continue
        wall = st["wall_h"] or (pilot_walls or {}).get(model)
        if model in (cfg.get("gate") or {}).get("models", []) and wall and not st["done"]:
            wall = wall * full_samples(model, direction) / cfg["gate"]["trials"]  # scale the short gate run up
        rows.append(dict(model=model, missing=missing, wall_h=wall, error_frac=st["error_frac"], note=""))
    for r in rows:
        if r["wall_h"]:
            r["time_h"] = min(b["time_cap_h"], max(b["min_time_h"], b["time_factor"] * r["wall_h"]))
        r["gpu_h"] = (r["wall_h"] or 0) * len(r["missing"])
    return rows


def budget(cfg):
    sc_walls = {r["model"]: r["wall_h"] for r in plan_direction(cfg, "sc2fc")}
    total, table = 0.0, {}
    for d in cfg["directions"]:
        rows = plan_direction(cfg, d, pilot_walls=sc_walls)
        table[d] = rows
        total += sum(r["gpu_h"] for r in rows)
    return total, table


def _fmt(rows):
    for r in rows:
        w = f"{r['wall_h']:.2f}" if r.get("wall_h") else "  -  "
        print(f"  {r['model']:34s} missing {str(r['missing']):16s} wall/seed {w} h  time {r.get('time_h', 0):.1f} h  "
              f"proj {r['gpu_h']:.1f} GPU-h  {r.get('note', '')}")


def cmd_status(cfg, args):
    for model in cfg["directions"][args.direction]["models"]:
        st = model_status(cfg, model, args.direction)
        print(f"{model:34s} done {st['done']} queued {st['queued']} failed {st['failed']} "
              f"wall {st['wall_h'] and round(st['wall_h'], 2)} h errors {st['error_frac']:.0%}")
    print("gate:", gate(cfg, args.direction))
    return 0


def cmd_budget(cfg, args):
    total, table = budget(cfg)
    for d, rows in table.items():
        print(d)
        _fmt(rows)
    cap = cfg["budget"]["gpu_h_cap"]
    print(f"projected remaining: {total:.1f} GPU-h (cap {cap})")
    return STOP if total > cap else 0


def cmd_launch(cfg, args):
    total, table = budget(cfg)
    if total > cfg["budget"]["gpu_h_cap"]:
        print(f"STOP: projected {total:.1f} GPU-h exceeds the cap {cfg['budget']['gpu_h_cap']}")
        return STOP
    rows = [r for r in table[args.direction] if r["missing"]]
    if args.models:
        rows = [r for r in rows if r["model"] in args.models]
    stop = []
    for r in rows:
        st = model_status(cfg, r["model"], args.direction)
        if st["error_frac"] > cfg["budget"]["max_error_frac"]:
            stop.append(f"{r['model']}: {st['error_frac']:.0%} of its trials errored")
            continue
        over = [s for s in r["missing"] if st["attempts"].get(s, 0) > cfg["budget"]["max_retries"]]
        if over:
            stop.append(f"{r['model']}: seeds {over} already failed {cfg['budget']['max_retries'] + 1} times")
            continue
        if not r.get("time_h"):
            stop.append(f"{r['model']}: no pilot wall time yet")
            continue
        job = submit.plan(cfg, args.direction, "full", [r["model"]], r["missing"], None)[0]
        retry = any(st["attempts"].get(s, 0) >= 1 for s in r["missing"])  # a seed that already ran and did not finish
        hours = math.ceil(min(cfg["budget"]["time_cap_h"], r["time_h"] * (2 if retry else 1)) * 4) / 4
        cmd = ["sbatch", "--parsable", f"--job-name={job['name']}", f"--time={int(hours)}:{int(hours % 1 * 60):02d}:00",
               f"--array={','.join(str(s) for s in r['missing'])}", str(HERE / "launch_model.sh"), r["model"],
               args.direction, str(job["samples"]), job["alg"], str(job["gpus"]), str(job["conc"])]
        if args.dry_run:
            print(" ".join(cmd[2:]))
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(f"{r['model']:34s} {'job ' + out.stdout.strip() if not out.returncode else 'FAILED ' + out.stderr.strip()}"
              f"  seeds {r['missing']}  --time {hours:.2f} h")
    for s in stop:
        print("STOP:", s)
    return STOP if stop else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["status", "gate", "budget", "launch"])
    ap.add_argument("--direction", choices=["sc2fc", "fc2sc"], default="sc2fc")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = cfg_load()
    if args.cmd == "gate":
        print(gate(cfg, args.direction))
        return 0
    return {"status": cmd_status, "budget": cmd_budget, "launch": cmd_launch}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
