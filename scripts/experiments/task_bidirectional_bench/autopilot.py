"""Spec v3 E0 orchestration under D1: status, pilot gate, budget and launch, all from this campaign's task logs
(results/logs/<job_prefix>_<condition>_<job>_<seed>.{out,err}), squeue and sacct. Host python (stdlib + yaml).

    python3 scripts/experiments/task_bidirectional_bench/autopilot.py status [--step E0.2]
    python3 scripts/experiments/task_bidirectional_bench/autopilot.py gate   --step E0.2
    python3 scripts/experiments/task_bidirectional_bench/autopilot.py budget
    python3 scripts/experiments/task_bidirectional_bench/autopilot.py launch --step E0.2 --stage pilot|full [--dry-run]

A (condition, seed) cell is done when a task log holds a best-trial summary that verifies (config, direction,
parcellation, condition, cohort size). `launch --stage full` needs the step's gate (E0.4: the gates of its
needs_pilots), submits only cells with no done and no queued task, retries a failed cell at most budget.max_retries
times, sets --time from the pilot wall time, and refuses to exceed the GPU-h cap (cut by queue_cut.factor once any
campaign task waited > queue_cut.wait_h in the queue). Exit code 2 = a D1 stop condition (report to the user).
"""
import argparse
import functools
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import submit  # noqa: E402

REPO_ROOT = submit.REPO_ROOT
LOG_DIR = REPO_ROOT / "results" / "logs"
MARKER = "Best Tune trial comprehensive summary:"
STOP = 2
DATE_FMT = "%a %b %d %I:%M:%S %p %Z %Y"
LEGACY_DIRS = ("task_fc2sc",)  # this folder's name for runs before the 2026-10-07 rename (logs keep that path)
KILL = re.compile(r"CANCELLED AT \S+ DUE to SIGNAL Terminated")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _when(line):
    m = re.search(r"at (\w{3} \w{3} +\d+ [\d:]+ [AP]M \w+ \d{4})", line)
    try:
        return datetime.strptime(re.sub(r" +", " ", m.group(1)), DATE_FMT) if m else None
    except ValueError:
        return None


def parse_summary(text):
    i = text.rfind(MARKER)
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[i + len(MARKER):].lstrip())[0]
    except ValueError:
        return None


def verify(cfg, step, cond, summary):
    """'' if the best-trial summary ran this step's config / direction / parcellation / condition / cohort, else why."""
    st, run = cfg["steps"][step], summary.get("run") or {}
    d = cfg["directions"][st["direction"]]
    want = [f"scripts/experiments/task_bidirectional_bench/configs/{st['parcellation']}/{cond}.yml"]
    want += [w.replace("/task_bidirectional_bench/", f"/{old}/") for w in list(want) for old in LEGACY_DIRS]
    want += [p.format(cond=cond) for p in st.get("legacy_configs", [])]
    errs = []
    if run.get("config_path") not in want:
        errs.append(f"config {run.get('config_path')}")
    if (run.get("source"), run.get("target")) != (d["source"], d["target"]):
        errs.append(f"{run.get('source')}->{run.get('target')}")
    if "parcellation" in run or not st.get("legacy_configs"):  # E0.1 logs predate these summary fields
        if run.get("parcellation") != st["parcellation"]:
            errs.append(f"parcellation {run.get('parcellation')}")
        if run.get("fc_source_condition") != cond:
            errs.append(f"condition {run.get('fc_source_condition')}")
        if run.get("n_subjects") != cfg["cohort"][st["parcellation"]]:
            errs.append(f"n_subjects {run.get('n_subjects')}")
    return "; ".join(errs)


@functools.lru_cache(maxsize=None)
def _campaign_logs():
    """One listing of results/logs: the campaign's .out files (e0_taskfc*), oldest first."""
    names = [n for n in os.listdir(LOG_DIR) if n.startswith("e0_taskfc") and n.endswith(".out")]
    return sorted((str(LOG_DIR / n) for n in names), key=os.path.getmtime)


@functools.lru_cache(maxsize=None)
def _queue():
    """{job name: {array task ids pending or running}} for this user, one squeue call."""
    out = _run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%j|%K"])
    q = {}
    for line in out.splitlines():
        name, _, task = line.partition("|")
        if task.isdigit():
            q.setdefault(name, set()).add(int(task))
    return q


def task_logs(cfg, step, cond):
    return _task_logs(step, cond, json.dumps(cfg, sort_keys=True, default=str))


@functools.lru_cache(maxsize=None)
def _task_logs(step, cond, cfg_json):
    """Every task log of this step x condition (exact name, so rest never matches rest_S1)."""
    cfg = json.loads(cfg_json)
    name = submit.job_name(cfg, step, cond)
    pat = re.compile(rf"{re.escape(name)}_(\d+)_(\d+)\.out")
    out = []
    for path in _campaign_logs():
        m = pat.fullmatch(os.path.basename(path))
        if not m:
            continue
        text = Path(path).read_text(errors="replace")
        seed = re.search(r"Seed=(\d+)", text)
        if not seed:
            continue
        lines = text.splitlines()
        start = next((_when(l) for l in lines if l.startswith("Starting job")), None)
        end = next((_when(l) for l in reversed(lines) if l.startswith("Job Over")), None)
        status = re.findall(r"Trial status: ([^│\n]*)", text)
        last = status[-1] if status else ""
        err_path = path[:-4] + ".err"
        err = Path(err_path).read_text(errors="replace") if os.path.exists(err_path) else ""
        summary = parse_summary(text)
        out.append(dict(path=path, job=f"{m.group(1)}_{m.group(2)}", seed=int(seed.group(1)), summary=summary,
                        bad=(verify(cfg, step, cond, summary) if summary else ""),
                        samples=int((re.search(r"\((\d+) samples", text) or [0, 0])[1]),
                        n_term=int((re.search(r"(\d+) TERMINATED", last) or [0, 0])[1]),
                        n_err=int((re.search(r"(\d+) ERROR", last) or [0, 0])[1]),
                        wall_h=((end - start).total_seconds() / 3600) if (start and end) else None,
                        killed=bool(KILL.search(err)), ended=bool(end) or bool(KILL.search(err))))
    return out


def queued(cfg, step, cond):
    return set(_queue().get(submit.job_name(cfg, step, cond), set()))


def cell_status(cfg, step, cond):
    logs = task_logs(cfg, step, cond)
    done = {l["seed"]: l for l in logs if l["summary"] and not l["bad"]}
    q = queued(cfg, step, cond)
    return dict(logs=logs, done=done, queued=q,
                failed={s for s in {l["seed"] for l in logs if l["ended"]} if s not in done and s not in q},
                attempts={s: sum(1 for l in logs if l["seed"] == s) for s in cfg["seeds"]},
                mismatched=[(l["seed"], l["bad"], os.path.basename(l["path"])) for l in logs if l["bad"]],
                kills={s: sum(1 for l in logs if l["seed"] == s and l["killed"]) for s in cfg["seeds"]})


def gate(cfg, step):
    """(True/False, detail), or (None, detail) while pilot tasks are unfinished."""
    g, p = cfg["gate"], cfg["pilot"]
    detail, ok = {}, True
    for cond in p["conditions"]:
        cs = cell_status(cfg, step, cond)
        if cs["mismatched"]:
            return False, {"mismatch": cs["mismatched"]}
        for seed in p["seeds"]:
            if seed not in cs["done"]:
                if seed in cs["failed"] and cs["attempts"][seed] > cfg["budget"]["max_retries"]:
                    return False, {f"{cond}/{seed}": "failed after retry"}
                return None, {f"{cond}/{seed}": "pending"}
            l = cs["done"][seed]
            s = l["summary"]
            val = (s.get("selected_by") or {}).get("value")
            test = ((s.get("metrics") or {}).get("test") or {})
            vals = [val] + [test.get(k) for k in ("pearson", "demeaned_pearson", "avg_rank", "top1_acc")]
            finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
            trials = l["n_term"] >= g["trials"] and l["n_err"] == 0 and l["samples"] == g["trials"]
            r = dict(val_dr=val, test_dr=test.get("demeaned_pearson"), trials=f"{l['n_term']}/{l['samples']}",
                     finite=finite, wall_h=l["wall_h"] and round(l["wall_h"], 2))
            ok &= finite and trials
            if cond == "rest":
                r["rest_val_ok"] = bool(finite and val > g["min_rest_val_dr"])
                ok &= r["rest_val_ok"]
                ref = (g.get("reference") or {}).get(step)
                if ref and seed == 0:
                    r["reference"] = ref["test_demeaned_pearson"]
                    r["reference_ok"] = bool(finite and abs(test["demeaned_pearson"] - ref["test_demeaned_pearson"]) <= ref["tol"])
                    ok &= r["reference_ok"]
            detail[f"{cond}/{seed}"] = r
    return bool(ok), detail


def step_gate(cfg, step):
    """Gate that unlocks this step's full run: its own pilot, or every pilot in needs_pilots."""
    needs = cfg["steps"][step].get("needs_pilots") or ([step] if step in cfg["pilot"]["steps"] else [])
    res = {s: gate(cfg, s) for s in needs}
    if any(v[0] is False for v in res.values()):
        return False, res
    if any(v[0] is None for v in res.values()):
        return None, res
    return True, res


def _walls(cfg, step):
    return [l["wall_h"] for c in cfg["conditions"] for l in task_logs(cfg, step, c) if l["summary"] and l["wall_h"]]


def wall_estimate(cfg, step, how=max):
    """Finished wall time (h). Projections (mean): this step, else the pilots that unlock it (4S456 pilot for E0.4),
    else E0.1. --time (max): the longest run of this step or of any step on the same parcellation (pilots are only
    2 tasks; E0.1 shows the spread across nodes), so a slow node does not hit the limit."""
    if how is max:
        parc = cfg["steps"][step]["parcellation"]
        same = [s for s, st in cfg["steps"].items() if st["parcellation"] == parc]
        walls = [w for s in same for w in _walls(cfg, s)]
        if walls:
            return max(walls)
    for s in [step] + list(cfg["steps"][step].get("needs_pilots") or []) + ["E0.1"]:  # E0.1: last-resort prior
        walls = _walls(cfg, s)
        if walls:
            return how(walls)
    return None


def _mean(xs):
    return sum(xs) / len(xs)


@functools.lru_cache(maxsize=None)
def _sacct():
    return _run(["sacct", "-u", os.environ.get("USER", ""), "-X", "-n", "-P", "-S", "2026-10-06",
                 "--format=JobName,Submit,Start,State"])


def queue_wait_h(cfg):
    """Longest queue wait (h) of any campaign task: sacct Start - Submit, or now - Submit while pending."""
    prefixes = tuple(st["job_prefix"] for k, st in cfg["steps"].items() if k != "E0.1")
    out = _sacct()
    worst, now = 0.0, datetime.now()
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 4 or not parts[0].startswith(prefixes):
            continue
        try:
            sub = datetime.fromisoformat(parts[1])
            start = datetime.fromisoformat(parts[2]) if parts[2] not in ("Unknown", "None", "") else now
        except ValueError:
            continue
        worst = max(worst, (start - sub).total_seconds() / 3600)
    return worst


def budget(cfg):
    """(used GPU-h, projected remaining GPU-h, cap) over the D1 steps (one GPU per task)."""
    b = cfg["budget"]
    used, proj = 0.0, 0.0
    for step in [s for s in cfg["steps"] if s != "E0.1"]:
        wall = wall_estimate(cfg, step, _mean) or 0.5
        for cond in cfg["conditions"]:
            cs = cell_status(cfg, step, cond)
            used += sum(l["wall_h"] or 0 for l in cs["logs"])
            todo = [s for s in cfg["seeds"] if s not in cs["done"]]
            proj += wall * len(todo)
    cap = b["gpu_h_cap"]
    if queue_wait_h(cfg) > b["queue_cut"]["wait_h"]:
        cap *= b["queue_cut"]["factor"]
    return used, proj, cap


def stops(cfg, step):
    """D1 stop conditions for this step: [] if none."""
    b, out = cfg["budget"], []
    trials = errs = 0
    for cond in cfg["conditions"]:
        cs = cell_status(cfg, step, cond)
        if cs["mismatched"]:
            out.append(f"{cond}: log mismatch {cs['mismatched']}")
        for s in cs["failed"]:
            if cs["attempts"][s] > b["max_retries"]:
                out.append(f"{cond}/{s}: failed {cs['attempts'][s]}x")
        for s, k in cs["kills"].items():
            if k >= 2:
                out.append(f"{cond}/{s}: killed {k}x (GPU-underuse policy?)")
        for l in cs["done"].values():
            trials += l["n_term"] + l["n_err"]
            errs += l["n_err"]
    if trials and errs / trials > b["max_error_frac"]:
        out.append(f"trial error fraction {errs / trials:.0%} > {b['max_error_frac']:.0%}")
    return out


def _time_str(h):
    m = int(math.ceil(h * 60))
    return f"{m // 60}:{m % 60:02d}:00"


def cmd_status(cfg, args):
    for step in ([args.step] if args.step else list(cfg["steps"])):
        st = cfg["steps"][step]
        print(f"{step} {st['parcellation']} {st['direction']} ({st['job_prefix']})")
        for cond in cfg["conditions"]:
            cs = cell_status(cfg, step, cond)
            print(f"  {cond:10s} done {sorted(cs['done'])} queued {sorted(cs['queued'])} failed {sorted(cs['failed'])}"
                  + (f" MISMATCH {cs['mismatched']}" if cs["mismatched"] else ""))
        if step != "E0.1":
            print("  gate:", step_gate(cfg, step)[0], " stops:", stops(cfg, step) or "none")
    used, proj, cap = budget(cfg)
    print(f"budget: used {used:.1f} GPU-h, projected remaining {proj:.1f}, cap {cap:.0f} "
          f"(max queue wait {queue_wait_h(cfg):.1f} h)")
    return 0


def cmd_gate(cfg, args):
    ok, detail = step_gate(cfg, args.step)
    print(json.dumps({"gate": ok, "detail": detail}, indent=1, default=str))
    return STOP if ok is False else 0


def cmd_budget(cfg, args):
    used, proj, cap = budget(cfg)
    print(f"used {used:.1f} + projected {proj:.1f} = {used + proj:.1f} GPU-h vs cap {cap:.0f}")
    return STOP if used + proj > cap else 0


def cmd_launch(cfg, args):
    step, b = args.step, cfg["budget"]
    if args.stage == "pilot":
        if step not in cfg["pilot"]["steps"]:
            raise SystemExit(f"{step} has no pilot (config.yml pilot.steps)")
        conds, seeds, time = cfg["pilot"]["conditions"], cfg["pilot"]["seeds"], cfg["pilot"]["time"]
        plan = {c: [s for s in seeds if s not in (cell_status(cfg, step, c)["done"].keys() | cell_status(cfg, step, c)["queued"])]
                for c in conds}
    else:
        ok, detail = step_gate(cfg, step)
        if ok is not True:
            print(f"{step}: gate {ok}: {json.dumps(detail, default=str)}")
            return STOP if ok is False else 0
        found = stops(cfg, step)
        if found:
            print(f"{step}: STOP {found}")
            return STOP
        wall = wall_estimate(cfg, step)
        time = _time_str(min(b["time_cap_h"], max(b["min_time_h"], b["time_factor"] * wall))) if wall else None
        plan = {}
        for c in cfg["conditions"]:
            cs = cell_status(cfg, step, c)
            plan[c] = [s for s in cfg["seeds"] if s not in cs["done"] and s not in cs["queued"]
                       and cs["attempts"][s] <= b["max_retries"]]
        used, proj, cap = budget(cfg)
        new = sum(len(v) for v in plan.values()) * (wall_estimate(cfg, step, _mean) or 0.5)
        print(f"{step}: {sum(len(v) for v in plan.values())} tasks, ~{new:.1f} GPU-h; used {used:.1f}, "
              f"projected remaining (all steps) {proj:.1f}, cap {cap:.0f}; --time {time}")
        if used + proj > cap:
            print(f"{step}: STOP projected {used + proj:.1f} GPU-h > cap {cap:.0f}")
            return STOP
    for c, seeds in plan.items():
        if seeds:
            submit.submit(cfg, step, [c], seeds, time, dry_run=args.dry_run)
    if not any(plan.values()):
        print(f"{step} {args.stage}: nothing to submit")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "gate", "budget", "launch"])
    ap.add_argument("--step")
    ap.add_argument("--stage", choices=["pilot", "full"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = submit.cfg_load()
    if args.cmd in ("gate", "launch") and not args.step:
        raise SystemExit("--step is required")
    if args.cmd == "launch" and not args.stage:
        raise SystemExit("--stage is required")
    return {"status": cmd_status, "gate": cmd_gate, "budget": cmd_budget, "launch": cmd_launch}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
