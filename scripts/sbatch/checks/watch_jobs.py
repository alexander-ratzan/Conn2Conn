"""Watch this user's SLURM jobs by name prefix and print one line per event (for a notifying monitor).

Events (each printed once):
    HANG_RAY    a Tune task has run > --ray-grace s without logging "Ray cluster resources" (Ray start-up hang)
    SILENT      a running task's .out/.err logs have not changed for > --silence s (--silence-tune for Tune jobs)
    ENDED       a task left the queue; its final state from sacct (COMPLETED / FAILED / CANCELLED / TIMEOUT ...)
    BLOCKED     a pending task can never start (DependencyNeverSatisfied)
    ALL_DONE    no matching job is left in the queue (the watcher exits)

    python scripts/sbatch/checks/watch_jobs.py --prefix e1_ [--poll 60]
Logs are expected at results/logs/<job-name>_<array job>_<task>.{out,err} (the repo launcher convention).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
LOG_DIR = REPO_ROOT / "results" / "logs"


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def queue(prefix):
    """{task id: (name, state, elapsed_s, reason, array_job, task)} for this user's matching jobs."""
    out = _run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i|%j|%T|%M|%R|%F|%K"])
    jobs = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 7 or not parts[1].startswith(prefix):
            continue
        jid, name, state, elapsed, reason, array_job, task = parts
        jobs[jid] = (name, state, _seconds(elapsed), reason, array_job, task)
    return jobs


def _seconds(s):
    if not s or s in ("0:00", "INVALID"):
        return 0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    nums = [int(x) for x in s.split(":")]
    while len(nums) < 3:
        nums.insert(0, 0)
    return days * 86400 + nums[0] * 3600 + nums[1] * 60 + nums[2]


def final_state(jid):
    out = _run(["sacct", "-j", jid, "-X", "-n", "-P", "--format=State,Elapsed,ExitCode,NodeList"])
    return out.strip().splitlines()[0] if out.strip() else "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--ray-grace", type=int, default=300)
    ap.add_argument("--silence-tune", type=int, default=900)
    ap.add_argument("--silence", type=int, default=1800)
    args = ap.parse_args()
    seen, reported = {}, set()

    def emit(key, msg):
        if key not in reported:
            reported.add(key)
            print(msg, flush=True)

    while True:
        jobs = queue(args.prefix)
        for jid, (name, state, elapsed, reason, array_job, task) in jobs.items():
            seen[jid] = name
            if state == "PENDING" and "DependencyNeverSatisfied" in reason:
                emit(("blocked", jid), f"BLOCKED {jid} {name}: {reason}")
            if state != "RUNNING":
                continue
            stem = LOG_DIR / f"{name}_{array_job}_{task if task not in ('', 'N/A') else 0}"
            out, err = stem.with_suffix(".out"), stem.with_suffix(".err")
            is_tune = "stage1" in name
            if is_tune and elapsed > args.ray_grace and out.exists() and "Ray cluster resources" not in out.read_text(errors="replace"):
                emit(("ray", jid), f"HANG_RAY {jid} {name}: {elapsed}s without 'Ray cluster resources' ({out.name})")
            mtimes = [p.stat().st_mtime for p in (out, err) if p.exists()]
            if mtimes:
                quiet = time.time() - max(mtimes)
                limit = args.silence_tune if is_tune else args.silence
                if quiet > limit:
                    emit(("silent", jid, int(max(mtimes))), f"SILENT {jid} {name}: logs unchanged for {int(quiet)}s ({out.name})")
        for jid, name in list(seen.items()):
            if jid not in jobs:
                emit(("ended", jid), f"ENDED {jid} {name}: {final_state(jid)}")
                del seen[jid]
        if not jobs:
            print("ALL_DONE", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
