"""
Apply the pilot gate and the epochs rule (config.yml `autonomy:`) to the collected pilot tables and print the decision.

    python scripts/experiments/composite_loss/krakencoder/grid_runner.py collect --set pilot
    python scripts/experiments/composite_loss/krakencoder/checks/pilot_gate.py [--parity-tag kraken_default]

Reads tables/pilot_epoch_history.csv, the parity fit's epoch_history.csv (batch 41) and the pilot job logs; writes
tables/pilot_gate.json (decision, chosen epochs, projected grid GPU-h).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())

import yaml  # noqa: E402


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def job_wall_s(job_id: str) -> float | None:
    """Wall time of a finished pilot job from its log start/end stamps."""
    out = REPO_ROOT / "results" / "logs" / f"cl_kraken_{job_id}_0.out"
    if not out.exists():
        return None
    text = out.read_text()
    times = re.findall(r"(?:Starting job .* at|Job Over at) (\w{3} \w{3} +\d+ [\d:]+ [AP]M \w+ \d{4})", text)
    if len(times) < 2:
        return None
    import datetime as dt
    fmt = "%a %b %d %I:%M:%S %p %Z %Y"
    a, b = (dt.datetime.strptime(" ".join(t.split()), fmt) for t in times[:2])
    return (b - a).total_seconds()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity-tag", default="kraken_default")
    ap.add_argument("--pilot-job", help="pilot SLURM array job id (for the measured job wall time)")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    rows = read(HERE / "tables" / "pilot_epoch_history.csv")
    decision = {"checks": {}}

    # Gate 1: every pilot fit present with a final checkpoint at recipe.epochs.
    want = cfg["recipe"]["epochs"]
    fits = defaultdict(set)
    for r in rows:
        fits[r["combo_id"]].add(int(r["epoch"]))
    expected = cfg["sets"]["pilot"]["cells"]
    complete = {c: (want in fits.get(c, set())) for c in expected}
    decision["checks"]["all_pilot_fits_complete"] = all(complete.values())

    # Gate 2: kraken_default (batch 64) final val demeaned r within 0.01 of the batch-41 parity fit.
    parity = read(REPO_ROOT / "results" / "krakencoder" / args.parity_tag / "seed0" / "epoch_history.csv")
    par_last = {r["direction"]: float(r["val_demeaned_pearson"]) for r in parity if int(r["epoch"]) == max(int(x["epoch"]) for x in parity)}
    kd_last = {r["direction"]: float(r["val_demeaned_pearson"]) for r in rows
               if r["combo_id"] == "kraken_default" and int(r["epoch"]) == want}
    diffs = {d: kd_last.get(d, math.nan) - par_last[d] for d in par_last}
    decision["kraken_default_vs_parity_val_dr"] = {d: {"batch64": kd_last.get(d), "batch41": par_last[d], "diff": diffs[d]}
                                                   for d in par_last}
    decision["checks"]["kraken_default_within_0.01_of_parity"] = all(abs(v) <= 0.01 for v in diffs.values())

    # Epochs rule: mean over pilot cells per (direction, epoch).
    curve = defaultdict(lambda: defaultdict(list))
    for r in rows:
        curve[r["direction"]][int(r["epoch"])].append((float(r["val_demeaned_pearson"]), float(r["val_avg_rank"])))
    first_ok = {}
    for d, by_epoch in curve.items():
        epochs = sorted(by_epoch)
        dr = {e: sum(v[0] for v in by_epoch[e]) / len(by_epoch[e]) for e in epochs}
        rk = {e: sum(v[1] for v in by_epoch[e]) / len(by_epoch[e]) for e in epochs}
        dr_max, rk_max = max(dr.values()), max(rk.values())
        ok = [e for e in epochs if dr[e] >= 0.98 * dr_max and rk[e] >= rk_max - 0.005]
        first_ok[d] = {"epoch": ok[0] if ok else None, "val_dr_max": dr_max, "val_rank_max": rk_max,
                       "curve_dr": {e: round(dr[e], 4) for e in epochs}}
    e_star = max(v["epoch"] or want for v in first_ok.values())
    chosen = min(int(math.ceil(e_star * 1.2 / 100.0) * 100), 2000)
    decision["epochs_rule"] = {"per_direction": first_ok, "plateau_epoch": e_star, "chosen_epochs": chosen}

    # Budget: 27 grid jobs x measured packed pilot job wall, scaled by chosen / pilot epochs.
    wall = job_wall_s(args.pilot_job) if args.pilot_job else None
    import sys
    sys.path.insert(0, str(HERE))
    import grid_runner
    grid_fits = grid_runner.plan(cfg, "grid", argparse.Namespace(epochs=None, checkpoint_every=None, tag_suffix=""))
    n_jobs = math.ceil(len(grid_fits) / cfg["packing"]["fits_per_job"])
    if wall:
        projected = n_jobs * wall / 3600.0 * (chosen / want)
        decision["budget"] = {"pilot_job_wall_h": wall / 3600.0, "grid_jobs": n_jobs, "projected_gpu_h": projected,
                              "limit_gpu_h": cfg["autonomy"]["grid_budget_gpu_h"]}
        decision["checks"]["grid_within_budget"] = projected <= cfg["autonomy"]["grid_budget_gpu_h"]
    decision["proceed"] = all(decision["checks"].values())
    out = HERE / "tables" / "pilot_gate.json"
    out.write_text(json.dumps(decision, indent=1))
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
