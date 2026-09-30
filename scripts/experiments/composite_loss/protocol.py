"""Composite-loss protocol CLI (spec v2 E1). One entry point for every instance and stage.

    python scripts/experiments/composite_loss/protocol.py stage1    --instance linear_backbone   # CPU: summary + consensus
    python scripts/experiments/composite_loss/protocol.py consensus --instance linear_backbone   # GPU: E1.3 runs + scales
    python scripts/experiments/composite_loss/protocol.py grid      --instance linear_backbone --task-index 0 --tasks 4
    python scripts/experiments/composite_loss/protocol.py report    --instance linear_backbone   # CPU: E1.5 tables/figures

GPU steps go through launch_consensus.sh / launch_grid.sh. Several (combination, seed) runs share one GPU as separate
processes (`runtime.parallel` in the instance config). Exit code 2 = a stop condition from the instance config
tripped (spec v2 D3): the agent stops and reports instead of continuing.
"""
import argparse
import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))

from scripts.results_utils import loss_grid as lg  # noqa: E402

STOP = 2


def _stop(instance, reasons, **extra):
    lg.update_state(instance, last_stop={"reasons": reasons, **extra})
    for r in reasons:
        print(f"STOP: {r}", flush=True)
    sys.exit(STOP)


# ------------------------------------------------------------------------------------------ stage1 (CPU)
def cmd_stage1(args):
    import yaml
    cfg = lg.load_instance(args.instance)
    runs = lg.find_stage1_runs(cfg)
    seeds = [int(s) for s in cfg["seeds"]]
    missing = [s for s in seeds if s not in runs]
    if missing:
        print(f"Stage 1 not finished for seeds {missing} (finished: {sorted(runs)})")
        return 1
    trials, epochs = lg.collect_stage1_trials(cfg["model"], {s: runs[s] for s in seeds})
    summary, stop = lg.stage1_summary(trials, cfg.get("stop", {}).get("stage1_min_best_val"))
    search = yaml.safe_load((REPO_ROOT / cfg["stage1"]["config"]).read_text())["search_space"]
    consensus, detail = lg.consensus_config(trials, search)
    tables = lg.instance_dir(args.instance) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    trials.to_csv(tables / "stage1_trials.csv", index=False)
    epochs.to_csv(tables / "stage1_epochs.csv.gz", index=False)
    summary.to_csv(tables / "stage1_summary.csv", index=False)
    lg.update_state(args.instance, stage1_runs=runs, stage1_best_val={int(r.seed): float(r.best_val) for r in summary.itertuples()},
                    consensus=consensus, consensus_detail=detail)
    print(summary.to_string(index=False))
    print("consensus:", json.dumps(consensus))
    if stop:
        _stop(args.instance, stop)
    return 0


# ------------------------------------------------------------------------------------------ GPU runs
def _worker(job):
    instance, stage, combo_id, seed, trainer_overrides, grad_cosine = job
    try:
        rec = lg.run_one(instance, stage, combo_id, seed, trainer_overrides, grad_cosine=grad_cosine)
        return {"ok": True, "combo_id": combo_id, "seed": seed, "record": rec}
    except Exception:
        return {"ok": False, "combo_id": combo_id, "seed": seed, "error": traceback.format_exc()}


def _run_jobs(jobs, parallel):
    ctx = mp.get_context("spawn")  # each run gets its own CUDA context and HCP_Base
    with ctx.Pool(processes=max(1, min(parallel, len(jobs))), maxtasksperchild=1) as pool:
        return pool.map(_worker, jobs, chunksize=1)


def cmd_consensus(args):
    cfg = lg.load_instance(args.instance)
    state = cfg["state"]
    if "consensus" not in state:
        print("run the stage1 step first")
        return 1
    seeds = args.seeds or [int(s) for s in cfg["seeds"]]
    trainer = lg.mse_only_trainer(cfg["grid"]["batch_size"])
    jobs = [(args.instance, "consensus", "mse_only", s, trainer, None) for s in seeds]
    results = _run_jobs(jobs, args.parallel or cfg["runtime"]["parallel"])
    failed = [r for r in results if not r["ok"]]
    for r in failed:
        print(f"run failed: seed {r['seed']}\n{r['error']}", flush=True)
    if failed:
        _stop(args.instance, [f"consensus run failed for seeds {[r['seed'] for r in failed]}"])
    recs = {r["seed"]: r["record"] for r in results}
    ok, stats = lg.consensus_accepted({s: recs[s]["val_demeaned_r_last"] for s in seeds},
                                      {s: state["stage1_best_val"][s] for s in seeds})
    scales = lg.reference_scales({s: recs[s]["term_scales"] for s in seeds})
    lg.update_state(args.instance, consensus_check={"accepted": ok, **stats}, reference_scales=scales)
    print("consensus check:", json.dumps(lg._plain({"accepted": ok, **stats})))
    print("reference scales c_t:", json.dumps(scales["c"]), "| spread (cv):", json.dumps(scales["spread_cv"]))
    if not ok:
        _stop(args.instance, [f"consensus mean val {stats['consensus_mean']:.4f} misses the per-seed best mean "
                              f"{stats['best_mean']:.4f} by more than 1 SE ({stats['se']:.4f})"])
    import math
    bad = [t for t, v in scales["c"].items() if not (isinstance(v, float) and math.isfinite(v) and v > 0)]
    if bad:
        _stop(args.instance, [f"non-finite or non-positive reference scale for {bad}"])
    return 0


def cmd_grid(args):
    cfg = lg.load_instance(args.instance)
    state = cfg["state"]
    if "reference_scales" not in state or not state.get("consensus_check", {}).get("accepted"):
        print("run the consensus step first (and it must be accepted)")
        return 1
    combos = cfg["grid"]["combos"]
    if args.combos:
        combos = [c for c in combos if c["id"] in set(args.combos)]
    seeds = args.seeds or [int(s) for s in cfg["seeds"]]
    pairs = [(c, s) for c in combos for s in seeds]
    if args.tasks:
        pairs = pairs[args.task_index::args.tasks]  # interleaved so every task gets a mix of combinations
    scales = state["reference_scales"]["c"]
    batch = cfg["grid"]["batch_size"]
    grad_cosine = cfg.get("runtime", {}).get("grad_cosine")
    jobs = [(args.instance, "stage2", c["id"], s, lg.combo_loss_trainer(c, scales, batch), grad_cosine) for c, s in pairs]
    print(f"{len(jobs)} runs on this task: {[(c['id'], s) for c, s in pairs]}", flush=True)
    results = _run_jobs(jobs, args.parallel or cfg["runtime"]["parallel"])
    problems = []
    for (c, s), r in zip(pairs, results):
        if not r["ok"]:
            problems.append(f"{c['id']} seed {s} failed")
            print(r["error"], flush=True)
            continue
        rec = r["record"]
        if rec["loss_signature"] != lg.expected_signature(c):
            problems.append(f"{c['id']} seed {s}: signature {rec['loss_signature']} != {lg.expected_signature(c)}")
        vals = [rec.get(f"test_{m}") for m in ("demeaned_pearson", "avg_rank", "mse")]
        if any(v is None or v != v for v in vals):
            problems.append(f"{c['id']} seed {s}: non-finite test metrics")
    if problems:
        _stop(args.instance, problems, task_index=args.task_index)
    return 0


# ------------------------------------------------------------------------------------------ report (CPU)
def cmd_report(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("composite_loss_report", Path(__file__).with_name("report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    return report.build(args.instance)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage1", "consensus", "grid", "report"):
        p = sub.add_parser(name)
        p.add_argument("--instance", required=True)
        if name in ("consensus", "grid"):
            p.add_argument("--seeds", type=int, nargs="*")
            p.add_argument("--parallel", type=int)
        if name == "grid":
            p.add_argument("--combos", nargs="*")
            p.add_argument("--task-index", type=int, default=0)
            p.add_argument("--tasks", type=int)
    args = ap.parse_args()
    return {"stage1": cmd_stage1, "consensus": cmd_consensus, "grid": cmd_grid, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
