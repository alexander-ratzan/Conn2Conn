"""Spec v3 E3 / I1: fixed reference scales for roster models without a v2:E1 composite-loss instance (Sarwar).

As v2:E1.3 (`loss_grid.measure_term_scales` / `reference_scales`): per seed, train the model MSE-only at its E2.2
seed-best config (model_benchmark/mse/<direction>/records.json) at batch 64, measure each term's mean |raw| over full
training batches, then c_t = mean_seeds(s_t) / mean_seeds(s_mse). One (direction, seed) per call; --combine writes
composite/reference_scales.yml, which build_composite_configs.py reads (`scales_from: measured`).

    python scripts/experiments/model_benchmark/composite_scales.py --model Sarwar2020MLP --direction sc2fc --seed 0
    python scripts/experiments/model_benchmark/composite_scales.py --combine
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
PARTS = HERE / "composite" / "reference_scales_parts"
DIRS = {"sc2fc": ("SC", "FC"), "fc2sc": ("FC", "SC")}


def measure(model, direction, seed, batch):
    from main import Sim
    from scripts.results_utils.loss_grid import measure_term_scales, mse_only_trainer
    rec = [r for r in json.loads((HERE / "mse" / direction / "records.json").read_text())
           if r["model"] == model and int(r["seed"]) == seed]
    if not rec:
        raise SystemExit(f"no E2.2 record for {model} {direction} seed {seed}")
    best = rec[0]["config"]
    mse_cfg = HERE.parent.parent.parent / "models" / "configs" / "benchmark" / "mse" / f"{model}.yml"
    alt = mse_cfg.with_name(f"{model}_fc2sc.yml")
    cfg_path = alt if direction == "fc2sc" and alt.exists() else mse_cfg
    override = {"model": {k: v for k, v in (best.get("model") or {}).items() if k != "name"},
                "trainer": {**(best.get("trainer") or {}), **mse_only_trainer(batch)}}
    src, tgt = DIRS[direction]
    sim = Sim(model_name=model, config_path=str(cfg_path), source=src, target=tgt, shuffle_seed=seed,
              data_load_mode="precomputed", batch_size=batch)
    out = sim._run_learned_single(mode="dev", save_checkpoint=False, config_override=override, run_eval=False)
    scales = measure_term_scales(out["model"], sim.train_ds, batch)
    PARTS.mkdir(parents=True, exist_ok=True)
    part = {"model": model, "direction": direction, "seed": seed, "batch_size": batch, "config": override,
            "term_scales": scales}
    (PARTS / f"{model}__{direction}__seed{seed}.json").write_text(json.dumps(part, indent=1, default=str))
    print(f"{model} {direction} seed {seed}: " + ", ".join(f"{t} {v['abs_mean']:.4g}" for t, v in scales.items()), flush=True)


def combine():
    from scripts.results_utils.loss_grid import reference_scales
    by = {}
    for p in sorted(PARTS.glob("*.json")):
        r = json.loads(p.read_text())
        by.setdefault(r["model"], {}).setdefault(r["direction"], {})[int(r["seed"])] = r["term_scales"]
    out = {m: {d: reference_scales(s) for d, s in ds.items()} for m, ds in by.items()}
    (HERE / "composite" / "reference_scales.yml").write_text(
        "# Spec v3 E3 measured reference scales (composite_scales.py --combine); c_t = s_t / s_mse over seeds.\n"
        + yaml.safe_dump(json.loads(json.dumps(out)), sort_keys=False, width=110))
    for m, ds in out.items():
        for d, r in ds.items():
            print(m, d, "seeds", r["seeds"], {t: round(c, 3) for t, c in r["c"].items()},
                  "spread_cv", {t: round(c, 3) for t, c in r["spread_cv"].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--direction", choices=list(DIRS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()
    if args.combine:
        return combine()
    measure(args.model, args.direction, args.seed, args.batch)


if __name__ == "__main__":
    main()
