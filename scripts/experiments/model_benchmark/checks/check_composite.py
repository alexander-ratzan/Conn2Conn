"""Spec v3 I1 acceptance: a composite benchmark config with every weight at 0 trains exactly like the E2.2 MSE config.

Trains PCA-PLS learnable SC -> FC, seed 0, at its E2.2 seed-0 best config twice in one process (same seed):
(A) models/configs/benchmark/mse/CrossModal_PCA_PLS_learnable.yml, (B) the composite config with all loss weights 0.
Both use batch 64 (E2.2's value for this model). Pass: same loss signature ("mse") and val / test metrics equal to
1e-6. Also checks the generated configs (build_configs.py --check, build_composite_configs.py --check).

    python scripts/experiments/model_benchmark/checks/check_composite.py      # prints COMPOSITE_OK / COMPOSITE_FAIL
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MB = HERE.parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
MODEL, SEED = "CrossModal_PCA_PLS_learnable", 0
KEYS = ("pearson", "demeaned_pearson", "avg_rank", "top1_acc", "mse")


def train(cfg_path, override):
    import lightning.pytorch as pl
    from main import Sim
    pl.seed_everything(1234, workers=True)
    sim = Sim(model_name=MODEL, config_path=str(cfg_path), source="SC", target="FC", shuffle_seed=SEED,
              data_load_mode="precomputed", batch_size=64)
    out = sim._run_learned_single(mode="dev", save_checkpoint=False, config_override=override, run_eval=True)
    sig = dict(out["train_result"].pl_module.hparams).get("loss_signature")
    return sig, {s: {k: float(out["metrics"][s][k]) for k in KEYS} for s in ("val", "test")}


def main():
    ok = True
    for script in ("build_configs.py", "build_composite_configs.py"):
        r = subprocess.run([sys.executable, str(MB / script), "--check"], capture_output=True, text=True)
        print(f"{script} --check: {r.stdout.strip()}", flush=True)
        ok &= r.returncode == 0 or "no reference scales yet" in r.stdout  # Sarwar scales may still be pending
    rec = [r for r in json.loads((MB / "mse" / "sc2fc" / "records.json").read_text())
           if r["model"] == MODEL and int(r["seed"]) == SEED][0]
    best = {"model": {k: v for k, v in rec["config"]["model"].items() if k != "name"},
            "trainer": dict(rec["config"].get("trainer") or {})}
    sig_a, a = train(REPO_ROOT / "models/configs/benchmark/mse" / f"{MODEL}.yml", best)
    sig_b, b = train(REPO_ROOT / "models/configs/benchmark/composite" / f"{MODEL}.yml", best)
    diff = max(abs(a[s][k] - b[s][k]) for s in a for k in KEYS)
    same = sig_a == sig_b == "mse" and diff <= 1e-6
    print(f"A (E2.2 MSE config): {sig_a} val dr {a['val']['demeaned_pearson']:.6f} test dr {a['test']['demeaned_pearson']:.6f}")
    print(f"B (composite, weights 0): {sig_b} val dr {b['val']['demeaned_pearson']:.6f} test dr {b['test']['demeaned_pearson']:.6f}")
    print(f"max |diff| {diff:.2e} -> {'OK' if same else 'FAIL'}; E2.2 record (retrained report) test dr "
          f"{rec['test']['demeaned_pearson']:.6f}", flush=True)
    ok &= same
    print("COMPOSITE_OK" if ok else "COMPOSITE_FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
