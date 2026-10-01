"""Test-retest ceiling for the E1.10 cross-model comparison (spec v2 E1.10): `TestRetestPrecomputed` on SC -> FC splits.

Predicts each subject's session-1 FC from their session-2 FC (no learning), on the same family-preserving splits as
the E1 instances (shuffle seeds 0-4), and writes the test metrics that the cross-model tool draws as the ceiling.
Only subjects with both resting-state sessions enter (the model needs `expose_fc_sessions`), so the test set can be
a subset of the instances' test set; the per-seed subject count is recorded.

    python scripts/experiments/composite_loss/ceiling/test_retest.py [--seeds 0 1 2 3 4]   (via launch_ceiling.sh)
Writes ceiling/test_retest.csv (one row per seed) and ceiling/test_retest.json (mean and SE).
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
OUT = Path(__file__).resolve().parent
METRICS = ("demeaned_pearson", "avg_rank", "top1_acc", "pearson", "mse")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    from main import Sim
    rows = []
    for seed in args.seeds:
        sim = Sim(model_name="TestRetestPrecomputed", source="SC", target="FC", shuffle_seed=seed,
                  data_load_mode="precomputed")
        out = sim._run_closed_form_single(mode="dev", save_checkpoint=False, run_eval=True)
        test = out["metrics"]["test"] or {}
        ev = (out.get("evaluators") or {}).get("test")
        n_test = int(len(getattr(ev, "Y_pred", getattr(ev, "preds", [])))) if ev is not None else None
        rows.append({"seed": seed, "n_test": n_test, **{f"test_{m}": float(test[m]) for m in METRICS if m in test}})
        print(json.dumps(rows[-1]), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "test_retest.csv", index=False, float_format="%.6g")
    summ = {"model": "TestRetestPrecomputed", "direction": "SC->FC", "seeds": args.seeds}
    for m in METRICS:
        col = f"test_{m}"
        if col in df:
            summ[f"{m}_mean"] = float(df[col].mean())
            summ[f"{m}_se"] = float(df[col].std(ddof=1) / math.sqrt(len(df))) if len(df) > 1 else 0.0
    (OUT / "test_retest.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
