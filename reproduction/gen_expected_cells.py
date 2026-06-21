#!/usr/bin/env python3
"""Generate configs/expected_cells.csv — the completeness ground truth.

Enumerates every expected grid cell (one CSV row per (task, parc, seed, variant, input, target))
derived from the SAME constants the runners use (RECON_PAIRS / DOWNSTREAM_INPUTS / estimator
specs), so the manifest can never drift from what the runners actually produce. The verifier
diffs the produced CSVs against this.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import (  # noqa: E402
    ESTIMATOR_SPECS, SCALAR_ESTIMATOR_SPECS, variant_tag, PARCELLATIONS,
    LEAK_TARGETS, CONFIGS_DIR,
)
from run_reconstruction_grid import RECON_PAIRS  # noqa: E402
from run_downstream_grid import DOWNSTREAM_INPUTS, COG_RESULT_TARGETS  # noqa: E402

SEEDS = list(range(10))


def main():
    rows = []
    for parc in PARCELLATIONS:
        for seed in SEEDS:
            # reconstruction
            for (inp, tgt) in RECON_PAIRS:
                for est, specs in ESTIMATOR_SPECS.items():
                    for spec in specs:
                        rows.append({"task": "reconstruction", "parcellation": parc,
                                     "seed": seed, "estimator": est,
                                     "variant": variant_tag(est, spec["params"]),
                                     "input_set": inp, "target": tgt})
            # downstream
            for inp in DOWNSTREAM_INPUTS:
                for tgt in COG_RESULT_TARGETS + LEAK_TARGETS:
                    for est, specs in SCALAR_ESTIMATOR_SPECS.items():
                        for spec in specs:
                            rows.append({"task": "downstream", "parcellation": parc,
                                         "seed": seed, "estimator": est,
                                         "variant": variant_tag(est, spec["params"]),
                                         "input_set": inp, "target": tgt})
    df = pd.DataFrame(rows)
    out = CONFIGS_DIR / "expected_cells.csv"
    df.to_csv(out, index=False)
    n_recon = int((df.task == "reconstruction").sum())
    n_down = int((df.task == "downstream").sum())
    print(f"[expected] wrote {len(df)} expected cells "
          f"(reconstruction={n_recon}, downstream={n_down}) -> {out}")


if __name__ == "__main__":
    main()
