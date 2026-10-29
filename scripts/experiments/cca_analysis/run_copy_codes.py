"""Spec v4 E5: copy in the shared block-PCA code space (CPU), every ordered pair including SC, for the notebook.

The source is encoded in E2's variance-adjusted block-PCA codes (D2; first k components), z = W^T (x_a - mean_a) / s_a,
and decoded as the target, x_hat = mean_b + s_b W z: the edge-space copy restricted to the shared k-dimensional basis
(E4.0's B0-codes). Scored like every model (E2's `score` on the full connectome). SC has its own basis, so pairs with
SC copy coordinates between unrelated bases: a null.  -> tables/copy_codes_pairs.csv (k = 256, E1 / E4's code size)

    python scripts/experiments/cca_analysis/run_copy_codes.py
"""
import numpy as np
import pandas as pd

from e5_common import CFG, HERE, PAIRS, SeedData, score

K = 256


def main():
    rows = []
    for seed in CFG["seeds"]:
        sd = SeedData(seed, "main")
        for k in [K]:
            for a, b in PAIRS:
                pred = sd.decode(b, sd.code(a, "test", k))
                rows.append(dict(seed=seed, k=k, source=a, target=b,
                                 **score(pred, sd.edges[("test", b)], sd.train_mean[b])))
        print(f"seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "tables" / "copy_codes_pairs.csv", index=False)
    print("wrote tables/copy_codes_pairs.csv")


if __name__ == "__main__":
    main()
