#!/usr/bin/env python3
"""Single-seed preview: per-estimator sibling / MZ / DZ AUC from a family npz (rank-based AUC,
pure numpy, no sklearn). For a quick look before the full pooled finalize.

    python preview_seed.py 0
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ESTIMATORS = ["BR", "PLS", "obj1a", "obj2c", "obj2c_raw"]


def auc(pos, neg):
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort() + 1.0
    U = r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return U / (pos.size * neg.size)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d = np.load(HERE / "outputs" / "parts" / "family" / f"obj_family_s{seed}.npz")
    g = lambda k: d[k] if k in d.files else np.array([])
    print(f"seed {seed} — sibling / MZ / DZ AUC per objective (vs unrelated_matched)")
    print(f"{'estimator':12s}{'sibling':>9s}{'MZ':>8s}{'DZ':>8s}{'n_sib':>7s}")
    for e in ESTIMATORS:
        ref = g(f"{e}__unrelated_matched")
        print(f"{e:12s}{auc(g(f'{e}__sibling'), ref):>9.3f}"
              f"{auc(g(f'{e}__MZ'), ref):>8.3f}{auc(g(f'{e}__DZ'), ref):>8.3f}"
              f"{g(f'{e}__sibling').size:>7d}")


if __name__ == "__main__":
    main()
