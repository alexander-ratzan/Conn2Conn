#!/usr/bin/env python3
"""F6/F7 family-structure runner — ONE (parcellation, seed) unit.

Faithful port of STEP 8.2: build the 8 connectome variants for this seed's test split,
extract per-pair demeaned-cosine similarities bucketed by relation (MZ/DZ/sibling/unrelated),
and cache them to a per-unit npz. The pooled aggregation (AUC + bootstrap + perm + FDR) runs
once in finalize_fm.py. This is the heavy step (needs connectome data) — runs on Torch.

The 8 variants include `combined_pred_SC` and `pred_SC_raw`, so the same cache also yields the
F7 predictor/identifier-tradeoff evidence (combined collapses to chance on siblings; raw does not).

    python run_f6_family.py --parc Glasser --seed 0
    python run_f6_family.py --parc 4S456Parcels --seed 7
"""
from pathlib import Path
import argparse
import sys
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fm_common as fm

OUT_PARTS = _HERE / "outputs" / "parts"
# notebook seed-0 Glasser pair counts (STEP 8.0 sanity) — used as a self-check on Glasser/seed0
_NB_GLASSER_SEED0_COUNTS = {"MZ": 33, "DZ": 13, "sibling": 125, "unrelated_matched": 171}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", required=True, choices=fm.PARCELLATIONS)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    D = fm._data()
    D["set_parcellation"](args.parc)
    split = D["load_seed_split"](seed=args.seed, source="FC", target="SC")

    sims, pair_counts = fm.pair_sims_for_seed(split, args.seed)
    print(f"[{args.parc} seed{args.seed}] pair counts: {pair_counts}", flush=True)

    # self-check: Glasser seed-0 must reproduce the notebook's documented pair counts
    if args.parc == "Glasser" and args.seed == 0:
        for r, n in _NB_GLASSER_SEED0_COUNTS.items():
            assert pair_counts.get(r) == n, (
                f"Glasser/seed0 pair-count drift {r}: got {pair_counts.get(r)} expected {n} "
                f"— split or pairing diverged from the notebook.")
        print("  self-check OK: Glasser/seed0 pair counts match the notebook.", flush=True)

    out_dir = OUT_PARTS / args.parc
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {"seed": np.array([args.seed]),
            "pair_count_keys": np.array(list(pair_counts.keys())),
            "pair_counts": np.array(list(pair_counts.values()))}
    for v in fm.FAM_VARIANTS:
        for r in fm.RELATIONS:
            save[f"{v}__{r}"] = sims[v].get(r, np.array([], dtype=np.float32))
    out_path = out_dir / f"family_seed{args.seed}.npz"
    np.savez_compressed(out_path, **save)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
