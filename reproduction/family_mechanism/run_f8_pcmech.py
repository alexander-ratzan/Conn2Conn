#!/usr/bin/env python3
"""F8 PC-mechanism runner — ONE (parcellation, seed) unit.

Faithful port of depth1 (cells 3/5/7) + depth1.1 (cells 3/6): PCA the SC/FC train edges,
compute per-PC FC->PC R², per-PC family AUC, sex+bv confound R², PC1 residualization, and
node strength for rich-club. Saves a per-unit npz; cross-seed alignment + localization +
verdict run once in finalize_f8.py.

    python run_f8_pcmech.py --parc Glasser --seed 0
"""
from pathlib import Path
import argparse
import sys
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fm_common as fm
import _f8_common as f8

OUT_PARTS = _HERE / "outputs" / "parts_f8"
# notebook seed-0 Glasser anchors (depth1) for the self-check
_NB_G0 = {"fc_r2_pc1": 0.5928, "fc_r2_pc3": 0.2561, "conf_r2_pc1": 0.8914,
          "auc_sb_pc3": 0.5838, "pair_counts": {"MZ": 33, "DZ": 13, "sibling": 125, "unrelated_matched": 171}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", required=True, choices=fm.PARCELLATIONS)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    res = f8.compute_seed(args.parc, args.seed)
    pc = dict(zip(res["pair_count_keys"], res["pair_counts"]))
    print(f"[{args.parc} seed{args.seed}] pair counts: {pc}", flush=True)
    print(f"  FC->PC R2 (PC1..3): {res['fc_r2'][:3].round(4)}  conf_R2 PC1={res['conf_r2'][0]:.4f}", flush=True)

    if args.parc == "Glasser" and args.seed == 0:
        for r, n in _NB_G0["pair_counts"].items():
            assert int(pc[r]) == n, f"Glasser/seed0 pair-count drift {r}: {pc[r]} != {n}"
        assert abs(res["fc_r2"][0] - _NB_G0["fc_r2_pc1"]) < 5e-3, res["fc_r2"][0]
        assert abs(res["fc_r2"][2] - _NB_G0["fc_r2_pc3"]) < 5e-3, res["fc_r2"][2]
        assert abs(res["conf_r2"][0] - _NB_G0["conf_r2_pc1"]) < 5e-3, res["conf_r2"][0]
        assert abs(res["auc_sb"][2] - _NB_G0["auc_sb_pc3"]) < 5e-3, res["auc_sb"][2]
        print("  self-check OK: Glasser/seed0 matches the notebook (pair counts + FC R2 + confound + PC3 AUC).", flush=True)

    out_dir = OUT_PARTS / args.parc
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"f8_seed{args.seed}.npz"
    np.savez_compressed(out_path, **res)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
