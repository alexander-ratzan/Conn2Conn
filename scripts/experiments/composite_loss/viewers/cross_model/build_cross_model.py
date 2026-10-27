"""Composite-loss cross-model viewer (spec v2 E1.10 / E3): test demeaned r vs average rank for every instance's loss
combinations, per direction on fixed axes, with model toggles, the test-retest ceiling and a hover / click panel (weights,
block, mean ± SE, paired Δ vs the model's MSE-only, per-seed values). Replaces the per-instance tradeoff_interactive.html
pages (same points, one page).

Data: ../../compare.py `viewer_data()` (every instance's tables/seed_records.csv, ceiling/*.json). Page:
cross_model.template on the shared viewer_kit. Output, next to this file (generated, not tracked): cross_model.html.

    python scripts/experiments/composite_loss/viewers/cross_model/build_cross_model.py [--smoke]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path[:0] = [str(REPO_ROOT), str(EXP)]
import compare  # noqa: E402
from scripts.data_viewers.viewer_kit import Viewer  # noqa: E402


def main():
    ap = Viewer.add_args(argparse.ArgumentParser())
    a = ap.parse_args()
    viewer = Viewer(__file__)
    if a.from_cache:
        viewer.write(*viewer.load_cache(), cache=False, smoke=a.smoke)
        return 0
    got = compare.viewer_data()
    if got is None:
        print("no instance tables yet")
        return 1
    viewer.write(*got, smoke=a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
