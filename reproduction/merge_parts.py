#!/usr/bin/env python3
"""Merge per-unit part CSVs (one per parc x seed array task) into the grid-level CSVs.

  outputs/parts/recon_*.csv -> outputs/reconstruction.csv
  outputs/parts/down_*.csv  -> outputs/downstream.csv
  outputs/parts/leak_*.csv  -> outputs/leak_verdict.csv
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import OUTPUTS_DIR  # noqa: E402

PARTS = OUTPUTS_DIR / "parts"


def merge(glob, out):
    files = sorted(PARTS.glob(glob))
    if not files:
        print(f"[merge] WARNING no parts match {glob}")
        return 0
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[merge] {len(files)} parts ({glob}) -> {out}  [{len(df)} rows]")
    return len(df)


def main():
    merge("recon_*.csv", OUTPUTS_DIR / "reconstruction.csv")
    merge("down_*.csv", OUTPUTS_DIR / "downstream.csv")
    merge("leak_*.csv", OUTPUTS_DIR / "leak_verdict.csv")


if __name__ == "__main__":
    main()
