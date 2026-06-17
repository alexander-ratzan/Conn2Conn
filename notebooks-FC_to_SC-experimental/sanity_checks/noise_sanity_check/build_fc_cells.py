#!/usr/bin/env python3
"""PREP — build the 4-cell FC cache (the loader normally averages LR+RL away).

For each parcellation and each subject, read the four per-direction/run Pearson FC
relmat TSVs:
  task-rest_dir-{LR,RL}_run-{1,2}_space-fsLR_seg-{parc}_stat-pearsoncorrelation_relmat.tsv
keep only subjects with all 4 present, extract the upper triangle of each, and cache:
  /scratch/ans9868/noise_cache/fc_cells/parc-{parc}/{subject_ids.npy, cells.npy}
with cells shape (n_subj, 4, n_edges), order [R1LR, R1RL, R2LR, R2RL].

hemi='both' (all ROIs), so no atlas masking is needed — edges = upper triangle of the
full ROI×ROI matrix. Runs on Torch (TSVs live there). I/O-bound; parallelized.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noise_common import XCPD_DIR, NOISE_CACHE, PARCELLATIONS

DIRS = ("LR", "RL")
RUNS = (1, 2)  # run-1 = REST1, run-2 = REST2


def _tsv_path(subj_folder, d, run, parc):
    return (XCPD_DIR / subj_folder / "func" /
            f"{subj_folder}_task-rest_dir-{d}_run-{run}"
            f"_space-fsLR_seg-{parc}_stat-pearsoncorrelation_relmat.tsv")


def _load_subject_cells(args):
    """Return (subject_id:int, cells:(4, n_edges)) or (None, None) if any cell missing."""
    subj_folder, parc = args
    # order must match _noise_common: [R1LR, R1RL, R2LR, R2RL]
    order = [(1, "LR"), (1, "RL"), (2, "LR"), (2, "RL")]
    paths = [_tsv_path(subj_folder, d, run, parc) for (run, d) in order]
    if not all(p.exists() for p in paths):
        return None, None
    try:
        mats = [pd.read_csv(p, sep="\t", header=0, index_col=0).values.astype(np.float32)
                for p in paths]
    except Exception as e:
        print(f"[build] {subj_folder} {parc} read error: {e}", flush=True)
        return None, None
    n = mats[0].shape[0]
    iu, ju = np.triu_indices(n, k=1)
    cells = np.stack([m[iu, ju] for m in mats], axis=0)  # (4, n_edges)
    sid = int(subj_folder.replace("sub-", ""))
    return sid, cells.astype(np.float32)


def build_parc(parc):
    subj_folders = sorted(d.name for d in XCPD_DIR.iterdir() if d.name.startswith("sub-"))
    print(f"[build] {parc}: scanning {len(subj_folders)} subject folders ...", flush=True)
    args = [(sf, parc) for sf in subj_folders]
    sids, rows = [], []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for sid, cells in ex.map(_load_subject_cells, args, chunksize=8):
            if sid is not None:
                sids.append(sid); rows.append(cells)
    if not rows:
        print(f"[build] {parc}: NO subjects with all 4 cells — check paths!", flush=True)
        return
    order = np.argsort(sids)
    sids = np.asarray(sids)[order]
    cells = np.stack(rows, axis=0)[order]  # (n_subj, 4, n_edges)
    out = NOISE_CACHE / f"parc-{parc}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "subject_ids.npy", sids)
    np.save(out / "cells.npy", cells)
    print(f"[build] {parc}: cached {cells.shape[0]} subjects x 4 cells x "
          f"{cells.shape[2]} edges -> {out}", flush=True)


if __name__ == "__main__":
    for parc in PARCELLATIONS:
        build_parc(parc)
    print("[build] done.", flush=True)
