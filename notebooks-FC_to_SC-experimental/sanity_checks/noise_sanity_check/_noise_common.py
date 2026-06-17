"""Shared helpers for the FC noise sanity-check suite.

Data model: per subject we have 4 FC cells in a 2x2 design (HCP-YA, both parcellations):
  cells[:, 0] = REST1-LR   (session 1, dir LR)
  cells[:, 1] = REST1-RL   (session 1, dir RL)
  cells[:, 2] = REST2-LR   (session 2, dir LR)
  cells[:, 3] = REST2-RL   (session 2, dir RL)
Each cell is the upper-triangle of a Pearson FC matrix. Built by build_fc_cells.py.

Reuses full_panel_eval / pca_pls_predict from further_exploration/_setup.py so metrics
match the rest of the project exactly.
"""
from pathlib import Path
import sys
import numpy as np

# --- locate further_exploration/_setup.py (walk to Conn2Conn root, robust to moves) ---
_here = Path(__file__).resolve()
_root = _here
while _root.name != "Conn2Conn" and _root.parent != _root:
    _root = _root.parent
_FE = _root / "notebooks-FC_to_SC-experimental" / "further_exploration"
if (_FE / "_setup.py").exists():
    sys.path.insert(0, str(_FE))
else:
    raise RuntimeError(f"could not locate further_exploration/_setup.py from {_here}")
from _setup import full_panel_eval, pca_pls_predict, load_seed_split  # noqa: E402

# --- where the 4-cell FC cache lives (ans9868 scratch; built by build_fc_cells.py) ---
NOISE_CACHE = Path("/scratch/ans9868/noise_cache/fc_cells")
PARCELLATIONS = ["Glasser", "4S456Parcels"]

# HCP raw FC source (per-direction Pearson relmat TSVs)
HCP_DIR = Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200")
XCPD_DIR = HCP_DIR / "HCP1200_fMRI" / "xcpd-0-9-1"

# cell index map
IDX_R1LR, IDX_R1RL, IDX_R2LR, IDX_R2RL = 0, 1, 2, 3


def cell_dir(parc):
    return NOISE_CACHE / f"parc-{parc}"


def load_fc_cells(parc):
    """Return (subject_ids: list[int], cells: (n_subj, 4, n_edges) float32)."""
    d = cell_dir(parc)
    sids = np.load(d / "subject_ids.npy")
    cells = np.load(d / "cells.npy")
    return sids.tolist(), cells.astype(np.float32)


def session_means(cells):
    """Return (rest1, rest2) each (n_subj, n_edges) = LR/RL average within session."""
    rest1 = 0.5 * (cells[:, IDX_R1LR] + cells[:, IDX_R1RL])
    rest2 = 0.5 * (cells[:, IDX_R2LR] + cells[:, IDX_R2RL])
    return rest1.astype(np.float32), rest2.astype(np.float32)


def group_mean(X):
    """Across-subject mean vector for demeaning (the full_panel demean reference)."""
    return X.mean(axis=0).astype(np.float32)


def results_dir():
    p = Path(__file__).resolve().parent / "outputs"
    p.mkdir(exist_ok=True)
    return p
