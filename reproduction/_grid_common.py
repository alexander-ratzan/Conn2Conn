"""Shared helpers for the Conn2Conn reproducibility grid.

Builds on `notebooks-FC_to_SC-experimental/further_exploration/_setup.py` (the project's
single source of truth for data loading + closed-form predictors + metric panel) so every
grid number matches the rest of the project exactly.

Core responsibilities:
  - locate + import _setup (walk to Conn2Conn root, move-robust)
  - set_parcellation(parc)            -> switch Glasser <-> 4S456Parcels
  - capped_pca_pls / estimator dispatch with the LOW-DIM CAP baked in (BP/low-dim lesson)
  - subject_ids_for(base, idx)        -> canonical subject IDs for a partition
  - load_split_checked(seed, parc)    -> load data via _setup AND assert it matches the
                                         FROZEN split json (BP-2: frozen is the source of truth)
  - flat_metric_keys(...)             -> FLAT W&B keys metrics/{task}/{input}/{target}/{metric}
  - append_csv(path, row)             -> CSV mirror (the source of truth)
  - git_commit(), config_hash(dims)   -> provenance on every row/artifact
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import hashlib
import subprocess
import numpy as np
import pandas as pd

# --- locate Conn2Conn root + further_exploration/_setup.py ------------------
_here = Path(__file__).resolve()
REPRO_ROOT = _here.parent                       # .../Conn2Conn/reproduction
_root = _here
while _root.name != "Conn2Conn" and _root.parent != _root:
    _root = _root.parent
CONN2CONN_ROOT = _root
_FE = _root / "notebooks-FC_to_SC-experimental" / "further_exploration"
if not (_FE / "_setup.py").exists():
    raise RuntimeError(f"could not locate further_exploration/_setup.py from {_here}")
sys.path.insert(0, str(_FE))
import _setup                                    # noqa: E402
from _setup import (                             # noqa: E402
    load_seed_split, pca_pls_predict, br_per_component_predict, full_panel_eval,
)

# --- canonical grid paths (project-root reproduction/) ----------------------
SPLITS_DIR = REPRO_ROOT / "splits"
OUTPUTS_DIR = REPRO_ROOT / "outputs"
CONFIGS_DIR = REPRO_ROOT / "configs"
for _p in (SPLITS_DIR, OUTPUTS_DIR, CONFIGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

PARCELLATIONS = ["Glasser", "4S456Parcels"]
WANDB_PROJECT = "conn2conn-fc-to-sc-reproduction"
PANEL_METRICS = ["mse", "r2", "pearson", "demeaned_pearson", "top1_acc", "avg_rank"]


# ============================================================================
# parcellation switch (load_seed_split reads _setup.PARCELLATION at call time)
# ============================================================================
def set_parcellation(parc: str) -> None:
    assert parc in PARCELLATIONS, f"unknown parcellation {parc!r}"
    _setup.PARCELLATION = parc


# ============================================================================
# subject IDs (canonical order from metadata_df)
# ============================================================================
def subject_ids_for(base, idx) -> list[int]:
    sids = np.asarray(base.metadata_df["subject"]).astype(np.int64)
    return [int(s) for s in sids[idx]]


# ============================================================================
# FROZEN SPLIT contract (BP-2): produced via _setup, asserted == frozen json
# ============================================================================
def split_json_path(seed: int) -> Path:
    return SPLITS_DIR / f"seed{seed}.json"


def load_split_checked(seed: int, parc: str) -> dict:
    """Load the seed split for `parc` via _setup and HARD-ASSERT its train/val/test
    subject IDs match the frozen splits/seed{seed}.json (ordered). The frozen file is the
    source of truth; _setup must agree or we fail loudly (no silent drift / misalignment).
    Returns the _setup split dict augmented with train_ids/val_ids/test_ids.
    """
    set_parcellation(parc)
    sp = load_seed_split(seed=seed)
    base = sp["base"]
    part = base.trainvaltest_partition_indices
    train_ids = subject_ids_for(base, part["train"])
    val_ids = subject_ids_for(base, part["val"])
    test_ids = subject_ids_for(base, part["test"])

    fp = split_json_path(seed)
    if not fp.exists():
        raise FileNotFoundError(
            f"frozen split missing: {fp} — run freeze_splits.py before any grid runner (BP-2)")
    frozen = json.loads(fp.read_text())
    for name, got in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        exp = [int(x) for x in frozen[f"{name}_ids"]]
        if got != exp:
            # ordered mismatch -> check set, report precisely
            if set(got) == set(exp):
                raise AssertionError(
                    f"seed{seed} {parc}: {name} subject SET matches frozen but ORDER differs "
                    f"— alignment unsafe; investigate metadata_df ordering")
            raise AssertionError(
                f"seed{seed} {parc}: {name} subjects DIFFER from frozen split "
                f"(got n={len(got)}, frozen n={len(exp)}, symdiff={len(set(got)^set(exp))}). "
                f"_setup split drifted from the frozen contract.")
    sp["train_ids"], sp["val_ids"], sp["test_ids"] = train_ids, val_ids, test_ids
    return sp


# ============================================================================
# input / target builders
# ============================================================================
def build_input(sp: dict, name: str):
    """Return (X_train, X_test) for an input set name. (connectome+bv+demo handled
    separately with per-block scaling — BP-1 — not here.)"""
    m = {
        "bv":      ("bv_train", "bv_test"),
        "demo":    ("demo_train", "demo_test"),
        "bv+demo": ("bvdemo_train", "bvdemo_test"),
        "FC":      ("FC_train", "FC_test"),
        "SC":      ("SC_train", "SC_test"),
    }
    if name not in m:
        raise ValueError(f"unknown input set {name!r} (use BP-1 path for connectome+bv+demo)")
    a, b = m[name]
    return np.asarray(sp[a], np.float32), np.asarray(sp[b], np.float32)


def build_target(sp: dict, name: str):
    m = {"SC": ("SC_train", "SC_test"), "FC": ("FC_train", "FC_test")}
    if name not in m:
        raise ValueError(f"unknown target {name!r}")
    a, b = m[name]
    return np.asarray(sp[a], np.float32), np.asarray(sp[b], np.float32)


# ============================================================================
# estimators — LOW-DIM CAP baked in (k_src = min(256, width), k_pls = min(64, k_src))
# ============================================================================
def capped_pca_pls(X_tr, X_te, Y_tr):
    k_src = min(256, X_tr.shape[1])
    k_tgt = min(256, Y_tr.shape[1])
    k_pls = min(64, k_src, k_tgt)
    return pca_pls_predict(X_tr, X_te, Y_tr, k_src=k_src, k_tgt=k_tgt, k_pls=k_pls)


def capped_bayesian_ridge(X_tr, X_te, Y_tr):
    k_src = min(256, X_tr.shape[1])
    k_tgt = min(256, Y_tr.shape[1])
    return br_per_component_predict(X_tr, X_te, Y_tr, k_src=k_src, k_tgt=k_tgt)


ESTIMATORS = {
    "pca_pls": capped_pca_pls,
    "bayesian_ridge": capped_bayesian_ridge,
    # "kernel_ridge": <Phase B — RBF, 3x3 bandwidth x alpha>
}


# ============================================================================
# provenance + logging helpers
# ============================================================================
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(CONN2CONN_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def config_hash(dims: dict) -> str:
    return hashlib.md5(json.dumps(dims, sort_keys=True).encode()).hexdigest()[:8]


def flat_metric_keys(task: str, input_set: str, target: str, panel: dict) -> dict:
    """FLAT W&B keys: metrics/{task}/{input}/{target}/{metric} (no nested objects — W&B
    drops/flattens nested keys inconsistently; flat = full table visible)."""
    return {f"metrics/{task}/{input_set}/{target}/{m}": float(panel[m])
            for m in PANEL_METRICS if m in panel}


def append_csv(path: Path, row: dict) -> None:
    """Append one row to a CSV (the source of truth), creating header on first write.
    Union-of-columns safe: re-reads + concatenates so column sets can grow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
