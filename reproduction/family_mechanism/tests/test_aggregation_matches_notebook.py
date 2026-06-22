"""Validate the F6/F7 aggregation against the NOTEBOOK's real saved output.

The notebook (STEP 8.2) saved per-seed per-pair similarities to seed_*.npz, and (STEP 8.3)
saved the aggregated AUC table to aggregate_auc.csv. Those expensive per-pair sims required
the connectome data; the aggregation does NOT. So we feed the notebook's OWN seed_*.npz into
OUR aggregate_family() and assert we reproduce aggregate_auc.csv exactly (same RNG seeds:
perm default_rng(42), boot default_rng(43)).

This runs locally (no torch / no connectome data). It is the strongest "matches the notebook"
check available off-cluster: same inputs, our aggregation, must equal the notebook's numbers.

    python -m pytest reproduction/family_mechanism/tests/test_aggregation_matches_notebook.py -q
    # or just:  python reproduction/family_mechanism/tests/test_aggregation_matches_notebook.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))            # family_mechanism/
import _fm_common as fm                                  # noqa: E402

_ROOT = _HERE.parents[3]                                 # Conn2Conn/
FAM_DIR = _ROOT / "notebooks-FC_to_SC-experimental/model_overviews/results/family_structure_phase2"


def _load_notebook_pooled():
    """Pool the notebook's seed_*.npz exactly as STEP 8.3 step 1."""
    seed_files = sorted(FAM_DIR.glob("seed_*.npz"))
    assert seed_files, f"no seed_*.npz under {FAM_DIR}"
    pooled = {v: {r: [] for r in fm.RELATIONS} for v in fm.FAM_VARIANTS}
    for f in seed_files:
        data = np.load(f)
        for v in fm.FAM_VARIANTS:
            for r in fm.RELATIONS:
                key = f"{v}__{r}"
                if key in data.files:
                    pooled[v][r].append(data[key])
    pooled = {v: {r: (np.concatenate(lst) if lst else np.array([]))
                  for r, lst in by.items()} for v, by in pooled.items()}
    return pooled, len(seed_files)


def test_aggregation_matches_aggregate_auc_csv():
    ref = pd.read_csv(FAM_DIR / "aggregate_auc.csv")
    pooled, n_seeds = _load_notebook_pooled()
    got = fm.aggregate_family(pooled, n_perm=10_000, n_boot=1000)

    # align on (variant, relation)
    key = ["variant", "relation"]
    m = ref.merge(got, on=key, suffixes=("_ref", "_got"))
    assert len(m) == len(ref) == len(got), "row sets differ"

    # n_pairs must be EXACT (pure pooling)
    assert (m.n_pairs_ref == m.n_pairs_got).all(), "n_pairs mismatch"

    # AUC is deterministic given pooled sims -> must match to ~1e-6
    max_auc_err = float((m.auc_ref - m.auc_got).abs().max())
    assert max_auc_err < 1e-6, f"AUC mismatch up to {max_auc_err}"

    # Bootstrap CI: same rng(43) + same data -> should match very tightly
    max_lo = float((m.auc_lo_ref - m.auc_lo_got).abs().max())
    max_hi = float((m.auc_hi_ref - m.auc_hi_got).abs().max())
    assert max_lo < 1e-4 and max_hi < 1e-4, f"bootstrap CI drift lo={max_lo} hi={max_hi}"

    # Permutation p: same rng(42) -> should match exactly (deterministic shuffles)
    max_p = float((m.p_perm_ref - m.p_perm_got).abs().max())
    assert max_p < 1e-9, f"perm-p mismatch up to {max_p}"

    # FDR + significance flags must match
    assert (m.p_fdr_ref.round(6) == m.p_fdr_got.round(6)).all(), "p_fdr mismatch"
    assert (m.sig_fdr_ref.astype(bool) == m.sig_fdr_got.astype(bool)).all(), "sig_fdr mismatch"

    return dict(n_seeds=n_seeds, max_auc_err=max_auc_err, max_ci_err=max(max_lo, max_hi),
                max_p_err=max_p, headline={
                    "pred_SC_resid_bvdemo/sibling": float(
                        got.query("variant=='pred_SC_resid_bvdemo' and relation=='sibling'").auc.iloc[0]),
                    "combined_pred_SC/sibling": float(
                        got.query("variant=='combined_pred_SC' and relation=='sibling'").auc.iloc[0]),
                })


if __name__ == "__main__":
    info = test_aggregation_matches_aggregate_auc_csv()
    print("PASS  aggregation reproduces the notebook's aggregate_auc.csv")
    print(f"  seeds pooled         : {info['n_seeds']}")
    print(f"  max |AUC error|      : {info['max_auc_err']:.2e}")
    print(f"  max |bootstrap CI err|: {info['max_ci_err']:.2e}")
    print(f"  max |perm-p error|   : {info['max_p_err']:.2e}")
    print(f"  F6 headline pred_SC_resid_bvdemo sibling AUC = {info['headline']['pred_SC_resid_bvdemo/sibling']:.6f}  (notebook 0.809673)")
    print(f"  F7 headline combined_pred_SC      sibling AUC = {info['headline']['combined_pred_SC/sibling']:.6f}  (notebook 0.504936)")
