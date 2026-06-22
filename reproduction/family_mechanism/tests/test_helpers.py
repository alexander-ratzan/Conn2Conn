"""Correctness tests for the pure aggregation helpers (no torch / no data needed).

These pin the copied-from-notebook helpers against independent reference implementations and
basic statistical properties, so a future edit that silently breaks them gets caught.

    python reproduction/family_mechanism/tests/test_helpers.py
"""
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _fm_common as fm


def test_fdr_bh_matches_reference():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9, 0.6, 0.02])
    rej, padj = fm.fdr_bh(p, alpha=0.05)
    # reference: statsmodels if present, else textbook BH
    try:
        from statsmodels.stats.multitest import multipletests
        r_ref, p_ref, *_ = multipletests(p, alpha=0.05, method="fdr_bh")
    except Exception:
        n = len(p); order = np.argsort(p); ranked = p[order]
        adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1].clip(0, 1)
        p_ref = np.empty(n); p_ref[order] = adj; r_ref = p_ref <= 0.05
    assert np.allclose(padj, p_ref, atol=1e-12), (padj, p_ref)
    assert (rej == r_ref).all()
    # padj is monotone in p-rank and bounded
    assert padj.min() >= 0 and padj.max() <= 1


def test_auc_vs_unrelated_matches_sklearn():
    rng = np.random.default_rng(0)
    sims = {"MZ": rng.normal(1.0, 1, 40), "DZ": rng.normal(0.5, 1, 30),
            "sibling": rng.normal(0.2, 1, 60), "unrelated_matched": rng.normal(0.0, 1, 80)}
    out = fm.auc_vs_unrelated(sims)
    for rel in ("MZ", "DZ", "sibling"):
        y = np.concatenate([np.ones(sims[rel].size), np.zeros(sims["unrelated_matched"].size)])
        s = np.concatenate([sims[rel], sims["unrelated_matched"]])
        assert abs(out[rel] - roc_auc_score(y, s)) < 1e-12
    # separated distributions -> AUC ordering MZ > sibling
    assert out["MZ"] > out["sibling"]


def test_zscore_by_unrelated():
    sims = {"MZ": np.array([2.0, 3.0]), "unrelated_matched": np.array([0.0, 1.0, 2.0, 3.0])}
    z = fm.zscore_by_unrelated(sims)
    ref = sims["unrelated_matched"]
    mu, sd = ref.mean(), ref.std(ddof=1)
    assert np.allclose(z["unrelated_matched"], (ref - mu) / sd)
    assert np.allclose(z["MZ"], (sims["MZ"] - mu) / sd)
    # degenerate unrelated bucket (size < 2) -> short-circuits, returns input unchanged
    degen = {"unrelated_matched": np.array([5.0])}
    assert fm.zscore_by_unrelated(degen) is degen


def test_perm_and_bootstrap_determinism_and_range():
    rng = np.random.default_rng(0)
    sims = {"MZ": rng.normal(0.8, 1, 50), "DZ": rng.normal(0.4, 1, 40),
            "sibling": rng.normal(0.1, 1, 70), "unrelated_matched": rng.normal(0.0, 1, 90)}
    # determinism: same seed -> identical
    p1 = fm.perm_p_auc(sims, n_perm=500, rng=np.random.default_rng(42))
    p2 = fm.perm_p_auc(sims, n_perm=500, rng=np.random.default_rng(42))
    assert p1 == p2
    for rel in ("MZ", "DZ", "sibling"):
        assert 0.0 <= p1[rel] <= 1.0
    b1 = fm.bootstrap_auc(sims, n_boot=300, rng=np.random.default_rng(43))
    b2 = fm.bootstrap_auc(sims, n_boot=300, rng=np.random.default_rng(43))
    assert b1 == b2
    # CI brackets the point estimate
    aucs = fm.auc_vs_unrelated(sims)
    for rel in ("MZ", "DZ", "sibling"):
        lo, hi = b1[rel]
        assert lo <= aucs[rel] <= hi, (rel, lo, aucs[rel], hi)


def test_empty_buckets_safe():
    # missing/short buckets must not crash; return nan-ish
    sims = {"MZ": np.array([]), "DZ": np.array([1.0, 2.0]), "sibling": np.array([0.5]),
            "unrelated_matched": np.array([0.0, 1.0, 2.0])}
    out = fm.auc_vs_unrelated(sims)
    assert np.isnan(out["MZ"]) and np.isnan(out["sibling"])
    assert not np.isnan(out["DZ"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"PASS  {f.__name__}")
    print(f"\nAll {len(fns)} helper tests passed.")
