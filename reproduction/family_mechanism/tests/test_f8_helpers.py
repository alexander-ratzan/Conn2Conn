"""Correctness tests for F8 pure helpers (no torch / no data)."""
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _f8_common as f8


def test_n_regions_from_edges():
    assert f8.n_regions_from_edges(64620) == 360
    assert f8.n_regions_from_edges(456 * 455 // 2) == 456
    try:
        f8.n_regions_from_edges(100)  # not triangular
        assert False, "should reject non-triangular"
    except AssertionError as e:
        assert "triangular" in str(e)


def test_per_pc_aucs_matches_direct():
    rng = np.random.default_rng(1)
    z = rng.normal(size=40)
    # pairs over 40 test subjects
    pairs = {"MZ": [(0, 1), (2, 3), (4, 5)], "DZ": [(6, 7)], "sibling": [(8, 9), (10, 11)],
             "unrelated_matched": [(12, 13), (14, 15), (16, 17), (18, 19)]}
    out = f8.per_pc_aucs(z, pairs)
    # direct recompute for MZ
    diff = np.abs(z[:, None] - z[None, :])
    sim = -diff
    mz = np.array([sim[i, j] for i, j in pairs["MZ"]])
    un = np.array([sim[i, j] for i, j in pairs["unrelated_matched"]])
    y = np.concatenate([np.ones(mz.size), np.zeros(un.size)])
    s = np.concatenate([mz, un])
    assert abs(out["MZ"] - roc_auc_score(y, s)) < 1e-12


def test_cosine_alignment_identity():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 100))
    C = f8.cosine_sim_matrix(A, A)
    # self-similarity diagonal == 1, matrix symmetric, off-diag <= 1
    assert np.allclose(np.diag(C), 1.0, atol=1e-6)
    assert np.allclose(C, C.T, atol=1e-6)
    assert C.max() <= 1.0 + 1e-6
    # best match of each row to itself is index i
    assert (np.argmax(np.abs(C), axis=1) == np.arange(5)).all()


def test_energy_top1pct_bounds():
    # all energy in 1 edge of 1000 -> ~1.0 ; uniform -> ~0.01
    v = np.zeros(1000); v[0] = 5.0
    assert f8.energy_top1pct(v) > 0.99
    u = np.ones(1000)
    assert abs(f8.energy_top1pct(u) - 0.01) < 0.005


def test_extract_pair_sims_empty_safe():
    M = np.arange(16).reshape(4, 4).astype(float)
    out = f8.extract_pair_sims(M, {"MZ": [(0, 1)], "DZ": []})
    assert out["MZ"][0] == M[0, 1]
    assert out["DZ"].size == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"PASS  {f.__name__}")
    print(f"\nAll {len(fns)} F8 helper tests passed.")
