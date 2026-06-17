#!/usr/bin/env python3
"""F — multivariate reliability scalars from the two FC sessions (REST1 vs REST2):
discriminability and fingerprinting, as whole-connectome reliability summaries that
complement the per-edge view in B.

  fingerprint_top1 : fraction of subjects whose REST1 best-matches their own REST2
                     (by demeaned cosine) among all subjects = identifiability.
  discriminability : P(same-subject cross-session distance < different-subject distance),
                     averaged — Bridgeford et al. style scalar (0.5=chance, 1=perfect).

Output: outputs/f_discriminability.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noise_common import load_fc_cells, session_means, group_mean, results_dir, PARCELLATIONS


def _demean_unit(X, mu):
    Z = X - mu
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-12)


def fingerprint_and_discriminability(rest1, rest2):
    mu = group_mean(np.concatenate([rest1, rest2], axis=0))
    Z1 = _demean_unit(rest1, mu)
    Z2 = _demean_unit(rest2, mu)
    S = Z1 @ Z2.T                      # (n,n) cosine similarity, rows=REST1, cols=REST2
    n = S.shape[0]
    # fingerprint: is the diagonal the argmax of its row?
    top1 = float(np.mean(np.argmax(S, axis=1) == np.arange(n)))
    # discriminability via similarity: for each i, fraction of j!=i with S[i,i] > S[i,j]
    same = np.diag(S)[:, None]                 # (n,1)
    off = S.copy()
    np.fill_diagonal(off, -np.inf)
    # count off-diagonal entries strictly less than the same-subject similarity
    less = (off < same).sum(axis=1)            # per row
    disc = float(np.mean(less / (n - 1)))
    return top1, disc


rows = []
for parc in PARCELLATIONS:
    try:
        sids, cells = load_fc_cells(parc)
    except FileNotFoundError:
        print(f"[F] {parc}: no cell cache; skipping", flush=True)
        continue
    rest1, rest2 = session_means(cells)
    top1, disc = fingerprint_and_discriminability(rest1, rest2)
    rows.append({"parc": parc, "n": cells.shape[0],
                 "fingerprint_top1": top1, "discriminability": disc})
    print(f"[F] {parc}: fingerprint_top1={top1:.4f} discriminability={disc:.4f}", flush=True)

df = pd.DataFrame(rows)
out = results_dir() / "f_discriminability.csv"
df.to_csv(out, index=False)
print(f"\n[F] saved -> {out}\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n[F] high top1/discriminability + modest per-edge reliability (B) = the distributed-")
print("[F] signal reconciliation: connectomes match in aggregate even when edges are noisy.")
