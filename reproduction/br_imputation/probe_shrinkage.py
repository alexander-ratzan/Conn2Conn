#!/usr/bin/env python3
"""Mechanism probe: is BR's worse identity / better reconstruction explained by SHRINKAGE
toward the group mean in low-variance target PCs?

For each Glasser seed, recompute pred_SC with BOTH estimators from the SAME split (apples-to-
apples), then measure:
  (1) per-subject demeaned norm ||pred - mu|| vs ||true - mu||  -> does BR sit closer to the mean?
  (2) per-target-PC recovery: corr(pred_PC_k, true_PC_k) across subjects, and amplitude ratio
      std(pred_PC_k)/std(true_PC_k), binned top->tail -> does BR win top PCs and shrink the tail?

Prediction (objective-function story): BR (evidence-tuned per-component ridge, shrinks low-SNR
modes to the mean) should have SMALLER demeaned norm and LOWER tail-PC amplitude/corr than PLS
(max cross-covariance, no per-component shrinkage). The heritable fingerprint lives in the tail.

    python probe_shrinkage.py            # Glasser, seeds 0-9
"""
from pathlib import Path
import sys
import numpy as np

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO))
from _grid_common import load_split_checked, capped_bayesian_ridge, capped_pca_pls  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "probe_shrinkage.csv"
SEEDS = list(range(10))
BINS = [(0, 10), (10, 50), (50, 128), (128, 256)]   # PC index ranges (0-based, top->tail)


def colcorr(A, B):
    """Per-column Pearson corr between (n,k) A and B."""
    Az = A - A.mean(0); Bz = B - B.mean(0)
    num = (Az * Bz).sum(0)
    den = np.sqrt((Az**2).sum(0) * (Bz**2).sum(0)) + 1e-12
    return num / den


def main():
    rows = []
    norm_summary = []
    perpc = {b: {"corr_BR": [], "corr_PLS": [], "amp_BR": [], "amp_PLS": [], "var": []} for b in BINS}
    for seed in SEEDS:
        sp = load_split_checked(seed=seed, parc="Glasser")
        FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
        SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
        mu = SC_tr.mean(0)
        pBR = capped_bayesian_ridge(FC_tr, FC_te, SC_tr)
        pPLS = capped_pca_pls(FC_tr, FC_te, SC_tr)

        # (1) demeaned norms
        nT = np.linalg.norm(SC_te - mu, axis=1).mean()
        nBR = np.linalg.norm(pBR - mu, axis=1).mean()
        nPLS = np.linalg.norm(pPLS - mu, axis=1).mean()
        norm_summary.append((seed, nT, nBR, nPLS, nBR / nT, nPLS / nT))

        # (2) project demeaned onto target PCA (fit on SC_tr, the estimators' own basis)
        pca = PCA(n_components=256, random_state=0).fit(SC_tr)
        U = pca.components_                       # (256, n_edges)
        var = pca.explained_variance_ratio_       # (256,)
        T = (SC_te - mu) @ U.T                     # (n_te, 256) true PC scores
        B = (pBR - mu) @ U.T
        P = (pPLS - mu) @ U.T
        cB, cP = colcorr(B, T), colcorr(P, T)      # per-PC recovery corr
        aB = B.std(0) / (T.std(0) + 1e-12)         # per-PC amplitude ratio (1 = no shrink)
        aP = P.std(0) / (T.std(0) + 1e-12)
        for (lo, hi) in BINS:
            perpc[(lo, hi)]["corr_BR"].append(np.nanmean(cB[lo:hi]))
            perpc[(lo, hi)]["corr_PLS"].append(np.nanmean(cP[lo:hi]))
            perpc[(lo, hi)]["amp_BR"].append(np.nanmean(aB[lo:hi]))
            perpc[(lo, hi)]["amp_PLS"].append(np.nanmean(aP[lo:hi]))
            perpc[(lo, hi)]["var"].append(float(var[lo:hi].sum()))
        for k in range(256):
            rows.append({"seed": seed, "pc": k, "var_ratio": float(var[k]),
                         "corr_BR": float(cB[k]), "corr_PLS": float(cP[k]),
                         "amp_BR": float(aB[k]), "amp_PLS": float(aP[k])})
        print(f"[probe] seed{seed}: demeaned-norm true={nT:.4f} BR={nBR:.4f} ({nBR/nT:.3f}x) "
              f"PLS={nPLS:.4f} ({nPLS/nT:.3f}x)", flush=True)

    # ---- summary ----
    ns = np.array([r[1:] for r in norm_summary])  # nT,nBR,nPLS,rBR,rPLS
    print("\n=== (1) DEMEANED NORM ||pred - mu|| / ||true - mu|| (mean over 10 seeds) ===")
    print(f"  BR  retains {ns[:,3].mean():.3f}x of true individual-deviation amplitude")
    print(f"  PLS retains {ns[:,4].mean():.3f}x")
    print(f"  -> BR sits {'CLOSER to' if ns[:,3].mean()<ns[:,4].mean() else 'farther from'} "
          f"the group mean (smaller = more shrinkage)")

    print("\n=== (2) PER-TARGET-PC RECOVERY, binned top->tail (mean over 10 seeds) ===")
    print(f"  {'PC bin':10s}{'cum.var%':>9s}{'corr_BR':>9s}{'corr_PLS':>9s}{'amp_BR':>8s}{'amp_PLS':>8s}")
    for (lo, hi) in BINS:
        d = perpc[(lo, hi)]
        print(f"  {f'{lo+1}-{hi}':10s}{100*np.mean(d['var']):>8.1f}%"
              f"{np.mean(d['corr_BR']):>9.3f}{np.mean(d['corr_PLS']):>9.3f}"
              f"{np.mean(d['amp_BR']):>8.3f}{np.mean(d['amp_PLS']):>8.3f}")
    print("\n  amp = std(pred PC)/std(true PC); <1 = shrunk toward mean. "
          "Prediction: BR amp < PLS amp in the tail bins (low-variance heritable modes).")

    import csv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n[probe] wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
