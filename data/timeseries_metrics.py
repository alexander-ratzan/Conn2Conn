"""Per-region timeseries metrics: autocorrelation, intrinsic neural timescale, (multiscale) sample entropy.

Every function takes x (T, P) (time x regions) and returns one value per region. The `*_runs` variants take the
subject's run bounds ((n_runs, 2) [start, stop) rows, `TimeseriesCache.run_bounds`) and never span a run boundary.

- Intrinsic timescale (Parkes et al., "Task-evoked flexibility of intrinsic neural timescales ..."; Watanabe et al.
  2019): area under the autocorrelation function from lag 0 up to (not including) the first lag at which it is <= 0,
  in seconds: TR * sum_{k < k0} acf(k). Per run, then averaged over runs (Parkes average over scans).
- Sample entropy (Richman & Moorman 2000): SampEn = -log(A / B), where B counts pairs i < j of the N - m length-m
  templates (starts 0 .. N - m - 1) within Chebyshev distance d < r, and A counts the same pairs at length m + 1.
  Multiscale entropy (Costa et al. 2005): coarse-grain by non-overlapping means of s samples, keep r fixed at
  r_frac x SD of the scale-1 signal (SD per run). Across runs A and B are summed before the log. Undefined (A or B = 0)
  -> NaN (the LOFT toolbox returns 0 there). MSE summary = area under the SampEn-vs-scale curve (trapezoid, unit
  scale spacing; NaN if any scale is undefined).
  Defaults follow Wijesinghe et al. (NeuroImage 2026, "Lifespan trajectories of the brain's functional complexity
  characterized by multiscale sample entropy"): m = 2, r = 0.3 x SD, scales 1-13, AUC over scales; strict d < r and
  r fixed across scales as in their LOFT toolbox (github.com/kayjann/complexity, sample_entropy.m). Their TR is 1.4 s
  (scale 13 = 18 s) on high-pass-only data; with HCP's TR 0.72 s scale 13 spans 9.4 s, and the band-pass removes
  everything above 0.08 Hz, i.e. all the detail finer than scale ~9.

Caveats for the HCP xcp-d data: the timeseries are band-pass filtered (~0.01-0.08 Hz), which shapes the ACF and the
coarse-scale entropy, and the task design is not regressed out (no event timings on disk; Parkes do regress it).
Compare across regions, subjects and conditions within this dataset, not to absolute values in the literature.
"""
import numpy as np

TR = 0.72
SCALES = tuple(range(1, 14))
M, R_FRAC = 2, 0.3


def acf(x, max_lag=None):
    """Biased sample autocorrelation of each column of x (T, P), lags 0..max_lag (default T - 1). Returns (L, P)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    max_lag = T - 1 if max_lag is None else min(int(max_lag), T - 1)
    x = x - x.mean(0)
    nfft = 1 << int(np.ceil(np.log2(2 * T - 1)))
    f = np.fft.rfft(x, n=nfft, axis=0)
    ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:max_lag + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return ac / ac[:1]


def timescale(x, tr=TR):
    """Intrinsic timescale (s) of each column of x (T, P): TR x sum of the ACF before its first value <= 0."""
    r = acf(x)
    nonpos = r <= 0
    k0 = np.where(nonpos.any(0), nonpos.argmax(0), r.shape[0])
    keep = np.arange(r.shape[0])[:, None] < k0[None, :]
    out = tr * np.where(keep, r, 0.0).sum(0)
    out[~np.isfinite(r[0])] = np.nan
    return out


def timescale_runs(x, bounds, tr=TR):
    """Mean over runs of the per-run timescale. Returns (P,)."""
    return np.mean([timescale(x[a:b], tr) for a, b in bounds], axis=0)


def coarse_grain(x, scale):
    """Non-overlapping means of `scale` consecutive samples (x: (T, P)); a trailing remainder is dropped."""
    n = x.shape[0] // scale
    return np.asarray(x[:n * scale], dtype=np.float64).reshape(n, scale, -1).mean(1)


def sampen_counts(x, m, r, chunk=16):
    """Template-match counts for sample entropy of each column of x (T, P) with tolerance r (P,).

    Returns (A, B), float64 (P,): pairs i < j within Chebyshev distance < r at length m + 1 (A) and m (B), over the
    T - m template starts shared by both lengths.
    """
    x = np.asarray(x, dtype=np.float64)
    T, P = x.shape
    N = T - m
    A, B = np.zeros(P), np.zeros(P)
    if N < 2:
        return A, B
    upper = np.triu(np.ones((N, N), dtype=bool), k=1)[:, :, None]
    r = np.broadcast_to(np.asarray(r, dtype=np.float64), (P,))
    for c in range(0, P, chunk):
        xc = x[:, c:c + chunk]
        D = np.abs(xc[:N, None, :] - xc[None, :N, :])
        for k in range(1, m):
            np.maximum(D, np.abs(xc[k:k + N, None, :] - xc[None, k:k + N, :]), out=D)
        rc = r[c:c + chunk]
        B[c:c + chunk] = ((D < rc) & upper).sum((0, 1))
        np.maximum(D, np.abs(xc[m:m + N, None, :] - xc[None, m:m + N, :]), out=D)
        A[c:c + chunk] = ((D < rc) & upper).sum((0, 1))
    return A, B


def _entropy(A, B):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((A > 0) & (B > 0), -np.log(A / B), np.nan)


def sample_entropy(x, m=M, r_frac=R_FRAC):
    """Sample entropy of each column of x (T, P), r = r_frac x column SD. Returns (P,)."""
    x = np.asarray(x, dtype=np.float64)
    return _entropy(*sampen_counts(x, m, r_frac * x.std(0)))


def multiscale_entropy_runs(x, bounds, scales=SCALES, m=M, r_frac=R_FRAC):
    """Multiscale sample entropy pooled over runs: per run, coarse-grain and count with r = r_frac x the run's SD;
    sum the counts over runs; SampEn per scale. Returns (len(scales), P)."""
    x = np.asarray(x, dtype=np.float64)
    out = []
    for s in scales:
        A = B = 0.0
        for a, b in bounds:
            run = x[a:b]
            ca, cb = sampen_counts(coarse_grain(run, s), m, r_frac * run.std(0))
            A, B = A + ca, B + cb
        out.append(_entropy(A, B))
    return np.stack(out)


def multiscale_entropy(x, scales=SCALES, m=M, r_frac=R_FRAC):
    """Multiscale sample entropy of one contiguous segment (one run or one window). Returns (len(scales), P)."""
    return multiscale_entropy_runs(x, [(0, len(x))], scales, m, r_frac)


def mse_auc(mse):
    """Area under SampEn vs scale (trapezoid, unit spacing) from (n_scales, P); NaN where any scale is undefined."""
    mse = np.asarray(mse, dtype=np.float64)
    return 0.5 * (mse[:-1] + mse[1:]).sum(0)
