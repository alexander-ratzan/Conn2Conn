#!/usr/bin/env python3
"""B — variance-components decomposition (generalizability theory) on the 2x2
subject x session x direction FC design. Answers "how much of FC is trait / state /
within-session / noise" WITHOUT needing timeseries.

Design: subjects S (random) crossed with two facets, session T (2 levels) and direction
D (2 levels), single observation per cell. EMS variance components per edge:
  sigma2_S    = trait signal (stable individual differences)
  sigma2_ST   = subject x session = person-specific day-to-day STATE change
  sigma2_SD   = subject x direction = person-specific within-session (incl. distortion)
  sigma2_STD  = residual = MEASUREMENT NOISE (+ unmodeled)
  sigma2_T/D/TD = group-level facet effects (not individual-difference variance)

Individual-difference fractions (sum to 1):
  trait        = sigma2_S   / (S + ST + SD + STD)
  state        = sigma2_ST  / (...)
  within_sess  = sigma2_SD  / (...)
  noise        = sigma2_STD / (...)

Also the generalizability coefficient for the averaged (REST1+REST2, LR+RL) connectome:
  G = S / (S + ST/n_t + SD/n_d + STD/(n_t*n_d))   # reliability of what we actually use

Output: outputs/b_variance_decomposition.csv (per-parc aggregate + the per-edge npy).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noise_common import load_fc_cells, results_dir, PARCELLATIONS

N_T, N_D = 2, 2


def variance_components(cells):
    """cells (N,4,E) ordered [R1LR,R1RL,R2LR,R2RL] -> dict of (E,) component arrays."""
    N, _, E = cells.shape
    X = cells.reshape(N, N_T, N_D, E).astype(np.float64)  # [s, t, d, e]

    grand = X.mean(axis=(0, 1, 2))                 # (E,)
    m_s = X.mean(axis=(1, 2))                       # (N,E)
    m_t = X.mean(axis=(0, 2))                       # (T,E)
    m_d = X.mean(axis=(0, 1))                       # (D,E)
    m_st = X.mean(axis=2)                           # (N,T,E)
    m_sd = X.mean(axis=1)                           # (N,D,E)
    m_td = X.mean(axis=0)                           # (T,D,E)

    SS_S = N_T * N_D * ((m_s - grand) ** 2).sum(axis=0)
    SS_T = N * N_D * ((m_t - grand) ** 2).sum(axis=0)
    SS_D = N * N_T * ((m_d - grand) ** 2).sum(axis=0)
    SS_ST = N_D * ((m_st - m_s[:, None, :] - m_t[None, :, :] + grand) ** 2).sum(axis=(0, 1))
    SS_SD = N_T * ((m_sd - m_s[:, None, :] - m_d[None, :, :] + grand) ** 2).sum(axis=(0, 1))
    SS_TD = N * ((m_td - m_t[:, None, :] - m_d[None, :, :] + grand) ** 2).sum(axis=(0, 1))
    pred = (m_st[:, :, None, :] + m_sd[:, None, :, :] + m_td[None, :, :, :]
            - m_s[:, None, None, :] - m_t[None, :, None, :] - m_d[None, None, :, :]
            + grand)
    SS_STD = ((X - pred) ** 2).sum(axis=(0, 1, 2))

    df_S, df_T, df_D = N - 1, N_T - 1, N_D - 1
    df_ST, df_SD, df_TD = (N - 1) * (N_T - 1), (N - 1) * (N_D - 1), (N_T - 1) * (N_D - 1)
    df_STD = (N - 1) * (N_T - 1) * (N_D - 1)

    MS_S, MS_T, MS_D = SS_S / df_S, SS_T / df_T, SS_D / df_D
    MS_ST, MS_SD, MS_TD = SS_ST / df_ST, SS_SD / df_SD, SS_TD / df_TD
    MS_STD = SS_STD / df_STD

    v_STD = MS_STD
    v_ST = (MS_ST - MS_STD) / N_D
    v_SD = (MS_SD - MS_STD) / N_T
    v_TD = (MS_TD - MS_STD) / N
    v_S = (MS_S - MS_ST - MS_SD + MS_STD) / (N_T * N_D)
    v_T = (MS_T - MS_ST - MS_TD + MS_STD) / (N * N_D)
    v_D = (MS_D - MS_SD - MS_TD + MS_STD) / (N * N_T)

    clamp = lambda a: np.maximum(a, 0.0)
    return {k: clamp(v) for k, v in dict(
        S=v_S, T=v_T, D=v_D, ST=v_ST, SD=v_SD, TD=v_TD, STD=v_STD).items()}


rows = []
for parc in PARCELLATIONS:
    try:
        sids, cells = load_fc_cells(parc)
    except FileNotFoundError:
        print(f"[B] {parc}: no cell cache; skipping", flush=True)
        continue
    print(f"[B] {parc}: n={cells.shape[0]}, edges={cells.shape[2]} ...", flush=True)
    vc = variance_components(cells)
    # individual-difference denominator (subject-involving components)
    denom = vc["S"] + vc["ST"] + vc["SD"] + vc["STD"]
    denom = np.where(denom > 0, denom, np.nan)
    trait = vc["S"] / denom
    state = vc["ST"] / denom
    within = vc["SD"] / denom
    noise = vc["STD"] / denom
    G = vc["S"] / (vc["S"] + vc["ST"] / N_T + vc["SD"] / N_D + vc["STD"] / (N_T * N_D)
                   + 1e-20)

    # save per-edge for figures / downstream
    np.savez(results_dir() / f"b_variance_components_{parc}.npz",
             S=vc["S"], ST=vc["ST"], SD=vc["SD"], STD=vc["STD"],
             T=vc["T"], D=vc["D"], TD=vc["TD"],
             trait_frac=trait, state_frac=state, within_frac=within, noise_frac=noise, G=G)

    rows.append({
        "parc": parc, "n_edges": cells.shape[2],
        "trait_frac_mean": float(np.nanmean(trait)),
        "state_frac_mean": float(np.nanmean(state)),
        "within_sess_frac_mean": float(np.nanmean(within)),
        "noise_frac_mean": float(np.nanmean(noise)),
        "G_mean": float(np.nanmean(G)),
        "G_median": float(np.nanmedian(G)),
    })
    print(f"  trait={np.nanmean(trait):.3f} state={np.nanmean(state):.3f} "
          f"within={np.nanmean(within):.3f} noise={np.nanmean(noise):.3f} "
          f"G(avg-connectome)={np.nanmean(G):.3f}", flush=True)

df = pd.DataFrame(rows)
out = results_dir() / "b_variance_decomposition.csv"
df.to_csv(out, index=False)
print(f"\n[B] saved -> {out}\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n[B] fractions are of individual-difference (subject-involving) variance and sum to 1.")
