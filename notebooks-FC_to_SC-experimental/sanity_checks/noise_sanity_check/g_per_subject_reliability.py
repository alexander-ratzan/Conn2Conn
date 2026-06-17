#!/usr/bin/env python3
"""G — per-subject FC reliability distribution.

The 0.49 ceiling is a MEAN over subjects. This asks: how is per-subject reliability
distributed? Normal or skewed? Are some subjects systematically easier (more reliable)
than others, and is that a stable subject property?

Per subject i, demeaned cosine between their two session connectomes:
  rel_i = cos(REST1_i - mu, REST2_i - mu)        # between-session
  rel_within_i = cos(R1LR_i - mu, R1RL_i - mu)   # within-session (run 1)
mu = group-mean connectome. Then: distribution stats, normality test, and the
within-vs-between per-subject correlation (is reliability a stable trait?).

Outputs: outputs/g_per_subject_reliability.csv (per subject), g_per_subject_summary.csv,
g_reliability_hist.png
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noise_common import (load_fc_cells, session_means, group_mean, results_dir,
                           PARCELLATIONS, IDX_R1LR, IDX_R1RL)


def per_subject_cosine(A, B, mu):
    """cos(A_i - mu, B_i - mu) per subject -> (n,)."""
    da = A - mu
    db = B - mu
    num = (da * db).sum(axis=1)
    den = np.sqrt((da ** 2).sum(axis=1)) * np.sqrt((db ** 2).sum(axis=1))
    return num / np.maximum(den, 1e-12)


summ_rows = []
fig, axes = plt.subplots(1, len(PARCELLATIONS), figsize=(11, 4.2), dpi=160)
if len(PARCELLATIONS) == 1:
    axes = [axes]

for ax, parc in zip(axes, PARCELLATIONS):
    try:
        sids, cells = load_fc_cells(parc)
    except FileNotFoundError:
        print(f"[G] {parc}: no cache; skipping", flush=True)
        continue
    rest1, rest2 = session_means(cells)
    mu = group_mean(np.concatenate([rest1, rest2], axis=0))
    rel_between = per_subject_cosine(rest1, rest2, mu)
    rel_within = per_subject_cosine(cells[:, IDX_R1LR], cells[:, IDX_R1RL], mu)

    # stats
    m, md, sd = float(rel_between.mean()), float(np.median(rel_between)), float(rel_between.std())
    sk, ku = float(stats.skew(rel_between)), float(stats.kurtosis(rel_between))
    # normality (D'Agostino-Pearson); also fraction of "unreliable" subjects
    try:
        _, p_norm = stats.normaltest(rel_between)
    except Exception:
        p_norm = float("nan")
    frac_low = float((rel_between < 0.2).mean())
    # is reliability a stable subject property? within vs between per-subject corr
    rho_wb, _ = stats.spearmanr(rel_within, rel_between)

    pct = {q: float(np.percentile(rel_between, q)) for q in (1, 5, 25, 50, 75, 95, 99)}
    summ_rows.append({
        "parc": parc, "n": len(sids),
        "mean": m, "median": md, "std": sd,
        "min": float(rel_between.min()), "max": float(rel_between.max()),
        "p1": pct[1], "p5": pct[5], "p25": pct[25], "p75": pct[75], "p95": pct[95], "p99": pct[99],
        "skew": sk, "kurtosis": ku, "normaltest_p": p_norm,
        "frac_below_0.2": frac_low,
        "within_vs_between_spearman": float(rho_wb),
    })
    # save per-subject
    pd.DataFrame({"subject": sids, "rel_between_session": rel_between,
                  "rel_within_session_run1": rel_within}).to_csv(
        results_dir() / f"g_per_subject_reliability_{parc}.csv", index=False)

    ax.hist(rel_between, bins=40, color="#4682b4", alpha=0.85)
    ax.axvline(m, color="#cd3e4e", lw=2, label=f"mean={m:.3f}")
    ax.axvline(md, color="green", lw=1.5, ls="--", label=f"median={md:.3f}")
    ax.set_title(f"{parc}\nn={len(sids)}, std={sd:.3f}, skew={sk:+.2f}", fontsize=9)
    ax.set_xlabel("per-subject between-session reliability (demeaned cos)")
    ax.set_ylabel("count"); ax.legend(fontsize=8)
    print(f"[G] {parc}: mean={m:.3f} median={md:.3f} std={sd:.3f} range=[{rel_between.min():.3f},"
          f"{rel_between.max():.3f}] skew={sk:+.2f} normaltest_p={p_norm:.2g} "
          f"frac<0.2={frac_low:.3f} within-between rho={rho_wb:+.2f}", flush=True)

fig.suptitle("Per-subject FC between-session reliability distribution", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(results_dir() / "g_reliability_hist.png", dpi=160, bbox_inches="tight")

df = pd.DataFrame(summ_rows)
df.to_csv(results_dir() / "g_per_subject_summary.csv", index=False)
print(f"\n[G] saved -> {results_dir() / 'g_per_subject_summary.csv'}")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
