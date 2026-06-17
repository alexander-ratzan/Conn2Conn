#!/usr/bin/env python3
"""H — per-subject achieved-vs-ceiling + reliability-filtered re-run (follow-up to G).

The 0.49 ceiling and the 17%-of-ceiling SC->FC number are means over a heterogeneous
population (per-subject reliability ranges ~0 to 0.78, G). This asks:
  1. Does SC->FC prediction quality TRACK each subject's own FC reliability ceiling?
     (per-subject achieved demeaned_r vs per-subject between-session reliability)
  2. Does excluding the low-reliability tail SHARPEN the aggregate result?

achieved_i = per-subject demeaned cosine(predicted FC, observed FC_test), pooled over the
10 seeds in which subject i lands in the test split. ceiling_i = subject i's
between-session reliability (from G). Both SC->FC and bv+demo->FC.

Saves (outputs/):
  h_per_subject_achieved_vs_ceiling.csv   (subject, source, achieved_mean, n_test, ceiling, fraction)
  h_reliability_filtered_summary.csv      (source, filter, n_kept, mean_achieved, mean_ceiling, fraction_of_ceiling)
  h_correlations.csv                      (source, pearson/spearman achieved-vs-ceiling)
  h_achieved_vs_ceiling_scatter.png
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
from _noise_common import pca_pls_predict, load_seed_split, results_dir

N_SEEDS = 10
PARC = "Glasser"  # match the per-subject reliability (G) primary parcellation
SOURCES = ["SC", "bv+demo"]


def per_subj_demeaned_cos(pred, true, mu):
    dp = pred - mu
    dt = true - mu
    num = (dp * dt).sum(axis=1)
    den = np.sqrt((dp ** 2).sum(axis=1)) * np.sqrt((dt ** 2).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def source_arrays(sp, source):
    if source == "SC":
        return sp["SC_train"], sp["SC_test"]
    if source == "bv+demo":
        return (np.concatenate([sp["bv_train"], sp["demo_train"]], axis=1),
                np.concatenate([sp["bv_test"], sp["demo_test"]], axis=1))
    raise ValueError(source)


# --- collect per-(subject, seed) achieved demeaned_r on the test split ---
records = {s: [] for s in SOURCES}
for seed in range(N_SEEDS):
    sp = load_seed_split(seed=seed)
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    mu = FC_tr.mean(axis=0)
    test_ids = np.asarray(sp["base"].metadata_df["subject"]).astype(int)[sp["test_idx"]]
    for source in SOURCES:
        X_tr, X_te = source_arrays(sp, source)
        k = min(256, X_tr.shape[1])
        pred = pca_pls_predict(X_tr, X_te, FC_tr, k_src=k, k_pls=min(64, k))
        ach = per_subj_demeaned_cos(pred, FC_te, mu)
        for sid, a in zip(test_ids, ach):
            records[source].append((int(sid), float(a)))
    print(f"[H] seed {seed} done", flush=True)

# --- per-subject mean achieved + merge with reliability ceiling (G) ---
gpath = results_dir() / f"g_per_subject_reliability_{PARC}.csv"
g = pd.read_csv(gpath)[["subject", "rel_between_session"]].rename(
    columns={"rel_between_session": "ceiling"})
g["subject"] = g["subject"].astype(int)

per_subj_rows = []
corr_rows = []
for source in SOURCES:
    df = pd.DataFrame(records[source], columns=["subject", "achieved"])
    agg = df.groupby("subject").agg(achieved_mean=("achieved", "mean"),
                                    n_test=("achieved", "size")).reset_index()
    m = agg.merge(g, on="subject", how="inner")
    m["source"] = source
    m["fraction"] = m["achieved_mean"] / m["ceiling"].replace(0, np.nan)
    per_subj_rows.append(m)
    pear = stats.pearsonr(m["achieved_mean"], m["ceiling"])
    spear = stats.spearmanr(m["achieved_mean"], m["ceiling"])
    corr_rows.append({"source": source, "n": len(m),
                      "pearson_achieved_vs_ceiling": float(pear[0]), "pearson_p": float(pear[1]),
                      "spearman_achieved_vs_ceiling": float(spear[0]), "spearman_p": float(spear[1])})
    print(f"[H] {source}: n={len(m)} achieved-vs-ceiling pearson={pear[0]:+.3f} "
          f"spearman={spear[0]:+.3f}", flush=True)

per_subj = pd.concat(per_subj_rows, ignore_index=True)
per_subj.to_csv(results_dir() / "h_per_subject_achieved_vs_ceiling.csv", index=False)
pd.DataFrame(corr_rows).to_csv(results_dir() / "h_correlations.csv", index=False)

# --- reliability-filtered aggregate re-run ---
FILTERS = {
    "all": lambda c: np.ones(len(c), bool),
    "drop_rel_below_0.2": lambda c: c >= 0.2,
    "drop_bottom_10pct": lambda c: c >= np.percentile(c, 10),
    "keep_top_50pct": lambda c: c >= np.median(c),
}
filt_rows = []
for source in SOURCES:
    m = per_subj[per_subj.source == source]
    cc = m["ceiling"].values
    for fname, fn in FILTERS.items():
        keep = fn(cc)
        sub = m[keep]
        mean_ach = float(sub["achieved_mean"].mean())
        mean_ceil = float(sub["ceiling"].mean())
        filt_rows.append({"source": source, "filter": fname, "n_kept": int(keep.sum()),
                          "mean_achieved": mean_ach, "mean_ceiling": mean_ceil,
                          "fraction_of_ceiling": mean_ach / mean_ceil if mean_ceil else np.nan})
filt = pd.DataFrame(filt_rows)
filt.to_csv(results_dir() / "h_reliability_filtered_summary.csv", index=False)
print("\n[H] reliability-filtered summary:")
print(filt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --- scatter ---
fig, axes = plt.subplots(1, len(SOURCES), figsize=(11, 4.4), dpi=160)
if len(SOURCES) == 1:
    axes = [axes]
for ax, source in zip(axes, SOURCES):
    m = per_subj[per_subj.source == source]
    ax.scatter(m["ceiling"], m["achieved_mean"], s=10, alpha=0.4, color="#4682b4")
    lim = [min(m["ceiling"].min(), m["achieved_mean"].min()),
           max(m["ceiling"].max(), m["achieved_mean"].max())]
    ax.plot(lim, lim, "k--", lw=0.8, label="y=x (ceiling)")
    cr = next(r for r in corr_rows if r["source"] == source)
    ax.set_title(f"{source}->FC\nper-subject achieved vs ceiling (r={cr['pearson_achieved_vs_ceiling']:+.2f})",
                 fontsize=9)
    ax.set_xlabel("subject FC reliability ceiling"); ax.set_ylabel("subject achieved demeaned_r")
    ax.legend(fontsize=8)
fig.suptitle("Per-subject: does prediction track each subject's reliability ceiling?", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(results_dir() / "h_achieved_vs_ceiling_scatter.png", dpi=160, bbox_inches="tight")
print(f"\n[H] saved per-subject CSV, filtered summary, correlations, scatter -> {results_dir()}")
