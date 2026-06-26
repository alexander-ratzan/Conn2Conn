#!/usr/bin/env python3
"""Figure for the shrinkage mechanism: per-PC amplitude + recovery, BR vs PLS.
Reads outputs/probe_shrinkage.csv (per seed × PC), averages over seeds, saves a 2-panel PNG.

    python make_probe_figure.py
"""
from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "outputs" / "probe_shrinkage.csv"
PNG = HERE / "outputs" / "probe_shrinkage.png"

# aggregate per PC (mean over seeds)
acc = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(CSV)):
    pc = int(r["pc"])
    for k in ("var_ratio", "corr_BR", "corr_PLS", "amp_BR", "amp_PLS"):
        acc[pc][k].append(float(r[k]))
pcs = sorted(acc)
x = np.array([p + 1 for p in pcs])                      # 1-based PC index
def m(k): return np.array([np.nanmean(acc[p][k]) for p in pcs])
ampB, ampP = m("amp_BR"), m("amp_PLS")
corB, corP = m("corr_BR"), m("corr_PLS")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# Panel A: amplitude ratio (the shrinkage smoking gun)
ax1.plot(x, ampP, color="#1f77b4", lw=2, label="PLS (max covariance)")
ax1.plot(x, ampB, color="#d62728", lw=2, label="BR (evidence shrinkage)")
ax1.axhline(1.0, color="gray", ls=":", lw=1)
ax1.fill_between(x, ampB, ampP, where=(ampP > ampB), color="#d62728", alpha=0.08)
ax1.set_xscale("log")
ax1.set_xlabel("target PC index (log)")
ax1.set_ylabel(r"amplitude retained  std(pred PC) / std(true PC)")
ax1.set_title("A. BR flattens the low-variance tail toward the mean")
ax1.annotate("BR collapses the tail\n(0.38→0.04)", xy=(200, ampB[199]), xytext=(40, 0.18),
             fontsize=9, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728"))
ax1.annotate("PLS stays flat\n(0.52→0.38)", xy=(200, ampP[199]), xytext=(20, 0.55),
             fontsize=9, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax1.legend(loc="upper right", fontsize=9); ax1.set_ylim(0, 0.62)

# Panel B: per-PC recovery corr (crossover)
ax2.plot(x, corP, color="#1f77b4", lw=2, label="PLS")
ax2.plot(x, corB, color="#d62728", lw=2, label="BR")
ax2.axhline(0.0, color="gray", ls=":", lw=1)
ax2.axvline(50, color="green", ls="--", lw=1, alpha=0.6)
ax2.text(52, 0.30, "crossover ~PC 50", color="green", fontsize=9)
ax2.set_xscale("log")
ax2.set_xlabel("target PC index (log)")
ax2.set_ylabel("recovery  corr(pred PC, true PC)")
ax2.set_title("B. BR wins the top PCs, PLS wins the tail")
ax2.legend(loc="upper right", fontsize=9)

fig.suptitle("BR vs PLS imputation of SC (FC→SC, Glasser ×10): the reconstruct↔identify mechanism",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(PNG, dpi=140, bbox_inches="tight")
print(f"wrote {PNG}")
