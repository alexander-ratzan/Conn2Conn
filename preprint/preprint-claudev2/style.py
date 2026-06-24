"""Shared plotting style + data loaders for the Conn2Conn main figures.

Run with the dev-env interpreter:  /Users/user/dev-env/bin/python make_figXX.py
All paths are absolute so scripts can be run from anywhere.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths
REPO = Path("/Users/user/projects/Conn2Conn")
REPRO = REPO / "reproduction"
OUT = REPRO / "outputs"
FAM = REPRO / "family_mechanism" / "outputs"
EXP = REPO / "notebooks-FC_to_SC-experimental"
SANITY = EXP / "sanity_checks"
NLIN = EXP / "non-linear-sanity-check"
TRACT = EXP / "tractography_predict"

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- palette
# A restrained, modality-coded palette. FC is warm, SC is cool, baselines grey.
C = {
    "FC":      "#C0392B",   # functional  -> warm red
    "SC":      "#2471A3",   # structural  -> cool blue
    "fc_lt":   "#E8A39A",
    "sc_lt":   "#9DC3E0",
    "base":    "#7F8C8D",   # bv+demo baseline grey
    "bv":      "#6C7A89",
    "demo":    "#16A085",
    "pred":    "#8E44AD",   # imputed/predicted purple
    "good":    "#1E8449",
    "bad":     "#A93226",
    "ink":     "#222222",
    "grid":    "#D5D8DC",
    "accent":  "#D4AC0D",   # gold highlight for the selected mechanism mode
}
PARCS = ["Glasser", "4S456Parcels"]
PARC_LABEL = {"Glasser": "Glasser (360)", "4S456Parcels": "4S456Parcels (456)"}

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": C["grid"],
    "grid.linewidth": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def panel_tag(ax, tag, dx=-0.02, dy=1.06):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print(f"  wrote figures/{name}.png + .pdf")
    plt.close(fig)


# ---------------------------------------------------------------- loaders
def load_recon():
    return pd.read_csv(OUT / "reconstruction.csv")


def load_downstream():
    return pd.read_csv(OUT / "downstream.csv")


def load_family():
    return pd.read_csv(FAM / "family_auc.csv")


def recon_cell(r, parc, est, src, tgt, col="demeaned_pearson", block=False):
    s = r[(r.parcellation == parc) & (r.estimator == est) &
          (r.source == src) & (r.target == tgt) & (r.is_block == block)][col]
    s = s[np.isfinite(s)]
    return s.values


def down_cell(d, parc, inp, tgt, col, est="bayesian_ridge"):
    s = d[(d.parcellation == parc) & (d.estimator == est) &
          (d.input_set == inp) & (d.target == tgt)][col]
    s = s[np.isfinite(s)]
    return s.values


# ---- cross-estimator helpers (KR has 9 hyperparameter variants; aggregate them) ----
ESTIMATORS = ["pca_pls", "bayesian_ridge", "kernel_ridge"]
EST_LABEL = {"pca_pls": "PCA→PLS", "bayesian_ridge": "BayesianRidge",
             "kernel_ridge": "KernelRidge"}
EST_COLOR = {"pca_pls": "#2471A3", "bayesian_ridge": "#C0392B",
             "kernel_ridge": "#1E8449"}


def recon_est(r, parc, est, src, tgt, col="demeaned_pearson", block=False):
    """Mean over seeds; for kernel_ridge, median across its 9 HP variants per seed."""
    s = r[(r.parcellation == parc) & (r.estimator == est) &
          (r.source == src) & (r.target == tgt) & (r.is_block == block)]
    if est == "kernel_ridge":
        s = s.groupby("seed")[col].median()
        v = s.values
    else:
        v = s[col].values
    v = v[np.isfinite(v)]
    return v


def down_est(d, parc, est, inp, tgt, col):
    s = d[(d.parcellation == parc) & (d.estimator == est) &
          (d.input_set == inp) & (d.target == tgt)]
    if est == "kernel_ridge":
        v = s.groupby("seed")[col].median().values
    else:
        v = s[col].values
    v = v[np.isfinite(v)]
    return v


def m_sd(v):
    import numpy as _np
    v = _np.asarray(v, float)
    return (float(_np.mean(v)), float(_np.std(v))) if len(v) else (float("nan"), 0.0)
