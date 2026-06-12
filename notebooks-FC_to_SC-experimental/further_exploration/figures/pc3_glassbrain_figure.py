#!/usr/bin/env python3
"""PC3 glass-brain figure: top-K edges drawn on a 3D scatter of Glasser
parcel centroids (no nilearn dependency, pure matplotlib).

Inputs:
  - sc_pc_loadings.npy  (10 components x 64620 edges, from Depth 1)
  - data/atlas_info/Glasser_dseg_reformatted.csv  (360 regions x {community_yeo, hemisphere, mni_x/y/z})

Outputs (in same dir as this script, under ./output_glassbrain/):
  - pc3_glassbrain.png   (single figure, sagittal+axial+coronal panels)
  - pc3_top200_edges_labeled.csv  (region pairs + networks for top edges)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO_ROOT = Path(__file__).resolve().parents[3]   # .../Conn2Conn
THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "output_glassbrain"
OUT_DIR.mkdir(exist_ok=True)

# ---- Inputs ----
LOADINGS = REPO_ROOT / "notebooks-FC_to_SC-experimental/model_overviews/results/local_results/further_exploration/depth1_spectral_mechanism/sc_pc_loadings.npy"
ATLAS    = REPO_ROOT / "data/atlas_info/Glasser_dseg_reformatted.csv"
TOP_K    = 200

assert LOADINGS.exists(), f"Missing {LOADINGS}"
assert ATLAS.exists(), f"Missing {ATLAS}"
pc_loadings = np.load(LOADINGS)
assert pc_loadings.shape == (10, 64620), pc_loadings.shape
pc3 = pc_loadings[2].astype(np.float64)
# Orient: positive lead edge for readability.
if pc3[np.argmax(np.abs(pc3))] < 0:
    pc3 = -pc3
pc3_abs = np.abs(pc3)
print(f"PC3: shape={pc3.shape}, max|·|={pc3_abs.max():.4f}")

atlas = pd.read_csv(ATLAS).sort_values("id").reset_index(drop=True)
atlas["idx0"] = atlas["id"].astype(int) - 1
atlas = atlas.set_index("idx0").sort_index()
N = len(atlas)
assert N == 360, N
coords = atlas[["mni_x", "mni_y", "mni_z"]].values
yeo    = atlas["community_yeo"].values
hemi   = atlas["hemisphere"].values

triu_i, triu_j = np.triu_indices(N, k=1)
order = np.argsort(-pc3_abs)
top_idx = order[:TOP_K]
top_loadings = pc3[top_idx]
top_abs      = pc3_abs[top_idx]
ti = triu_i[top_idx]
tj = triu_j[top_idx]

# Per-edge network and hemisphere labels.
top_df = pd.DataFrame({
    "rank":     np.arange(1, TOP_K + 1),
    "loading":  top_loadings,
    "abs_load": top_abs,
    "region_i": atlas.loc[ti, "label"].values,
    "region_j": atlas.loc[tj, "label"].values,
    "net_i":    yeo[ti],
    "net_j":    yeo[tj],
    "hemi_i":   hemi[ti],
    "hemi_j":   hemi[tj],
})
top_df.to_csv(OUT_DIR / f"pc3_top{TOP_K}_edges_labeled.csv", index=False)
print(f"Saved {OUT_DIR / f'pc3_top{TOP_K}_edges_labeled.csv'}")

# Network color map (Yeo7 conventional colors).
NET_COLORS = {
    "default mode":       "#cd3e4e",
    "frontoparietal":     "#e69422",
    "dorsal attention":   "#00760e",
    "ventral attention":  "#c43afa",
    "somatosensory":      "#4682b4",
    "visual":             "#781286",
    "limbic":             "#dcf8a4",
}
# Edge color from the dominant network of its two endpoints (alphabetical pick).
def edge_color(net_a, net_b):
    if net_a == net_b:
        return NET_COLORS.get(net_a, "#888888")
    # Mix: pick the more "interesting" pair shown by the PC3 enrichment
    # (visual + DAN dominate); for visualization just pick the one with the lower
    # index alphabetically so it's deterministic.
    return NET_COLORS.get(min(net_a, net_b), "#888888")

edge_colors = [edge_color(top_df["net_i"].iloc[k], top_df["net_j"].iloc[k]) for k in range(TOP_K)]
# Line width scaled by |loading|.
norms = top_abs / top_abs.max()
widths = 0.5 + 3.0 * norms

# Node colors by network (for the scatter).
node_colors = [NET_COLORS.get(yeo[i], "#cccccc") for i in range(N)]
node_sizes  = np.full(N, 12.0)

# --- Make a 1x3 panel: sagittal (y,z), axial (x,y), coronal (x,z) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=180)
PANELS = [
    ("sagittal (right view, -x)", 1, 2, lambda c: (c[:, 1], c[:, 2])),    # y vs z, color by hemi
    ("axial (top view)",          0, 1, lambda c: (c[:, 0], c[:, 1])),
    ("coronal (back view)",       0, 2, lambda c: (c[:, 0], c[:, 2])),
]
for ax, (title, ax_a, ax_b, proj) in zip(axes, PANELS):
    xs, ys = proj(coords)
    # Draw all edges first (so they sit under nodes).
    segs = []
    for k in range(TOP_K):
        segs.append([(xs[ti[k]], ys[ti[k]]), (xs[tj[k]], ys[tj[k]])])
    lc = LineCollection(segs, colors=edge_colors, linewidths=widths, alpha=0.55)
    ax.add_collection(lc)
    # Nodes.
    ax.scatter(xs, ys, s=node_sizes, c=node_colors, edgecolors="none", alpha=0.9, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Legend.
handles = [
    plt.Line2D([0], [0], color=NET_COLORS[k], lw=3, label=k.title()) for k in NET_COLORS.keys()
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"SC-PC3 top-{TOP_K} edges by |loading|  (Yeo7-colored)\n"
             "Edge width ∝ |loading|; node positions are Glasser MNI centroids",
             fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
out_png = OUT_DIR / "pc3_glassbrain.png"
fig.savefig(out_png, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")

# Per-network breakdown for the writeup.
net_pair_top = np.array([" || ".join(sorted([a, b])) for a, b in zip(top_df["net_i"], top_df["net_j"])])
breakdown = pd.Series(net_pair_top).value_counts().head(10)
print(f"\nTop {TOP_K} edges by network pair (count and fraction):")
for pair, n in breakdown.items():
    print(f"  {pair:50s} {n:4d}  ({n/TOP_K*100:.1f}%)")

# Intra- vs inter-hemispheric.
intra = (top_df["hemi_i"] == top_df["hemi_j"]).sum()
print(f"\nIntra-hemispheric: {intra}/{TOP_K}  ({intra/TOP_K*100:.1f}%)")
print(f"Inter-hemispheric: {TOP_K - intra}/{TOP_K}  ({(TOP_K-intra)/TOP_K*100:.1f}%)")
