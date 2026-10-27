"""Multitask latent viewer: subject-wise FC encodings in the multitask CCA latent (spec v4 E1) vs pooled block-PCA codes.

For each seed (0-4) and held-out split (val, test), every subject of the split is encoded from each of the 8 FC
conditions in two spaces of equal dimension d (E1's SC-blind selected latent for that seed):
  - CCA latent:   g^m_i = z^m_i A_m           (MAXVAR over the 8 conditions, fit on train; E1 / I2)
  - pooled PCA:   the first d block-PCA codes z^m_i[:d]   (equal-variance pooled FC basis; v4 D2)
Each dimension is divided by its SD over the train encodings of all conditions (as the notebook), so the two spaces
are in comparable units. Per-subject statistics are computed in the full d-dimensional space over all subjects of the
split (not only the ones displayed): silhouette (Euclidean, cluster = subject's 8 encodings), dispersion (mean distance
of the 8 encodings to their centroid), separation (distance from the centroid to the nearest other subject's
centroid), and rest -> task identification (correlation rank of the subject's own rest encoding among all subjects',
for each task; as notebook section 5). Display coordinates: the first N_DIMS latent dimensions and a 2-D PCA of the
split's points.

Math and derivations: scripts/notebooks/EDA/FC_encoded_conditions_analysis.ipynb (sections 2 and 5).

Model and data: scripts/experiments/cca_latent_bench (E1 config, per-seed records for the selected latent) and E2's
per-seed cache (joint_flow_matching). Page: multitask_latent_viewer.template on the shared viewer_kit (one data block per
seed x split). Output, next to this file (generated, not tracked): multitask_latent_viewer.html and its data cache
multitask_latent_viewer_data.json (--from-cache rebuilds the page only).

    python scripts/data_viewers/multitask_latent_viewer/build_multitask_latent_viewer.py [--seeds 0 1 2 3 4] [--from-cache] [--smoke]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
EXP = REPO_ROOT / "scripts" / "experiments" / "cca_latent_bench"
sys.path.insert(0, str(EXP))
from run_seed import CFG, FC, SeedData  # noqa: E402  (E1 config and E2's per-seed cache)
from models.architectures.multiview_cca import maxvar  # noqa: E402
from models.eval.metrics import compute_corr_matrix  # noqa: E402
from sklearn.metrics import silhouette_samples  # noqa: E402
from scripts.data_viewers.viewer_kit import Viewer  # noqa: E402

N_DIMS = 10
TASKS = [v for v in FC if v != "rest"]


def r(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def subject_stats(E):
    """E: (n, C, d) encodings of one split. Per-subject and summary statistics in the full d-dim space."""
    n, C, d = E.shape
    pts = E.reshape(n * C, d)
    lab = np.repeat(np.arange(n), C)
    sil = silhouette_samples(pts, lab).reshape(n, C).mean(1)
    cen = E.mean(1)
    disp = np.linalg.norm(E - cen[:, None], axis=2).mean(1)
    D = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    np.fill_diagonal(D, np.inf)
    sep = D.min(1)
    ci_rest = FC.index("rest")
    ranks, hits = [], []
    for t in TASKS:
        src, tgt = E[:, ci_rest], E[:, FC.index(t)]
        cc = compute_corr_matrix(tgt - tgt.mean(0), src - src.mean(0))      # rows: task encodings; cols: rest
        own = np.diag(cc)
        ranks.append(1 - (cc > own[:, None]).sum(1) / n)                     # 1 = own rest encoding ranked first
        hits.append(cc.argmax(1) == np.arange(n))
    ranks, hits = np.stack(ranks, 1), np.stack(hits, 1)
    S = E - E.mean(axis=(0, 1), keepdims=True)
    between, within = S.mean(1).var(0).sum(), S.var(1).mean(0).sum()
    return dict(silhouette=r(sil), dispersion=r(disp), separation=r(sep), rest_rank=r(ranks.mean(1)),
                rest_top1_tasks=hits.sum(1).astype(int).tolist(),
                summary=dict(silhouette=float(sil.mean()), dispersion=float(disp.mean()), separation=float(sep.mean()),
                             between_share=float(between / (between + within)),
                             rest_top1=float(hits.mean()), rest_avg_rank=float(ranks.mean())))


def pca2(E):
    """2-D PCA of all (subject, condition) points of a split in one space."""
    n, C, d = E.shape
    X = E.reshape(n * C, d)
    X = X - X.mean(0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    return P.reshape(n, C, 2), (s[:2] ** 2 / (s ** 2).sum()).tolist()


def build_seed(seed):
    lat = pd.read_csv(REPO_ROOT / CFG["records_dir"] / f"seed{seed}_latent.csv")
    sel = lat[lat.selected_sc_blind.eq(True)].iloc[0]
    k, d, lam = int(sel.k_fc), int(sel.d), float(sel.lambda_rel)
    sd = SeedData(seed, "main")
    Z = {s: {v: sd.code(v, s, k).astype(np.float64) for v in FC} for s in ("train", "val", "test")}
    A, _, _ = maxvar(Z["train"], d, lam)
    scale_cca = np.concatenate([Z["train"][v] @ A[v] for v in FC]).std(0)
    scale_pca = np.concatenate([Z["train"][v][:, :d] for v in FC]).std(0)
    out = {}
    for split in ("val", "test"):
        ids = sd.subject_ids[sd.splits[split]].tolist()
        spaces = {}
        for name, E in (("cca", np.stack([Z[split][v] @ A[v] / scale_cca for v in FC], 1)),
                        ("pca", np.stack([Z[split][v][:, :d] / scale_pca for v in FC], 1))):
            P2, ev = pca2(E)
            spaces[name] = dict(dims=r(E[:, :, :N_DIMS]), pca2=r(P2), pca2_var=r(ev, 4), **subject_stats(E))
        out[f"{seed}_{split}"] = dict(seed=seed, split=split, n=len(ids), subjects=ids, k_fc=k, d=d, lambda_rel=lam,
                                      spaces=spaces)
        print(f"seed {seed} {split}: n {len(ids)}; silhouette CCA {spaces['cca']['summary']['silhouette']:.3f} "
              f"vs PCA {spaces['pca']['summary']['silhouette']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=CFG["seeds"])
    Viewer.add_args(ap)
    args = ap.parse_args()
    viewer = Viewer(__file__)
    if args.from_cache:
        viewer.write(*viewer.load_cache(), cache=False, smoke=args.smoke)
        return
    sets = {}
    for s in args.seeds:
        sets.update(build_seed(s))
    meta = dict(conditions=FC, n_dims=N_DIMS, notebook="scripts/notebooks/EDA/FC_encoded_conditions_analysis.ipynb",
                writeup="scripts/experiments/cca_latent_bench/cca_latent_bench.md")
    viewer.write(meta, sets, smoke=args.smoke)

if __name__ == "__main__":
    main()
