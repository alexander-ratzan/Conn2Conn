"""Spec v4 E2 shared pieces: config, per-seed data and code-space cache, pair lists, scoring.

The cache (one npz per seed x code-space variant, under `cache_dir`) holds the code space, every subject's codes for
every view at k_max, and the val / test edge vectors of every view, so GPU jobs never rebuild `HCP_Base`.

    python scripts/experiments/joint_flow_matching/common.py --seeds 0 1 2 3 4 [--variant main|per_view]   # build caches
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
from models.architectures.crossmodal_flow_matching import CodeSpace  # noqa: E402
from models.eval.metrics import compute_basic_regression_metrics, compute_corr_matrix  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
CFG = yaml.safe_load((EXP_DIR / "config.yml").read_text())
VARIANTS = ("main", "per_view")


def pairs(views, nested=()):
    """Ordered (source, target) pairs, excluding self pairs and nested pairs in both directions."""
    bad = {tuple(p) for p in nested} | {tuple(p[::-1]) for p in nested}
    return [(a, b) for a in views for b in views if a != b and (a, b) not in bad]


def load_views(seed):
    """All views (`session_views`) as float32 edge arrays on the matched cohort, plus split indices and subject ids."""
    from data.dataset_utils import FC_TASK_CONDITIONS
    from data.hcp_dataset import HCP_Base
    b = HCP_Base(parcellation=CFG["parcellation"], hemi="both", source="FC", target="SC",
                 data_load_mode="precomputed", shuffle_seed=seed,
                 fc_conditions=["rest", *FC_TASK_CONDITIONS], expose_fc_sessions=True)
    data = {v: b.fc_condition_upper_triangles[v] for v in ("rest", *FC_TASK_CONDITIONS)}
    data["rest_S1"] = b.fc_session1_upper_triangles
    data["rest_S2"] = b.fc_session2_upper_triangles
    data["SC"] = b.sc_upper_triangles
    data = {v: np.asarray(data[v], dtype=np.float32) for v in CFG["session_views"]}
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in b.trainvaltest_partition_indices.items()}
    ids = np.asarray([int(s) for s in b.metadata_df["subject"]], dtype=np.int64)
    return data, splits, ids


def cache_path(seed, variant):
    return Path(CFG["cache_dir"]) / f"seed{seed}_{variant}.npz"


def build_cache(seed, variant="main"):
    data, splits, ids = load_views(seed)
    tr = splits["train"]
    fit_views = CFG["views"]                  # the code space is fit on the main views only (D2)
    cs = CodeSpace({v: data[v][tr] for v in fit_views}, pooled_views=[v for v in fit_views if v != "SC"],
                   k_max=CFG["code_space"]["k_max"], scaling=CFG["code_space"]["scaling"],
                   per_view=(variant == "per_view"))
    for v in CFG["session_views"]:
        if v not in cs.views:                     # sessions: own mean / scale, rest's basis; they do not shape it
            cs.add_view(v, data[v][tr], like="rest")
    out = {f"cs__{k}": v for k, v in cs.state().items()}
    for v in CFG["session_views"]:
        out[f"codes__{v}"] = cs.encode(v, data[v])
        out[f"train_mean__{v}"] = data[v][tr].mean(0)
        for split in ("val", "test"):
            out[f"edges__{split}__{v}"] = data[v][splits[split]]
    for k, v in splits.items():
        out[f"split__{k}"] = v
    out["subject_ids"] = ids
    p = cache_path(seed, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    return p


class SeedData:
    """Everything a job needs for one seed x code-space variant, loaded from the cache."""

    def __init__(self, seed, variant="main"):
        z = np.load(cache_path(seed, variant), allow_pickle=False)
        self.seed, self.variant = seed, variant
        self.cs = CodeSpace.from_state({k[4:]: z[k] for k in z.files if k.startswith("cs__")})
        self.splits = {k[7:]: z[k] for k in z.files if k.startswith("split__")}
        self.codes = {k[7:]: z[k] for k in z.files if k.startswith("codes__")}
        self.train_mean = {k[12:]: z[k] for k in z.files if k.startswith("train_mean__")}
        self.edges = {}
        for k in z.files:
            if k.startswith("edges__"):
                _, split, v = k.split("__")
                self.edges[(split, v)] = z[k]
        self.subject_ids = z["subject_ids"]

    def decode(self, view, Z):
        return self.cs.decode(view, Z)

    def code(self, view, split, k):
        return self.codes[view][self.splits[split], :k]


def score(pred_edges, true_edges, train_mean):
    """v3:E0 / Evaluator metrics on edges: Pearson r, demeaned r, avg rank, top-1 (rows = targets), plus MSE and the
    spread ratio (mean norm of the predicted deviation from the train mean / mean norm of the true deviation)."""
    pred = np.asarray(pred_edges, dtype=np.float64)
    true = np.asarray(true_edges, dtype=np.float64)
    mu = np.asarray(train_mean, dtype=np.float64)
    cc = compute_corr_matrix(true, pred)
    ccd = compute_corr_matrix(true - mu, pred - mu)
    m = compute_basic_regression_metrics(pred, true, corr_matrix=cc, corr_matrix_demeaned=ccd)
    m["spread_ratio"] = float(np.linalg.norm(pred - mu, axis=1).mean() / np.linalg.norm(true - mu, axis=1).mean())
    return {k: float(v) for k, v in m.items() if k != "r2"}


def demeaned_r_torch(pred, true, mu):
    """Mean row-wise Pearson r of (pred - mu) with (true - mu); torch, for fast val selection."""
    a = pred - mu
    b = true - mu
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-10)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=CFG["seeds"])
    ap.add_argument("--variant", choices=VARIANTS, default="main")
    args = ap.parse_args()
    for s in args.seeds:
        print(f"seed {s} {args.variant}: {build_cache(s, args.variant)}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
