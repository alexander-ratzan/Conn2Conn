"""Spec v4 E2.0d: what the leading components of the pooled FC basis (total-norm block scaling, rest + 7 tasks) are.

For PCs 1-5 (seed 0; seed 1 for stability):
  - edge loadings summarized as Yeo-7 network blocks (mean loading per within / between network block);
  - sign agreement of the loadings and their correlation with the group-mean FC (is it a global-strength axis?);
  - each view's share of the component's train variance;
  - subject scores: correlation with each subject's mean edge (global FC) per view; cross-view consistency (held-out
    subjects, correlation of a subject's score in rest with their score in each task);
  - top correlations of held-out rest scores with numeric subject metadata;
  - seed stability: |correlation| of seed-0 and seed-1 loadings.

    python scripts/experiments/joint_flow_matching/checks/pooled_pc_interpretation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basis_spectrum import FC_VIEWS  # noqa: E402
from pooled_basis_variance import REPO_ROOT, FC_TASK_CONDITIONS, HCP_Base, top_basis  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "pooled_basis_variance"
N_PC = 5


def build(seed):
    b = HCP_Base(parcellation="Glasser", hemi="both", source="FC", target="SC", data_load_mode="precomputed",
                 shuffle_seed=seed, fc_conditions=["rest", *FC_TASK_CONDITIONS])
    tr = np.asarray(b.trainvaltest_partition_indices["train"])
    ho = np.concatenate([b.trainvaltest_partition_indices["val"], b.trainvaltest_partition_indices["test"]])
    raw = {v: b.fc_condition_upper_triangles[v].astype(np.float64) for v in FC_VIEWS}
    mu = {v: raw[v][tr].mean(0) for v in FC_VIEWS}
    norm = {v: np.linalg.norm(raw[v][tr] - mu[v]) for v in FC_VIEWS}
    W, _ = top_basis(np.vstack([(raw[v][tr] - mu[v]) / norm[v] for v in FC_VIEWS]), N_PC)
    group_mean = np.mean([mu[v] for v in FC_VIEWS], axis=0)
    W = W * np.sign(W.T @ group_mean)        # orient: positive loading along the group-mean FC pattern
    return b, tr, ho, raw, mu, norm, W, group_mean


def main():
    b, tr, ho, raw, mu, norm, W, group_mean = build(0)
    *_, W1, _ = build(1)
    roi = pd.read_csv(REPO_ROOT / "data" / "atlas_info" / "Glasser_dseg_reformatted.csv")
    net = roi["community_yeo"].fillna("none").to_numpy()
    iu = np.triu_indices(len(roi), k=1)
    nets = sorted(set(net) - {"none"})
    meta = b.metadata_df.reset_index(drop=True)
    numeric = meta.select_dtypes("number").drop(columns=["subject"], errors="ignore")

    summary, blocks = [], []
    for j in range(N_PC):
        w = W[:, j]
        row = dict(pc=j + 1,
                   seed_stability=abs(np.corrcoef(w, W1[:, j])[0, 1]),
                   frac_positive=float((w > 0).mean()),
                   corr_with_group_mean_fc=float(np.corrcoef(w, group_mean)[0, 1]))
        lam = {v: np.square(((raw[v][tr] - mu[v]) / norm[v]) @ w).sum() for v in FC_VIEWS}
        for v in FC_VIEWS:
            row[f"share_{v}"] = lam[v] / sum(lam.values())
        scores = {v: ((raw[v] - mu[v]) / norm[v]) @ w for v in FC_VIEWS}
        for v in FC_VIEWS:
            row[f"r_globalFC_{v}"] = float(np.corrcoef(scores[v][ho], raw[v][ho].mean(1))[0, 1])
            if v != "rest":
                row[f"r_rest_{v}"] = float(np.corrcoef(scores["rest"][ho], scores[v][ho])[0, 1])
        r_meta = {c: np.corrcoef(scores["rest"][ho], numeric[c].to_numpy()[ho])[0, 1] for c in numeric
                  if numeric[c].iloc[ho].notna().all() and numeric[c].iloc[ho].std() > 0}
        top = sorted(r_meta.items(), key=lambda kv: -abs(kv[1]))[:3]
        row["top_metadata"] = "; ".join(f"{c} {r:+.2f}" for c, r in top)
        summary.append(row)
        for a_i, a in enumerate(nets):
            for b_n in nets[a_i:]:
                m = ((net[iu[0]] == a) & (net[iu[1]] == b_n)) | ((net[iu[0]] == b_n) & (net[iu[1]] == a))
                blocks.append(dict(pc=j + 1, block=f"{a} - {b_n}", mean_loading=float(w[m].mean()) * np.sqrt(len(w))))
    s, bl = pd.DataFrame(summary), pd.DataFrame(blocks)
    OUT_DIR.mkdir(exist_ok=True)
    s.to_csv(OUT_DIR / "pc_interpretation.csv", index=False)
    bl.to_csv(OUT_DIR / "pc_network_blocks.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 120)
    print(s.round(3).T)
    for j in range(1, N_PC + 1):
        t = bl[bl.pc == j].sort_values("mean_loading")
        print(f"\nPC{j} network blocks (loading x sqrt(E); most negative / most positive):")
        print(pd.concat([t.head(4), t.tail(4)]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
