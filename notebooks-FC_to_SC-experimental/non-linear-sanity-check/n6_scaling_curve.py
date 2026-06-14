#!/usr/bin/env python3
"""N6 — data-scaling learning curve: is the nonlinear gap model-limited or data-limited?

Subsample the train set at n = 100, 200, 400, full(~683) and, at each size, measure the
NONLINEAR GAP = (residual-boosted final) - (linear template), on the multimodal SINK,
for BOTH downstream cognition and FC reconstruction. Test set is FIXED + full at every n
so performance is comparable. 10 seeds (each = a different family split AND subsample draw).

Interpretation:
  - gap flat ~0 across n  -> nonlinearity isn't coming with more subjects; ceiling is
    structural. Definitive null.
  - gap grows with n      -> signal is n-limited; the move is a bigger cohort, not a
    bigger model. Fundable direction.

Model complexity is held at a fixed moderate k across n (k_per_block=32, k_tgt=128,
k_pls=64), auto-capped down by the helpers only when n forces it (small-n). This is the
textbook learning-curve design: vary data, hold model fixed; the gap isolates the
nonlinear contribution at each n.

Outputs:
  n6_scaling_results.csv  (task, n_sub, seed, linear_perf, final_perf, gap)
  n6_scaling_summary.csv  (task, n_sub: median linear/final/gap + Wilcoxon gap>0)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import load_seed_split_with_r2t, target_train_test, full_panel_eval
from _residual import residual_cognition_blocks, residual_reconstruct_blocks

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
N_SUBS = [100, 200, 400, 100000]   # 100000 = use full train (clamped)
K_PER_BLOCK = 32
K_TGT = 128
K_PLS = 64
COG_TARGET = "CogTotalComp_Unadj"   # the composite; most stable target

COG_CSV = next(p for p in [
    Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv"),
    Path("/scratch/ans9868/Conn2Conn/data/HCP1200_UNRESTRICTED.csv"),
] if p.exists())
cog_lookup = pd.read_csv(COG_CSV)[["Subject", COG_TARGET]].astype({"Subject": int}).set_index("Subject")


def cog_vec(split):
    subj = np.asarray(split["base"].metadata_df["subject"]).astype(int)
    def pull(idx):
        return np.array([cog_lookup[COG_TARGET].get(s, np.nan) for s in subj[idx]], dtype=np.float64)
    return pull(split["train_idx"]), pull(split["test_idx"])


def cog_pearson(pred, true):
    ok = ~np.isnan(true)
    return float(pearsonr(pred[ok], true[ok])[0]) if ok.sum() >= 3 else np.nan


rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    n_full = len(split["train_idx"])
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(n_full)

    # Cognition sink blocks (train arrays are subsampled per n_sub below).
    cog_tr_full, cog_te = cog_vec(split)
    FC_tr_full, FC_te, FC_mean = target_train_test(split, "FC")

    # Pre-grab full train block arrays.
    fc_tr, sc_tr, r2_tr = split["FC_train"], split["SC_train"], split["r2t_flat_train"]
    bv_tr, dm_tr = split["bv_train"], split["demo_train"]
    fc_te, sc_te, r2_te = split["FC_test"], split["SC_test"], split["r2t_flat_test"]
    bv_te, dm_te = split["bv_test"], split["demo_test"]

    for n_req in N_SUBS:
        n_use = min(n_req, n_full)
        sub = perm[:n_use]

        # ---- cognition sink: [FC, SC, r2t, bv, demo] ----
        cog_btr = [fc_tr[sub], sc_tr[sub], r2_tr[sub], bv_tr[sub], dm_tr[sub]]
        cog_bte = [fc_te, sc_te, r2_te, bv_te, dm_te]
        c_final, c_tmpl = residual_cognition_blocks(cog_btr, cog_bte, cog_tr_full[sub],
                                                    k_per_block=K_PER_BLOCK)
        lin_c = cog_pearson(c_tmpl, cog_te)
        fin_c = cog_pearson(c_final, cog_te)
        rows.append({"task": "cognition", "n_sub": n_use, "seed": seed,
                     "linear_perf": lin_c, "final_perf": fin_c, "gap": fin_c - lin_c})

        # ---- reconstruction sink: [SC, r2t, bv, demo] -> FC ----
        rec_btr = [sc_tr[sub], r2_tr[sub], bv_tr[sub], dm_tr[sub]]
        rec_bte = [sc_te, r2_te, bv_te, dm_te]
        r_final, r_tmpl = residual_reconstruct_blocks(rec_btr, rec_bte, FC_tr_full[sub],
                                                      k_tgt=K_TGT, k_pls=K_PLS,
                                                      k_per_block=K_PER_BLOCK)
        lin_r = full_panel_eval(r_tmpl, FC_te, FC_mean)["demeaned_pearson"]
        fin_r = full_panel_eval(r_final, FC_te, FC_mean)["demeaned_pearson"]
        rows.append({"task": "reconstruction", "n_sub": n_use, "seed": seed,
                     "linear_perf": lin_r, "final_perf": fin_r, "gap": fin_r - lin_r})
        print(f"  n={n_use:4d}  cog: lin={lin_c:.3f} gap={fin_c-lin_c:+.4f}  "
              f"recon: lin={lin_r:.4f} gap={fin_r-lin_r:+.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n6_scaling_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n6_scaling_results.csv'}")

# Summary per (task, n_sub): median linear/final/gap + Wilcoxon gap>0.
srows = []
for task in ["cognition", "reconstruction"]:
    for n_use in sorted(df[df.task == task]["n_sub"].unique()):
        g = df[(df.task == task) & (df.n_sub == n_use)]
        gaps = g["gap"].values
        try:
            _, p = wilcoxon(gaps, alternative="greater")
        except ValueError:
            p = float("nan")
        srows.append({"task": task, "n_sub": int(n_use), "n_seeds": len(gaps),
                      "median_linear": float(g["linear_perf"].median()),
                      "median_final": float(g["final_perf"].median()),
                      "median_gap": float(np.median(gaps)),
                      "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
                      "wilcoxon_p_gap_gt0": float(p)})
summary = pd.DataFrame(srows)
summary.to_csv(THIS_DIR / "n6_scaling_summary.csv", index=False)

print("\n=== Learning curve: nonlinear gap vs n ===")
for task in ["cognition", "reconstruction"]:
    print(f"\n  {task}:")
    sub = summary[summary.task == task]
    for _, r in sub.iterrows():
        star = "*" if r["wilcoxon_p_gap_gt0"] < 0.05 else " "
        print(f"    n={r['n_sub']:4d}  linear={r['median_linear']:+.4f}  "
              f"final={r['median_final']:+.4f}  gap={r['median_gap']:+.4f} "
              f"[{r['gap_min']:+.4f},{r['gap_max']:+.4f}] p={r['wilcoxon_p_gap_gt0']:.3f}{star}")
    # Trend: Spearman of gap vs n across all per-seed points.
    from scipy.stats import spearmanr
    gg = df[df.task == task]
    rho, pp = spearmanr(gg["n_sub"], gg["gap"])
    print(f"    TREND gap-vs-n: Spearman rho={rho:+.3f} p={pp:.4f}  "
          f"({'GROWS with n' if rho > 0.2 and pp < 0.05 else 'flat / no growth'})")
