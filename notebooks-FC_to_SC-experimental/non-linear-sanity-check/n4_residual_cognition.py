#!/usr/bin/env python3
"""N4-cog — additive residual (boosted) downstream cognition.

Architecture A: template = OOF-BayesianRidge, NL (KernelRidge) learns the residual,
final = template + NL. Does handing the NL model the linear cognition prediction for
free let it find nonlinear cognition signal in tractography?

For each rep in {bv+demo, SC, r2t, r2t_corr, SC_r2t, FC}, each target
{CogTotal,Fluid,Crystal}, 10 seeds: report Pearson, Spearman (rank), R² for BOTH
template (BR) and final (BR+KR), plus lift over bv+demo floor.

Outputs:
  n4_cog_results.csv  (rep, target, variant, seed, pearson, spearman, r2)
  n4_cog_summary.csv  (medians + final-minus-template delta + lift over floor)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import load_seed_split_with_r2t, source_train_test
from _residual import residual_cognition

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
COG_TARGETS = ["CogTotalComp_Unadj", "CogFluidComp_Unadj", "CogCrystalComp_Unadj"]
REPS = ["bv+demo", "SC", "r2t", "r2t_corr", "SC_r2t", "FC"]

COG_CSV = next(p for p in [
    Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv"),
    Path("/scratch/ans9868/Conn2Conn/data/HCP1200_UNRESTRICTED.csv"),
] if p.exists())
cog_lookup = pd.read_csv(COG_CSV)[["Subject"] + COG_TARGETS].astype({"Subject": int}).set_index("Subject")


def cog_vec(split, target):
    subj = np.asarray(split["base"].metadata_df["subject"]).astype(int)
    def pull(idx):
        return np.array([cog_lookup[target].get(s, np.nan) for s in subj[idx]], dtype=np.float64)
    return pull(split["train_idx"]), pull(split["test_idx"])


def src_for(split, rep):
    if rep == "bv+demo":
        return (np.concatenate([split["bv_train"], split["demo_train"]], axis=1),
                np.concatenate([split["bv_test"], split["demo_test"]], axis=1))
    return source_train_test(split, rep)


def metrics(pred, true):
    ok = ~np.isnan(true)
    if ok.sum() < 3:
        return dict(pearson=np.nan, spearman=np.nan, r2=np.nan)
    return dict(pearson=float(pearsonr(pred[ok], true[ok])[0]),
                spearman=float(spearmanr(pred[ok], true[ok])[0]),
                r2=float(r2_score(true[ok], pred[ok])))


rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    for target in COG_TARGETS:
        cog_tr, cog_te = cog_vec(split, target)
        for rep in REPS:
            X_tr, X_te = src_for(split, rep)
            final, template = residual_cognition(X_tr, X_te, cog_tr)
            rows.append({"rep": rep, "target": target, "variant": "template",
                         "seed": seed, **metrics(template, cog_te)})
            rows.append({"rep": rep, "target": target, "variant": "final",
                         "seed": seed, **metrics(final, cog_te)})
        print(f"  {target} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n4_cog_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n4_cog_results.csv'}")

# Summary: median per (rep, target, variant) + final-template delta + lift over floor.
summ = df.groupby(["rep", "target", "variant"])[["pearson", "spearman", "r2"]].median().reset_index()
summ.to_csv(THIS_DIR / "n4_cog_summary.csv", index=False)

# Floor = bv+demo final (best linear+NL on demographics).
print("\n=== Residual-boosted cognition: template vs final, lift over bv+demo ===")
for target in COG_TARGETS:
    sub = summ[summ.target == target]
    floor_final = sub[(sub.rep == "bv+demo") & (sub.variant == "final")]["pearson"].iloc[0]
    print(f"\n  {target} (bv+demo final floor pearson={floor_final:.3f}):")
    for rep in REPS:
        t = sub[(sub.rep == rep) & (sub.variant == "template")]["pearson"].iloc[0]
        f = sub[(sub.rep == rep) & (sub.variant == "final")]["pearson"].iloc[0]
        lift = f - floor_final
        print(f"    {rep:9s} template={t:+.3f} final={f:+.3f} Δ(final-tmpl)={f-t:+.4f} "
              f"lift_over_floor={lift:+.3f}")

# Decision: does residual-boost let a tractography rep clear the floor?
print("\n=== DECISION ===")
TRACT = ["SC", "r2t", "r2t_corr", "SC_r2t"]
hits = []
for target in COG_TARGETS:
    sub = summ[summ.target == target]
    floor_final = sub[(sub.rep == "bv+demo") & (sub.variant == "final")]["pearson"].iloc[0]
    for rep in TRACT:
        f = sub[(sub.rep == rep) & (sub.variant == "final")]["pearson"].iloc[0]
        if f - floor_final >= 0.02:
            hits.append((rep, target, f - floor_final))
if hits:
    print("  Tractography reps clearing floor by >=0.02 under residual-boost:")
    for rep, t, l in hits:
        print(f"    {rep} / {t}: lift={l:+.3f}")
else:
    print("  NONE clear the bv+demo floor even with the linear answer handed in free.")
    print("  -> residual-boost does not unlock tractography cognition signal.")
