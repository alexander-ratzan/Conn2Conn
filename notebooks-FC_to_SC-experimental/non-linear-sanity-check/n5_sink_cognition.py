#!/usr/bin/env python3
"""N5-cog — multimodal SINK residual-boost for cognition.

The one test that can see CROSS-MODAL interactions: put all modalities in one feature
space and let a nonlinear model cross them. Compares four predictors per target/seed:
  bv+demo        (floor)
  FC             (current champion)
  sink_linear    [FC ‖ SC ‖ r2t ‖ bv ‖ demo], per-block PCA + linear BR
  sink_residual  sink_linear template + KernelRidge residual (OOF, no leakage)

Metrics: Pearson, Spearman (rank), R². Decision rules:
  - cross-modal complementarity: sink_linear − FC >= +0.02
  - nonlinear cross-modal unlock: sink_residual − sink_linear >= +0.02, Wilcoxon p<0.05

Outputs: n5_cog_results.csv, n5_cog_summary.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import load_seed_split_with_r2t, source_train_test
from _residual import residual_cognition, residual_cognition_blocks

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
COG_TARGETS = ["CogTotalComp_Unadj", "CogFluidComp_Unadj", "CogCrystalComp_Unadj"]

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


def metrics(pred, true):
    ok = ~np.isnan(true)
    if ok.sum() < 3:
        return dict(pearson=np.nan, spearman=np.nan, r2=np.nan)
    return dict(pearson=float(pearsonr(pred[ok], true[ok])[0]),
                spearman=float(spearmanr(pred[ok], true[ok])[0]),
                r2=float(r2_score(true[ok], pred[ok])))


def sink_blocks(split):
    """[FC, SC, r2t, bv, demo] train/test block lists."""
    btr = [split["FC_train"], split["SC_train"], split["r2t_flat_train"],
           split["bv_train"], split["demo_train"]]
    bte = [split["FC_test"], split["SC_test"], split["r2t_flat_test"],
           split["bv_test"], split["demo_test"]]
    return btr, bte


rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    bd_tr = np.concatenate([split["bv_train"], split["demo_train"]], axis=1)
    bd_te = np.concatenate([split["bv_test"], split["demo_test"]], axis=1)
    fc_tr, fc_te = source_train_test(split, "FC")
    btr, bte = sink_blocks(split)
    for target in COG_TARGETS:
        cog_tr, cog_te = cog_vec(split, target)
        # bv+demo floor (residual_cognition gives final; template is the linear floor).
        bd_final, bd_tmpl = residual_cognition(bd_tr, bd_te, cog_tr)
        rows.append({"rep": "bv+demo", "target": target, "seed": seed, **metrics(bd_tmpl, cog_te)})
        # FC alone (linear template).
        fc_final, fc_tmpl = residual_cognition(fc_tr, fc_te, cog_tr)
        rows.append({"rep": "FC", "target": target, "seed": seed, **metrics(fc_tmpl, cog_te)})
        # sink linear (template) + sink residual (final).
        s_final, s_tmpl = residual_cognition_blocks(btr, bte, cog_tr)
        rows.append({"rep": "sink_linear", "target": target, "seed": seed, **metrics(s_tmpl, cog_te)})
        rows.append({"rep": "sink_residual", "target": target, "seed": seed, **metrics(s_final, cog_te)})
        print(f"  {target}: FC={metrics(fc_tmpl,cog_te)['pearson']:.3f} "
              f"sink_lin={metrics(s_tmpl,cog_te)['pearson']:.3f} "
              f"sink_res={metrics(s_final,cog_te)['pearson']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n5_cog_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n5_cog_results.csv'}")

summ = df.groupby(["rep", "target"])[["pearson", "spearman", "r2"]].median().reset_index()
summ.to_csv(THIS_DIR / "n5_cog_summary.csv", index=False)

print("\n=== Sink cognition (median Pearson) ===")
for target in COG_TARGETS:
    s = summ[summ.target == target].set_index("rep")["pearson"]
    fc, sl, sr, fl = s["FC"], s["sink_linear"], s["sink_residual"], s["bv+demo"]
    print(f"\n  {target}:")
    print(f"    bv+demo floor   = {fl:+.3f}")
    print(f"    FC              = {fc:+.3f}  (lift {fc-fl:+.3f})")
    print(f"    sink_linear     = {sl:+.3f}  (vs FC {sl-fc:+.3f}, lift {sl-fl:+.3f})")
    print(f"    sink_residual   = {sr:+.3f}  (vs sink_linear {sr-sl:+.4f})")

# Decision with Wilcoxon on the residual-over-linear-sink delta.
print("\n=== DECISION ===")
unlock = False
for target in COG_TARGETS:
    sl = df[(df.rep == "sink_linear") & (df.target == target)].sort_values("seed")["pearson"].values
    sr = df[(df.rep == "sink_residual") & (df.target == target)].sort_values("seed")["pearson"].values
    fc = df[(df.rep == "FC") & (df.target == target)].sort_values("seed")["pearson"].values
    d_res = sr - sl
    try:
        _, p_res = wilcoxon(d_res, alternative="greater")
    except ValueError:
        p_res = float("nan")
    compl = np.median(sl - fc)
    print(f"  {target}: sink_linear−FC median={compl:+.4f} | "
          f"residual−sink_linear median={np.median(d_res):+.4f} p={p_res:.3f}")
    if np.median(d_res) >= 0.02 and p_res < 0.05:
        unlock = True
if unlock:
    print("  -> NONLINEAR CROSS-MODAL UNLOCK: residual on the multimodal sink beats the")
    print("     linear sink by >=0.02. Cross-modal interaction signal exists. Investigate.")
else:
    print("  -> No nonlinear cross-modal unlock: residual adds <0.02 over the linear sink.")
