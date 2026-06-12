#!/usr/bin/env python3
"""E4 — PC3-equivalent localization on the r2t bundle representation.

The Depth 1.1 finding: SC-PC3 is the dorsal visual stream / DAN intra-hemispheric
backbone. If r2t carries the same structural mode, we should see PC3 in r2t-space
emphasize a small set of named bundles consistent with the dorsal stream story
(candidates: ILF, IFOF, SLF, vertical occipital, optic radiation).

Procedure (per seed, 10 seeds):
  1. PCA(r2t_flat, K=10) on the train set → top-10 r2t modes
  2. Align modes across seeds to seed-0 anchors (max |cos sim|)
  3. For each anchor mode in {3, 4, 5}, identify the top (region, bundle) entries
     by absolute loading; report which bundles dominate

Then a cross-check: how strongly does seed-0's SC-PC3 loading project onto seed-0's
r2t-PCs? (Spearman of per-subject scores: SC-PC3 score vs each r2t-PC score across
the test set.) If a single r2t-PC carries most of the SC-PC3 score variance, the
two PCAs are picking up the same physical mode.

Outputs:
  e4_r2t_pc_stability.csv          (median |cos|, FC->R², per mode 1..10)
  e4_r2t_top_bundles_per_mode.csv  (top-30 (region, bundle) loadings per mode 3,4,5)
  e4_sc_pc3_to_r2t_pc_projection.csv  (Spearman of seed-0 SC-PC3 scores vs r2t-PCs)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tract_setup import (load_seed_split_with_r2t, PCA, BayesianRidge,
                          pair_indices_by_relation, extract_pair_sims,
                          auc_vs_unrelated, N_REGIONS, N_TRACTS)

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
K_R2T_TOP = 10
K_PCA_FC = 256

# Tract names from data/dataset_utils.py (n=66 standard subset).
TRACT_LIST = ['AssociationArcuateFasciculusL', 'AssociationArcuateFasciculusR',
    'AssociationCingulumL', 'AssociationCingulumR',
    'AssociationExtremeCapsuleL', 'AssociationExtremeCapsuleR',
    'AssociationFrontalAslantTractL', 'AssociationFrontalAslantTractR',
    'AssociationHippocampusAlveusL', 'AssociationHippocampusAlveusR',
    'AssociationInferiorFrontoOccipitalFasciculusL', 'AssociationInferiorFrontoOccipitalFasciculusR',
    'AssociationInferiorLongitudinalFasciculusL', 'AssociationInferiorLongitudinalFasciculusR',
    'AssociationMiddleLongitudinalFasciculusL', 'AssociationMiddleLongitudinalFasciculusR',
    'AssociationParietalAslantTractL', 'AssociationParietalAslantTractR',
    'AssociationSuperiorLongitudinalFasciculusL', 'AssociationSuperiorLongitudinalFasciculusR',
    'AssociationUncinateFasciculusL', 'AssociationUncinateFasciculusR',
    'AssociationVerticalOccipitalFasciculusL', 'AssociationVerticalOccipitalFasciculusR',
    'CerebellumCerebellumL', 'CerebellumCerebellumR',
    'CerebellumInferiorCerebellarPeduncleL', 'CerebellumInferiorCerebellarPeduncleR',
    'CerebellumMiddleCerebellarPeduncle', 'CerebellumSuperiorCerebellarPeduncle',
    'CerebellumVermis', 'CommissureCorpusCallosum',
    'CranialNerveCNIIIL', 'CranialNerveCNIIIR', 'CranialNerveCNIIL', 'CranialNerveCNIIR',
    'CranialNerveCNVIIIL', 'CranialNerveCNVIIIR', 'CranialNerveCNVL', 'CranialNerveCNVR',
    'ProjectionBasalGangliaAcousticRadiationL', 'ProjectionBasalGangliaAcousticRadiationR',
    'ProjectionBasalGangliaAnsaLenticularisL', 'ProjectionBasalGangliaAnsaLenticularisR',
    'ProjectionBasalGangliaAnsaSubthalamicaL', 'ProjectionBasalGangliaAnsaSubthalamicaR',
    'ProjectionBasalGangliaCorticostriatalTractL', 'ProjectionBasalGangliaCorticostriatalTractR',
    'ProjectionBasalGangliaFasciculusLenticularisL', 'ProjectionBasalGangliaFasciculusLenticularisR',
    'ProjectionBasalGangliaFasciculusSubthalamicusL', 'ProjectionBasalGangliaFasciculusSubthalamicusR',
    'ProjectionBasalGangliaFornixL', 'ProjectionBasalGangliaFornixR',
    'ProjectionBasalGangliaOpticRadiationL', 'ProjectionBasalGangliaOpticRadiationR',
    'ProjectionBasalGangliaThalamicRadiationL', 'ProjectionBasalGangliaThalamicRadiationR',
    'ProjectionBrainstemCorticopontineTractL', 'ProjectionBrainstemCorticopontineTractR',
    'ProjectionBrainstemCorticospinalTractL', 'ProjectionBrainstemCorticospinalTractR',
    'ProjectionBrainstemMedialForebrainBundleL', 'ProjectionBrainstemMedialForebrainBundleR',
    'ProjectionBrainstemNonDecussatingDentatorubrothalamicTractL',
    'ProjectionBrainstemNonDecussatingDentatorubrothalamicTractR']
assert len(TRACT_LIST) == N_TRACTS

# Atlas for region labels.
for _p in [Path("/scratch/ans9868/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
           Path("/Users/user/projects/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv")]:
    if _p.exists():
        atlas = pd.read_csv(_p).sort_values("id").reset_index(drop=True)
        atlas["idx0"] = atlas["id"].astype(int) - 1
        atlas_idx = atlas.set_index("idx0")
        break
else:
    raise RuntimeError("atlas CSV not found")
REGION_LABELS = atlas["label"].values

# ---------- Per-seed r2t-PCA fits + per-mode FC->R² ----------
print("=== Stage 1: per-seed r2t PCA fits ===", flush=True)
per_seed = {}
for seed in range(N_SEEDS):
    split = load_seed_split_with_r2t(seed=seed)
    r2t_flat_tr = split["r2t_flat_train"]
    r2t_flat_te = split["r2t_flat_test"]
    FC_tr = split["FC_train"]
    FC_te = split["FC_test"]

    pca = PCA(n_components=K_R2T_TOP, random_state=0).fit(r2t_flat_tr)
    r2t_scores_tr = pca.transform(r2t_flat_tr)
    r2t_scores_te = pca.transform(r2t_flat_te)

    # Per-mode FC -> r2t-PC R².
    pca_fc = PCA(n_components=K_PCA_FC, random_state=0).fit(FC_tr)
    Z_FC_tr = pca_fc.transform(FC_tr)
    Z_FC_te = pca_fc.transform(FC_te)
    fc_r2 = np.zeros(K_R2T_TOP, dtype=np.float64)
    for k in range(K_R2T_TOP):
        m = BayesianRidge(max_iter=300).fit(Z_FC_tr, r2t_scores_tr[:, k])
        fc_r2[k] = r2_score(r2t_scores_te[:, k], m.predict(Z_FC_te))

    per_seed[seed] = {
        "loadings": pca.components_.copy(),
        "expl_var": pca.explained_variance_ratio_.copy(),
        "fc_r2":    fc_r2,
        "scores_te": r2t_scores_te,
    }
    print(f"  seed {seed}: expl_var top-10 = "
          f"{[f'{x:.3f}' for x in pca.explained_variance_ratio_]}")
    print(f"    FC->r2t-PC R² = {[f'{x:.3f}' for x in fc_r2]}")

# ---------- Stability via cosine alignment to seed 0 ----------
print("\n=== Stage 2: stability via cosine alignment ===", flush=True)
anchor_L = per_seed[0]["loadings"]
def cos_sim_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T

stab_rows = []
for k in range(K_R2T_TOP):
    cos_seeds, fc_r2_seeds, var_seeds = [], [per_seed[0]["fc_r2"][k]], [per_seed[0]["expl_var"][k]]
    for s in range(1, N_SEEDS):
        sims = cos_sim_rows(anchor_L[k:k+1], per_seed[s]["loadings"])[0]
        j = int(np.argmax(np.abs(sims)))
        cos_seeds.append(float(np.abs(sims[j])))
        fc_r2_seeds.append(per_seed[s]["fc_r2"][j])
        var_seeds.append(per_seed[s]["expl_var"][j])
    stab_rows.append({
        "anchor_mode": k + 1,
        "median_abs_cos": float(np.median(cos_seeds)),
        "min_abs_cos":    float(np.min(cos_seeds)),
        "median_expl_var": float(np.median(var_seeds)),
        "median_FC_to_R2T_R2": float(np.median(fc_r2_seeds)),
    })
stab_df = pd.DataFrame(stab_rows)
stab_df.to_csv(THIS_DIR / "e4_r2t_pc_stability.csv", index=False)
print(stab_df.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

# ---------- Stage 3: top-(region, bundle) entries for modes 3, 4, 5 ----------
print("\n=== Stage 3: top (region, bundle) entries for modes 3, 4, 5 ===", flush=True)
TOP_K_ENTRIES = 30
top_rows = []
for k in [3, 4, 5]:
    anchor = anchor_L[k - 1]
    # Orient: positive lead entry.
    if anchor[np.argmax(np.abs(anchor))] < 0:
        anchor = -anchor
    # Reshape (N_REGIONS * N_TRACTS,) -> (N_REGIONS, N_TRACTS) so we can read off
    # which (region, bundle) cells dominate.
    L_mat = anchor.reshape(N_REGIONS, N_TRACTS)
    order = np.argsort(-np.abs(anchor))[:TOP_K_ENTRIES]
    for rank, idx in enumerate(order, start=1):
        r = idx // N_TRACTS
        t = idx % N_TRACTS
        top_rows.append({
            "anchor_mode": k,
            "rank": rank,
            "region_idx": int(r),
            "region": REGION_LABELS[r],
            "bundle": TRACT_LIST[t],
            "loading": float(L_mat[r, t]),
        })
top_df = pd.DataFrame(top_rows)
top_df.to_csv(THIS_DIR / "e4_r2t_top_bundles_per_mode.csv", index=False)
print(f"\nTop 5 entries per mode preview:")
for k in [3, 4, 5]:
    print(f"\n  Mode {k}:")
    for _, row in top_df[top_df["anchor_mode"] == k].head(5).iterrows():
        print(f"    {row['region']:25s}  {row['bundle']:50s}  loading={row['loading']:+.4f}")
print(f"\nFull table -> {THIS_DIR / 'e4_r2t_top_bundles_per_mode.csv'}")

# Bundle-frequency aggregation: which named bundles dominate each mode's top entries?
print("\n=== Bundle frequency in top entries per mode ===")
for k in [3, 4, 5]:
    bf = top_df[top_df["anchor_mode"] == k]["bundle"].value_counts().head(8)
    print(f"\n  Mode {k}:")
    for b, n in bf.items():
        print(f"    {b:55s}  {n}/{TOP_K_ENTRIES}")

# ---------- Stage 4: cross-check SC-PC3 vs r2t-PC modes (seed 0) ----------
print("\n=== Stage 4: SC-PC3 score vs r2t-PC scores (seed 0) ===", flush=True)
# Need seed-0 SC-PC3 scores on the test set.
SC_LOADINGS = next(p for p in [
    Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/model_overviews/results/local_results/further_exploration/depth1_spectral_mechanism/sc_pc_loadings.npy"),
    Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/model_overviews/results/local_results/further_exploration/depth1_spectral_mechanism/sc_pc_loadings.npy"),
] if p.exists())
sc_loadings = np.load(SC_LOADINGS)
sc_pc3_loading = sc_loadings[2]
sp0 = load_seed_split_with_r2t(seed=0)
# Project test-set SC onto SC-PC3.
SC_te = sp0["SC_test"]
SC_train_mean = sp0["SC_train"].mean(axis=0)
sc_pc3_score = (SC_te - SC_train_mean) @ sc_pc3_loading

r2t_scores_te = per_seed[0]["scores_te"]
proj_rows = []
for k in range(K_R2T_TOP):
    rho, p = spearmanr(sc_pc3_score, r2t_scores_te[:, k])
    proj_rows.append({"r2t_mode": k + 1, "spearman_with_SC_PC3": float(rho),
                       "p_value": float(p)})
proj_df = pd.DataFrame(proj_rows).sort_values("spearman_with_SC_PC3", key=lambda s: -s.abs())
proj_df.to_csv(THIS_DIR / "e4_sc_pc3_to_r2t_pc_projection.csv", index=False)
print(proj_df.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

best = proj_df.iloc[0]
print(f"\nBest match: r2t-mode {int(best['r2t_mode'])} (Spearman={best['spearman_with_SC_PC3']:+.3f})")
if abs(best["spearman_with_SC_PC3"]) >= 0.5:
    print("=> Strong projection: a single r2t-mode carries most of SC-PC3's subject-level variance.")
    print("   The two PCAs are picking up substantially the same physical mode.")
elif abs(best["spearman_with_SC_PC3"]) >= 0.3:
    print("=> Moderate projection. SC-PC3 has some r2t correlate but not a clean 1-to-1 mode.")
else:
    print("=> Weak projection. SC-PC3 does not correspond cleanly to any single r2t-mode.")
