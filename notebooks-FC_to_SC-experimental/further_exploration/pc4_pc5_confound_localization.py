#!/usr/bin/env python3
"""PC4 and PC5 confound + localization test — second-tier mechanism check.

PC3 is the headline structural mode. PC4 and PC5 are also stable across 10 seeds
(median |cos| 0.90 and 0.87 per Depth 1.1), with FC->R² ≈ 0.12 and 0.11. Do they
tell a coherent secondary story, or are they noise modes?

For each PC in {4, 5}:
  1. 10-seed alignment to seed-0 (signed cos sim of loading vectors)
  2. Confound test: OLS [sex || bv] -> PC_k scores; report train+test R²
  3. PC_k FC-predictability: median across 10 seeds, raw vs residualized on [sex, bv]
  4. Network enrichment (Yeo7 pairs), interhemi fraction, rich-club fraction, energy concentration
  5. Spotlight top-30 edges with region labels

Outputs (in same dir as this script, under pc4_pc5_results/):
  pc{k}_stability_aligned.csv
  pc{k}_confound.csv
  pc{k}_localization_per_seed.csv
  pc{k}_enrichment_agg.csv
  pc{k}_top30_edges.csv
  pc{k}_verdict.txt
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Locate _setup.py.
for _cand in [Path.cwd(),
              Path(__file__).resolve().parent,
              Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration"),
              Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration")]:
    if (_cand / "_setup.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError(f"_setup.py not found from cwd={Path.cwd()}")
from _setup import (load_seed_split, PCA, BayesianRidge, LinearRegression,
                     pair_indices_by_relation, extract_pair_sims, auc_vs_unrelated)
from sklearn.metrics import r2_score

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "pc4_pc5_results"
OUT_DIR.mkdir(exist_ok=True)

N_SEEDS = 10
K_PCA_TOP = 10
K_PCA_FC = 256
PCS_TO_TEST = [4, 5]   # 1-indexed for human-readability; converted to 0-indexed below
N_REGIONS = 360

# Atlas (for network labels in enrichment).
ATLAS_PATH = next(p for p in [
    Path("/scratch/ans9868/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
    Path("/Users/user/projects/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv"),
] if p.exists())
atlas = pd.read_csv(ATLAS_PATH).sort_values("id").reset_index(drop=True)
atlas["idx0"] = atlas["id"].astype(int) - 1
atlas_idx = atlas.set_index("idx0")
NETWORK_COL = "community_yeo"

triu_i, triu_j = np.triu_indices(N_REGIONS, k=1)
N_EDGES = len(triu_i)

# Per-edge atlas labels (precompute).
hemi_i_all = atlas_idx.loc[triu_i, "hemisphere"].values
hemi_j_all = atlas_idx.loc[triu_j, "hemisphere"].values
net_i_all  = atlas_idx.loc[triu_i, NETWORK_COL].values
net_j_all  = atlas_idx.loc[triu_j, NETWORK_COL].values
net_pair_all = np.array([" || ".join(sorted([a, b])) for a, b in zip(net_i_all, net_j_all)])
all_pair_counts = pd.Series(net_pair_all).value_counts()
interhemi_base = float((hemi_i_all != hemi_j_all).mean())

# ============================================================================
# Stage 1: Refit PCAs per seed (10 seeds), compute per-PC FC->R^2 and AUCs.
# Cache in memory.
# ============================================================================
print("=== Stage 1: per-seed PCA fits ===", flush=True)
per_seed = {}
for seed in range(N_SEEDS):
    print(f"  seed {seed} ...", flush=True)
    sp = load_seed_split(seed=seed)
    pca_sc = PCA(n_components=K_PCA_TOP, random_state=0).fit(sp["SC_train"])
    sc_scores_te = pca_sc.transform(sp["SC_test"])
    sc_scores_tr = pca_sc.transform(sp["SC_train"])
    pca_fc = PCA(n_components=K_PCA_FC, random_state=0).fit(sp["FC_train"])
    Z_FC_tr = pca_fc.transform(sp["FC_train"])
    Z_FC_te = pca_fc.transform(sp["FC_test"])

    # Per-PC FC->R² (top-10)
    fc_r2 = np.zeros(K_PCA_TOP, dtype=np.float64)
    for k in range(K_PCA_TOP):
        m = BayesianRidge(max_iter=300).fit(Z_FC_tr, sc_scores_tr[:, k])
        fc_r2[k] = r2_score(sc_scores_te[:, k], m.predict(Z_FC_te))

    # Per-PC AUCs.
    rng = np.random.default_rng(42 + seed)
    pairs = pair_indices_by_relation(sp["base"].metadata_df, sp["test_idx"], rng, age_tol_yrs=3.0)
    auc_mz = np.full(K_PCA_TOP, np.nan)
    auc_dz = np.full(K_PCA_TOP, np.nan)
    auc_sb = np.full(K_PCA_TOP, np.nan)
    for k in range(K_PCA_TOP):
        score_vec = sc_scores_te[:, k]
        diff = np.abs(score_vec[:, None] - score_vec[None, :])
        sims_by_rel = extract_pair_sims(-diff, pairs)
        a = auc_vs_unrelated(sims_by_rel)
        auc_mz[k] = a.get("MZ", np.nan)
        auc_dz[k] = a.get("DZ", np.nan)
        auc_sb[k] = a.get("sibling", np.nan)

    per_seed[seed] = {
        "loadings":   pca_sc.components_.copy(),
        "expl_var":   pca_sc.explained_variance_ratio_.copy(),
        "sc_pc_tr":   sc_scores_tr,
        "sc_pc_te":   sc_scores_te,
        "fc_r2":      fc_r2,
        "auc_mz":     auc_mz,
        "auc_dz":     auc_dz,
        "auc_sb":     auc_sb,
    }

anchor_loadings = per_seed[0]["loadings"]


def cosine_sim_matrix(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


# ============================================================================
# Stage 2: For each PC in PCS_TO_TEST, do the full battery.
# ============================================================================
for pc_num in PCS_TO_TEST:
    k_anchor = pc_num - 1   # 0-indexed
    print(f"\n=== PC{pc_num}: stability, confound, localization ===", flush=True)

    # --- Alignment across seeds ---
    aligned = []
    for s in range(N_SEEDS):
        sim_row = cosine_sim_matrix(anchor_loadings[k_anchor:k_anchor+1],
                                     per_seed[s]["loadings"])[0]
        j = int(np.argmax(np.abs(sim_row)))
        signed = float(sim_row[j])
        sign = float(np.sign(signed))
        aligned.append({
            "seed": s, "matched_pc_1based": j + 1,
            "signed_cos": signed, "abs_cos": float(np.abs(signed)),
            "expl_var":  float(per_seed[s]["expl_var"][j]),
            "fc_r2":     float(per_seed[s]["fc_r2"][j]),
            "auc_mz":    float(per_seed[s]["auc_mz"][j]),
            "auc_dz":    float(per_seed[s]["auc_dz"][j]),
            "auc_sb":    float(per_seed[s]["auc_sb"][j]),
            "pc_signed_loadings": sign * per_seed[s]["loadings"][j],   # for downstream
        })
    align_df = pd.DataFrame(aligned)[
        ["seed","matched_pc_1based","signed_cos","abs_cos","expl_var",
         "fc_r2","auc_mz","auc_dz","auc_sb"]]
    align_df.to_csv(OUT_DIR / f"pc{pc_num}_stability_aligned.csv", index=False)
    print(f"  median |cos| = {align_df['abs_cos'].median():.4f}, "
          f"min = {align_df['abs_cos'].min():.4f}")
    print(f"  median FC->R² = {align_df['fc_r2'].median():.4f}")
    print(f"  median AUC_MZ={align_df['auc_mz'].median():.4f} "
          f"AUC_DZ={align_df['auc_dz'].median():.4f} "
          f"AUC_sib={align_df['auc_sb'].median():.4f}")

    # --- Confound test on seed 0 ---
    sp0 = load_seed_split(seed=0)
    sc_pc_tr0 = per_seed[0]["sc_pc_tr"][:, k_anchor]   # anchor (seed 0 always matches anchor exactly)
    sc_pc_te0 = per_seed[0]["sc_pc_te"][:, k_anchor]
    sex_tr = sp0["base"].sex_oh[sp0["train_idx"]]
    sex_te = sp0["base"].sex_oh[sp0["test_idx"]]
    bv_tr  = sp0["bv_train"]
    bv_te  = sp0["bv_test"]
    X_conf_tr = np.concatenate([sex_tr, bv_tr], axis=1)
    X_conf_te = np.concatenate([sex_te, bv_te], axis=1)
    ols = LinearRegression().fit(X_conf_tr, sc_pc_tr0)
    r2_conf_tr = r2_score(sc_pc_tr0, ols.predict(X_conf_tr))
    r2_conf_te = r2_score(sc_pc_te0, ols.predict(X_conf_te))
    # Residualize and re-test FC predictability.
    pc_resid_tr = sc_pc_tr0 - ols.predict(X_conf_tr)
    pc_resid_te = sc_pc_te0 - ols.predict(X_conf_te)
    pca_fc0 = PCA(n_components=K_PCA_FC, random_state=0).fit(sp0["FC_train"])
    Z_FC_tr0 = pca_fc0.transform(sp0["FC_train"])
    Z_FC_te0 = pca_fc0.transform(sp0["FC_test"])
    br_raw   = BayesianRidge(max_iter=300).fit(Z_FC_tr0, sc_pc_tr0)
    r2_raw   = r2_score(sc_pc_te0, br_raw.predict(Z_FC_te0))
    br_resid = BayesianRidge(max_iter=300).fit(Z_FC_tr0, pc_resid_tr)
    r2_resid = r2_score(pc_resid_te, br_resid.predict(Z_FC_te0))
    conf_df = pd.DataFrame([{
        "pc": pc_num,
        "confound_R2_train": float(r2_conf_tr),
        "confound_R2_test":  float(r2_conf_te),
        "FC_to_PC_R2_raw":   float(r2_raw),
        "FC_to_PC_R2_resid": float(r2_resid),
        "drop":              float(r2_raw - r2_resid),
    }])
    conf_df.to_csv(OUT_DIR / f"pc{pc_num}_confound.csv", index=False)
    print(f"  confound R² (test)  = {r2_conf_te:.4f}")
    print(f"  FC->PC R² raw       = {r2_raw:.4f}")
    print(f"  FC->PC R² residual  = {r2_resid:.4f}  (drop {r2_raw-r2_resid:+.4f})")

    # --- Localization probes per seed at K=200 ---
    K = 200
    loc_rows = []
    enr_rows = []
    pc_aligned_for_seed0 = aligned[0]["pc_signed_loadings"]  # seed 0 anchor
    if pc_aligned_for_seed0[np.argmax(np.abs(pc_aligned_for_seed0))] < 0:
        pc_aligned_for_seed0 = -pc_aligned_for_seed0
    for s in range(N_SEEDS):
        sp_s = load_seed_split(seed=s)
        # Recompute SC node strength per seed for rich-club hubs.
        SC_tr_s = sp_s["SC_train"]
        sc_mean_edge = SC_tr_s.mean(axis=0)
        sym = np.zeros((N_REGIONS, N_REGIONS), dtype=np.float32)
        sym[triu_i, triu_j] = sc_mean_edge
        sym[triu_j, triu_i] = sc_mean_edge
        node_strength = sym.mean(axis=1)
        hub_thresh = np.quantile(node_strength, 0.80)
        is_hub = node_strength >= hub_thresh
        richclub_base = float((is_hub[triu_i] & is_hub[triu_j]).mean())

        pc_s = aligned[s]["pc_signed_loadings"]
        if pc_s[np.argmax(np.abs(pc_s))] < 0:
            pc_s = -pc_s

        # Energy concentration.
        sq = pc_s ** 2
        sq_sorted = np.sort(sq)[::-1]
        n_top1pct = max(1, N_EDGES // 100)
        energy_top1pct = float(sq_sorted[:n_top1pct].sum() / sq_sorted.sum())

        order = np.argsort(-np.abs(pc_s))
        top_idx = order[:K]
        ti_t, tj_t = triu_i[top_idx], triu_j[top_idx]
        interhemi_frac = float((atlas_idx.loc[ti_t, "hemisphere"].values
                                 != atlas_idx.loc[tj_t, "hemisphere"].values).mean())
        richclub_frac = float((is_hub[ti_t] & is_hub[tj_t]).mean())
        loc_rows.append({
            "seed": s, "K": K,
            "interhemi_frac_top": interhemi_frac,
            "interhemi_frac_base": interhemi_base,
            "richclub_frac_top":  richclub_frac,
            "richclub_frac_base": richclub_base,
            "energy_top1pct":     energy_top1pct,
        })

        top_pair = np.array([" || ".join(sorted([a, b]))
                              for a, b in zip(atlas_idx.loc[ti_t, NETWORK_COL].values,
                                              atlas_idx.loc[tj_t, NETWORK_COL].values)])
        for pair, n_obs in pd.Series(top_pair).value_counts().items():
            n_exp = all_pair_counts.get(pair, 0) * (K / N_EDGES)
            enr = (n_obs / n_exp) if n_exp > 0 else np.nan
            enr_rows.append({"seed": s, "K": K, "net_pair": pair,
                              "n_obs": int(n_obs), "n_exp": float(n_exp),
                              "enrichment": float(enr)})

    loc_df = pd.DataFrame(loc_rows)
    enr_df = pd.DataFrame(enr_rows)
    loc_df.to_csv(OUT_DIR / f"pc{pc_num}_localization_per_seed.csv", index=False)

    # Aggregate enrichment.
    agg = enr_df.groupby("net_pair").agg(
        n_seeds=("seed", "nunique"),
        median_enrichment=("enrichment", "median"),
        min_enrichment=("enrichment", "min"),
        median_n_obs=("n_obs", "median"),
    ).reset_index().sort_values("median_enrichment", ascending=False)
    agg.to_csv(OUT_DIR / f"pc{pc_num}_enrichment_agg.csv", index=False)
    print(f"\n  Top 5 network pairs (median enrichment):")
    for _, r in agg.head(5).iterrows():
        print(f"    {r['net_pair']:50s} median={r['median_enrichment']:.2f}x  min={r['min_enrichment']:.2f}x")
    print(f"\n  Localization probes (median across 10 seeds):")
    print(f"    interhemi top vs base: {loc_df['interhemi_frac_top'].median():.4f} vs "
          f"{loc_df['interhemi_frac_base'].iloc[0]:.4f}")
    print(f"    richclub  top vs base: {loc_df['richclub_frac_top'].median():.4f} vs "
          f"{loc_df['richclub_frac_base'].median():.4f}")
    print(f"    energy top-1%        : {loc_df['energy_top1pct'].median():.4f}")

    # --- Top-30 edges at seed 0, labeled ---
    pc_anchor = aligned[0]["pc_signed_loadings"]
    if pc_anchor[np.argmax(np.abs(pc_anchor))] < 0:
        pc_anchor = -pc_anchor
    order0 = np.argsort(-np.abs(pc_anchor))[:30]
    top30 = pd.DataFrame({
        "rank":     np.arange(1, 31),
        "region_i": atlas_idx.loc[triu_i[order0], "label"].values,
        "region_j": atlas_idx.loc[triu_j[order0], "label"].values,
        "net_i":    atlas_idx.loc[triu_i[order0], NETWORK_COL].values,
        "net_j":    atlas_idx.loc[triu_j[order0], NETWORK_COL].values,
        "hemi_i":   atlas_idx.loc[triu_i[order0], "hemisphere"].values,
        "hemi_j":   atlas_idx.loc[triu_j[order0], "hemisphere"].values,
        "loading":  pc_anchor[order0],
    })
    top30.to_csv(OUT_DIR / f"pc{pc_num}_top30_edges.csv", index=False)

    # --- Verdict text ---
    stable = align_df["abs_cos"].median() >= 0.85
    is_confound = r2_conf_te >= 0.60
    has_fc_signal = align_df["fc_r2"].median() >= 0.10
    interhemi_anti = (loc_df["interhemi_frac_top"].median() < 0.30 * loc_df["interhemi_frac_base"].iloc[0])
    richclub_elev = loc_df["richclub_frac_top"].median() > 1.30 * loc_df["richclub_frac_base"].median()
    energy_peak   = loc_df["energy_top1pct"].median() > 0.10
    stable_pairs  = agg[(agg["n_seeds"] >= 8) & (agg["min_enrichment"] >= 1.5)
                        & (agg["median_enrichment"] >= 2.0)]

    lines = [
        f"PC{pc_num} verdict",
        "=" * 60,
        f"stable across seeds (|cos| >= 0.85)?       {stable}  ({align_df['abs_cos'].median():.3f})",
        f"FC-predictable (median R² >= 0.10)?        {has_fc_signal}  ({align_df['fc_r2'].median():.3f})",
        f"PC{pc_num} is sex+BV confound (R²_test >= 0.60)?  {is_confound}  ({r2_conf_te:.3f})",
        f"  -> drop in FC->R² after partialling      {r2_raw - r2_resid:+.3f}",
        f"intra-hemispheric (top << base)?           {interhemi_anti}",
        f"rich-club elevated (>1.3x base)?           {richclub_elev}",
        f"energy peaked (top-1% > 0.10)?             {energy_peak}",
        f"# stable enriched network pairs:           {len(stable_pairs)}",
        "",
    ]
    if is_confound:
        lines.append(f"=> PC{pc_num} is largely sex + brain-volume; report as demographic-confound, not structural mode.")
    elif stable and has_fc_signal and (len(stable_pairs) >= 1 or richclub_elev or interhemi_anti) and energy_peak:
        lines.append(f"=> PC{pc_num} is a stable, FC-predictable structural mode beyond demographics.")
        if len(stable_pairs) > 0:
            lines.append(f"   Stable enriched network pairs:")
            for _, r in stable_pairs.iterrows():
                lines.append(f"     - {r['net_pair']}: median enrichment {r['median_enrichment']:.2f}x")
    elif stable and has_fc_signal:
        lines.append(f"=> PC{pc_num} is FC-predictable but lacks a strong anatomical localization signature.")
    else:
        lines.append(f"=> PC{pc_num} does not pass the mechanism bar: too unstable or no clear structural story.")

    verdict = "\n".join(lines)
    print("\n" + verdict)
    (OUT_DIR / f"pc{pc_num}_verdict.txt").write_text(verdict + "\n")

print(f"\nAll outputs in {OUT_DIR}")
