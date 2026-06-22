#!/usr/bin/env python3
"""Finalize F8 — cross-seed alignment, confound summary, PC3 localization + verdict, per parc.

Faithful port of depth1 cell 9 + depth1.1 cells 4/8/10/11. Torch-free: consumes the per-unit
npz (loadings, fc_r2, AUCs, confounds, node strength) + the atlas dseg. Writes merged CSVs with
a `parcellation` column and prints the automated verdict. Regression guard: Glasser must match
the notebook's saved per-PC / localization numbers.

    python finalize_f8.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fm_common as fm
import _f8_common as f8

OUT_PARTS = _HERE / "outputs" / "parts_f8"
OUT = _HERE / "outputs"
PRIMARY_K = 200
STABLE_COS = 0.85


def load_parc(parc):
    files = sorted((OUT_PARTS / parc).glob("f8_seed*.npz"),
                   key=lambda p: int(p.stem.split("seed")[1]))
    assert files, f"no f8_seed*.npz for {parc}"
    per = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        s = int(d["seed"][0])
        per[s] = {k: d[k] for k in d.files}
    return per, sorted(per)


def per_pc_table(per, seeds, parc):
    """All-seeds per-PC mechanism table; seed-0 slice == depth1 per_pc_heritability/fc_pred + confound."""
    rows = []
    for s in seeds:
        p = per[s]
        for k in range(f8.K_PCA_TOP):
            rows.append(dict(parcellation=parc, seed=s, pc=k + 1,
                             explained_var_ratio=float(p["expl_var"][k]),
                             FC_to_PC_R2=float(p["fc_r2"][k]),
                             AUC_MZ=float(p["auc_mz"][k]), AUC_DZ=float(p["auc_dz"][k]),
                             AUC_sibling=float(p["auc_sb"][k]),
                             confound_R2_test=float(p["conf_r2"][k])))
    return pd.DataFrame(rows)


def stability(per, seeds, parc):
    """Align each anchor PC (seed0) to its best |cos| match per seed; medians across seeds."""
    anchor = per[seeds[0]]["loadings"]
    rows = []
    for k in range(f8.K_PCA_TOP):
        cos, fc, mz, dz, sb, var = [], [per[seeds[0]]["fc_r2"][k]], [per[seeds[0]]["auc_mz"][k]], \
            [per[seeds[0]]["auc_dz"][k]], [per[seeds[0]]["auc_sb"][k]], [per[seeds[0]]["expl_var"][k]]
        for s in seeds[1:]:
            sim = f8.cosine_sim_matrix(anchor[k:k + 1], per[s]["loadings"])[0]
            j = int(np.argmax(np.abs(sim)))
            cos.append(float(abs(sim[j])))
            fc.append(per[s]["fc_r2"][j]); mz.append(per[s]["auc_mz"][j])
            dz.append(per[s]["auc_dz"][j]); sb.append(per[s]["auc_sb"][j]); var.append(per[s]["expl_var"][j])
        rows.append(dict(parcellation=parc, anchor_pc=k + 1,
                         median_abs_cos=float(np.median(cos)), min_abs_cos=float(np.min(cos)),
                         median_expl_var=float(np.median(var)),
                         median_FC_to_PC_R2=float(np.median(fc)),
                         median_AUC_MZ=float(np.nanmedian(mz)), median_AUC_DZ=float(np.nanmedian(dz)),
                         median_AUC_sibling=float(np.nanmedian(sb))))
    return pd.DataFrame(rows)


def pc3_localization(per, seeds, parc):
    """Align each seed's PC3 to seed0 PC3, localize using saved node strength + atlas; aggregate."""
    atlas, netcol = f8.load_atlas(parc)
    n_edges = per[seeds[0]]["loadings"].shape[1]
    N = f8.n_regions_from_edges(n_edges)
    assert len(atlas) == N, f"{parc}: atlas {len(atlas)} != inferred {N} regions"
    ti, tj = np.triu_indices(N, k=1)
    all_pair_counts, interhemi_base = f8.base_rates(atlas, netcol, ti, tj, n_edges)

    anchor = per[seeds[0]]["loadings"][2]
    if anchor[np.argmax(np.abs(anchor))] < 0:
        anchor = -anchor
    loc_rows, enr_rows = [], []
    for s in seeds:
        L = per[s]["loadings"]
        sim = (L @ anchor) / (np.linalg.norm(L, axis=1) * np.linalg.norm(anchor) + 1e-12)
        j = int(np.argmax(np.abs(sim)))
        pc3 = np.sign(sim[j]) * L[j]
        if pc3[np.argmax(np.abs(pc3))] < 0:
            pc3 = -pc3
        is_hub = per[s]["is_hub"].astype(bool)
        rcb = float(per[s]["richclub_base"][0])
        loc, enr = f8.pc3_localization(pc3, atlas, netcol, ti, tj, all_pair_counts,
                                       interhemi_base, is_hub, rcb)
        for r in loc:
            r.update(parcellation=parc, seed=s, anchor_match_signedcos=float(sim[j]))
            loc_rows.append(r)
        for r in enr:
            r.update(parcellation=parc, seed=s)
            enr_rows.append(r)
    loc_df = pd.DataFrame(loc_rows)
    enr_df = pd.DataFrame(enr_rows)
    prim = enr_df[enr_df.K == PRIMARY_K]
    agg = prim.groupby("net_pair").agg(
        n_seeds=("seed", "nunique"), median_enrichment=("enrichment", "median"),
        min_enrichment=("enrichment", "min"), median_n_obs=("n_obs", "median"),
    ).reset_index().sort_values("median_enrichment", ascending=False)
    agg.insert(0, "parcellation", parc)
    return loc_df, agg


def verdict(parc, stab_df, loc_df, agg):
    prim = loc_df[loc_df.K == PRIMARY_K]
    energy = prim.energy_top1pct_frac.median()
    interhemi_elev = prim.interhemi_frac_top.median() > 1.30 * prim.interhemi_frac_base.iloc[0]
    richclub_elev = prim.richclub_frac_top.median() > 1.30 * prim.richclub_frac_base.median()
    stable = agg[(agg.n_seeds >= 8) & (agg.min_enrichment >= 1.5) & (agg.median_enrichment >= 2.0)]
    pc3 = stab_df[stab_df.anchor_pc == 3].iloc[0]
    print(f"\n[{parc}] PC3: median|cos|={pc3.median_abs_cos:.3f}  FC->R2={pc3.median_FC_to_PC_R2:.3f}  "
          f"AUC_sib={pc3.median_AUC_sibling:.3f}  energy_top1%={energy:.3f}  "
          f"interhemi_elev={interhemi_elev}  richclub_elev={richclub_elev}  stable_pairs={len(stable)}")
    if energy <= 0.10 and len(stable) == 0:
        print("  -> (c) SMEAR: no coherent localization.")
    elif len(stable) >= 1 and (interhemi_elev or richclub_elev):
        sig = "interhemispheric/callosal" if interhemi_elev else "rich-club/hub"
        print(f"  -> (a) HEADLINE: PC3 is a {sig} structural mode ({len(stable)} stable enriched pairs).")
    elif len(stable) >= 1:
        print("  -> (b) DISTRIBUTED: enriched in specific network pairs, no interhemi/rich-club signature.")
    else:
        print("  -> mixed: read the enrichment table manually.")


def main():
    parcs = [p for p in fm.PARCELLATIONS if (OUT_PARTS / p).exists()]
    assert parcs, f"no F8 part dirs under {OUT_PARTS}"
    OUT.mkdir(parents=True, exist_ok=True)
    per_pc_all, stab_all, loc_all, agg_all = [], [], [], []
    for parc in parcs:
        per, seeds = load_parc(parc)
        per_pc_all.append(per_pc_table(per, seeds, parc))
        stab = stability(per, seeds, parc); stab_all.append(stab)
        loc, agg = pc3_localization(per, seeds, parc); loc_all.append(loc); agg_all.append(agg)
        print(f"[{parc}] pooled {len(seeds)} seeds")
        verdict(parc, stab, loc, agg)

    pd.concat(per_pc_all, ignore_index=True).to_csv(OUT / "f8_per_pc.csv", index=False)
    pd.concat(stab_all, ignore_index=True).to_csv(OUT / "f8_stability.csv", index=False)
    pd.concat(loc_all, ignore_index=True).to_csv(OUT / "f8_pc3_localization.csv", index=False)
    pd.concat(agg_all, ignore_index=True).to_csv(OUT / "f8_pc3_enrichment_agg.csv", index=False)
    print(f"\nwrote f8_per_pc.csv / f8_stability.csv / f8_pc3_localization.csv / "
          f"f8_pc3_enrichment_agg.csv to {OUT}")

    # regression guard: Glasser seed-0 per-PC vs the notebook
    nb = _HERE.parents[1] / ("notebooks-FC_to_SC-experimental/model_overviews/results/"
                             "local_results/further_exploration/depth1_spectral_mechanism")
    if "Glasser" in parcs and (nb / "per_pc_fc_predictability.csv").exists():
        ref = pd.read_csv(nb / "per_pc_fc_predictability.csv")
        got = pd.concat(per_pc_all, ignore_index=True)
        g0 = got[(got.parcellation == "Glasser") & (got.seed == 0)].sort_values("pc")
        err = float((ref.FC_to_PC_R2.values - g0.FC_to_PC_R2.values).__abs__().max())
        print(f"\n[guard] Glasser seed0 FC->PC R2 vs notebook: max|err|={err:.2e} "
              f"-> {'PASS' if err < 1e-3 else 'WARN'}")


if __name__ == "__main__":
    main()
