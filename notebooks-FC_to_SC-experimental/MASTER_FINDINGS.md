# Conn2Conn — master findings + evidence index

Single-document synthesis of the full experimental arc on HCP-YA (Glasser parcellation,
64,620 edges, ~683 train / ~195 test, family-aware splits, closed-form PCA→PLS unless
noted). **Every finding below lists the exact repo-relative `.md` / `.csv` evidence files
backing it.** Paths are relative to the repo root (`notebooks-FC_to_SC-experimental/…`).
Branch `adel-temp`.

**One-paragraph summary.** Predicting structure from function (FC→SC) reliably beats the
reverse (SC→FC) by ~1.4–1.8× in individual-level prediction; the effect survives anatomy/
demographic controls, every metric/estimator/reduction tested, and localizes to a stable,
heritable visual / dorsal-attention structural mode (SC-PC3). SC is anatomy-driven, FC is
demographics-driven; a predicted connectome inherits its *source* modality's information,
so the ~1.5× reconstruction asymmetry becomes ~2–2.5× downstream-utility asymmetry. A
brain-volume + demographics (bv+demo) baseline accounts for most apparent cross-modal
cognition signal and is proposed as a required baseline. Richer tractography (named-bundle
r2t) adds nothing, and the connectome→cognition relationship is linearly saturated
(structural ceiling, not method- or sample-size-limited). Effect sizes are research-grade,
not diagnostic-grade.

---

## PART 1 — FINDINGS (each with evidence trail)

### F1 — The FC→SC > SC→FC asymmetry (motivating finding)
Raw demeaned-r FC→SC ≈ 0.134 vs SC→FC ≈ 0.085 (~1.5×); survives anatomy+demographic
stripping (~1.4×); holds on all 6 metrics (avg_rank ratio 1.25× most stable); clears the
1.15 gate in 9–10/10 seeds. Reproduces Krakencoder's 0.16/0.09 with a deterministic model.
- **Evidence (CSV)**:
  - `model_overviews/results/notebook_snapshot/phase1/section2_asymmetry.csv` — single-seed asymmetry, all residual bases, ratios for demeaned/top1/avg_rank
  - `model_overviews/results/Exp7_stringent_ratio_10_seed/aggregate_full_panel_ratios.csv` — 10-seed ratios, all 6 metrics, target_only + double_sided, CIs + n_above_1.15
  - `model_overviews/results/Exp7_stringent_ratio_10_seed/per_seed.csv` + `full_panel_seed_{0..9}.csv` — per-seed raw
  - `model_overviews/results/Step11_grand_10seed/aggregate_ratios.csv` — 10-seed ratios across every basis (none/bv/demo/bv+demo), t/p vs 1.00 and 1.15
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high.* Robustness in Appendix A1–A3.

### F2 — Modality dissociation
SC is anatomy-driven (bv→SC 0.162, "anatomy wins"); FC is demographics-driven (demo→FC
0.114, "demo wins"). Same data, opposite drivers, 10/10 seeds.
- **Evidence (CSV)**: `model_overviews/results/notebook_snapshot/phase1/section1_baselines.csv` — bv / demo / bv+demo → SC and → FC, demeaned_r + top1 + avg_rank
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high.*

### F3 — Imputation inherits the source modality (utility asymmetry)
pred_SC-from-FC beats real SC for cognition (1.25–1.45×); pred_FC-from-SC loses 40–50%.
The ~1.5× reconstruction asymmetry becomes **2.06–2.54× downstream-utility asymmetry**
(CogTotal 2.06×, Fluid 2.43×, Cryst 2.54×). A predicted connectome carries its *source*
modality's information.
- **Evidence (CSV)**: `model_overviews/results/downstream_prediction_phase2/aggregate.csv` — per-input (obs_/pred_ SC/FC), per-target cognition scores + CIs (pred/obs ratios derived here)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high — novel framing.*

### F4 — FC's cognition signal is real; SC's is mostly demographics
Fraction surviving bv+demo removal: obs_FC 60/60/73% (Total/Fluid/Cryst) vs obs_SC
27/18/31%.
- **Evidence (CSV)**:
  - `model_overviews/results/downstream_prediction_phase2/aggregate.csv` — obs_FC / obs_SC / *_resid_bvdemo inputs per target
  - `model_overviews/results/downstream_prediction_phase2/perm_vs_bvdemo.csv` — paired permutation test vs bv+demo, delta_mean, p_fdr (incl. the sex/age leak diagnostics)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high on direction.*

### F5 — Clinical cost–benefit (the citable core)
T1→bv+demo r=0.353 (cheap, ~universal); +resting fMRI (obs FC) 0.434 (**only real gain**);
+diffusion (obs SC) 0.258 (**below the free baseline**). → acquire fMRI, skip diffusion;
report a bv+demo baseline. FC significantly beats bv+demo on crystallized/total (paired
permutation test, not CI-overlap).
- **Evidence (CSV)**:
  - `model_overviews/results/downstream_prediction_phase2/aggregate.csv` — bv+demo / obs_FC / obs_SC per target, mean + ci_lo/ci_hi
  - `model_overviews/results/downstream_prediction_phase2/perm_vs_bvdemo.csv` — p_vs_bvdemo, p_fdr (the correct "beats baseline" test)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high direction; magnitudes medium (n=195 test).*

### F6 — Predicted connectomes carry heritable family signal
pred_SC_resid_bvdemo separates siblings from strangers at AUC **0.810** vs bv+demo baseline
0.563 (Δ +0.247). FC→SC prediction captures family-specific wiring beyond shared anatomy.
- **Evidence (CSV)**: `model_overviews/results/family_structure_phase2/aggregate_auc.csv` — per-variant × relation (MZ/DZ/sibling/unrelated) AUC + permutation p + FDR + sig flag
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: medium-high — from an older run; flagged for regeneration (Appendix, close-out #1).*

### F7 — The predictor / identifier tradeoff
combined_pred_SC (optimized for reconstruction) wins on cognition but collapses to chance
on sibling separation (AUC 0.505): the bv+demo content dominates the reconstruction
objective and drowns the within-family signal. Reconstruct OR discriminate, not both from
one loss.
- **Evidence (CSV)**:
  - `model_overviews/results/family_structure_phase2/aggregate_auc.csv` — combined_pred_SC row (sibling 0.505) vs pred_SC_resid_bvdemo (0.810)
  - `model_overviews/results/local_results/further_exploration/depth2_robustness_sensitivity/combined_followup.csv` — 4-perturbation diagnostic confirming the collapse is structural (identical ~0.538), not a bug
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`; `sanity_checks/preprocessing_check/findings.md` (Section C perturbations)
- *Confidence: high.*

### F8 — Mechanism: FC predicts a visual / dorsal-attention heritable structural mode (SC-PC3)
SC's third principal mode (~1.5% variance): stable across 10 seeds (cos 0.89),
FC-predictable (R²≈0.22), heritable beyond demographics (AUC MZ 0.71 > DZ 0.60 > sibling
0.58), spatially concentrated (62% energy in 1% of edges), within-hemisphere visual / DAN
(enrichment 11.7× / 8.2× / 4.9×). Distributed across PC3–PC5. PC1 is a 89% sex+BV confound.
- **Evidence (CSV) — spectral / predictability**:
  - `model_overviews/results/local_results/further_exploration/depth1_spectral_mechanism/synthesis_cross_tab.csv` — per-PC explained-var × heritability AUC × FC→PC R²
  - `…/depth1_spectral_mechanism/per_pc_heritability.csv`, `…/per_pc_fc_predictability.csv`, `…/pc2_top_edges.csv`
  - `…/depth1_spectral_mechanism/sc_pc_loadings.npy`, `…/sc_pca_mean.npy` — raw PC loadings
- **Evidence (CSV) — 10-seed stability, confound, localization**:
  - `…/depth1.1_pc_stability_and_confounds/stability_aligned_to_seed0.csv` — cross-seed loading-cosine alignment, median |cos|, FC→R², AUCs per mode
  - `…/depth1.1_pc_stability_and_confounds/pc_confound_r2.csv` — OLS [sex‖bv] → PC_k (PC1 89%, PC3 ≈0)
  - `…/depth1.1_pc_stability_and_confounds/pc3_enrichment_agg.csv`, `pc3_enrichment_per_seed.csv`, `pc3_localization_per_seed.csv` — Yeo7 enrichment, interhemi, rich-club, energy
  - `…/depth1.1_pc_stability_and_confounds/pc3_top50_edges_seed0_labeled.csv` — top edges w/ region+network labels
- **Evidence (CSV) — PC4/PC5 extension**: `further_exploration/pc4_pc5_results/pc{4,5}_stability_aligned.csv`, `pc{4,5}_confound.csv`, `pc{4,5}_enrichment_agg.csv`, `pc{4,5}_localization_per_seed.csv`, `pc{4,5}_top30_edges.csv`
- **Evidence (figure + edges)**: `further_exploration/figures/output_glassbrain/pc3_glassbrain.png`, `further_exploration/figures/output_glassbrain/pc3_top200_edges_labeled.csv`
- **Evidence (MD)**: `further_exploration/README.md`
- *Confidence: medium-high — single parcellation (hardening target).*

### F9 — Richer tractography (named-bundle r2t) is a dead end *(arc beyond the PDF)*
r2t bundle profile predicts FC worse than counts (0.049 vs 0.085 demeaned-r) on every
metric; adds nothing marginal over SC (Δ=−0.001, p=0.98); count-SC is a sufficient
statistic. SC-PC3 doesn't map to any single bundle mode (best Spearman −0.32). r2t carries
no cognition above the bv+demo floor; synthetic-FC-from-r2t doesn't recover FC's signal.
- **Evidence (CSV)**:
  - `tractography_predict/e1_source_rep_results.csv` — SC / r2t / r2t_corr / SC_r2t / kitchen_sink → FC, all 6 metrics, 10 seeds
  - `tractography_predict/e2_asymmetry_summary.csv` (+ `e2_asymmetry_results.csv`) — asymmetry per rep × metric
  - `tractography_predict/e3_marginal_summary.csv` (+ `e3_marginal_results.csv`) — paired Δ(SC_r2t − SC)
  - `tractography_predict/e4_r2t_pc_stability.csv`, `e4_sc_pc3_to_r2t_pc_projection.csv`, `e4_r2t_top_bundles_per_mode.csv` — bundle-mode analysis + SC-PC3 projection
  - `tractography_predict/e5_downstream_results.csv` + `e5_downstream_summary.csv` — cognition per rep incl. r2t→synthFC substitution
  - `tractography_predict/tractography_synthesis.csv` — compact verdict table
- **Evidence (MD)**: `tractography_predict/findings.md`, `tractography_predict/findings_in_depth.md`, `tractography_predict/README.md`
- *Confidence: high.*

### F10 — The connectome→cognition relationship is linearly saturated *(arc beyond the PDF)*
Four orthogonal nonlinear tests all null: model class (KernelRidge/HGB), residual-boost
(OOF), cross-modal sink, and the data-scaling curve (gap flat n=100→683 → structural, not
sample-size). Closes the PDF's "re-run under KRR" close-out item; only MLP unrun (argued
against). Asymmetry ratios identical linear vs KRR (SC 1.62→1.64).
- **Evidence (CSV)**:
  - N1–N3 (plain nonlinear): `non-linear-sanity-check/n1_cognition_summary.csv`, `n2_reconstruction_summary.csv`, `n3_marginal_summary.csv` (+ `*_results.csv`), `nonlinear_synthesis.csv`
  - N4 (residual-boost): `non-linear-sanity-check/n4_cog_summary.csv`, `n4_recon_summary.csv` (+ `*_results.csv`), `residual_synthesis.csv`
  - N5 (cross-modal sink): `non-linear-sanity-check/n5_cog_summary.csv`, `n5_recon_summary.csv` (+ `*_results.csv`)
  - N6 (data-scaling): `non-linear-sanity-check/n6_scaling_summary.csv`, `n6_scaling_results.csv`, `scaling_synthesis.csv`
- **Evidence (MD)**: `non-linear-sanity-check/findings_nonlinear.md`, `findings_residual.md`, `findings_scaling.md`, `DESIGN_residual_learning.md`, `README.md`
- *Confidence: high.*

**Meta-finding.** Every lever is capped by biology + n≈878, not method. Contribution is
neuroscience (asymmetry, non-substitutability, dissociation) + methods (the bv+demo
baseline), not a clinical biomarker.

---

## PART 2 — ROBUSTNESS & CORRECTNESS APPENDIX (with evidence)

### A1 — Reduction-axis robustness (defends F1)
Asymmetry holds across 5 reductions: no-reduction full PLS 1.81×, learned PCA 1.62×, three
JL variants 1.39–1.55×; all reject ratio=1.0 at p≤0.001. PCA does not inject the effect.
- **CSV**: `sanity_checks/preprocessing_check/reduction_axis_summary.csv` (per method × metric, FC-wins + Wilcoxon), `reduction_axis_synthesis.csv`, `method_a_results.csv` (PCA), `method_b_results.csv` (full PLS), `method_c_results.csv` (JL ×3)
- **MD**: `sanity_checks/preprocessing_check/findings.md`, `README.md`

### A2 — Estimator robustness (defends F1)
PLS 1.56× vs BayesianRidge 1.75× (amplified). Also KernelRidge ≈ PLS (F10/N2).
- **CSV**: `model_overviews/results/notebook_snapshot/phase1/exp5_br_robustness.csv`; `tractography_predict/e2_asymmetry_summary.csv` (KR vs PLS); `non-linear-sanity-check/n2_reconstruction_summary.csv`
- **MD**: `model_overviews/results/FINDINGS.md`

### A3 — K_PCA / K_PLS sensitivity (defends F1)
Stable 1.5–1.9× over K_PCA{64,128,256}; inflation only at K=512 via weaker SC→FC
denominator.
- **CSV**: `model_overviews/results/local_results/further_exploration/depth2_robustness_sensitivity/k_sweep.csv`, `estimator_comparison.csv`
- **MD**: `further_exploration/README.md`

### A4 — PC mechanism verification (defends F8)
PC2-disappearance caught (earlier "PC2 R²=0.26" → median 0.017, single-seed artifact);
PCs 1–5 stable (cos≥0.85), 6+ noise; PC1 = 89% sex+BV confound.
- **CSV**: `…/depth1.1_pc_stability_and_confounds/stability_aligned_to_seed0.csv`, `pc_confound_r2.csv`; `…/depth1_spectral_mechanism/synthesis_cross_tab.csv`

### A5 — PC3 reliability (defends F8)
Visual/DAN localization survives partialling edge strength+distance (enrichment rises
12.0→13.3×) and an independent FC scan-rescan reliability proxy (joint R² 41.2% vs 41.0%).
- **CSV**: `sanity_checks/tract_check/enrichment_residual_top200.csv`, `retest_icc_results/enrichment_residual_with_fc_reliability.csv`, `retest_icc_results/fc_reliability_summary.csv`
- **MD**: `sanity_checks/tract_check/findings.md`, `README.md`; `sanity_checks/tract_check/retest_check_note.py` (rigorous-ICC caveat)

### Negatives kept / deliberately not chased
- SC carries no non-demographic cognition signal (F4); cognition ceiling is structural
  (F10); CNN/autoencoder not chased (wrong inductive bias); rigorous HCP-retest ICC not
  run (proxy passed); genomics/clinical out of scope.

---

## PART 3 — HARDENING / CLOSE-OUT AGENDA (consolidation mode)

Open items before write-up (from `Preliminary_Results` close-out + added robustness):
1. **Regenerate every table from one consistent run** — F6 family-structure CSVs are from
   an older pass (`family_structure_phase2/aggregate_auc.csv`). The one real correctness debt.
2. **Leak guardrail** — auto-fail any downstream input predicting sex>0.99 / age>0.85
   (diagnostics already in `downstream_prediction_phase2/perm_vs_bvdemo.csv`).
3. **KRR + MLP re-run** — KRR ✅ done (F10/N-arc); MLP optional (argued against).
4. **Parcellation replication** — re-run F1/F2/F5/F8 on 4S456Parcels (cache exists) to
   kill the "Glasser artifact" objection. *Highest-value add.*
5. **Bootstrap CIs** on every headline number (ratios, lifts) alongside Wilcoxon/perm p.
6. **WandB + dataset parameterization → port a second HCP cohort → auto-regenerate figures.**

---

## MAP OF THE WORK
| Area | Directory | Key docs |
|---|---|---|
| Main closed-form notebook, Phase 0/1/2 (F1–F7) | `model_overviews/` | `results/FINDINGS.md`, `results/NEXT_STEPS.md` |
| Mechanism PC3/PC4/PC5 (F8) | `further_exploration/` | `README.md` |
| Reduction robustness (A1), combined-pred (F7) | `sanity_checks/preprocessing_check/` | `findings.md` |
| PC3 reliability (A5) | `sanity_checks/tract_check/` | `findings.md` |
| Tractography r2t (F9) | `tractography_predict/` | `findings.md`, `findings_in_depth.md` |
| Nonlinear / residual / sink / scaling (F10) | `non-linear-sanity-check/` | `findings_nonlinear.md`, `findings_residual.md`, `findings_scaling.md` |

*All quantitative values trace to the CSVs cited above (executed notebooks / SLURM runs).*
