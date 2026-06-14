# Conn2Conn — master findings (FC↔SC cross-modal, mechanism, downstream, robustness)

Single-document synthesis of the full experimental arc on HCP-YA (Glasser, 64,620 edges,
~957 subjects, family-aware train/val/test splits, PCA-closed-form pipeline unless noted).
Each section links the detailed findings doc + result CSVs. Branch: `adel-temp`.

**One-paragraph summary.** FC→SC connectome prediction beats SC→FC by ~1.5–1.8× and the
asymmetry is robust across all 6 metrics, all input reductions, and a named anatomical
locus (the dorsal visual-stream / dorsal-attention intra-hemispheric backbone, SC-PC3).
Downstream, FC is the *only* representation that predicts cognition above a demographic
floor; SC, brain-anatomy, and richer tractography (named-bundle r2t) add nothing. The
connectome→cognition relationship is **linearly saturated** — no nonlinear model, no
cross-modal interaction, no residual-boost, and no extra data up to n≈683 surfaces signal
linear models miss. The bottleneck is information, not method.

---

## 1. The FC↔SC asymmetry (headline) — robust and multi-metric

**Finding**: FC→SC > SC→FC, ratio ~1.5–1.8×, significant on **all 6 metrics** at
Wilcoxon p=0.001 across 10 seeds.

- Main-model PCA→PLS→PCA: demeaned_pearson ratio **1.62×**; also significant on pearson,
  top1_acc (2.56×), avg_rank (1.23×), mse (2.59×), r2 (diff +0.036). All p=0.001.
- **Robust to input reduction** (`sanity_checks/preprocessing_check/findings.md`): no
  reduction (full PLS on 64,620 edges) **1.81×**; learned PCA(256) 1.62×; three
  Johnson-Lindenstrauss random projections (Gaussian/sparse-auto/sparse-1/3)
  1.39/1.39/1.55×. All reject ratio=1.0 at p≤0.001. The learned PCA basis is not
  load-bearing; the asymmetry is a property of the data, not the pipeline.
- **Robust to K_PCA / K_PLS** (Depth 2): stable 1.5–1.9× across K_PCA∈{64,128,256}.

→ Detail: `sanity_checks/preprocessing_check/findings.md`, `model_overviews/results/FINDINGS.md`

## 2. Mechanism — the asymmetry has a named anatomical locus (SC-PC3)

**Finding**: FC predicts SC's third principal component — a stable, heritable, anatomically
specific structural mode — and that mode is the dorsal visual-stream / dorsal-attention
backbone.

- SC-PC3 (Depth 1.1): stable across 10 seeds (median |cos| 0.89), FC-predictable
  (R²≈0.22), heritable beyond demographics (AUC MZ 0.71 > DZ 0.60 > sibling 0.58; sex+BV
  confound R²≈0), ~1.5% SC variance.
- Spatially: **visual‖visual 11.7×, DAN‖DAN 8.2×, DAN‖visual 4.9×** edge enrichment
  (10/10 seeds), 62% of L2 energy in 1% of edges, 98.5% intra-hemispheric. Glass-brain:
  `further_exploration/figures/output_glassbrain/pc3_glassbrain.png`.
- Extends to PC4 (visual-heavy) and PC5 (DAN-heavy) — a small *family* of dorsal-stream
  modes, all non-confounds.
- **Caveat resolved**: SC-PC1 (largest variance) is 89% sex+brain-volume — a demographic
  confound, reported as such, NOT a structural-heritability finding.

→ Detail: `further_exploration/depth1.1_pc_stability_and_confounds.ipynb` outputs

## 3. Reliability sanity — PC3 localization is not a tractography artifact

**Finding**: the visual/DAN localization survives partialling out reconstruction
reliability proxies.

- Edge strength + inter-region distance explain 41% of |PC3|, but after partialling them
  out the residual top-200 enrichment **increases** for visual‖visual (13.3×) and stays
  high for DAN‖DAN (5.7×), DAN‖visual (5.0×).
- Independent FC scan-rescan reliability (REST1 vs REST2, per-edge r, mean 0.45) adds
  nothing beyond strength+distance (joint R² 41.2% vs 41.0%); localization unchanged.
- Caveat: FC reliability ≠ SC reliability; gold-standard SC test-retest ICC needs the HCP
  retest dMRI release (not in cache).

→ Detail: `sanity_checks/tract_check/findings.md`

## 4. Downstream cognition — FC is the ceiling, nothing else clears the floor

**Finding**: FC is the only representation that predicts NIH-Toolbox cognition above the
bv+demo floor; SC, anatomy, and tractography do not.

Median test Pearson (10 seeds), lift over bv+demo floor:
| rep | CogCrystal | CogFluid | CogTotal | clears floor? |
|---|---|---|---|---|
| **FC** | 0.451 (+0.105) | 0.342 (+0.033) | 0.451 (+0.078) | **yes** |
| SC | 0.267 (−0.079) | 0.180 (−0.130) | 0.261 (−0.111) | no |
| r2t (bundle) | 0.164 (−0.183) | 0.130 (−0.179) | 0.196 (−0.176) | no |
| r2t→synthetic-FC | 0.180 (−0.166) | 0.197 (−0.113) | 0.202 (−0.170) | no |

- After residualizing demographics, FC retains 0.37 crystallized / 0.22 fluid; every
  structural rep collapses to ~0.
- Synthetic-FC-generated-from-tractography does NOT recover FC's cognition signal.

→ Detail: `tractography_predict/findings.md`, `findings_in_depth.md`

## 5. Richer tractography (r2t named bundles) is a dead end

**Finding**: the region-to-tract bundle representation predicts FC *worse* than counts,
adds nothing marginal, and contains no cognition signal.

- r2t→FC demeaned_pearson 0.049 vs count-SC 0.085 (counts win on every metric).
- Marginal Δ([SC‖r2t]−SC)→FC = −0.001 (p=0.98): count-SC is a sufficient statistic.
- The FC↔SC asymmetry actually *amplifies* in the bundle representation (FC→r2t 2.26× vs
  FC→SC 1.62× on demeaned_pearson) — but it's metric-dependent (fails on cross-scale mse).
- SC-PC3 does not map to any single r2t bundle mode (best Spearman −0.32): the dorsal-
  stream backbone is a count-edge phenomenon the 66-bundle atlas is too coarse to express.

→ Detail: `tractography_predict/findings.md`, `findings_in_depth.md`

## 6. Nonlinear sanity — the relationship is linearly saturated (4 independent tests)

**Finding**: no nonlinear method extracts signal linear models miss, across four
orthogonal axes. The connectome→cognition ceiling is structural, not a method limit.

| Test | What it probes | Result |
|---|---|---|
| **N1–N3** model class (HGB, KernelRidge) | does a flexible estimator beat linear? | NULL (KR preserves FC signal, finds nothing in tractography; HGB overfits at n≈683) |
| **N4** residual-boost (OOF) | hand the model the linear answer free, learn only the residual | cognition NULL; reconstruction +0.005 real-but-negligible (p=0.001) |
| **N5** multimodal sink | cross-modal FC×SC×r2t interactions | NULL; combining modalities is *worse* than FC alone (−0.05 to −0.08) |
| **N6** data-scaling curve | does the nonlinear gap grow with n? | NULL — gap flat/≤0 at n=100→683; **structural ceiling, not data-limited** |

- The N6 nuance (logged so it isn't misread): a positive Spearman *slope* on the
  reconstruction gap is an overfitting-penalty vanishing, NOT signal — the gap is negative
  at every n and asymptotes at zero, never positive. Linear performance *does* grow with n
  (the curve works); the nonlinear *advantage* stays at zero.
- Asymmetry ratios identical under linear vs KernelRidge (SC 1.62→1.64, r2t 2.26→2.27):
  the asymmetry is model-class robust too.

→ Detail: `non-linear-sanity-check/findings_nonlinear.md`, `findings_residual.md`,
`findings_scaling.md`, `DESIGN_residual_learning.md`

---

## Writeup-ready claims (each backed above)

1. **FC→SC connectome prediction is asymmetrically easier than SC→FC** (~1.5–1.8×),
   robust across all 6 metrics, all linear reductions (incl. none and random projection),
   and K choices. p=0.001, 10 seeds.
2. **The asymmetry is mechanistically localized** to a stable, heritable-beyond-
   demographics dorsal visual-stream / DAN intra-hemispheric structural backbone
   (SC-PC3+PC4+PC5), and this localization survives reliability partialling.
3. **FC uniquely carries non-demographic cognition signal**; SC, FreeSurfer anatomy, and
   richer tractography do not clear the demographic floor, and synthetic FC from
   tractography does not recover it.
4. **Count-SC is a sufficient statistic** for cross-modal prediction; the named-bundle
   representation is strictly worse and adds nothing.
5. **The connectome→cognition relationship is linearly saturated** — robust to model
   class, residual-boosting, cross-modal interactions, and sample size to n≈683. The
   ceiling is structural/information, not capacity or data; a bigger model is not
   indicated.

## Honest caveats (carried in the detail docs)

- Single parcellation (Glasser); single cohort (HCP-YA); n≈683 train.
- SC test-retest ICC not available (FC reliability used as proxy cross-check).
- Cannot rule out a tiny positive nonlinear gap at n≈10⁴ (Biobank scale), but the
  data shows convergence to zero, not growth above it; if chased, the move is a bigger
  cohort, never a bigger model.
- KernelRidge hyperparameters fixed (median-heuristic gamma, alpha=1.0); decision margins
  wide vs effect sizes.

## Map of the work

| Area | Directory |
|---|---|
| Main closed-form notebook + Phase 1/2 | `model_overviews/` (`results/FINDINGS.md`) |
| Mechanism (PC3 spectral, confounds, glass-brain) | `further_exploration/` |
| Reduction-axis robustness | `sanity_checks/preprocessing_check/` |
| Tractography reliability of PC3 | `sanity_checks/tract_check/` |
| Tractography representations (r2t) + downstream | `tractography_predict/` |
| Nonlinear / residual / sink / scaling | `non-linear-sanity-check/` |
