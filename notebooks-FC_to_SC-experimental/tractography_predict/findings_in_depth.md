# Tractography-predict — in-depth findings

Companion to [`findings.md`](findings.md) with full tables, per-seed spreads, every
metric, statistical tests, and real + relative paths to every artifact. All numbers
are 10-seed medians (with [min, max] where shown) on the HCP-YA Glasser parcellation
(64,620 edges, ~957 canonical subjects, family-aware train/val/test ≈ 683/79/195),
PCA-closed-form pipeline (PCA 256 → PLS 64 → inverse-PCA unless noted).

- **Run date**: 2026-06-12
- **Branch**: `adel-temp` (commit `963dd8f` + result CSVs in `112e64e`)
- **Compute**: 6 SLURM jobs in parallel on NYU Torch (`cpu_short`), `:ro` apptainer
  overlay. e1 at 16 cpu (job 10688656); e2–retest at 4 cpu each (jobs 10689496–500)
  to fit under the 32-core per-user QOS cap.

## Contents
1. [Data representations](#1-data-representations)
2. [E1 — source-rep → FC, all metrics](#2-e1)
3. [E2 — FC↔SC asymmetry across reps, all 6 metrics](#3-e2)
4. [E3 — marginal contribution of r2t over SC](#4-e3)
5. [E4 — bundle-level PCA + does SC-PC3 live in r2t?](#5-e4)
6. [E5 — downstream cognition](#6-e5)
7. [Retest reliability cross-check](#7-retest)
8. [File index (real + relative paths)](#8-file-index)
9. [Methods notes & caveats](#9-methods)

---

## 1. Data representations <a name="1-data-representations"></a>

All derived from the same SIFT2 tractography pipeline; loaded by `_tract_setup.py`.

| Rep | Per-subject shape | Dim used | Source |
|---|---|---|---|
| `SC` | (360,360) → upper-tri | 64,620 | `base.sc_upper_triangles` — SIFT2 vol-normalized streamline counts, log1p |
| `r2t` | (360, 66) flattened | 23,760 | `base.sc_r2t_matrices` — region × 66 named anatomical bundles |
| `r2t_corr` | (360,360) → upper-tri | 64,620 | `corrcoef(r2t rows)` per subject (`base.sc_r2t_corr_matrices`) |
| `SC_r2t` | per-block | 256+256 latents | per-block PCA of [SC, r2t] then concat |
| `kitchen_sink` | per-block | 256+256+16+9 | per-block PCA of [SC, r2t] + raw-standardized [bv, demo] |
| `FC` | (360,360) → upper-tri | 64,620 | `base.fc_upper_triangles` |

The 66 bundles include arcuate, cingulum, ILF/IFOF/SLF/MdLF, uncinate, vertical
occipital, corticospinal, corpus callosum, optic/thalamic radiation, fornix,
cerebellar peduncles, cranial nerves (full list in `e4_pc3_localization_on_r2t.py`).

---

## 2. E1 — predicting FC from each structural representation <a name="2-e1"></a>

Median [min, max] across 10 seeds. Source: `e1_source_rep_results.csv` (50 rows).
Higher is better for all except `mse` (lower) and `r2` (higher, here negative).

| rep | demeaned_pearson | pearson | top1_acc | avg_rank | mse | r2 |
|---|---|---|---|---|---|---|
| **SC** | **0.0847** [0.073, 0.101] | 0.8292 | 0.0513 [0.031, 0.067] | 0.7128 [0.687, 0.744] | 0.0143 | −0.057 |
| **kitchen_sink** | **0.0942** [0.085, 0.117] | 0.8288 | 0.0487 [0.021, 0.072] | **0.7228** [0.708, 0.758] | 0.0143 | −0.057 |
| SC_r2t | 0.0791 [0.072, 0.107] | 0.8272 | 0.0359 [0.015, 0.067] | 0.7020 [0.678, 0.750] | 0.0145 | −0.064 |
| r2t | 0.0493 [0.038, 0.060] | 0.8153 | 0.0205 [0.000, 0.036] | 0.6090 [0.595, 0.638] | 0.0154 | −0.127 |
| r2t_corr | 0.0362 [0.024, 0.049] | 0.8231 | 0.0179 [0.005, 0.031] | 0.6070 [0.586, 0.636] | 0.0148 | −0.091 |

**Reading.**
- Ranking is consistent across *every* metric: `SC > SC_r2t > r2t > r2t_corr` for the
  pure-tractography reps; `kitchen_sink` (adds bv+demo) is the only thing that beats SC.
- Count-SC carries ~1.7× the demeaned_pearson of bundle-r2t (0.085 vs 0.049) and ~2.5×
  the top-1 identifiability (0.051 vs 0.021).
- `pearson` (raw row correlation) is high and flat (~0.82–0.83) for all reps — it's
  dominated by the shared population mean structure and barely discriminates; this is
  exactly why the project uses **demeaned_pearson** as the primary metric.
- Negative `r2` for all reps is expected: these closed-form predictions don't beat the
  per-edge mean on raw scale; the subject-specific signal lives in the demeaned/rank
  metrics.

---

## 3. E2 — FC↔SC asymmetry across representations, all 6 metrics <a name="3-e2"></a>

For each rep X, both directions (FC→X and X→FC) over 10 seeds. "FC-wins" is oriented so
>1 (ratio) or >0 (r2 difference) always means FC→X beats X→FC. Directionality:
higher-better metrics use ratio FC/X; `mse` (lower-better) uses ratio X/FC; `r2`
(negatives) uses difference FC−X. Source: `e2_asymmetry_summary.csv` (long format, 18 rows).

| rep | metric | FC-wins | [min, max] | Wilcoxon p | FC→X med | X→FC med |
|---|---|---|---|---|---|---|
| **SC** | demeaned_pearson | **1.621×** | [1.32, 1.86] | **0.001** | 0.1355 | 0.0847 |
| SC | pearson | 1.101× | [1.096, 1.111] | **0.001** | 0.9125 | 0.8292 |
| SC | top1_acc | 2.562× | [1.50, 5.00] | **0.001** | 0.1179 | 0.0513 |
| SC | avg_rank | 1.225× | [1.17, 1.26] | **0.001** | 0.8787 | 0.7128 |
| SC | mse | 2.594× | [2.52, 2.64] | **0.001** | 0.0055 | 0.0143 |
| SC | r2 (diff) | +0.036 | [+0.017, +0.051] | **0.001** | −0.022 | −0.057 |
| **r2t** | demeaned_pearson | **2.264×** | [1.87, 2.83] | **0.001** | 0.1113 | 0.0493 |
| r2t | pearson | 1.037× | [1.029, 1.040] | **0.001** | 0.8453 | 0.8153 |
| r2t | top1_acc | 1.500× | [0.67, 3.00] | 0.039 | 0.0282 | 0.0205 |
| r2t | avg_rank | 1.185× | [1.12, 1.23] | **0.001** | 0.7212 | 0.6090 |
| r2t | mse | ⚠ 2.8e-8 | — | 1.000 (n.s.) | 536456 | 0.0154 |
| r2t | r2 (diff) | +0.446 | [−0.35, +0.74] | 0.002 | 0.323 | −0.127 |
| r2t_corr | demeaned_pearson | 1.634× | [1.18, 2.39] | **0.001** | 0.0582 | 0.0362 |
| r2t_corr | pearson | 0.981× | [0.975, 0.990] | 1.000 (n.s.) | 0.8071 | 0.8231 |
| r2t_corr | top1_acc | 1.825× | [0.40, 5.00] | 0.063 | 0.0256 | 0.0179 |
| r2t_corr | avg_rank | 1.175× | [1.11, 1.24] | **0.001** | 0.7092 | 0.6070 |
| r2t_corr | mse | 0.854× | [0.83, 0.87] | 1.000 (n.s.) | 0.0174 | 0.0148 |
| r2t_corr | r2 (diff) | −6.88 | [−1040, +0.025] | 0.990 (n.s.) | −6.98 | −0.091 |

**Reading.**
- **SC asymmetry is bulletproof: FC→SC beats SC→FC on all 6 metrics, every one
  p=0.001.** Strongest multi-metric statement of the project's headline.
- **r2t amplifies the asymmetry on the shape/rank metrics** (demeaned_pearson 2.26× vs
  SC's 1.62×) but is metric-dependent: it passes demeaned_pearson, pearson, avg_rank,
  r2-diff, and (weakly) top1, but **fails mse**.
- ⚠ **The r2t `mse` cell is uninterpretable**: FC→r2t predicts the *raw bundle counts*
  (median target magnitude ~5×10⁵), while r2t→FC predicts FC (~0–1). The two MSEs live
  on incomparable scales (536456 vs 0.015), so their ratio is meaningless — not a real
  "FC loses" result. Report demeaned_pearson/rank for r2t, not mse.
- r2t_corr is a weak/mixed asymmetry carrier (fails pearson, mse, r2).

---

## 4. E3 — marginal contribution of r2t over SC <a name="4-e3"></a>

Paired Δ per seed between `SC → FC` and `[SC‖r2t] → FC`, both via identical per-block
PCA (SC represented the same way in both arms). Source: `e3_marginal_summary.csv`.

| n_seeds | median dp SC | median dp SC_r2t | median Δ | [min, max] Δ | Wilcoxon p (two-sided) | p (one-sided, SC_r2t greater) |
|---|---|---|---|---|---|---|
| 10 | 0.0847 | 0.0791 | **−0.0013** | [−0.0119, +0.0061] | 0.049 | 0.981 |

**Reading.** Adding the bundle representation does **not** improve FC prediction over
counts — Δ is essentially zero and slightly negative (two-sided p=0.049 indicates
SC_r2t is marginally *worse*, from the extra uninformative latents). **Count-SC is a
sufficient statistic for cross-modal prediction.**

---

## 5. E4 — bundle-level PCA + does SC-PC3 live in r2t? <a name="5-e4"></a>

### 5a. r2t PC stability across 10 seeds
Source: `e4_r2t_pc_stability.csv`. (median |cos| of loadings aligned to seed-0 anchor.)

| anchor mode | median \|cos\| | min \|cos\| | median expl-var | median FC→r2t-PC R² |
|---|---|---|---|---|
| 1 | 0.989 | 0.976 | 0.0646 | 0.100 |
| 2 | 0.980 | 0.958 | 0.0400 | **0.153** |
| 3 | 0.948 | 0.894 | 0.0256 | 0.037 |
| 4 | 0.880 | 0.711 | 0.0220 | 0.029 |
| 5 | 0.792 | 0.669 | 0.0208 | 0.085 |
| 6–10 | 0.68–0.75 | 0.26–0.66 | ~0.015–0.018 | ≤0.045 (some negative) |

Modes 1–3 are highly stable (|cos| ≥ 0.95); 4–5 moderate; 6+ unstable. r2t-mode 2 is the
most FC-predictable (R²=0.15).

### 5b. Top (region, bundle) loadings for stable modes
Source: `e4_r2t_top_bundles_per_mode.csv` (top 30/mode; top 5 shown).

- **Mode 3** — IFOF + ILF, occipital seeds: `Left_V1 × IFOF` (+0.59), `Right_V1 × IFOF`
  (−0.45), `Left_V1 × ILF` (+0.19), `Left_PGp × ILF` (−0.18). → occipital-to-frontal
  visual association stream.
- **Mode 4** — arcuate, temporal seeds: `Left_TE1p × Arcuate` (+0.39), `Right_TE1p ×
  Arcuate` (−0.31), `Right_TE2a × Arcuate` (+0.21). → language/arcuate axis.
- **Mode 5** — MdLF + IFOF: `Right_A4 × MdLF` (+0.27), `Right_a47r × IFOF` (−0.25),
  `Right_VIP × MdLF` (+0.19). → middle-longitudinal/parietal.

### 5c. Does SC-PC3 (dorsal-stream backbone from depth1.1) map to an r2t mode?
Source: `e4_sc_pc3_to_r2t_pc_projection.csv` — Spearman of seed-0 SC-PC3 subject scores
vs each r2t-mode score (test set, n≈195).

| r2t mode | Spearman vs SC-PC3 | p |
|---|---|---|
| 10 | −0.320 | 5e-6 |
| 7 | −0.226 | 0.0015 |
| 4 | +0.192 | 0.0071 |
| 5 | +0.174 | 0.0148 |
| (others) | \|ρ\| ≤ 0.16 | — |

**Reading.** No single r2t mode carries SC-PC3 (best \|ρ\|=0.32, and it's a high/unstable
mode). **The dorsal visual-stream / DAN backbone is a count-SC edge phenomenon** that
the coarse 66-bundle atlas cannot express — consistent with E1 (counts > bundles) and the
depth1.1 localization being an intra-parietal/occipital *edge* pattern.

---

## 6. E5 — downstream cognition <a name="6-e5"></a>

Predict NIH-Toolbox composites; rep → PCA(256) → BayesianRidge → score; test Pearson,
10-seed median. `pearson_resid` = predicting cognition residualized on bv+demo (isolates
non-demographic signal). `lift_over_bvdemo_raw` = pearson_raw − bv+demo floor.
Source: `e5_downstream_results.csv` (210 rows) + `e5_downstream_summary.csv`.

### CogCrystallized
| rep | pearson_raw | pearson_resid | lift over bv+demo |
|---|---|---|---|
| **FC** | **0.451** | **0.374** | **+0.105** |
| bv+demo (floor) | 0.346 | — | 0 |
| SC | 0.267 | 0.086 | −0.079 |
| r2t→synthFC | 0.180 | — | −0.166 |
| r2t | 0.164 | 0.027 | −0.183 |
| SC_r2t | 0.160 | 0.028 | −0.187 |
| r2t_corr | 0.103 | −0.009 | −0.244 |

### CogFluid
| rep | pearson_raw | pearson_resid | lift over bv+demo |
|---|---|---|---|
| **FC** | **0.342** | **0.218** | **+0.033** |
| bv+demo (floor) | 0.309 | — | 0 |
| r2t→synthFC | 0.197 | — | −0.113 |
| SC | 0.180 | 0.006 | −0.130 |
| r2t_corr | 0.157 | 0.079 | −0.152 |
| SC_r2t | 0.135 | 0.072 | −0.174 |
| r2t | 0.130 | 0.061 | −0.179 |

### CogTotal
| rep | pearson_raw | pearson_resid | lift over bv+demo |
|---|---|---|---|
| **FC** | **0.451** | **0.264** | **+0.078** |
| bv+demo (floor) | 0.372 | — | 0 |
| SC | 0.261 | 0.040 | −0.111 |
| r2t→synthFC | 0.202 | — | −0.170 |
| r2t | 0.196 | 0.049 | −0.176 |
| SC_r2t | 0.196 | 0.048 | −0.177 |
| r2t_corr | 0.151 | −0.029 | −0.221 |

**Reading.**
- **FC is the only representation that beats the bv+demo floor** on all three targets
  (+0.105 / +0.033 / +0.078). After residualizing demographics it retains 0.37 / 0.22 /
  0.26 — real non-demographic cognition signal.
- **Every tractography representation falls below the demographic floor** (negative lift):
  age+sex+brain-volume predicts cognition better than count-SC, bundle-r2t,
  bundle-similarity, or their combination.
- **Substitution fails**: `r2t → synthetic-FC → cognition` (0.18–0.20 raw) does not
  recover FC's signal. Synthetic FC generated from tractography is not a cognitive
  biomarker — FC's cognition-predictive structure is not reconstructable from
  tractography.

---

## 7. Retest reliability cross-check <a name="7-retest"></a>

FC scan-rescan (REST1 vs REST2) per-edge Pearson across 957 subjects, as an independent
data-driven reliability proxy added to the strength+distance partial from `tract_check`.
Source: `../sanity_checks/tract_check/retest_icc_results/`.

- Per-edge FC reliability r: mean **0.452**, median 0.441, p5 0.229, p95 0.697, 0 NaN
  edges (`fc_reliability_summary.csv`, `fc_edge_reliability_pearson.npy`).
- Adding FC reliability to the |PC3| ~ strength+distance partial: joint R² = **41.2%**
  vs strength+distance-only **41.0%** → FC reliability adds essentially nothing.
- PC3 visual/DAN localization under the 3-proxy residual: **visual‖visual 13.3×,
  DAN‖DAN 5.7×, DAN‖visual 5.0×** — unchanged from the 2-proxy result.
- **Conclusion**: PC3 localization is not a reliability artifact (independent
  confirmation). Caveat: FC reliability ≠ SC reliability; gold-standard SC ICC needs the
  HCP retest dMRI release (not in this cache; see `retest_check_note.py`).

---

## 8. File index (real + relative paths) <a name="8-file-index"></a>

Real root: `/Users/user/projects/Conn2Conn/`
Torch root: `/scratch/ans9868/Conn2Conn/`
All paths below are relative to repo root.

### Scripts
| Relative path | Purpose |
|---|---|
| `notebooks-FC_to_SC-experimental/tractography_predict/_tract_setup.py` | data loader (adds r2t/r2t_corr), `block_pca_pls_predict`, panel eval |
| `notebooks-FC_to_SC-experimental/tractography_predict/e1_source_rep_comparison.py` | E1 |
| `notebooks-FC_to_SC-experimental/tractography_predict/e2_asymmetry_across_reps.py` | E2 |
| `notebooks-FC_to_SC-experimental/tractography_predict/e3_marginal_r2t.py` | E3 |
| `notebooks-FC_to_SC-experimental/tractography_predict/e4_pc3_localization_on_r2t.py` | E4 |
| `notebooks-FC_to_SC-experimental/tractography_predict/e5_downstream_cognition.py` | E5 |
| `notebooks-FC_to_SC-experimental/tractography_predict/synthesize_tractography.py` | E1/E2/E3 synthesis |
| `notebooks-FC_to_SC-experimental/tractography_predict/run_one.sbatch` | parameterized parallel runner (`:ro` overlay) |
| `notebooks-FC_to_SC-experimental/tractography_predict/run_all.sbatch` | sequential runner (legacy) |
| `notebooks-FC_to_SC-experimental/sanity_checks/tract_check/retest_icc_pipeline.py` | retest FC-ICC + 3-proxy partial |

### Result CSVs / outputs
| Relative path | Contents |
|---|---|
| `…/tractography_predict/e1_source_rep_results.csv` | 50 rows (5 reps × 10 seeds), all metrics |
| `…/tractography_predict/e2_asymmetry_results.csv` | 60 rows (3 reps × 2 dirs × 10 seeds) |
| `…/tractography_predict/e2_asymmetry_summary.csv` | 18 rows (3 reps × 6 metrics) FC-wins + Wilcoxon |
| `…/tractography_predict/e3_marginal_results.csv` / `_summary.csv` | per-seed + paired Δ test |
| `…/tractography_predict/e4_r2t_pc_stability.csv` | 10 modes × stability/expl-var/FC-R² |
| `…/tractography_predict/e4_r2t_top_bundles_per_mode.csv` | top-30 (region,bundle) for modes 3/4/5 |
| `…/tractography_predict/e4_sc_pc3_to_r2t_pc_projection.csv` | Spearman SC-PC3 vs r2t modes |
| `…/tractography_predict/e5_downstream_results.csv` | 210 rows (7 reps × 3 targets × 10 seeds) |
| `…/tractography_predict/e5_downstream_summary.csv` | 21 rows medians + lift over floor |
| `…/tractography_predict/tractography_synthesis.csv` | compact one-row-per-finding |
| `…/tractography_predict/*_output.txt` | captured stdout per experiment |
| `…/sanity_checks/tract_check/retest_icc_results/fc_edge_reliability_pearson.npy` | (64620,) per-edge FC reliability |
| `…/sanity_checks/tract_check/retest_icc_results/fc_reliability_summary.csv` | reliability summary stats |
| `…/sanity_checks/tract_check/retest_icc_results/enrichment_residual_with_fc_reliability.csv` | 3-proxy residual enrichment |
| `…/sanity_checks/tract_check/retest_icc_results/retest_findings.txt` | retest verdict |

### Source data (Torch, read-only)
| Path | Contents |
|---|---|
| `/scratch/asr655/neuroinformatics/Conn2Conn_data/sc/parc-Glasser_*_log1p-1/r2t_matrices.npy` | (n, 360, 66) region-to-tract |
| `/scratch/asr655/neuroinformatics/Conn2Conn_data/sc/parc-Glasser_*_log1p-1/upper_triangles.npy` | (n, 64620) SC |
| `/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv` | NIH-Toolbox cognition |
| `/Users/user/projects/Conn2Conn/data/atlas_info/Glasser_dseg_reformatted.csv` | 360 regions × Yeo7/hemisphere/MNI |

---

## 9. Methods notes & caveats <a name="9-methods"></a>

- **Per-block PCA fix (E1/E3)**: naive concatenation of SC (64,620-dim) and r2t
  (23,760-dim) into one PCA was dominated entirely by r2t's magnitude, making SC/bv/demo
  invisible (all combined reps returned byte-identical results to r2t alone). Fix in
  `_tract_setup.block_pca_pls_predict`: PCA each block to its own 256-dim latent, then
  concat. Narrow blocks (bv 16-dim, demo 9-dim) are z-scored and passed through raw.
- **r2t mse is cross-scale** (§3 ⚠): FC→r2t predicts raw bundle counts (~10⁵ magnitude),
  r2t→FC predicts FC (~0–1). MSE not comparable across these directions; use
  demeaned_pearson/rank instead.
- **r2 negative throughout**: closed-form predictions don't beat the per-edge mean on raw
  scale; the subject-specific signal lives in demeaned/rank metrics. Hence the project's
  primary metric is `demeaned_pearson`.
- **Substitution-fidelity setup (E5)**: cognition model fit on REAL FC_train (teacher),
  applied to synthetic-FC_test (predicted from r2t). Measures whether synthetic FC lands
  in the same cognition-predictive subspace as real FC.
- **Indexing**: `base.sc_r2t_matrices` is already canonical-sliced by `HCP_Base`;
  canonical = train+val+test. Index with train_idx/test_idx exactly like
  `sc_upper_triangles`.
- **HPC parallelism**: per-user QOS caps are ~32 cores and a total-memory cap (6×96G was
  rejected). Run parallel jobs at ~4 cpu / 24G each. apptainer overlay must be `:ro`
  for concurrent jobs (can't open `:rw` twice); set `PYTHONDONTWRITEBYTECODE=1`.
- **Single parcellation**: all results are Glasser. 4S456Parcels cache exists
  (`parc-4S456Parcels_*`) but was not run here.
