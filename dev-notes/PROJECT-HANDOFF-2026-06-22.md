# Conn2Conn — Project Handoff (2026-06-22)

A self-contained orientation to the whole project: what it is, how it's organized, what we've
found, how each finding is sanity-checked, and the history of how we got here. Read this first;
it points to the canonical detail docs for everything else.

Companion docs (canonical detail):
- `notebooks-FC_to_SC-experimental/MASTER_FINDINGS.md` — the findings ledger (F1–F10), with
  confidence levels and evidence CSVs.
- `planning/reproducibility_and_grid_plan_{theory,todo,runlog}.md` — the reproduction grid's
  *why* / *what-now* / *what-it-cost*.
- `reproduction/exploration/{FINDINGS_EXPLORATION,DISCREPANCY_RESOLUTION}.md` — deep re-analysis
  of the grid + resolution of the "numbers don't match the PDF" scare.
- `dev-notes/CLAUDE-CONTEXT-2026-06-18.md` — earlier full session dump (more granular).

---

## 1. What the project is

**Conn2Conn** studies **cross-modal connectome prediction** in HCP-YA (Human Connectome Project,
Young Adults): given one connectivity modality, predict the other, and ask what that
predictability does and doesn't buy you.

- **FC** = functional connectivity (resting-state correlation matrix).
- **SC** = structural connectivity (diffusion/tractography streamline counts).
- Both are represented as **upper-triangle edge vectors** of a region×region matrix.
- **Two parcellations** (atlases): **Glasser** (360 regions → 64,620 edges) and **4S456Parcels**
  (456 regions → 103,740 edges; 400 cortical Schaefer + 56 subcortical/cerebellar).
- **Cohort:** ~957 subjects with both modalities; family-aware **train ≈ 683 / test ≈ 195**
  (siblings/twins kept on the same side of the split).

It is a **methods / redirect paper**, not a leaderboard-chasing paper. The thesis that emerged:

> Cross-modal connectome prediction is **strongly directional** (FC→SC ≫ SC→FC), **linearly
> saturated** (bigger models / more data don't help), **FC is the ceiling** for predicting
> cognition (SC and *imputed* connectomes add nothing or hurt), yet predicted connectomes still
> **carry heritable family signal** — but you **cannot optimize one connectome to both reconstruct
> and identify**. A small, fragile spectral mechanism (a heritable, FC-predictable low-variance SC
> mode) exists but is parcellation-sensitive.

### The metric that matters: `demeaned_pearson`
Connectomes share a huge group-mean structure, so **raw** Pearson/r² between any predicted and
true connectome is ~0.8–0.95 and meaningless (population-mean-dominated; cross-modal r² < 0 is
normal). The **primary metric everywhere** is **`demeaned_pearson`**: per-subject cosine of the
connectome *after subtracting the training group-mean*, averaged over test subjects. It measures
how well you recover *individual deviation* from the average brain. Identifiability is tracked
with `avg_rank` / `top1_acc` (fingerprinting).

### Estimators (and which to quote)
- **`pca_pls`** — PCA(256)→PLS(64)→inverse-PCA. The closed-form workhorse for *reconstruction*.
- **`bayesian_ridge`** — PCA(256)→BayesianRidge per target component. **The only stable estimator
  for scalar/cognition targets** (see §6 discrepancy note) and the strongest oracle.
- **`kernel_ridge`** — RBF, a 3×3 gamma×alpha sweep (9 variants). Turned out **degenerate** (HP
  spread ~0.004 → all 9 ≈ 1 variant).
- Reporting rule: **state the estimator next to every number.** Reconstruction headlines use
  `pca_pls` or `bayesian_ridge`; **downstream/cognition must use `bayesian_ridge`.**

---

## 2. How the project is organized (the parts)

```
Conn2Conn/
├── main.py, models/                  # Sim() data loader + model/eval infra (the engine)
├── data/
│   ├── dataset_utils.py              # load_metadata(): connectomes + HCP restricted (family) + demo
│   └── atlas_info/                   # {Glasser,4S456Parcels}_dseg_reformatted.csv (network labels)
│
├── notebooks-FC_to_SC-experimental/  # ORIGINAL research (mostly Glasser, ad-hoc runs)
│   ├── MASTER_FINDINGS.md            # the findings ledger (F1–F10)
│   ├── model_overviews/
│   │   └── crossmodal_pca_pls_closed_form_overview.ipynb   # STEP 6 predictors; STEP 8 family; STEP 9 downstream
│   ├── further_exploration/
│   │   ├── _setup.py                 # SINGLE SOURCE OF TRUTH: load_seed_split + predictors + family helpers
│   │   ├── depth1_spectral_mechanism.ipynb        # F8 seed-0 PC mechanism
│   │   ├── depth1.1_pc_stability_and_confounds.ipynb  # F8 10-seed stability + confounds + PC3 localization
│   │   ├── depth2_robustness_sensitivity.ipynb    # F7 robustness (combined collapse not-a-bug)
│   │   └── depth3_cognition_ceiling.ipynb         # why not chase cognition further
│   ├── tractography_predict/         # F9: richer structure (r2t bundles) — _tract_setup.py + e1..e5
│   ├── non-linear-sanity-check/      # F10: nonlinear nulls — N1..N6 + findings_{nonlinear,residual,scaling}.md
│   └── sanity_checks/
│       ├── noise_sanity_check/       # Ceiling A (cross-session reproducibility ≈ 0.49)
│       └── preprocessing_check/      # preprocessing/leakage sanity
│
├── reproduction/                     # THE CLEAN GRID SUITE (deterministic, both parcs × 10 seeds)
│   ├── _grid_common.py               # builds on _setup; set_parcellation, frozen-split assert (BP-2),
│   │                                 #   capped estimators, leak helpers, provenance, CSV append
│   ├── freeze_splits.py + splits/     # the 10 FROZEN seeds (the BP-2 source of truth, versioned)
│   ├── run_reconstruction_grid.py    # 12 input→target pairs × 11 estimator-variants
│   ├── make_handoff_artifacts.py     # saves pred_{SC,FC}_{train,test}.npy per (parc,seed)
│   ├── run_downstream_grid.py        # 10 inputs × 5 targets × 11 variants (cognition + sex/age leak)
│   ├── run_leak_checks.py            # demographic-leak guardrail (EXPECTED_SIGNAL logic)
│   ├── gen_expected_cells.py, merge_parts.py, verify_completeness.py, summarize.py
│   ├── run_unit.sbatch, finalize.sbatch, run_downstream_only.sbatch
│   ├── outputs/{reconstruction,downstream,leak_verdict}.csv   # SOURCE OF TRUTH (CSV)
│   ├── reports/reproduction_findings.md
│   ├── exploration/                  # deep re-analysis: explore.py, make_figures.py, FINDINGS_*, DISCREPANCY_*
│   ├── family_mechanism/             # F6/F7/F8 grid (ports of STEP 8 + depth1/1.1), both parcs
│   │   ├── _fm_common.py, _f8_common.py, run_f6_family.py, run_f8_pcmech.py,
│   │   ├── finalize_fm.py, finalize_f8.py, run_{fm,f8}_unit.sbatch, finalize_*.sbatch, submit_{fm,f8}.sh
│   │   ├── tests/                    # 4 test files; reproduce notebook numbers bit-exact (no torch)
│   │   └── outputs/{family_auc, f8_per_pc, f8_stability, f8_pc3_localization, f8_pc3_enrichment_agg}.csv
│   ├── upload_to_wandb.py            # replay all CSVs -> W&B tables + headline scalars
│   └── HANDOFF_FOR_ANALYSIS.md       # primer for the desktop Claude data-analysis app
│
├── planning/                         # reproducibility_and_grid_plan_{theory,todo,runlog}.md
│                                     #   + family_mechanism_grid_plan.md
├── dev-notes/                        # session handoffs (this file; CLAUDE-CONTEXT-2026-06-18.md)
└── memory/                           # Claude auto-memory (Torch paths, ops rules)
```

**Two-layer design.** The notebooks are the *exploratory* layer (where findings were discovered,
mostly Glasser, single runs). `reproduction/` is the *confirmatory* layer: it imports the
notebooks' `_setup.py` so the math is identical, then re-runs everything deterministically on
**both parcellations × 10 frozen seeds** with completeness + leak guardrails. **The CSV outputs
are the source of truth**; W&B and reports are downstream views.

---

## 3. The findings (F1–F10)

Status key: ✅ replicated on both parcellations (grid); 🟡 Glasser-only / exploratory.

| # | Finding | Result | Status |
|---|---------|--------|--------|
| **F1** | **Asymmetry**: FC→SC ≫ SC→FC | 1.6–1.7× (pca_pls); ratio robust across estimators | ✅ both |
| **F2** | **Double dissociation** | anatomy(bv)→structure(SC); demographics(demo)→function(FC) | ✅ both |
| **F3** | **Imputation utility** | `pred_*` connectomes saved + usable, but see F5 | ✅ both |
| **F4** | **FC→cognition is real** | `obs_FC` lifts CogCryst over bv+demo (60–80% seeds sig) | ✅ both |
| **F5** | **SC underperforms; imputation doesn't transfer** | `obs_SC` lift ≤ 0; **`pred_FC` actively harmful** (−0.11..−0.14) | ✅ both |
| **Ceiling A** | data reproducibility | FC day1↔day2 ≈ 0.49 (cross-session) | 🟡 |
| **Ceiling B** | model oracle | FC→FC ≈ 0.63–0.67, SC→SC ≈ 0.62–0.65 (BR); B > A as predicted | ✅ both |
| **F6** | **Predicted connectomes carry heritable family signal** | `pred_SC_resid_bvdemo` sibling AUC **0.810 / 0.816** vs baseline 0.563/0.565 | ✅ both |
| **F7** | **Predictor / identifier tradeoff** | `combined_pred_SC` collapses to chance (**0.505 / 0.501**, n.s.); `pred_SC_raw` separates (0.680/0.670) | ✅ both |
| **F8** | **PC mechanism** | PC1 = sex+BV confound (R² 0.89/0.93); a heritable FC-predictable mode exists (R²≈0.22–0.24, sib AUC≈0.58) **but at PC3 Glasser / PC4 4S456** | 🟡 partial |
| **F9** | **Richer structure doesn't help** | r2t bundles predict FC *worse* than counts; count-SC is a sufficient statistic | 🟡 Glasser |
| **F10** | **Nonlinearity & more data won't help** | 4 orthogonal nulls (KernelRidge/HGB, residual-boost, multimodal sink, n-scaling) | 🟡 Glasser |

### Nuances worth carrying (from `exploration/FINDINGS_EXPLORATION.md`)
1. **F1 is not a reliability artifact.** At the oracle, FC and SC are equally self-predictable
   (FC→FC / SC→SC ≈ 1.0), so FC→SC > SC→FC is genuine directional information, not "SC is a noisier
   target." Kills the obvious reviewer objection.
2. **F5 is stronger than "useless":** imputed FC is *actively harmful* downstream (worst cells in
   the grid), pred_SC ≈ neutral.
3. **Cross-modal reaches only ~21–24% of the within-modal oracle ceiling.** Honest framing.
4. **4S456 systematically boosts →SC prediction (+8–14%) and slightly dampens →FC** — structured,
   interpretable, the genuinely-new cross-parcellation evidence.
5. **F8's mechanism mode migrates index across atlases** (PC3 Glasser ↔ PC4 4S456). The phenomenon
   replicates; the "PC3" label is Glasser-specific. PC2 was a single-seed false positive (caught).

---

## 4. Sanity checks (how each claim is defended)

The project is unusually disciplined about *not* fooling itself. The checks:

- **Frozen splits (BP-2).** Splits are derived ONCE (`reproduction/splits/seed{0..9}.json`),
  loaded everywhere, and **asserted** against on every load (`load_split_checked`). No silent
  re-derivation. Parcellation-independent (verified: same subjects regardless of atlas).
- **Per-block scaling (BP-1).** `connectome + bv + demo` inputs are scaled per-block so the tiny
  subject-info block isn't swamped by the connectome PCA.
- **Completeness contract.** `gen_expected_cells.py` writes a manifest; `verify_completeness.py`
  **hard-fails** if any expected cell is missing or non-finite. The spine grid is 13,640/13,640.
- **Leak guardrail.** `run_leak_checks.py` flags demographic leakage. Verdicts: `ok` /
  `EXEMPT_FLAGGED` (input contains bv+demo) / `EXPECTED_SIGNAL` (raw connectome predicting sex/age
  = real biology, not leak) / `LEAK_FAIL` (hard-fail). The grid has **0 LEAK_FAIL**; only the
  *combined* obs_FC+obs_SC crosses the sex threshold (complementary info).
- **Ceiling A vs B.** Cross-session reproducibility (0.49) is the data noise floor; the model
  oracle (0.63–0.67) sits above it — exactly the ordering you want (`sanity_checks/noise_sanity_check`).
- **Preprocessing check** (`sanity_checks/preprocessing_check`) — no train/test preprocessing leak.
- **F10 nonlinear nulls** are themselves a sanity check on F1/F5: four independent "is there extra
  signal?" probes, four nulls → the linear ceiling is structural, not a modeling failure. (HGB is
  flagged non-load-bearing — it fails the FC sanity probe at n≈683; KernelRidge is the trustworthy
  probe.)
- **Reproduction-vs-notebook validation.** The F6/F7/F8 ports are checked **bit-exact** against the
  notebooks' saved outputs, off-cluster (no torch): F6/F7 aggregation reproduces `aggregate_auc.csv`
  to 1e-16 (AUC) / 0 (perm-p); F8 localization reproduces the saved seed-0 loadings exactly
  (energy 0.630678, top-50 edge mapping 50/50). Each runner self-checks Glasser/seed0 against the
  notebook; each finalize has a Glasser regression guard.

---

## 5. The reproduction grid (the confirmatory engine)

- **Spine grid = 13,640 cells** = 2,640 reconstruction (2 parc × 10 seeds × 12 pairs × 11 variants)
  + 11,000 downstream (2 parc × 10 seeds × 10 inputs × 5 targets × 11 variants). Verified complete,
  0 non-finite, 0 LEAK_FAIL. Reproduces F1/F2/F4/F5 + Ceiling B on both parcs.
- **Family-mechanism grid (F6/F7/F8)** lives in `reproduction/family_mechanism/`, same frozen
  splits, 20 units (2 parc × 10 seeds) each, dependency-chained finalize.
- **Outputs (source of truth, all local + in git):**
  - `reproduction/outputs/{reconstruction,downstream,leak_verdict}.csv`
  - `reproduction/family_mechanism/outputs/{family_auc,f8_per_pc,f8_stability,f8_pc3_localization,f8_pc3_enrichment_agg}.csv`
  - `reproduction/configs/expected_cells.csv`, `reports/reproduction_findings.md`
- **W&B:** project `conn2conn-fc-to-sc-reproduction`; everything is replayed from CSV (no
  recompute) via `reproduction/upload_to_wandb.py` → 8 tables + headline scalars. Latest run:
  `…/runs/jmwgpvi9`.

### Running on NYU Torch (HPC) — the operational rules
- Account `torch_pr_60_tandon_priority`, partition `cpu_short`. **CPU jobs are capped at 4h
  cluster-wide** (5h/8h rejected). `--exclude` is policy-blocked (use cancel+retry on bad nodes).
- **Light grids (F6/F7/F8): 16G / 8 CPU / 1h / array %10** (peak RSS ~4–8 GB). Spine grid was
  24G (peak ~10–14 GB); 48G over-asked and tripped `QOSMaxMemoryPerUser`.
- **Sync via git, never scp.** Bridge: `git push torch:… adel-temp:refs/heads/laptop-incoming`
  → `ssh torch git merge --ff-only` → `git push origin` → laptop fetch. Container = apptainer with
  `:ro` overlay (export HOME/cache to scratch or wandb-core hits disk-quota).
- **Never poll `squeue`** (admin emails). Watch passive **file sentinels** under `sentinels/`.
- One-command launch: `bash submit_fm.sh` / `bash submit_f8.sh` (array + afterok finalize).
- Per-unit timing ≈ 3 min (Glasser) / 5 min (4S456); a 20-unit grid is ~15–20 min wall + queue.

---

## 6. The "numbers don't match the PDF" scare — resolved

An analyst flagged the grid's oracle (0.38) and cognition baseline (0.129) as far below the PDF
(0.647 / 0.359). **Both were estimator confusion**, not real changes (`exploration/DISCREPANCY_RESOLUTION.md`):
- The analyst read **`pca_pls`** rows; the PDF used **`bayesian_ridge`**. Holding the estimator
  fixed, everything matches to the digit (BR SC→SC = 0.648 ≈ PDF 0.647; BR bv+demo→CogTotal = 0.359).
- **Bonus:** `pca_pls`/`kernel_ridge` *scalar* regression is ill-conditioned (r² = −50 to −540, and
  not even numerically reproducible across nominally-identical input) → **only `bayesian_ridge` is
  reportable for downstream.** This validates the report's choice.

---

## 7. History / timeline (how we got here)

1. **Exploratory phase (notebooks).** FC↔SC prediction prototyped in
   `crossmodal_pca_pls_closed_form_overview.ipynb` (Glasser, ad-hoc seeds): discovered F1–F5,
   Ceiling B, F6/F7 (STEP 8/9). Then `further_exploration/` (F8 PC mechanism, robustness, cognition
   ceiling), `tractography_predict/` (F9), `non-linear-sanity-check/` (F10). Findings logged in
   `MASTER_FINDINGS.md`. Weakness: Glasser-only, numbers from different runs.
2. **Reproduction grid build.** Split the plan into theory/todo/runlog; froze 10 seeds; built
   `reproduction/` as deterministic Python; resolved BP-1/BP-2; added completeness + leak
   guardrails; ran the **13,640-cell spine grid** on both parcs on Torch (with the node-contention
   and memory-rightsizing war stories in `runlog.md`). Reproduced F1/F2/F4/F5 + Ceiling B on both.
3. **Deep re-analysis.** `exploration/` — confirmed all findings, surfaced 8 nuances, resolved the
   estimator-confusion discrepancy.
4. **Family-mechanism grid (this week).** Ported STEP 8 (F6/F7) and depth1/1.1 (F8) into
   `family_mechanism/`, validated bit-exact against the notebooks, ran both parcs × 10 seeds on
   Torch. **F6/F7 replicate cleanly; F8 replicates the phenomenon but the PC index migrates.**
5. **W&B upload.** All CSVs (spine + family) replayed into one W&B run.

### Open threads (see `planning/.../todo.md` + memory)
- **F8 refinement:** the PC3-localization step hard-aligns a fixed index; should select the
  mechanism mode **by property** (most FC-predictable non-confounded PC) so 4S456 (PC4) compares
  like-for-like. The W&B *scalars* already do this; the localization table doesn't.
- **F10 nonlinear-nulls grid:** port `non-linear-sanity-check/` (N1–N6) to both parcs × 10 seeds
  (currently Glasser-only) — concrete plan in `planning/reproducibility_and_grid_plan_todo.md` item 5.
- **Fold grid numbers into `MASTER_FINDINGS.md`** (retire "numbers from different runs").
- **Deferred:** PCA-cache optimization (~30–50% wall-time, bit-identical) before any full re-run;
  bootstrap CIs on headline numbers; Ceiling A↔B reconciliation in `findings_noise.md`.

### Standing operational rules (from memory; do not violate)
- Sync via **git**, not scp. Never poll **squeue**. Never `scancel -u` (only specific job IDs).
  Never run heavy compute on the login node. Don't attempt interactive ssh/MFA in autonomous mode.
- Torch paths/aliases: see `memory/reference_torch_conn2conn_paths.md`.

---

## 8. Quick start for a new contributor / agent

```bash
# read the data (laptop has all result CSVs in git; connectomes live on Torch)
column -s, -t reproduction/outputs/reconstruction.csv | head
column -s, -t reproduction/family_mechanism/outputs/family_auc.csv | head

# validate the family/F8 ports against the notebooks (local, no torch needed)
python reproduction/family_mechanism/tests/test_aggregation_matches_notebook.py
python reproduction/family_mechanism/tests/test_f8_localization_matches_notebook.py

# re-derive the exploration digest + figures
python reproduction/exploration/explore.py && python reproduction/exploration/make_figures.py

# push results to W&B (needs a key; CSV is source of truth)
.wandb-venv/bin/python reproduction/upload_to_wandb.py        # or --offline
```
Lead reconstruction with `demeaned_pearson`; lead downstream with `lift_over_bvdemo` + paired-perm
`lift_perm_p`; always name the estimator (BR for cognition/oracle). Treat sex/age as leak-checks,
not findings.
