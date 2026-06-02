# Next steps — what's missing to make the analysis exhaustive

_Updated 2026-05-29 after the overnight 10-seed run + post-hoc audit._

See [FINDINGS.md](FINDINGS.md) for what we have. This file lists what we **don't** have, organized in tiers from "must-do for paper-grade symmetry" to "reviewer-defense polish" to "out-of-scope generalization".

---

## Quick orientation: what's covered now

| Question | Status |
|---|---|
| Does FC→SC beat SC→FC on raw cross-modal prediction? | ✅ 10-seed for FC→SC; SC→FC only single-seed |
| Does the asymmetry survive anatomy + demographics control? | ✅ paper-grade: target-only 1.505 ± 0.20, double-sided 1.553 ± 0.12, 10/10 seeds > 1.15 |
| Does it survive on every metric (not just demeaned-r)? | ✅ all 6 metrics × 10 seeds for `bv+demo` basis |
| Does it survive the BR vs PLS model class switch? | ⚠️ single-seed only, on `bv+demo` basis only |
| Does the FC→SC predicted SC preserve family-specific signal beyond `bv+demo`? | ✅ 10-seed, AUC delta +0.038 MZ / +0.023 DZ / +0.247 sibling |
| Does FC→SC predicted SC have downstream utility for cognition? | ✅ 10-seed, substitution fidelity 1.25–1.44× vs `obs_SC` |
| Symmetric: does SC→FC predicted FC have downstream utility? | ❌ NOT computed (`pred_FC_raw` and `combined_pred_FC` absent from Analysis 2 input set) |

---

## Tier 1 — Required for full symmetry and basis coverage

These are the gaps I'd close before submitting a paper. The user explicitly requested both items in this tier.

### 1.1 — Symmetric downstream prediction (Phase 2 Analysis 2)

**Gap**: Phase 2 Analysis 2 has 9 inputs but none of them are predicted FC. The current 9 are:
```
bv+demo, obs_FC, obs_SC, obs_FC+obs_SC,
pred_SC_raw, combined_pred_SC,
pca_FC, pca_SC, pca_FC+pca_SC
```

**Action**: add 2 more inputs to STEP 9.1's `_phase2_build_inputs_for_seed`:
- `pred_FC_raw` — SC→FC predicted via the same PCA+PLS architecture (just direction flipped)
- `combined_pred_FC` — `[PCA(SC) ‖ bv ‖ demo] → FC` combined predictor

Re-run STEP 9.2 (10-seed loop) and STEP 9.3 (aggregation) with 11 inputs instead of 9.

**What this enables**: compute the **asymmetry of imputation utility**:
- `pred_SC_raw / obs_SC` fidelity vs `pred_FC_raw / obs_FC` fidelity, per cognitive target
- If `pred_SC` fidelity > `pred_FC` fidelity, the FC→SC asymmetry shows up as imputation-utility asymmetry too (Phase 1 finding re-validated through a clinical-prediction lens)

**Cost**: ~25 min — patch cell 52, `rm` stale `seed_*.parquet`, re-run cells 52-54.

### 1.2 — 10-seed cross-modal on missing bases (Phase 1 Exp 7 extension)

**Gap**: Exp 7's per-seed records only contain cross-modal demeaned-r for the `bv+demo` basis. The single-seed Section 2 has values for `none`, `bv`, and `demo` bases too, but no 10-seed numbers:

| Basis | FC→SC (single) | SC→FC (single) | FC→SC (10-seed) | SC→FC (10-seed) |
|---|---|---|---|---|
| none | 0.132 | 0.085 | ✅ 0.1341 ± 0.005 (STEP 2) | ❌ |
| bv | 0.078 | 0.056 | ❌ | ❌ |
| demo | 0.089 | 0.057 | ❌ | ❌ |
| bv+demo | 0.066 | 0.048 | ✅ 0.069 ± 0.005 | ✅ 0.046 ± 0.005 |

**Action**: extend the Exp 7 cell (or write a parallel cell) to also compute cross-modal residual PCA+PLS for `none`, `bv-only`, and `demo-only` bases per seed. Save expanded per-seed CSVs.

**What this enables**: 10-seed ratios + CIs for every basis row in the Section 2 table. Today only the `bv+demo` row has paper-grade CIs.

**Cost**: ~15 min — 3 extra bases × 2 directions × 1 framework × 10 seeds × ~3 sec per PLS call.

---

## Tier 2 — Natural completeness given the work in Tier 1

Same scope of compute, but extends to model-class and combined-predictor coverage that's currently single-seed-only.

### 2.1 — 10-seed BR robustness for all bases, not just bv+demo

**Gap**: Exp 5 (BR vs PLS) ran single-seed on bv+demo only. We don't have 10-seed BR cross-modal for any basis.

**Action**: per seed, also run BR-per-component cross-modal on `none`, `bv`, `demo`, `bv+demo` × target-only + double-sided. Match the Exp 7 PLS structure but with BR.

**What this enables**: confirm the asymmetry direction is model-class-invariant across ALL bases (currently only bv+demo single-seed in Exp 5).

**Cost**: ~50 min — BR is the slow path (256 fits per direction × ~1 sec each). 4 bases × 2 directions × 2 frameworks × 10 seeds = 160 BR runs, but many can share precomputed PCAs.

### 2.2 — 10-seed combined predictor

**Gap**: Exp 6 is single-seed. `combined_pred_SC` (and `combined_pred_FC` after Tier 1.1) need 10-seed numbers for direct comparison vs the cross-modal-alone baselines.

**Action**: extend Exp 6 to a 10-seed loop. Saves per-seed CSV like Exp 7 does.

**What this enables**: paper-grade comparison of `combined_pred_SC` (the clinical "everything-in" model) against `pred_SC_raw` (cross-modal-only) and `bv+demo` (subject-info-only). Especially important given the surprising age-prediction result (combined → age = 0.576).

**Cost**: ~20 min.

### 2.3 — 10-seed identifiability (Section 3 expansion)

**Gap**: Section 3 is single-seed. Identifiability metrics (`top1`, `avg_rank`) are noisy, so 10-seed CIs are especially useful.

**Action**: derives mostly from Tier 1's per-seed full panel — just needs aggregation. Trivial post-processing.

**Cost**: trivial (data already in Tier 1 outputs).

---

## Tier 3 — Hyperparameter sensitivity / reviewer defense

Optional. Useful if reviewers ask "but does this hold at K=X?" — currently we can only point at the project's default K and say "same as the broader literature".

### 3.1 — K_PCA dimension sweep

**Current**: K_PCA = 256 everywhere (project default).

**Action**: re-run Exp 7 (10-seed, bv+demo, both frameworks, PLS) at K_PCA ∈ {128, 256, 512}. Plot the asymmetry ratio as a function of K_PCA.

**What this checks**: is the headline 1.5× ratio an artifact of K=256? Hopefully not; expectation is the ratio is stable across K from ~64 onward.

**Cost**: ~60 min (3 alternates × ~20 min each).

### 3.2 — K_PLS components sweep

**Current**: K_PLS = 64 everywhere.

**Action**: re-run at K_PLS ∈ {32, 64, 128}.

**What this checks**: model-flexibility sensitivity. Likely stable.

**Cost**: ~20 min.

### 3.3 — Demographics decomposition

**Current**: `demo` basis combines age + sex + race_eth into a single 10-dim vector.

**Action**: also run with `age-only` (1 dim), `sex-only` (2 dim), `race_eth-only` (~7 dim) as separate residualization bases. See which component of demographics is doing the work.

**What this checks**: is `demo → FC = 0.114` mostly age (which correlates with global FC variance) or mostly sex/race_eth? Important for interpretation.

**Cost**: ~30 min.

---

## Tier 4 — Generalization (different data, out of scope this session)

Future work. These require data setup not currently in the project.

### 4.1 — 4S456Parcels parcellation

**Current**: only Glasser parcellation (64,620 edges).

**Action**: re-run Phase 1 + Phase 2 with 4S456Parcels (a different atlas, larger N). The data already exists in the project; just needs parcellation toggle.

**What this checks**: parcellation robustness. If the headline asymmetry is Glasser-specific, that's a concerning limitation.

**Cost**: ~hours. Worth doing if pivot direction is locked.

### 4.2 — HCP-Aging / HCP-Development cohorts

**Current**: only HCP-Young Adult (1206 subjects, ages 22-37).

**Action**: re-run on HCP-Aging (older adults) and HCP-Development (children/adolescents).

**What this checks**: lifespan generalization. Currently we make a claim about young adults only.

**Cost**: requires data preprocessing setup not in repo. Significant work.

---

## Tier 5 — Methodology refinements

Polish that strengthens the paper but isn't load-bearing.

### 5.1 — Cross-fit train predictions in Phase 2 Analysis 2

**Current**: `_phase2_build_inputs_for_seed` uses `_pca_pls_predict(_FC_tr, _FC_tr, _SC_tr)` for train predictions — naive in-sample (the same PLS fit predicts on train). This is consistently applied across all rows so the comparison is fair, but it's slightly optimistic.

**Action**: implement K-fold cross-fit for the train predictions. For each subject in the train set, predict using a PLS fit that doesn't include that subject. Use the same scheme for all "predicted" input rows.

**What this fixes**: removes the in-sample optimism caveat. Estimates of `pred_SC` downstream utility become unbiased.

**Cost**: ~30 min implementation + ~30 min re-run.

### 5.2 — Family-structure AUC: per-seed AUCs with hierarchical CIs

**Current**: pools all pairs across 10 seeds, then bootstraps the pooled set for CIs.

**Action**: alternatively, compute AUC per seed, then mean ± CI of AUCs across the 10 seeds. Slightly different statistical semantics (assumes seeds are independent observations, which they roughly are family-aware-split-wise).

**What this fixes**: cleaner inference story; some statisticians prefer the per-seed approach for cross-validated AUCs.

**Cost**: ~10 min.

### 5.3 — Alternate anatomy basis: cortical thickness

**Current**: 16 FreeSurfer volume features.

**Action**: also run with cortical-thickness vectors (~340 dims if per-parcel) as a higher-rank anatomy basis. See if the bv→SC = 0.167 number changes substantially.

**What this checks**: anatomy basis dimensionality sensitivity. Currently anatomy is low-rank by design.

**Cost**: ~30 min — needs cortical-thickness data already in the project.

---

## Cost summary

| Tier | Compute | Cumulative | What you get |
|---|---|---|---|
| Tier 1 (your ask) | ~40 min | 40 min | Full symmetry, every basis × both directions at 10 seeds |
| + Tier 2 (completeness) | +70 min | 110 min | Model-class + combined-predictor coverage |
| + Tier 3 (hyperparam) | +110 min | 220 min | Reviewer-defense robustness checks |
| + Tier 5 methodology | +70 min | 290 min | Cross-fit train + alternate anatomy basis |

**Recommended scope**: Tier 1 + Tier 2 (~2h compute). That gets you paper-grade results for every cell we've discussed. Tier 3 / 5 are reviewer-defense add-ons; do them if and when reviewers push.

**Skip Tier 4** — different data is its own paper, not a completeness fix.

---

## Suggested execution path

If you want to do Tier 1 + Tier 2 in one session:

1. **Request a fresh SLURM allocation**: `srun -A torch_pr_60_general --partition=cpu_short --cpus-per-task=4 --mem=96G --time=04:00:00` (matches last night; sub-minute landing expected).

2. **Patch the notebook** to add ONE new cell named something like:

    > `# ============= STEP 11: GRAND 10-SEED RUN — all bases × all model classes × both frameworks × full panel =============`

    This cell:
    - Loops 10 seeds (cache-resumable like Exp 7 and Phase 2)
    - Per seed: builds residuals for `bv`, `demo`, `bv+demo`
    - Runs PCA+PLS cross-modal for each (basis, framework, direction) × both directions
    - Runs BR cross-modal for each (basis, framework, direction)
    - Runs combined predictor in both directions (Exp 6 10-seed extension)
    - Saves per-seed CSV with full 6-metric panel
    - Aggregates at end with mean ± std + t-tests vs 1.00 and 1.15

3. **Patch STEP 9.1** (Phase 2 Analysis 2 helpers): add `pred_FC_raw` + `combined_pred_FC` to the `_phase2_build_inputs_for_seed` return dict; update `_INPUTS` in STEP 9.2 from 9 → 11; STEP 9.3 picks them up automatically.

4. **Delete stale caches**: `rm /scratch/ans9868/Conn2Conn/results/local_results/downstream_prediction_phase2/seed_*.parquet` so the 11-input run starts fresh.

5. **Run** the new STEP 11 + re-run STEP 9.2-9.3.

6. **Pull results** locally with the same rsync pattern used to populate this directory.

7. **Update FINDINGS.md** with the 11-input downstream table and the per-basis 10-seed ratios.

The Tier 1+2 patch is one new cell + one modified cell (52) + one `rm`. About 80 lines of code total.

---

## What I'd NOT do without asking first

- **Anything that changes existing CSV file names or breaks current downstream consumers.** Add new files alongside existing ones; never rename or delete the paper-grade CSVs that already exist.

- **Anything that runs a substantially different model class** (e.g., the project's deep learning models — Sarwar2020MLP, Chen2024GCN, Nodal GNN). Those have their own sbatch scripts and aren't part of the closed-form notebook's scope.

- **Tier 4 generalization runs without an explicit pivot decision.** Different data is a different paper.

---

## Status snapshot

| | |
|---|---|
| Phase 0 baselines | ✅ done, replicated |
| Phase 1 Sections 1-3 + Exp 4-6 | ✅ single-seed done |
| Phase 1 Exp 7 (10-seed bv+demo) | ✅ paper-grade |
| Phase 1 Exp 7 (10-seed all bases) | ❌ Tier 1.2 |
| Phase 1 Exp 5 (BR robustness, 10-seed all bases) | ❌ Tier 2.1 |
| Phase 1 Exp 6 (combined predictor, 10-seed) | ❌ Tier 2.2 |
| Phase 2 Analysis 1 (family-structure, 10-seed) | ✅ done |
| Phase 2 Analysis 2 (downstream, 9 inputs, 10-seed) | ✅ done, leak-fixed |
| Phase 2 Analysis 2 (symmetric: pred_FC_raw + combined_pred_FC) | ❌ Tier 1.1 |
| Sanity check 1 (brain-vol decomposition) | ✅ done, verdict recorded |
| Sanity check 2-5 (future checks listed in sanity_checks.ipynb) | ❌ optional |
| Hyperparameter sensitivity (K_PCA, K_PLS, demo decomp) | ❌ Tier 3 |
| Cross-fit train predictions | ❌ Tier 5.1 |
| Generalization (parcellation, cohort) | ❌ Tier 4 |
