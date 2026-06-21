# CLAUDE-CONTEXT — 2026-06-18

A full context dump of everything established this working span on **Conn2Conn** (HCP-YA
cross-modal connectome prediction, FC↔SC). Intended as a handoff so a future session (or
human) can pick up without re-deriving. Branch: `adel-temp`. Everything below is committed
and synced to `origin/adel-temp` (latest at write time: `9e82229`).

---

## 1. What this project is

Predicting one connectome modality from the other on HCP-YA:
- **FC** = functional connectivity (resting-state fMRI, Pearson correlation matrices)
- **SC** = structural connectivity (diffusion MRI, SIFT2-weighted streamline counts)
- Closed-form pipeline: **PCA → PLS → inverse-PCA** (also BayesianRidge, KernelRidge as
  robustness estimators). Family-aware train/val/test splits, 10 seeds.

The work has **pivoted from a discovery paper to a methods/redirect paper.** The
contribution is **(1) a required `bv+demo` baseline** the field isn't using, and **(2) a
rigorous demonstration that the cognition ceiling is structural, not method/sample limited.**
The FC→SC asymmetry is *established* (Krakencoder) and used as scaffolding; the PC3 mechanism
is an exploratory, hedged bonus.

---

## 2. Data (HCP-YA) — what we have, what we don't

- ~957 canonical subjects (FC ∩ SC ∩ FreeSurfer ∩ node-features ∩ metadata).
- **Parcellations: exactly 2 available** — `Glasser` (360 ROIs → 64,620 edges; multimodal
  HCP-MMP1.0, NOT anatomical) and `4S456Parcels` (456 ROIs → 103,740 edges; AtlasPack
  multi-resolution). **FC is the hard cap** — raw xcpd output only has `seg-Glasser` and
  `seg-4S456Parcels`. A 3rd would require re-running xcpd (out of scope). Verified:
  subject availability is **identical across parcellations** (SC 1063=1063, FC 1090=1090,
  symdiff 0).
- **FC test-retest IS available**: `expose_fc_sessions=True` gives REST1 (run-1) & REST2
  (run-2) — two different days, same scanner — plus LR/RL phase-encode directions. So per
  subject there are **4 FC cells** (REST1-LR, REST1-RL, REST2-LR, REST2-RL); the standard
  loader averages them. **No BOLD timeseries** (inputs are precomputed Pearson `relmat`
  TSVs) → no half-split / timeframe bootstrap.
- **SC test-retest is NOT available**: single dMRI acquisition per subject, **no
  tractograms (.tck)**, no retest. So SC reliability is unmeasurable from our cache —
  needs the HCP retest dMRI release (~45 subj) or a literature ICC plug-in. (The friend
  Slack ask is about sourcing this.)
- **Richer tractography**: the SC cache has `r2t_matrices.npy` = region-to-tract profiles
  (360 regions × 66 named bundles) — used in `tractography_predict`.
- Cognition: NIH Toolbox `CogTotalComp_Unadj`, `CogFluidComp_Unadj`, `CogCrystalComp_Unadj`
  from `/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv`.
- Cache root: `DEFAULT_CONN2CONN_CACHE_ROOT = /scratch/asr655/neuroinformatics/Conn2Conn_data`
  (asr655's — read-only for us; we write our own caches under `/scratch/ans9868/`).

---

## 3. The findings (canonical = `notebooks-FC_to_SC-experimental/MASTER_FINDINGS.md`)

MASTER_FINDINGS is the registry; every finding carries its repo-relative `.md`/`.csv`
evidence trail. Methods-paper framing: **spine = baseline (C1) + ceiling (C2=F5/F9/F10)**;
F1 demoted to scaffolding; F8 hedged exploratory.

- **F1 — FC→SC > SC→FC asymmetry** (~1.5–1.8×, all 6 metrics p=0.001, robust to reduction
  [no-reduction 1.81×, PCA 1.62×, 3 JL variants 1.39–1.55×] and K). **Established
  (Krakencoder), used as scaffolding, not claimed novel.**
- **F2 — Modality dissociation**: SC anatomy-driven (bv→SC 0.162), FC demographics-driven
  (demo→FC 0.114).
- **F3 — Imputation inherits the source modality** (utility asymmetry 2.06–2.54×). ⚠ ratios
  CI-less.
- **F4 — FC cognition signal real (60/60/73% survive bv+demo), SC's mostly demographics
  (27/18/31%).**
- **F5 / C1 — the bv+demo baseline (PRIMARY contribution)**: CogCryst T1→bv+demo r=0.353;
  +fMRI 0.434 (only real gain); +diffusion 0.258 (**below the free baseline**). "Report a
  bv+demo baseline; acquire fMRI, skip diffusion for cognition."
- **F6 — predicted connectomes carry heritable family signal** (pred_SC_resid_bvdemo
  sibling AUC 0.810 vs bv+demo 0.563). ⚠ from an OLDER run — regenerate from one consistent
  pass (the one real correctness debt).
- **F7 — predictor/identifier tradeoff** (combined_pred_SC wins cognition, collapses to
  chance 0.505 on siblings; confirmed not-a-bug via 4 perturbations).
- **F8 — Mechanism (EXPLORATORY, medium pending 4S456 replication)**: FC predicts SC-PC3, a
  visual/dorsal-attention, intra-hemispheric, heritable-beyond-demographics structural mode
  (R²≈0.22, sibling AUC 0.58, enrichment 11.7×/8.2×/4.9×, 62% energy in 1% edges). PC1 is a
  89% sex+BV confound. PC2 was a single-seed false positive. Distributed across PC3–5.
- **F9 — richer tractography (r2t) is a dead end**: predicts FC worse than counts (0.049 vs
  0.085), Δ marginal=−0.001, count-SC is a sufficient statistic, no cognition above floor,
  synthetic-FC-from-tractography doesn't recover FC's signal.
- **F10 — connectome→cognition is linearly saturated**: 4 orthogonal nonlinear nulls —
  model class (HGB/KernelRidge), residual-boost (OOF), cross-modal sink, **data-scaling
  curve (gap flat n=100→683 → structural, not sample-size)**.

**Meta**: everything capped by biology + n≈878, not method. Research-grade, not
diagnostic-grade.

---

## 4. The noise module — CLOSED, filed as F10-supporting (MASTER A6)

`notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/` (`findings_noise.md`).
Answers "how much of the scan is noise" via FC test-retest (the only modality we can).

**The honest bottom line (level-dependent):**
1. A single FC measurement is **~64% noise** (per-edge individual-difference variance).
2. The averaged usable connectome is **~41% noise** (reliability G ≈ 0.59).
3. The reproducible individual signal tops out at **demeaned-r 0.49** (between-session
   ceiling), heterogeneous across people (**0 to 0.78**, std 0.12, left-skewed, a stable
   subject trait ρ=0.41).
4. **Of that reproducible 0.49, SC explains ~17%, flat across subjects.**

**Key result (H):** SC→FC prediction quality is **uncorrelated** with per-subject FC
reliability (r=0.01, vs bv+demo baseline r=0.15), and reliability-filtering does NOT sharpen
it (fraction 0.169→0.137 as ceiling rises but achieved stays pinned ~0.08). So the SC→FC gap
is **not FC-measurement-noise and not per-subject reliability** (airtight). "SC doesn't
*contain* it" is the strong interpretation, pending SC retest (a uniform SC noise floor would
also give a flat line). Fingerprint top1=0.93, discriminability=0.998 (distributed-signal
reconciliation). **Parcellation-robust** (4S456 ≈ Glasser).

**Two open review notes in `findings_noise.md` (`#REVIEW AND QUESTION`):**
- Disattenuation (Spearman 1904) is a **population** theorem, not per-subject. Ceiling valid
  at population level; per-subject use "reliability distribution + correlation," not pointwise
  "fraction of ceiling" (that column should be descriptive only).
- **Two ceilings**: (A) data/reproducibility (FC_day1↔day2 = 0.49, model-free, the
  *biological* denominator); (B) model/oracle (FC→FC, SC→SC through same pipeline; SC→SC≈0.647
  from earlier work, FC→FC = to-compute in grid). B is *available for both directions &
  never exceeded per-subject* but is a *looser* bound (fits non-reproducible session-specific
  signal). Use A for the biological 17%, B for model-capacity/cross-modal-loss. Don't
  substitute B for A.

Modules: build_fc_cells (PREP, 4-cell cache at `/scratch/ans9868/noise_cache/`), A
(reliability ceiling), B (2×2 G-theory variance decomp), E (cross-modal disattenuation,
SC→FC÷ceiling), F (fingerprint/discriminability), G (per-subject distribution), H
(achieved-vs-ceiling + reliability-filtered). All sbatch, results CSVs in `outputs/`.

---

## 5. The reproducibility grid — PLANNED, build-ready, not yet built

`planning/reproducibility_and_grid_plan.md` (+ `Overall_plan.md`, `cleanup_plan.md`,
`grid_plan.md`, `roadmap/noise-sanity-check.md`). Purpose: re-derive the confirmatory claims
from scratch in ONE consistent parameterized pipeline (fixes F6 debt), with CIs + a run
ledger.

**Locked decisions:**
- **Two orthogonal axes**: input/feature set × estimator. `bv+demo` is an *input*, not a model.
- **Estimators (locked)**: PCA→PLS, BayesianRidge, KernelRidge(RBF). NOT: Krakencoder,
  learnable PLS, MLP, CovProjector.
- **2 parcellations** (Glasser + 4S456 — STOP gate resolved, only 2 exist, FC-capped).
- **10 seeds**, **frozen once** → `configs/splits/seed{0..9}.json`, loaded by every cell
  (Option B; A==B verified since subject availability identical across parcellations).
- **W&B offline** (`WANDB_MODE=offline`, **FLAT keys** `metrics/{task}/{input}/{target}/{metric}`),
  **CSV is the source of truth**.
- **Two claim-tables**: reconstruction (connectome targets; incl. FC→FC/SC→SC oracle =
  Ceiling B) and downstream (cognition + sex/age leak-checks).
- **Recon→downstream handoff contract** (the fragile seam, now explicit): reconstruction
  writes 4 fixed-PCA→PLS imputed connectomes (`pred_{SC,FC}_{train,test}`, train via in-sample
  `X_tr→X_tr`), `recon_per_subject`, `split_index` — all keyed `(parcellation, seed)`.
  Downstream HARD-asserts they exist + joins on `subject_id` before running imputation rows.
  Hard ordering: recon → downstream → leak_check → summarize.
- **Leak guardrail**: any input predicting sex>0.99 / age>0.85 hard-fails; `combined_pred_*`
  exempt-but-flagged.
- **Completeness contract**: enumerate `configs/expected_cells.csv` up front (per-task valid
  (input,target) pairs; KR 3×3 = 9 rows each); per-cell write-time assertion (sentinel+reason
  for expected NaN); `verify_completeness.py` hard-fails on any missing cell after the grid.
- **Metric reporting order**: reconstruction → demeaned_r, avg_rank (then top1/mse/r2/pearson
  caveated); downstream → lift_over_bvdemo + **paired permutation p** (marginal CIs mislead).

**Two breakage risks LOCKED (silently-wrong, not crashes):**
- **BP-1**: `connectome+bv+demo` must use per-block scaling (reuse `_blocks_to_latents` —
  PCA edges→256, bv+demo raw z-scored 26, concat) or subject-info vanishes in PCA.
- **BP-2**: freeze splits once, load everywhere, align by `subject_id` (never re-derive →
  no order misalignment, no split-logic drift).

**Provenance / what's new**: the entire F1–F10 story is **Glasser-only**; only the noise
module ran 4S456. So the grid's value-add is running the whole spine on 4S456 *for the first
time* — **watch the 4S456 F1–F5 cells first** (the genuinely new evidence; everything Glasser
is reproduction). **Smoke-test on 4S456** (the worst case: 103,740 edges).

**Next build steps**: write `grid.yml` → freeze splits + `expected_cells.csv` → reconstruction
runner (handoff artifacts, per-subject, flat keys) → downstream runner (subject_id join) →
leak checks → smoke test on 4S456 via sbatch → launch → verify_completeness → summarize.

---

## 6. Repo map (under `notebooks-FC_to_SC-experimental/`)

| Area | Dir | Key docs |
|---|---|---|
| Main notebook, Phase 0/1/2 (F1–F7, baseline C1) | `model_overviews/` | `results/FINDINGS.md` |
| Mechanism PC3/4/5 (F8, exploratory) | `further_exploration/` | `README.md` |
| Reduction robustness (A1), combined-pred (F7) | `sanity_checks/preprocessing_check/` | `findings.md` |
| PC3 reliability (A5) | `sanity_checks/tract_check/` | `findings.md` |
| FC noise / reliability ceiling (A6, CLOSED) | `sanity_checks/noise_sanity_check/` | `findings_noise.md` |
| Tractography r2t (F9) | `tractography_predict/` | `findings.md`, `findings_in_depth.md` |
| Nonlinear/residual/sink/scaling (F10) | `non-linear-sanity-check/` | `findings_nonlinear.md`, `findings_residual.md`, `findings_scaling.md` |
| Grid plan | `../planning/` | `reproducibility_and_grid_plan.md` |

Shared helpers: `further_exploration/_setup.py` (data load, `pca_pls_predict`,
`full_panel_eval`, walks to `Conn2Conn` root — move-robust); `tractography_predict/_tract_setup.py`
(adds r2t + `block_pca_pls_predict`); `non-linear-sanity-check/_residual.py` (OOF residual +
`_blocks_to_latents`). Research papers in `dev-notes/research-papers/` (incl.
`2024_JimenezMarin_MultiScaleStructureFunctionNeurogenetics_MPI-LEMON.pdf` — bha2/LEMON, a
good *second cohort* for replication but NOT test-retest).

---

## 7. HPC / ops knowledge (NYU Torch) — IMPORTANT operational rules

- SSH alias `torch`; ControlMaster holds an MFA handshake for the session window. **MFA
  drops repeatedly** — when SSH says "Permission denied / Too many auth failures," the user
  must `ssh torch` once to re-establish; **do not hammer SSH** (lockout risk).
- **Sync via git, NEVER scp** (user rule + memory). Working method this session:
  `git push torch:/scratch/ans9868/Conn2Conn adel-temp:refs/heads/laptop-incoming` →
  `ssh torch 'cd repo && git merge --ff-only laptop-incoming && git branch -d laptop-incoming
  && git push origin adel-temp'` → laptop `git fetch && git merge --ff-only`. (Laptop's own
  GitHub creds are stale/wrong-account → push to origin only works *from Torch*.)
- **Never run compute on the login node** — always `sbatch`. (I slipped once running E on the
  login node; killed the specific PIDs and resubmitted via sbatch. Don't repeat.)
- **Never poll squeue/sacct in a loop** (admin emails). Use passive file sentinels
  (`DONE.sentinel`/`ERROR.sentinel`) + a single remote `until [ -f sentinel ]; do sleep; done`.
- **SLURM**: account `torch_pr_60_tandon_priority`, partition `cpu_short`. QOS caps:
  ~**32 CPU per user** total, and a per-job memory ceiling (~120G). For parallel jobs use
  ~4 CPU / 24G each so several fit under 32 cores.
- **apptainer overlay**: `/scratch/ans9868/kraken_env/unlocked_kraken_env.ext3` on image
  `/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`. **Mount `:ro` for
  parallel jobs** — `:rw` can't be opened by two processes at once (broke an early parallel
  attempt). Set `PYTHONDONTWRITEBYTECODE=1` with `:ro`. conda env
  `/ext3/miniforge3/envs/kraken_env`.
- Jupyter runbook: `dev-notes/torch-jupyter-tmux-skill.md`.
- **statsmodels is NOT installed** in kraken_env (use inline BH-FDR / scipy).

---

## 8. Recurring method gotchas (learned the hard way)

- **Low-dim inputs** (bv 16, demo 9, bv+demo 26) can't go through PCA(256): cap
  `k_src=min(256, width)`, `k_pls=min(64,k_src)`. (Crashed the noise-module E once.)
- **Heterogeneous concat** (edges + bv+demo) → raw PCA swamps the low-dim block → use
  per-block PCA/standardize then concat (`_blocks_to_latents`).
- **`demeaned_pearson` is the primary metric** — raw pearson is population-mean-dominated
  (~0.81 even for test-retest), r² is negative cross-modally (expected). Demean by the
  train/group mean; it's per-subject cosine of (y−μ).
- **Reliability/disattenuation is population-level**, not per-subject (see A6 review note).
- **Split must be frozen + loaded**, not re-derived (BP-2).
- **PLSRegression on raw 64,620-dim** materializes `coef_` = (64620×64620) = 33 GB → OOM.
  Predict manually via `x_rotations_ @ y_loadings_.T` (done in preprocessing_check method B).
- Multi-source agents writing CSVs: commit on the source machine, then git-bridge — don't
  hand-merge.

---

## 9. Open items / immediate next steps

1. **Build the grid** (steps in §5) — the de-risked, fully-specified next action.
2. **F6 regeneration** — family-structure CSVs are from an older pass; regenerate from the
   grid's consistent run (the one real correctness debt).
3. **Bootstrap CIs** on CI-less headline numbers (F3 utility ratios, F4 fractions, F5 lifts).
4. **SC test-retest** — the one data-blocked item (friend Slack ask; HCP retest dMRI release).
   Unblocks SC reliability + the "SC doesn't contain it" → genuine-independence last 5%.
5. **Phase-2 physical reorg** of `notebooks-FC_to_SC-experimental/` into numbered dirs
   (`cleanup_plan.md`) — paused; the grid can proceed without it.
6. **Second-cohort replication** — LEMON/bha2 (Zenodo 8158914) is a good external cohort
   (SC+FC derived, MRtrix/SIFT2 like ours), though single-session (not test-retest).

---

*Written 2026-06-18. All findings/evidence committed to `origin/adel-temp` (≈`9e82229`).
The grid plan is locked and build-ready; the noise module is closed; the science is
Glasser-validated with the cross-parcellation replication being the grid's main new evidence.*
