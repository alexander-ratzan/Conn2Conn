# Conn2Conn Reproduction Grid — Analysis Handoff

Self-contained primer for analyzing the grid results (e.g. in the Claude desktop app).
**Give this file + the three CSVs to the analyst.** No cluster access needed.

## Files (all local, in this folder / `outputs/`)
- `outputs/reconstruction.csv` — connectome→connectome prediction (2,640 rows)
- `outputs/downstream.csv` — connectome→cognition/sex/age prediction (11,000 rows)
- `outputs/leak_verdict.csv` — demographic-leak guardrail verdicts (4,400 rows)
- `configs/expected_cells.csv` — the completeness manifest (every expected cell)

## What this is
A reproducibility grid for **HCP-YA cross-modal connectome prediction** (FC = functional,
SC = structural). One consistent pipeline computed every confirmatory claim across:
**2 parcellations** (Glasser 360-region, 4S456Parcels 456-region) × **10 seeds** (frozen
train/val/test splits) × **3 estimators** (`pca_pls`, `bayesian_ridge`, `kernel_ridge` — the
last a 3×3 gamma×alpha sweep = 9 variants, so 11 estimator-variants total). All cells verified
present (13,640/13,640) and finite.

## Column dictionary
**Common keys:** `parcellation`, `seed`, `estimator`, `variant` (e.g. `kernel_ridge[alpha=1.0,gamma_mult=0.5]`),
`input_set`, `target`, `git_commit`, `config_hash`.

**reconstruction.csv** — predict a connectome from an input; metrics:
- `demeaned_pearson` — **PRIMARY**: per-subject cosine of (connectome − group-mean), averaged.
- `avg_rank`, `top1_acc` — identifiability (normalized rank / fingerprint; 1 = perfect).
- `pearson`, `r2`, `mse` — completeness; **caveated** (see below).
- `input_set→target` pairs: `FC→SC`, `SC→FC` (asymmetry); `bv→*`, `demo→*`, `bv+demo→*`
  (dissociation/baseline); `FC+bv+demo→SC`, `SC+bv+demo→FC` (does connectome add over subject-info);
  `FC→FC`, `SC→SC` (within-modality **oracle = Ceiling B**).

**downstream.csv** — predict cognition/sex/age from an input; metrics:
- `lift_over_bvdemo` — **PRIMARY**: score minus the bv+demo baseline (same estimator/seed).
- `lift_perm_p` — paired permutation p for that lift (use this, not marginal CIs).
- `residualized_pearson` — predicting cognition after removing the bv+demo OLS term.
- `pearson`/`spearman`/`r2` — raw; `balanced_acc` — for `target==sex`; `n_eval`.
- `is_leak_target` (sex/age), `contains_bvdemo` (input includes subject-info).
- inputs: `bv+demo` (baseline), `obs_FC`, `obs_SC`, `obs_FC+obs_SC`, `pred_SC`, `pred_FC`
  (imputed connectomes), and the `*+bv+demo` combinations.

**leak_verdict.csv** — `verdict ∈ {ok, EXPECTED_SIGNAL, EXEMPT_FLAGGED, LEAK_FAIL}`,
`leak_score`, `threshold`, `is_connectome`, `contains_bvdemo`.

## Metric reporting order (lead with the load-bearing numbers)
- **Reconstruction:** `demeaned_pearson` → `avg_rank` → (`top1_acc`/`pearson`/`r2`/`mse` caveated).
- **Downstream:** `lift_over_bvdemo` + `lift_perm_p` → `residualized_pearson` → (raw pearson/spearman;
  sex/age are leak-checks, **not** results).

## Caveats (important for honest analysis)
- **Use `demeaned_pearson`, not raw `pearson`.** Raw pearson/r2 are population-mean-dominated
  (raw pearson ≈ 0.8–0.9 even for weak prediction). `r2 < 0` cross-modally is expected.
- **For "beats baseline" use the paired permutation p (`lift_perm_p`)**, not marginal CIs.
- **sex/age targets are leak-checks**, not findings. Raw connectomes predicting sex/age well is
  expected biology (`EXPECTED_SIGNAL`), not a leak. Only `LEAK_FAIL` (none here) is a problem.
- Each row is one seed; aggregate as **mean ± std across the 10 seeds**.

## The claims to verify (and the expected pattern)
- **F1 — asymmetry:** `FC→SC` demeaned_r ≈ 1.5–1.7× `SC→FC`, on both parcellations.
- **F2 — dissociation:** `bv→SC` (anatomy) strong; `demo→FC` (demographics) relatively stronger.
- **C1 — baseline:** `bv+demo` is the bar; report it explicitly.
- **F4 — FC cognition real:** `obs_FC` lifts significantly over `bv+demo` for cognition.
- **F5 — SC underperforms:** `obs_SC` lift ≤ 0; `pred_SC` adds ~nothing downstream.
- **Ceiling B — oracle:** `FC→FC` / `SC→SC` (the within-modality ceiling).
- **Cross-parcellation replication:** the 4S456 numbers should track Glasser (the new evidence).

## Suggested first analyses
1. Pivot reconstruction `demeaned_pearson` by `input→target` × `parcellation` (estimator=`pca_pls`),
   mean±std over seeds; compute the FC→SC/SC→FC ratio per parcellation.
2. Downstream: per cognition target, rank inputs by `lift_over_bvdemo`; flag `lift_perm_p < 0.05`.
3. Glasser-vs-4S456 scatter of each claim's effect (replication check).
4. KR 3×3 flatness: confirm `demeaned_pearson` is stable across the 9 `kernel_ridge` variants.
