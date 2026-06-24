# Sanity-Check Gap Plan


This file separates checks already closed from existing repository outputs from checks that require new data, saved predictions, or new model runs.


## Closed With Existing Data


- Reconstruction quality is decoupled from cognitive lift; see `supplement_sanity_checks.md`, section `Existing-Data Gap Closure`.

- The current downstream grid covers three cognition composites plus age/sex controls.

- Seed-level lift bounds show predicted SC alone is near-zero or negative for cognition, while observed FC plus bv+demo is detectably positive.

- Leak verdicts show zero genuine `LEAK_FAIL` cells.

- Family AUC shows predicted connectomes can preserve identity/family signal when the representation selects for it.

- Per-subject FC reliability does not explain SC->FC achieved performance.


## Requires New Data Or New Runs


- **SC-side reliability/noise**
  - Needed: Repeat dMRI, split-half tractography, bootstrap streamlines, or saved tractography perturbations.
  - Feasibility: medium if raw diffusion/tractography pipeline is available; low from summary CSVs alone.
  - Why it matters: Current repository bounds FC noise well but cannot produce a true SC test-retest ceiling.

- **Objective interpolation**
  - Needed: New training grid mixing reconstruction, fingerprint/family, and cognition objectives on frozen splits.
  - Feasibility: medium-high computationally; no new cohort needed.
  - Why it matters: Would turn the objective-mismatch argument from inferential to causal.

- **Broader phenotype families**
  - Needed: Additional HCP behavioral, personality, motor, emotion, and latent phenotype targets.
  - Feasibility: high if phenotypes are already local; medium if target cleaning is needed.
  - Why it matters: Current grid covers three cognition composites plus age/sex controls, not all behavior.

- **Predicted-SC calibration/topology**
  - Needed: Saved predicted matrices or regenerated predictions; compare degree, strength, sparsity, modularity, hubs.
  - Feasibility: medium if predictions are cached; medium-low if all predictions must be regenerated.
  - Why it matters: Correlation can look acceptable while graph topology is biologically distorted.

- **Edge-class stratification**
  - Needed: Per-edge predictions plus distance, network labels, reliability, and streamline-strength bins.
  - Feasibility: medium with saved matrices and atlas metadata.
  - Why it matters: Could show whether useful signal is concentrated in short/long, intra/inter-network, or high-reliability edges.

- **Split stress tests**
  - Needed: Rerun selected cells under random, family-aware, age/sex-balanced, high-motion-excluded, and low-motion-only splits.
  - Feasibility: medium; compute-heavy but no new data.
  - Why it matters: Would make the split-drift/leakage defense harder to attack.

- **External validity**
  - Needed: Second cohort with FC, dMRI-derived SC, demographics, and comparable cognition/behavior targets.
  - Feasibility: low-medium depending on access.
  - Why it matters: HCP-YA-only evidence supports an internal claim, not a universal population claim.

- **Full hyperparameter leakage audit**
  - Needed: Static/code audit plus small rerun proving PCA, scaling, residualization, and target transforms fit inside folds.
  - Feasibility: high for code audit; medium for rerun.
  - Why it matters: Leak verdicts cover output behavior; this would document every transform boundary.
