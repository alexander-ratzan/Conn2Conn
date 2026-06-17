# Cleanup Plan For `notebooks-FC_to_SC-experimental`

## Scope

This phase reorganizes the existing FC-to-SC workspace. It is not the new reproducibility
grid. The goal is to preserve the current evidence while making it easier to understand,
rerun, and cite.

All cleanup should be done locally on the laptop. Do not reorganize files on Torch.

## Part A — Track And Save Current Information

### Current meaningful untracked artifacts to preserve

These were synced from Torch and should be tracked before path cleanup:

```text
notebooks-FC_to_SC-experimental/further_exploration/pc4_pc5_results/pc4_verdict.txt
notebooks-FC_to_SC-experimental/further_exploration/pc4_pc5_results/pc5_verdict.txt
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_0.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_1.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_2.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_3.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_4.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_5.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_6.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_7.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_8.csv
notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/_method_b_per_seed/seed_9.csv
notebooks-FC_to_SC-experimental/sanity_checks/tract_check/pc4_pc5_output.txt
```

### Files not to mix into the science commit

```text
.claude/scheduled_tasks.lock
```

Recommendation: restore or separately handle this file before the evidence-preservation
commit so the commit contains only project evidence.

### Scheduler clutter to avoid tracking

Do not track by default:

```text
DONE_*.sentinel
slurm*.out
slurm-one-*.out
```

Exception: if a captured output file is the only human-readable evidence for an analysis
and is referenced by findings, keep it. `pc4_pc5_output.txt` fits that exception.

## Part B — Proposed Physical Layout

Target layout:

```text
notebooks-FC_to_SC-experimental/
  README.md
  MASTER_FINDINGS.md
  EVIDENCE_REGISTRY.md
  RUNBOOK.md

  00_main_story/
  01_mechanism/
  02_sanity_checks/
  03_tractography/
  04_nonlinear/
  05_reproduction/
  99_legacy_or_archive/
```

Suggested moves:

```text
model_overviews/                     -> 00_main_story/model_overviews/
tier_extensions.ipynb                -> 00_main_story/tier_extensions.ipynb
sanity_checks.ipynb                  -> 02_sanity_checks/sanity_checks.ipynb

further_exploration/                 -> 01_mechanism/further_exploration/

sanity_checks/preprocessing_check/   -> 02_sanity_checks/preprocessing_check/
sanity_checks/tract_check/           -> 02_sanity_checks/tract_check/

tractography_predict/                -> 03_tractography/tractography_predict/

non-linear-sanity-check/             -> 04_nonlinear/nonlinear_sanity_check/

EDA/                                 -> 99_legacy_or_archive/EDA/
```

This is a starting structure, not a law. The key principle is that readers should be able
to tell whether a folder contains the main story, a mechanism probe, a sanity check, a
tractography test, a nonlinear robustness test, or the new reproducibility suite.

## Part C — Path And Link Audit

After moving files, update:

- Markdown links.
- Notebook relative imports and `sys.path` setup cells.
- Python script imports.
- CSV/result output paths.
- sbatch paths.
- README file indexes.
- references in `MASTER_FINDINGS.md`.

Useful checks:

```bash
rg "notebooks-FC_to_SC-experimental|further_exploration|tractography_predict|non-linear-sanity-check|sanity_checks"
rg "../|results/|local_results"
```

## Part D — Findings Audit

For every finding or sanity check, record:

```text
Claim:
Status: confirmatory / exploratory / stale / superseded / needs rerun
Evidence files:
Scripts:
Outputs:
Caveats:
Next action:
```

Important audit targets:

- JL versus PCA / reduction-axis checks.
- Whether FC and SC scans/inputs are normalized in the intended way.
- Demeaned Pearson formula validation.
- Tractography reliability checks:
  - edge strength proxy
  - distance proxy
  - FC scan-rescan reliability proxy
  - missing gold-standard SC ICC
- PC2 false-positive history and PC3/4/5 revised mechanism status.
- Which family-structure files are from older runs and need regeneration.
- Which derived quantities still lack bootstrap CIs.

## Part E — New Index Documents

Add or update:

```text
notebooks-FC_to_SC-experimental/README.md
notebooks-FC_to_SC-experimental/EVIDENCE_REGISTRY.md
notebooks-FC_to_SC-experimental/RUNBOOK.md
```

`EVIDENCE_REGISTRY.md` should map:

```text
Finding -> status -> scripts -> CSVs -> markdown docs -> caveats -> rerun command
```

`RUNBOOK.md` should explain how to rerun each suite after the paths are cleaned.

## Part F — Cleanup Commit Strategy

Suggested commit sequence:

1. `sync: preserve HPC-derived FC-to-SC analysis artifacts`
2. `docs: add FC-to-SC workspace cleanup plan`
3. `reorg: restructure FC-to-SC experimental workspace`
4. `docs: update FC-to-SC evidence registry and runbook`

Keep the physical move separate from findings edits where possible.

