# Cross-model benchmark (spec v2 E2.2: MSE-only)

**Status:** E2.2.1 build done; pilot not yet run · **Owner:** agent:modeling · **Spec:** `context_packages/repo_spec_docs/spec_doc_v2.md` E2.2

## Question

With every model tuned on the same splits under MSE only, how are test Pearson r, demeaned r, average rank and top-1
accuracy distributed across model types? The ranking picks the models for E2.3 (composite loss, tuned weights);
E3 repeats the benchmark for FC → SC.

## Layout

```
models/configs/benchmark/mse/<Model>.yml        # one MSE-only benchmark config per model (both directions)
models/configs/benchmark/mse/<Model>_fc2sc.yml  # only where FC -> SC needs a different model setting (Sarwar)
scripts/experiments/model_benchmark/
├── build_configs.py      # generates the configs above from the model defaults + declared narrowing (SPECS)
├── config.yml            # roster per direction, model types (figure groups), packing, reused results, latent gate
├── submit.py             # submits one SLURM array per (model, direction): job name e2_mse_<Model>_<direction>
├── launch_model.sh       # one seed per array task: Tune + best-trial report
├── run.py                # task logs -> records.json -> tables + figures, per direction
├── checks/check_benchmark.py
└── sc2fc/ , fc2sc/       # records.json, tables/, figures/ (generated)
```

## How to run

| Step | Command |
|---|---|
| (Re)generate configs | `python scripts/experiments/model_benchmark/build_configs.py` (`--check` to verify) |
| Pilot (1 seed per model + latent gate on seeds 0–1) | `python scripts/experiments/model_benchmark/submit.py --direction sc2fc --stage pilot` |
| Full run | `python scripts/experiments/model_benchmark/submit.py --direction sc2fc --stage full [--models ...]` |
| Collect + figures | `python scripts/experiments/model_benchmark/run.py --direction sc2fc` (`--cached` re-renders) |
| Checks (CPU) | `python scripts/experiments/model_benchmark/checks/check_benchmark.py` |

## Figures (`<direction>/figures/`)

`bars_<metric>.png` for Pearson r, demeaned r, average rank and top-1 accuracy, plus `bars_all_metrics.png` (2 × 2):
bars grouped and coloured by model type, groups sorted by their mean on the metric and models within a group by their
own mean; mean ± SE over seeds with per-seed points; native-objective models hatched (*); PCA null as a dotted
line; test-retest ceiling as a dashed line, or noted in the title when off scale.

## Results

Pending.
