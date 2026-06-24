# Preprint Figure Manifest

Generated with:

```bash
/Users/user/dev-env/bin/python preprint/preprint-codex/scripts/make_preprint_figures.py
```

Output directory:

`/Users/user/projects/Conn2Conn/preprint/preprint-codex/figures`

## Main Figures

| Figure | File stem | Manuscript role |
|---|---|---|
| Figure 1 | `fig1_evaluation_spine` | Evaluation design, frozen splits, reconstruction vs utility branches, and sanity checks |
| Figure 2 | `fig2_directional_translation` | FC->SC vs SC->FC asymmetry, seed ratios, same-modality oracles, anatomy/demo double dissociation |
| Figure 3 | `fig3_utility_checkpoint` | Cognition prediction and lift over `bv+demo`, observed vs imputed connectome contrast |
| Figure 4 | `fig4_closed_escape_routes` | Oracle fractions, reduction robustness, tractography/null nonlinear checks, scaling, and FC-noise accounting |
| Figure 5 | `fig5_signal_changes_with_objective` | Family AUCs, reconstruction/identification tradeoff, and property-selected mechanism mode |

Each figure is written as both PNG and PDF.

## Read-Only Sources

- `reproduction/outputs/reconstruction.csv`
- `reproduction/outputs/downstream.csv`
- `reproduction/family_mechanism/outputs/family_auc.csv`
- `reproduction/family_mechanism/outputs/f8_stability.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/reduction_axis_synthesis.csv`
- `notebooks-FC_to_SC-experimental/tractography_predict/e1_source_rep_results.csv`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/n2_reconstruction_summary.csv`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/n6_scaling_summary.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/outputs/a_reliability_ceiling.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/outputs/h_reliability_filtered_summary.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/outputs/h_correlations.csv`

## Notes

- The figures are manuscript-facing composites, not replacements for the exploratory figures in `reproduction/exploration/figures`.
- Figure 5 follows the manuscript caveat: the mechanism panel highlights Glasser PC3 and 4S456 PC4 as property-selected low-variance modes rather than treating one PC index as universal.
- Figure 4 keeps FC test-retest reliability as an FC-side accounting result and uses the same-modality oracle separately.
