# Updated Results Section - Full Reproduction Grid

This section supersedes the June 18 preliminary-results PDF for the confirmatory grid. It keeps the same scientific story, but uses the completed reproduction grid: 10 seeds, Glasser plus 4S456Parcels, PCA->PLS / BayesianRidge / KernelRidge, frozen family-aware splits, leak checks, and CSV-first provenance.

## Headline

The core result survives intact: FC->SC prediction is consistently stronger than SC->FC, subject-info baselines explain much of the apparent cross-modal signal, and observed FC is the only connectome representation that reliably adds cognition signal above bv+demo. The full grid adds two important refinements: (1) use BayesianRidge/KernelRidge same-modality rows for Ceiling B, not the bottlenecked PLS oracle; (2) state estimator next to every absolute number, because PLS, BayesianRidge, and KernelRidge can move magnitudes even when directions are stable.

## F1 - FC->SC > SC->FC
| Parcellation | FC->SC | SC->FC | ratio |
|---|---|---|---|
| Glasser | 0.136 | 0.084 | 1.61x |
| 4S456Parcels | 0.146 | 0.087 | 1.68x |

The asymmetry is now cross-parcellation evidence, not just Glasser reproduction. It clears the 1.15x gate in every seed on both parcellations.

## F2 - Anatomy predicts structure; demographics predict function
| Parcellation | bv->SC | demo->SC | bv->FC | demo->FC | bv+demo->SC | bv+demo->FC |
|---|---|---|---|---|---|---|
| Glasser | 0.162 | 0.130 | 0.049 | 0.111 | 0.169 | 0.100 |
| 4S456Parcels | 0.185 | 0.133 | 0.045 | 0.105 | 0.191 | 0.094 |

This is stronger as a 2x2 dissociation than as a single baseline result: anatomy wins for SC, demographics wins for FC. The 4S456 grid strengthens the structure side.

## Ceiling B - model oracle reconciliation
| Parcellation | PLS SC->SC | BR SC->SC | KR SC->SC | BR FC->FC |
|---|---|---|---|---|
| Glasser | 0.376 | 0.648 | 0.645 | 0.673 |
| 4S456Parcels | 0.386 | 0.622 | 0.618 | 0.632 |

The preliminary PDF's SC->SC = 0.647 was not contradicted. It matches the completed grid's BayesianRidge/KernelRidge same-modality oracle. The lower PLS SC->SC number is a bottlenecked PCA->PLS->inverse-PCA self-map and should not be used as the disattenuation denominator.

## F3 - imputation inherits source information, but does not beat bv+demo
| Parcellation | Target | pred_SC / obs_SC | pred_FC / obs_FC | utility asym |
|---|---|---|---|---|
| Glasser | CogTotal | 1.37x | 0.55x | 2.49x |
| Glasser | CogFluid | 1.41x | 0.57x | 2.48x |
| Glasser | CogCryst | 1.36x | 0.45x | 3.03x |
| 4S456Parcels | CogTotal | 1.38x | 0.61x | 2.27x |
| 4S456Parcels | CogFluid | 1.66x | 0.66x | 2.52x |
| 4S456Parcels | CogCryst | 1.32x | 0.52x | 2.56x |

The imputation asymmetry is stronger in the full grid than the preliminary PDF. However, the clean wording is now: pred_SC beats observed SC downstream because it inherits FC-like information, but pred_SC is roughly neutral against bv+demo. Pred_FC is actively harmful relative to bv+demo.

## F4 - residual cognition signal after bv+demo
| Parcellation | Input | Total | Fluid | Cryst |
|---|---|---|---|---|
| Glasser | obs_FC | 67% | 66% | 78% |
| Glasser | obs_SC | 16% | 5% | 24% |
| Glasser | pred_SC | 49% | 51% | 57% |
| Glasser | pred_FC | 17% | 27% | -0% |
| 4S456Parcels | obs_FC | 65% | 64% | 74% |
| 4S456Parcels | obs_SC | 22% | 19% | 26% |
| 4S456Parcels | pred_SC | 51% | 56% | 54% |
| 4S456Parcels | pred_FC | 29% | 43% | 10% |

Observed FC retains most of its cognition signal after removing bv+demo, especially for crystallized cognition. Observed SC mostly collapses. Pred_SC retains some FC-derived residual signal; pred_FC carries little residual functional cognition signal.

## F5 - cost-benefit baseline
| Parcellation | Input | Total r | Total lift | Total p | Cryst r | Cryst lift | Cryst p |
|---|---|---|---|---|---|---|---|
| Glasser | bv+demo | 0.359 | +0.000 | 1 | 0.354 | +0.000 | 1 |
| Glasser | obs_FC | 0.447 | +0.088 | 0.0802 | 0.487 | +0.133 | 0.0307 |
| Glasser | obs_SC | 0.248 | -0.111 | 0.939 | 0.253 | -0.101 | 0.817 |
| Glasser | obs_FC+bv+demo | 0.474 | +0.115 | 0.0157 | 0.516 | +0.162 | 0.00225 |
| Glasser | obs_SC+bv+demo | 0.309 | -0.050 | 0.824 | 0.319 | -0.035 | 0.604 |
| 4S456Parcels | bv+demo | 0.359 | +0.000 | 1 | 0.354 | +0.000 | 1 |
| 4S456Parcels | obs_FC | 0.445 | +0.086 | 0.0905 | 0.476 | +0.122 | 0.0407 |
| 4S456Parcels | obs_SC | 0.256 | -0.103 | 0.927 | 0.259 | -0.095 | 0.883 |
| 4S456Parcels | obs_FC+bv+demo | 0.468 | +0.109 | 0.0192 | 0.500 | +0.146 | 0.008 |
| 4S456Parcels | obs_SC+bv+demo | 0.308 | -0.052 | 0.831 | 0.312 | -0.042 | 0.695 |

The clean citable claim is: bv+demo is the required baseline; observed FC adds real cognition signal, strongest for CogCryst; observed SC underperforms the cheap baseline. The most direct 'add fMRI to cheap baseline' comparison is obs_FC+bv+demo, which is significant for CogTotal and CogCryst on both parcellations in the grid.

## New grid takeaways worth exploring

- The asymmetry is not a target-reliability artifact: BR/KR FC->FC and SC->SC oracles are nearly matched, while cross-modal FC->SC still beats SC->FC by about 1.6-1.7x.
- 4S456 specifically boosts ->SC prediction, especially bv->SC and bv+demo->SC, while ->FC changes little or drops slightly. This looks like finer parcellation adding structural detail rather than generic metric noise.
- SC can be actively counterproductive downstream: obs_SC and obs_SC+bv+demo stay below bv+demo for cognition. That is a stronger message than 'SC adds little.'
- pred_FC is worse than useless for cognition in the grid, consistently below bv+demo and below obs_FC. That sharpens the non-substitutability result.
- The PC3/visual-DAN mechanism remains the main non-confirmatory piece. F1-F5 now replicate across parcellations; F8 still needs a 4S456 mechanism pass before it can be phrased as more than hypothesis-generating.

## Suggested write-up posture

Lead with the baseline and ceiling logic rather than biomarker language. The field story is: simple deterministic models reproduce the FC->SC asymmetry, but subject-level anatomy and demographics explain a large fraction of apparent connectome signal; observed FC is the only modality with reliable cognition lift above that floor; richer SC, imputed FC, and nonlinear models do not rescue the cognition claim in HCP-YA.
