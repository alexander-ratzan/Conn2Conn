# E2.1 addendum: un-anchored flows, targets block FC PCA codes (no CCA) (test, 5 seeds; point metrics are means over the 8 conditions); flows retrained, cap 3000 epochs, min 50

| model | demeaned r | average rank | energy (lower is better) | set average rank | set top-1 | spread r |
|---|---|---|---|---|---|---|
| ridge_cond | 0.0441 | 0.6621 | — | — | — | — |
| gauss | — | — | 0.0028 | 0.6386 | 0.0238 | -0.0300 |
| flow_linear | 0.0402 | 0.6921 | 0.0086 | 0.7760 | 0.0671 | -0.0039 |
| flow_free | 0.0345 | 0.6710 | 0.0094 | 0.7288 | 0.0368 | -0.0301 |
| flow_linear_shuffled | -0.0004 | 0.5032 | 0.0088 | 0.4977 | 0.0119 | -0.0007 |
| flow_free_shuffled | 0.0023 | 0.4970 | 0.0098 | 0.4976 | 0.0076 | -0.0682 |

flow_linear: demeaned r − ridge -0.0038 ± 0.0007 (0 / 5 seeds > 0); Gaussian − flow_linear energy -0.0058 ± 0.0000 (0 / 5 seeds flow better)

flow_free: demeaned r − ridge -0.0096 ± 0.0029 (0 / 5 seeds > 0); Gaussian − flow_free energy -0.0066 ± 0.0001 (0 / 5 seeds flow better)
