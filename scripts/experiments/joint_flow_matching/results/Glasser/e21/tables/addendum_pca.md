# E2.1 addendum: un-anchored flows, targets block FC PCA codes (no CCA) (test, 5 seeds; point metrics are means over the 8 conditions)

| model | demeaned r | average rank | energy (lower is better) | set average rank | set top-1 | spread r |
|---|---|---|---|---|---|---|
| ridge_cond | 0.0441 | 0.6621 | — | — | — | — |
| gauss | — | — | 0.0028 | 0.6386 | 0.0238 | -0.0300 |
| flow_linear | 0.0379 | 0.6793 | 0.0030 | 0.7326 | 0.0498 | 0.0427 |
| flow_free | 0.0400 | 0.6645 | 0.0027 | 0.7130 | 0.0335 | 0.0087 |
| flow_linear_shuffled | -0.0001 | 0.5043 | 0.0032 | 0.4989 | 0.0000 | 0.0481 |
| flow_free_shuffled | 0.0006 | 0.5037 | 0.0029 | 0.4995 | 0.0043 | 0.0066 |

flow_linear: demeaned r − ridge -0.0061 ± 0.0008 (0 / 5 seeds > 0); Gaussian − flow_linear energy -0.0002 ± 0.0000 (0 / 5 seeds flow better)

flow_free: demeaned r − ridge -0.0041 ± 0.0007 (0 / 5 seeds > 0); Gaussian − flow_free energy +0.0001 ± 0.0000 (5 / 5 seeds flow better)
