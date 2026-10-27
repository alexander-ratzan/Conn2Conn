# E2.1 addendum: un-anchored flows, targets block FC PCA codes (no CCA) (test, 5 seeds; point metrics are means over the 8 conditions); flows retrained, cap 3000 epochs, min 50

| model | demeaned r | average rank | energy (lower is better) | set average rank | set top-1 | spread r |
|---|---|---|---|---|---|---|
| ridge_cond | 0.0441 | 0.6621 | — | — | — | — |
| gauss | — | — | 0.0028 | 0.6386 | 0.0238 | -0.0300 |
| flow_linear | 0.0380 | 0.6793 | 0.0030 | 0.7360 | 0.0487 | -0.0230 |
| flow_free | 0.0382 | 0.6638 | 0.0028 | 0.7000 | 0.0270 | 0.0107 |
| flow_linear_shuffled | -0.0001 | 0.5046 | 0.0032 | 0.5014 | 0.0043 | -0.0231 |
| flow_free_shuffled | 0.0003 | 0.5015 | 0.0029 | 0.4936 | 0.0022 | -0.0116 |

flow_linear: demeaned r − ridge -0.0061 ± 0.0009 (0 / 5 seeds > 0); Gaussian − flow_linear energy -0.0002 ± 0.0000 (0 / 5 seeds flow better)

flow_free: demeaned r − ridge -0.0058 ± 0.0007 (0 / 5 seeds > 0); Gaussian − flow_free energy +0.0000 ± 0.0000 (3 / 5 seeds flow better)
