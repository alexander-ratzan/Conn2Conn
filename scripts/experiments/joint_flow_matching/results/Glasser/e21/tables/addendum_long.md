# E2.1 addendum: un-anchored flows, targets CCA latent (test, 5 seeds; point metrics are means over the 8 conditions); flows retrained, cap 3000 epochs, min 50

| model | demeaned r | average rank | energy (lower is better) | set average rank | set top-1 | spread r |
|---|---|---|---|---|---|---|
| ridge_cond | 0.0490 | 0.6758 | — | — | — | — |
| gauss | — | — | 0.0462 | 0.8437 | 0.0649 | -0.0175 |
| flow | 0.0473 | 0.6704 | 0.0469 | 0.8483 | 0.0790 | 0.0001 |
| flow_linear | 0.0442 | 0.6600 | 0.0615 | 0.8232 | 0.0628 | 0.0406 |
| flow_free | 0.0376 | 0.6323 | 0.0924 | 0.7684 | 0.0606 | 0.0202 |
| flow_linear_shuffled | 0.0003 | 0.5060 | 0.0733 | 0.5118 | 0.0054 | 0.0482 |
| flow_free_shuffled | 0.0010 | 0.5047 | 0.1042 | 0.5149 | 0.0065 | 0.0032 |
| flow_shuffled | -0.0001 | 0.5038 | 0.0506 | 0.5004 | 0.0065 | 0.0145 |

flow_linear: demeaned r − ridge -0.0048 ± 0.0012 (0 / 5 seeds > 0); Gaussian − flow_linear energy -0.0153 ± 0.0071 (0 / 5 seeds flow better)

flow_free: demeaned r − ridge -0.0114 ± 0.0016 (0 / 5 seeds > 0); Gaussian − flow_free energy -0.0461 ± 0.0054 (0 / 5 seeds flow better)
