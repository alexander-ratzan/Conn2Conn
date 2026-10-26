# E2.1 addendum: un-anchored flows (test, 5 seeds; point metrics are means over the 8 conditions)

| model | demeaned r | average rank | energy (lower is better) | set average rank | set top-1 | spread r |
|---|---|---|---|---|---|---|
| ridge_cond | 0.0490 | 0.6758 | — | — | — | — |
| gauss | — | — | 0.0462 | 0.8437 | 0.0649 | -0.0175 |
| flow | 0.0473 | 0.6704 | 0.0469 | 0.8483 | 0.0790 | 0.0001 |
| flow_linear | 0.0462 | 0.6685 | 0.0486 | 0.8418 | 0.0843 | 0.0269 |
| flow_free | 0.0444 | 0.6624 | 0.0481 | 0.8182 | 0.0637 | 0.0097 |
| flow_linear_shuffled | 0.0013 | 0.5081 | 0.0527 | 0.5186 | 0.0065 | 0.0232 |
| flow_free_shuffled | -0.0003 | 0.5023 | 0.0509 | 0.5043 | 0.0043 | 0.0200 |
| flow_shuffled | -0.0001 | 0.5038 | 0.0506 | 0.5004 | 0.0065 | 0.0145 |

flow_linear: demeaned r − ridge -0.0028 ± 0.0006 (0 / 5 seeds > 0); Gaussian − flow_linear energy -0.0023 ± 0.0003 (0 / 5 seeds flow better)

flow_free: demeaned r − ridge -0.0046 ± 0.0021 (0 / 5 seeds > 0); Gaussian − flow_free energy -0.0019 ± 0.0002 (0 / 5 seeds flow better)
