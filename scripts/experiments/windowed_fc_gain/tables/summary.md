# windowed_fc_gain — gates 1-6 summary

Estimator `demean_concat`; thresholds (config, fixed before results): G1 r ≥ 0.98, G2 halves demeaned avg rank ≥ 0.9, G3 ≥ 0.95, G4 shift ratio ≤ 0.25, G6 included ≥ 0.85.

| condition | timing_group | G1 r(est, target) | G1 | G2 halves avg_rank_dm | G2 | G3 halves avg_rank_dm | G3 | G4 halves shift | G4 | G6 included | G6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rest | rest | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.140 | pass | 0.920 | pass |
| emotion | matched | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.026 | pass | 0.996 | pass |
| gambling | matched | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.028 | pass | 0.990 | pass |
| language | unmatched | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.060 | pass | 0.960 | pass |
| motor | unmatched | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.038 | pass | 0.998 | pass |
| relational | matched | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.029 | pass | 0.846 | FAIL |
| social | mostly_matched | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.039 | pass | 0.987 | pass |
| wm | unmatched | 1.000 | pass | 1.000 | pass | 1.000 | pass | 0.091 | pass | 0.916 | pass |

Part schemes (G2 part vs part, G3 part vs full target; demeaned identification, chance 0.5):

| condition | scheme | s | G2 avg rank dm | G2 top-1 dm | G3 avg rank dm | G3 top-1 dm | G3 own r dm | mean of parts G3 avg rank dm | G4 shift |
|---|---|---|---|---|---|---|---|---|---|
| rest | halves | 1728 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 0.140 |
| rest | quarters | 864 | 1.000 | 0.998 | 1.000 | 1.000 | 0.810 | 1.000 | 0.211 |
| rest | direction | 1728 | 0.999 | 0.973 | 1.000 | 1.000 | 0.871 | 1.000 | 0.189 |
| rest | session | 1728 | 0.996 | 0.925 | 1.000 | 1.000 | 0.870 | 1.000 | 0.118 |
| emotion | halves | 127 | 1.000 | 1.000 | 1.000 | 1.000 | 0.934 | 1.000 | 0.026 |
| emotion | quarters | 63 | 1.000 | 1.000 | 1.000 | 1.000 | 0.837 | 1.000 | 0.036 |
| emotion | direction | 127 | 0.792 | 0.044 | 1.000 | 0.999 | 0.697 | 1.000 | 0.119 |
| gambling | halves | 181 | 1.000 | 1.000 | 1.000 | 1.000 | 0.924 | 1.000 | 0.028 |
| gambling | quarters | 91 | 1.000 | 1.000 | 1.000 | 1.000 | 0.818 | 1.000 | 0.040 |
| gambling | direction | 182 | 0.910 | 0.199 | 1.000 | 0.999 | 0.717 | 1.000 | 0.145 |
| language | halves | 228 | 1.000 | 1.000 | 1.000 | 1.000 | 0.915 | 1.000 | 0.060 |
| language | quarters | 114 | 1.000 | 1.000 | 1.000 | 1.000 | 0.796 | 1.000 | 0.087 |
| language | direction | 228 | 0.970 | 0.407 | 1.000 | 1.000 | 0.733 | 1.000 | 0.243 |
| motor | halves | 204 | 1.000 | 1.000 | 1.000 | 1.000 | 0.919 | 1.000 | 0.038 |
| motor | quarters | 102 | 1.000 | 1.000 | 1.000 | 1.000 | 0.806 | 1.000 | 0.054 |
| motor | direction | 204 | 0.911 | 0.149 | 1.000 | 1.000 | 0.712 | 1.000 | 0.181 |
| relational | halves | 167 | 1.000 | 1.000 | 1.000 | 1.000 | 0.927 | 1.000 | 0.029 |
| relational | quarters | 84 | 1.000 | 1.000 | 1.000 | 1.000 | 0.822 | 1.000 | 0.041 |
| relational | direction | 167 | 0.870 | 0.112 | 1.000 | 1.000 | 0.708 | 1.000 | 0.118 |
| social | halves | 197 | 1.000 | 1.000 | 1.000 | 1.000 | 0.922 | 1.000 | 0.039 |
| social | quarters | 98 | 1.000 | 1.000 | 1.000 | 1.000 | 0.813 | 1.000 | 0.058 |
| social | direction | 197 | 0.903 | 0.159 | 1.000 | 1.000 | 0.720 | 1.000 | 0.176 |
| wm | halves | 291 | 1.000 | 1.000 | 1.000 | 1.000 | 0.895 | 1.000 | 0.091 |
| wm | quarters | 145 | 1.000 | 1.000 | 1.000 | 1.000 | 0.759 | 1.000 | 0.128 |
| wm | direction | 292 | 0.983 | 0.554 | 1.000 | 1.000 | 0.747 | 1.000 | 0.203 |
