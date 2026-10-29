# E6 — fingerprint embedding checks (test, seeds 0–4, mean ± SE)

Closed form B = (8/7)W − I/7 vs the fitted map: max |diff| 9.4e-16. Directions with fingerprint share w > 0.5: seed 0 56, seed 1 56, seed 2 55, seed 3 56, seed 4 56 of 256.

## 1. Rest session 1 ↔ session 2 (different days)

| model | average rank | top-1 | demeaned r |
|---|---|---|---|
| copy (codes) | 0.967 ± 0.003 | 0.789 ± 0.006 | 0.374 ± 0.002 |
| copy (edges) | 0.989 ± 0.003 | 0.947 ± 0.007 | 0.494 ± 0.004 |
| embedding | 0.994 ± 0.001 | 0.891 ± 0.010 | 0.294 ± 0.002 |

## 2a. Confound proxies → representation (test R²)

| representation | of | R² |
|---|---|---|
| codes | fingerprint | 0.039 ± 0.002 |
| codes | rest | 0.143 ± 0.007 |
| embedding | fingerprint | 0.054 ± 0.003 |
| embedding | rest | 0.048 ± 0.002 |

## 2b. Cross-condition identification (56 pairs, code space), confounds removed or not

| representation | confounds removed | average rank | top-1 |
|---|---|---|---|
| codes | False | 0.832 ± 0.002 | 0.153 ± 0.003 |
| codes | True | 0.830 ± 0.002 | 0.154 ± 0.002 |
| embedding | False | 0.965 ± 0.001 | 0.540 ± 0.004 |
| embedding | True | 0.962 ± 0.001 | 0.529 ± 0.005 |

## 3. Behaviour (test Pearson r of ridge predictions)

| target | fingerprint | fingerprint embedding | rest codes | rest edges | rest embedding |
|---|---|---|---|---|---|
| CogCrystalComp_AgeAdj | 0.493 ± 0.033 | 0.464 ± 0.035 | 0.482 ± 0.025 | 0.509 ± 0.020 | 0.408 ± 0.032 |
| CogFluidComp_AgeAdj | 0.389 ± 0.051 | 0.360 ± 0.035 | 0.277 ± 0.045 | 0.297 ± 0.050 | 0.249 ± 0.022 |
| CogTotalComp_AgeAdj | 0.502 ± 0.040 | 0.488 ± 0.038 | 0.424 ± 0.028 | 0.472 ± 0.025 | 0.370 ± 0.029 |
| PMAT24_A_CR | 0.385 ± 0.028 | 0.365 ± 0.027 | 0.294 ± 0.028 | 0.353 ± 0.030 | 0.275 ± 0.033 |

## 4. SC → targets

Against the own training target (codes):

| route | R² | average rank | top-1 |
|---|---|---|---|
| SC → fingerprint (ridge) | 0.021 ± 0.002 | 0.825 ± 0.012 | 0.096 ± 0.014 |
| SC → fingerprint embedding (ridge) | 0.073 ± 0.005 | 0.852 ± 0.007 | 0.122 ± 0.017 |
| SC → rest FC (ridge) | 0.010 ± 0.003 | 0.720 ± 0.013 | 0.038 ± 0.006 |

Against measured rest FC (decoded edges):

| route | average rank | top-1 | demeaned r |
|---|---|---|---|
| SC → condition, multiview CCA (E4 B2) | 0.719 ± 0.004 | 0.056 ± 0.007 | 0.097 ± 0.002 |
| SC → condition, ridge (E4 B1) | 0.726 ± 0.013 | 0.041 ± 0.006 | 0.092 ± 0.005 |
| SC → fingerprint (ridge) | 0.800 ± 0.009 | 0.061 ± 0.004 | 0.089 ± 0.002 |
| SC → fingerprint embedding (ridge) | 0.822 ± 0.005 | 0.067 ± 0.004 | 0.096 ± 0.002 |
| SC → rest FC (ridge) | 0.726 ± 0.012 | 0.041 ± 0.004 | 0.091 ± 0.004 |

Against measured FC, mean over the 8 conditions (decoded edges):

| route | average rank | top-1 | demeaned r |
|---|---|---|---|
| SC → condition, multiview CCA (E4 B2) | 0.661 ± 0.004 | 0.029 ± 0.002 | 0.046 ± 0.002 |
| SC → condition, ridge (E4 B1) | 0.661 ± 0.004 | 0.018 ± 0.001 | 0.044 ± 0.001 |
| SC → fingerprint (ridge) | 0.722 ± 0.006 | 0.037 ± 0.003 | 0.048 ± 0.002 |
| SC → fingerprint embedding (ridge) | 0.744 ± 0.006 | 0.043 ± 0.005 | 0.052 ± 0.002 |
| SC → rest FC (ridge) | 0.662 ± 0.006 | 0.025 ± 0.002 | 0.038 ± 0.002 |
