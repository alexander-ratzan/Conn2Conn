# E6 — fingerprint embedding checks (test, seeds 0–4, mean ± SE)

Closed form B = (8/7)W − I/7 vs the fitted map: max |diff| 9.4e-16. Directions with fingerprint share w > 0.5: seed 0 56, seed 1 56, seed 2 55, seed 3 56, seed 4 56 of 256.

## SC → rest FC routes

| route | group | what SC is trained to predict |
|---|---|---|
| 1  Rest codes | single condition | SC → the subject's rest FC (block-PCA codes), ridge |
| 2  CCA, SC as a view | CCA latents | CCA over SC + 8 FC conditions; SC's own encoder into the latent (no regression) |
| 3  CCA consensus | CCA latents | CCA over the 8 FC conditions; target = mean of the subject's 8 encodings (E1), ridge |
| 4  Mean fingerprint | block-PCA fingerprints | target = mean of the subject's 8 conditions' block-PCA codes, ridge |
| 5  Fingerprint embedding | block-PCA fingerprints | target = mean fingerprint × pooled map (reliability-weighted), ridge |

Stage 1, pairing with the route's own target (not comparable across routes):

| route | R² | average rank | top-1 |
|---|---|---|---|
| 1  Rest codes | 0.010 ± 0.003 | 0.720 ± 0.013 | 0.038 ± 0.006 |
| 2  CCA, SC as a view | -0.219 ± 0.067 | 0.802 ± 0.021 | 0.042 ± 0.005 |
| 3  CCA consensus | 0.019 ± 0.018 | 0.891 ± 0.007 | 0.129 ± 0.017 |
| 3b CCA consensus, E1's latent (d 32, λ 0.1) | 0.048 ± 0.012 | 0.870 ± 0.007 | 0.088 ± 0.010 |
| 4  Mean fingerprint | 0.021 ± 0.002 | 0.825 ± 0.012 | 0.096 ± 0.014 |
| 5  Fingerprint embedding | 0.073 ± 0.005 | 0.852 ± 0.007 | 0.122 ± 0.017 |

Stage 2, predicted vs measured rest FC (ceiling: the true target decoded the same way):

| route | average rank | top-1 | demeaned r | ceiling average rank | ceiling demeaned r |
|---|---|---|---|---|---|
| 1  Rest codes | 0.726 ± 0.012 | 0.041 ± 0.004 | 0.091 ± 0.004 | 1.000 | 0.678 |
| 2  CCA, SC as a view | 0.732 ± 0.010 | 0.036 ± 0.007 | 0.100 ± 0.004 | 0.996 | 0.459 |
| 3  CCA consensus | 0.749 ± 0.013 | 0.037 ± 0.006 | 0.101 ± 0.005 | 0.985 | 0.351 |
| 3b CCA consensus, E1's latent (d 32, λ 0.1) | 0.741 ± 0.010 | 0.028 ± 0.007 | 0.100 ± 0.005 | 0.969 | 0.303 |
| 4  Mean fingerprint | 0.800 ± 0.009 | 0.061 ± 0.004 | 0.089 ± 0.002 | 1.000 | 0.421 |
| 5  Fingerprint embedding | 0.822 ± 0.005 | 0.067 ± 0.004 | 0.096 ± 0.002 | 0.998 | 0.339 |
| E4.0c B1+SC (= route 1) | 0.726 ± 0.013 | 0.041 ± 0.006 | 0.092 ± 0.005 | – | – |
| E4.0c B2 (CCA chosen on FC → FC) | 0.719 ± 0.004 | 0.056 ± 0.007 | 0.097 ± 0.002 | – | – |

Against measured FC, mean over the 8 conditions:

| route | average rank | top-1 | demeaned r |
|---|---|---|---|
| 1  Rest codes | 0.662 ± 0.006 | 0.025 ± 0.002 | 0.038 ± 0.002 |
| 2  CCA, SC as a view | 0.663 ± 0.005 | 0.019 ± 0.002 | 0.047 ± 0.003 |
| 3  CCA consensus | 0.682 ± 0.006 | 0.022 ± 0.003 | 0.049 ± 0.002 |
| 3b CCA consensus, E1's latent (d 32, λ 0.1) | 0.677 ± 0.006 | 0.019 ± 0.002 | 0.050 ± 0.003 |
| 4  Mean fingerprint | 0.722 ± 0.006 | 0.037 ± 0.003 | 0.048 ± 0.002 |
| 5  Fingerprint embedding | 0.744 ± 0.006 | 0.043 ± 0.005 | 0.052 ± 0.002 |
| E4.0c B1+SC (= route 1) | 0.661 ± 0.004 | 0.018 ± 0.001 | 0.044 ± 0.001 |
| E4.0c B2 (CCA chosen on FC → FC) | 0.661 ± 0.004 | 0.029 ± 0.002 | 0.046 ± 0.002 |

## Rest session 1 ↔ session 2 (different days)

| prediction | average rank | top-1 | demeaned r |
|---|---|---|---|
| copy (codes) | 0.967 ± 0.003 | 0.789 ± 0.006 | 0.374 ± 0.002 |
| copy (edges) | 0.989 ± 0.003 | 0.947 ± 0.007 | 0.494 ± 0.004 |
| embedding | 0.994 ± 0.001 | 0.891 ± 0.010 | 0.294 ± 0.002 |

## Confound proxies

| representation | of | R² explained by the proxies |
|---|---|---|
| codes | fingerprint | 0.039 ± 0.002 |
| codes | rest | 0.143 ± 0.007 |
| embedding | fingerprint | 0.054 ± 0.003 |
| embedding | rest | 0.048 ± 0.002 |

| representation | proxies removed | cross-condition average rank | top-1 |
|---|---|---|---|
| codes | False | 0.832 ± 0.002 | 0.153 ± 0.003 |
| codes | True | 0.830 ± 0.002 | 0.154 ± 0.002 |
| embedding | False | 0.965 ± 0.001 | 0.540 ± 0.004 |
| embedding | True | 0.962 ± 0.001 | 0.529 ± 0.005 |

## Behaviour (test r)

| target | fingerprint | fingerprint embedding | rest codes | rest edges | rest embedding |
|---|---|---|---|---|---|
| CogCrystalComp_AgeAdj | 0.493 ± 0.033 | 0.464 ± 0.035 | 0.482 ± 0.025 | 0.509 ± 0.020 | 0.408 ± 0.032 |
| CogFluidComp_AgeAdj | 0.389 ± 0.051 | 0.360 ± 0.035 | 0.277 ± 0.045 | 0.297 ± 0.050 | 0.249 ± 0.022 |
| CogTotalComp_AgeAdj | 0.502 ± 0.040 | 0.488 ± 0.038 | 0.424 ± 0.028 | 0.472 ± 0.025 | 0.370 ± 0.029 |
| PMAT24_A_CR | 0.385 ± 0.028 | 0.365 ± 0.027 | 0.294 ± 0.028 | 0.353 ± 0.030 | 0.275 ± 0.033 |
