Paired per-seed differences vs `CrossModal_PCA_PLS_mse` (best input-matched linear model by mean test demeaned r; shared seeds). † = extra inputs, not input-matched.

| Model | Seeds | Δ Pearson r | Δ demeaned r | Δ avg rank | Δ top-1 |
|---|---|---|---|---|---|
| PCA-PLS + covariates † | 5 | -0.0037 ± 0.0015 | +0.0421 ± 0.0044 | +0.0949 ± 0.0093 | +0.2882 ± 0.0322 |
| PCA-PLS (E2.2) | 5 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| PCA-PLS learnable | 5 | -0.0002 ± 0.0009 | -0.0063 ± 0.0072 | +0.0256 ± 0.0066 | +0.0431 ± 0.0140 |
| Conditional Gaussian (E2.2) | 5 | +0.0003 ± 0.0004 | -0.0154 ± 0.0020 | +0.0175 ± 0.0068 | +0.0574 ± 0.0115 |
| Linear backbone | 5 | +0.0003 ± 0.0005 | -0.0230 ± 0.0141 | +0.0260 ± 0.0111 | +0.0451 ± 0.0057 |
| Krakencoder (paper loss) | 5 | -0.0003 ± 0.0003 | -0.0425 ± 0.0038 | +0.0310 ± 0.0110 | +0.0523 ± 0.0103 |
| Sarwar MLP | 5 | -0.0376 ± 0.0067 | -0.0703 ± 0.0081 | +0.0153 ± 0.0098 | +0.0379 ± 0.0242 |
| PCA null (MSE, E2.2) | 5 | -0.0517 ± 0.0034 | -0.1408 ± 0.0040 | -0.3540 ± 0.0051 | -0.0841 ± 0.0112 |
