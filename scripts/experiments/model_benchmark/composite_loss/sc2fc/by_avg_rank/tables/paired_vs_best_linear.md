Paired per-seed differences vs `CrossModal_ConditionalGaussian_mse` (best input-matched linear model by mean test demeaned r; shared seeds). † = extra inputs, not input-matched.

| Model | Seeds | Δ Pearson r | Δ demeaned r | Δ avg rank | Δ top-1 |
|---|---|---|---|---|---|
| Conditional Gaussian (E2.2) | 5 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| PCA-PLS (E2.2) | 5 | +0.0020 ± 0.0033 | -0.0012 ± 0.0080 | -0.0346 ± 0.0234 | -0.0195 ± 0.0071 |
| PCA-PLS + covariates † | 5 | +0.0025 ± 0.0028 | -0.0093 ± 0.0099 | +0.1232 ± 0.0224 | +0.0636 ± 0.0189 |
| Krakencoder (paper loss) | 5 | -0.0009 ± 0.0028 | -0.0133 ± 0.0072 | +0.0577 ± 0.0237 | +0.0113 ± 0.0158 |
| Linear backbone | 5 | +0.0023 ± 0.0029 | -0.0187 ± 0.0052 | +0.1314 ± 0.0261 | +0.0472 ± 0.0174 |
| PCA-PLS learnable | 5 | +0.0018 ± 0.0030 | -0.0192 ± 0.0110 | +0.1121 ± 0.0262 | +0.0492 ± 0.0148 |
| Sarwar MLP | 5 | -0.0214 ± 0.0052 | -0.0409 ± 0.0093 | +0.0090 ± 0.0249 | -0.0062 ± 0.0150 |
| PCA null (MSE, E2.2) | 5 | -0.0082 ± 0.0035 | -0.0825 ± 0.0113 | -0.2282 ± 0.0330 | -0.0431 ± 0.0107 |
