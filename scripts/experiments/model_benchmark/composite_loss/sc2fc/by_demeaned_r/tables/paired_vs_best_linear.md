Paired per-seed differences vs `CrossModal_linear_backbone` (best input-matched linear model by mean test demeaned r; shared seeds). † = extra inputs, not input-matched.

| Model | Seeds | Δ Pearson r | Δ demeaned r | Δ avg rank | Δ top-1 |
|---|---|---|---|---|---|
| PCA-PLS + covariates † | 5 | -0.0008 ± 0.0009 | +0.0070 ± 0.0083 | -0.0169 ± 0.0189 | +0.0031 ± 0.0130 |
| Linear backbone | 5 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| Conditional Gaussian (E2.2) | 5 | -0.0029 ± 0.0029 | -0.0049 ± 0.0035 | -0.0156 ± 0.0195 | +0.0021 ± 0.0050 |
| PCA-PLS learnable | 5 | -0.0018 ± 0.0012 | -0.0054 ± 0.0027 | -0.0236 ± 0.0192 | -0.0113 ± 0.0127 |
| PCA-PLS (E2.2) | 5 | -0.0008 ± 0.0008 | -0.0061 ± 0.0048 | -0.0502 ± 0.0165 | -0.0174 ± 0.0080 |
| Krakencoder (paper loss) | 5 | -0.0038 ± 0.0002 | -0.0183 ± 0.0043 | +0.0421 ± 0.0232 | +0.0133 ± 0.0187 |
| Sarwar MLP | 5 | -0.0512 ± 0.0102 | -0.0255 ± 0.0075 | -0.0672 ± 0.0225 | -0.0246 ± 0.0135 |
| PCA null (MSE, E2.2) | 5 | -0.0111 ± 0.0008 | -0.0874 ± 0.0080 | -0.2438 ± 0.0243 | -0.0410 ± 0.0139 |
