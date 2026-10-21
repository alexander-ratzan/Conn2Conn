Paired per-seed differences vs `CrossModal_linear_backbone` (best input-matched linear model by mean test demeaned r; shared seeds). † = extra inputs, not input-matched.

| Model | Seeds | Δ Pearson r | Δ demeaned r | Δ avg rank | Δ top-1 |
|---|---|---|---|---|---|
| PCA-PLS + covariates † | 5 | -0.0124 ± 0.0038 | +0.0554 ± 0.0041 | +0.0644 ± 0.0204 | +0.2318 ± 0.0176 |
| Linear backbone | 5 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| PCA-PLS (E2.2) | 5 | -0.0010 ± 0.0005 | -0.0031 ± 0.0037 | -0.0034 ± 0.0112 | -0.0164 ± 0.0124 |
| PCA-PLS learnable | 5 | -0.0012 ± 0.0019 | -0.0049 ± 0.0063 | -0.0158 ± 0.0165 | -0.0103 ± 0.0166 |
| Conditional Gaussian (E2.2) | 5 | -0.0007 ± 0.0006 | -0.0186 ± 0.0050 | +0.0141 ± 0.0167 | +0.0410 ± 0.0161 |
| Sarwar MLP | 5 | -0.0176 ± 0.0074 | -0.0236 ± 0.0090 | -0.0356 ± 0.0263 | -0.0277 ± 0.0270 |
| Krakencoder (paper loss) | 5 | -0.0013 ± 0.0004 | -0.0457 ± 0.0068 | +0.0276 ± 0.0176 | +0.0359 ± 0.0179 |
| PCA null (MSE, E2.2) | 5 | -0.0526 ± 0.0035 | -0.1439 ± 0.0041 | -0.3574 ± 0.0151 | -0.1005 ± 0.0200 |
