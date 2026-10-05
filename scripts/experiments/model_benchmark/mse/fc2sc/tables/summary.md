| Model | Type | Seeds | Pearson r | Demeaned r | Average rank | Top-1 accuracy |
|---|---|---|---|---|---|---|
| PCA-PLS + covariates † | Linear (learned) | 5 | 0.9179 ± 0.0002 | 0.2209 ± 0.0031 | 0.9732 ± 0.0014 | 0.4472 ± 0.0166 |
| PCA-PLS learnable | Linear (learned) | 5 | 0.9162 ± 0.0002 | 0.1617 ± 0.0050 | 0.8889 ± 0.0087 | 0.1292 ± 0.0154 |
| Linear backbone | Linear (learned) | 5 | 0.9157 ± 0.0003 | 0.1577 ± 0.0019 | 0.8586 ± 0.0116 | 0.0759 ± 0.0124 |
| PCA-PLS | Linear (closed-form) | 5 | 0.9143 ± 0.0004 | 0.1473 ± 0.0038 | 0.8669 ± 0.0100 | 0.0933 ± 0.0110 |
| Krakencoder (MSE) | Deep-learning baseline | 5 | 0.9150 ± 0.0002 | 0.1341 ± 0.0020 | 0.9019 ± 0.0050 | 0.1364 ± 0.0084 |
| Conditional Gaussian | Linear (closed-form) | 5 | 0.9146 ± 0.0005 | 0.1319 ± 0.0022 | 0.8844 ± 0.0082 | 0.1508 ± 0.0151 |
| Sarwar MLP | Deep-learning baseline | 5 | 0.9150 ± 0.0002 | 0.1309 ± 0.0057 | 0.8643 ± 0.0143 | 0.0882 ± 0.0164 |
| PLS-SVD | Linear (closed-form) | 5 | 0.9087 ± 0.0015 | 0.1273 ± 0.0016 | 0.8610 ± 0.0052 | 0.1179 ± 0.0138 |
| Masked MLP pretrainer ‡ | Latent / pretrained | 5 | 0.9141 ± 0.0003 | 0.1217 ± 0.0020 | 0.8320 ± 0.0035 | 0.0933 ± 0.0062 |
| Krakencoder (paper loss) | Deep-learning baseline | 5 | 0.9140 ± 0.0002 | 0.1048 ± 0.0016 | 0.8979 ± 0.0053 | 0.1456 ± 0.0066 |
| PCA null | Null / ceiling | 5 | 0.8627 ± 0.0034 | 0.0065 ± 0.0034 | 0.5129 ± 0.0057 | 0.0092 ± 0.0038 |
