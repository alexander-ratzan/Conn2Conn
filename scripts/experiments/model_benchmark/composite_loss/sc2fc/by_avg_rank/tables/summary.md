| Model | Type | Seeds | Pearson r | Demeaned r | Average rank | Top-1 accuracy |
|---|---|---|---|---|---|---|
| Conditional Gaussian (E2.2) | Linear (closed-form) | 5 | 0.8333 ± 0.0044 | 0.0934 ± 0.0073 | 0.7440 ± 0.0253 | 0.0554 ± 0.0121 |
| PCA-PLS (E2.2) | Linear (closed-form) | 5 | 0.8354 ± 0.0021 | 0.0922 ± 0.0052 | 0.7095 ± 0.0130 | 0.0359 ± 0.0087 |
| PCA-PLS + covariates † | Linear (learned) | 5 | 0.8359 ± 0.0018 | 0.0841 ± 0.0038 | 0.8672 ± 0.0108 | 0.1190 ± 0.0152 |
| Krakencoder (paper loss) | Deep-learning baseline | 5 | 0.8324 ± 0.0018 | 0.0801 ± 0.0011 | 0.8017 ± 0.0062 | 0.0667 ± 0.0046 |
| Linear backbone | Linear (learned) | 5 | 0.8356 ± 0.0017 | 0.0747 ± 0.0034 | 0.8755 ± 0.0104 | 0.1026 ± 0.0175 |
| PCA-PLS learnable | Linear (learned) | 5 | 0.8351 ± 0.0020 | 0.0742 ± 0.0049 | 0.8561 ± 0.0042 | 0.1046 ± 0.0142 |
| Sarwar MLP | Deep-learning baseline | 5 | 0.8119 ± 0.0072 | 0.0525 ± 0.0048 | 0.7530 ± 0.0179 | 0.0492 ± 0.0101 |
| PCA null (MSE, E2.2) | Null / ceiling | 5 | 0.8251 ± 0.0013 | 0.0110 ± 0.0048 | 0.5158 ± 0.0088 | 0.0123 ± 0.0042 |
