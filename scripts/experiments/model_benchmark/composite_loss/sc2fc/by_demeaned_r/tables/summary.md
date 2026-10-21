| Model | Type | Seeds | Pearson r | Demeaned r | Average rank | Top-1 accuracy |
|---|---|---|---|---|---|---|
| PCA-PLS + covariates † | Linear (learned) | 5 | 0.8354 ± 0.0021 | 0.1053 ± 0.0076 | 0.7428 ± 0.0127 | 0.0564 ± 0.0065 |
| Linear backbone | Linear (learned) | 5 | 0.8362 ± 0.0017 | 0.0983 ± 0.0043 | 0.7597 ± 0.0206 | 0.0533 ± 0.0155 |
| Conditional Gaussian (E2.2) | Linear (closed-form) | 5 | 0.8333 ± 0.0044 | 0.0934 ± 0.0073 | 0.7440 ± 0.0253 | 0.0554 ± 0.0121 |
| PCA-PLS learnable | Linear (learned) | 5 | 0.8344 ± 0.0028 | 0.0929 ± 0.0049 | 0.7361 ± 0.0236 | 0.0421 ± 0.0059 |
| PCA-PLS (E2.2) | Linear (closed-form) | 5 | 0.8354 ± 0.0021 | 0.0922 ± 0.0052 | 0.7095 ± 0.0130 | 0.0359 ± 0.0087 |
| Krakencoder (paper loss) | Deep-learning baseline | 5 | 0.8324 ± 0.0018 | 0.0801 ± 0.0011 | 0.8017 ± 0.0062 | 0.0667 ± 0.0046 |
| Sarwar MLP | Deep-learning baseline | 5 | 0.7850 ± 0.0107 | 0.0729 ± 0.0058 | 0.6924 ± 0.0205 | 0.0287 ± 0.0068 |
| PCA null (MSE, E2.2) | Null / ceiling | 5 | 0.8251 ± 0.0013 | 0.0110 ± 0.0048 | 0.5158 ± 0.0088 | 0.0123 ± 0.0042 |
