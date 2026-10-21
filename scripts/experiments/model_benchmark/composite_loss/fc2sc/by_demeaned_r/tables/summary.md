| Model | Type | Seeds | Pearson r | Demeaned r | Average rank | Top-1 accuracy |
|---|---|---|---|---|---|---|
| PCA-PLS + covariates † | Linear (learned) | 5 | 0.9030 ± 0.0039 | 0.2059 ± 0.0031 | 0.9347 ± 0.0118 | 0.3415 ± 0.0226 |
| Linear backbone | Linear (learned) | 5 | 0.9153 ± 0.0003 | 0.1505 ± 0.0062 | 0.8703 ± 0.0187 | 0.1097 ± 0.0198 |
| PCA-PLS (E2.2) | Linear (closed-form) | 5 | 0.9143 ± 0.0004 | 0.1473 ± 0.0038 | 0.8669 ± 0.0100 | 0.0933 ± 0.0110 |
| PCA-PLS learnable | Linear (learned) | 5 | 0.9141 ± 0.0017 | 0.1455 ± 0.0083 | 0.8546 ± 0.0268 | 0.0995 ± 0.0266 |
| Conditional Gaussian (E2.2) | Linear (closed-form) | 5 | 0.9146 ± 0.0005 | 0.1319 ± 0.0022 | 0.8844 ± 0.0082 | 0.1508 ± 0.0151 |
| Sarwar MLP | Deep-learning baseline | 5 | 0.8977 ± 0.0072 | 0.1268 ± 0.0043 | 0.8347 ± 0.0112 | 0.0821 ± 0.0104 |
| Krakencoder (paper loss) | Deep-learning baseline | 5 | 0.9140 ± 0.0002 | 0.1048 ± 0.0016 | 0.8979 ± 0.0053 | 0.1456 ± 0.0066 |
| PCA null (MSE, E2.2) | Null / ceiling | 5 | 0.8627 ± 0.0034 | 0.0065 ± 0.0034 | 0.5129 ± 0.0057 | 0.0092 ± 0.0038 |
