| Model | Type | Seeds | Pearson r | Demeaned r | Average rank | Top-1 accuracy |
|---|---|---|---|---|---|---|
| PCA-PLS + covariates † | Linear (learned) | 5 | 0.9106 ± 0.0017 | 0.1894 ± 0.0057 | 0.9619 ± 0.0059 | 0.3815 ± 0.0401 |
| PCA-PLS (E2.2) | Linear (closed-form) | 5 | 0.9143 ± 0.0004 | 0.1473 ± 0.0038 | 0.8669 ± 0.0100 | 0.0933 ± 0.0110 |
| PCA-PLS learnable | Linear (learned) | 5 | 0.9142 ± 0.0009 | 0.1411 ± 0.0085 | 0.8926 ± 0.0104 | 0.1364 ± 0.0163 |
| Conditional Gaussian (E2.2) | Linear (closed-form) | 5 | 0.9146 ± 0.0005 | 0.1319 ± 0.0022 | 0.8844 ± 0.0082 | 0.1508 ± 0.0151 |
| Linear backbone | Linear (learned) | 5 | 0.9147 ± 0.0002 | 0.1243 ± 0.0121 | 0.8930 ± 0.0058 | 0.1385 ± 0.0078 |
| Krakencoder (paper loss) | Deep-learning baseline | 5 | 0.9140 ± 0.0002 | 0.1048 ± 0.0016 | 0.8979 ± 0.0053 | 0.1456 ± 0.0066 |
| Sarwar MLP | Deep-learning baseline | 5 | 0.8767 ± 0.0068 | 0.0770 ± 0.0056 | 0.8823 ± 0.0066 | 0.1313 ± 0.0180 |
| PCA null (MSE, E2.2) | Null / ceiling | 5 | 0.8627 ± 0.0034 | 0.0065 ± 0.0034 | 0.5129 ± 0.0057 | 0.0092 ± 0.0038 |
