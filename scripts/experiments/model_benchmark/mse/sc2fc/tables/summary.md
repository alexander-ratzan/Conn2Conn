| Model | Type | Seeds | Pearson r | Demeaned r | Average rank | Top-1 accuracy |
|---|---|---|---|---|---|---|
| Test-retest ceiling | Null / ceiling | 5 | 0.8151 ± 0.0031 | 0.4924 ± 0.0040 | 0.9866 ± 0.0042 | 0.9251 ± 0.0114 |
| Linear backbone | Linear (learned) | 5 | 0.8362 ± 0.0016 | 0.0983 ± 0.0043 | 0.7439 ± 0.0170 | 0.0328 ± 0.0090 |
| PCA-PLS + covariates † | Linear (learned) | 5 | 0.8363 ± 0.0018 | 0.0974 ± 0.0080 | 0.6961 ± 0.0174 | 0.0390 ± 0.0077 |
| Conditional Gaussian | Linear (closed-form) | 5 | 0.8333 ± 0.0044 | 0.0934 ± 0.0073 | 0.7440 ± 0.0253 | 0.0554 ± 0.0121 |
| PCA-PLS | Linear (closed-form) | 5 | 0.8354 ± 0.0021 | 0.0922 ± 0.0052 | 0.7095 ± 0.0130 | 0.0359 ± 0.0087 |
| Masked MLP pretrainer | Latent / pretrained | 5 | 0.8327 ± 0.0019 | 0.0900 ± 0.0053 | 0.7080 ± 0.0118 | 0.0359 ± 0.0043 |
| Krakencoder (MSE) | Deep-learning baseline | 5 | 0.8321 ± 0.0018 | 0.0860 ± 0.0031 | 0.6825 ± 0.0028 | 0.0462 ± 0.0067 |
| PCA-PLS learnable | Linear (learned) | 5 | 0.8354 ± 0.0019 | 0.0860 ± 0.0074 | 0.6709 ± 0.0251 | 0.0287 ± 0.0095 |
| Krakencoder (paper loss) | Deep-learning baseline | 5 | 0.8324 ± 0.0018 | 0.0801 ± 0.0011 | 0.8017 ± 0.0062 | 0.0667 ± 0.0046 |
| Sarwar MLP | Deep-learning baseline | 5 | 0.8202 ± 0.0021 | 0.0682 ± 0.0053 | 0.6449 ± 0.0143 | 0.0308 ± 0.0054 |
| PLS-SVD | Linear (closed-form) | 5 | 0.8245 ± 0.0064 | 0.0658 ± 0.0079 | 0.6352 ± 0.0207 | 0.0287 ± 0.0101 |
| Chen GCN | Deep-learning baseline | 5 | 0.7580 ± 0.0020 | 0.0216 ± 0.0021 | 0.5867 ± 0.0053 | 0.0164 ± 0.0041 |
| Nodal MLP | Pairwise nodal | 5 | 0.7508 ± 0.0037 | 0.0120 ± 0.0053 | 0.5572 ± 0.0025 | 0.0082 ± 0.0013 |
| Nodal GNN | Pairwise nodal | 5 | 0.7189 ± 0.0220 | 0.0111 ± 0.0060 | 0.5486 ± 0.0058 | 0.0082 ± 0.0026 |
| PCA null | Null / ceiling | 5 | 0.8251 ± 0.0013 | 0.0110 ± 0.0048 | 0.5158 ± 0.0088 | 0.0123 ± 0.0042 |
