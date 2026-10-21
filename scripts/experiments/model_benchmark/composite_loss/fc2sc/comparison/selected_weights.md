| Model | Selection | `varmatch` per seed (0-4) | `correye_dm` per seed (0-4) | `neidist` per seed (0-4) | Seeds with any term on |
|---|---|---|---|---|---|
| PCA-PLS learnable | selected on val demeaned r | 0.5, 0.1, 0.1, 0.1, 0 | 0, 0, 0.3, 0.5, 0.1 | 0.5, 0, 0.3, 0.3, 0.1 | 5/5 |
| PCA-PLS learnable | selected on val avg rank | 0.1, 0.1, 0, 0, 0 | 0.5, 0, 0.1, 0, 0.1 | 0.3, 0, 0.5, 0, 0.3 | 4/5 |
| Linear backbone | selected on val demeaned r | 0, 0.5, 0.5, 0, 0.1 | 0, 0, 0, 0.3, 0.1 | 0, 0.1, 0.1, 0.1, 0 | 4/5 |
| Linear backbone | selected on val avg rank | 0, 0.3, 0.3, 0.5, 0 | 0.5, 0, 0.1, 0.5, 0.1 | 0, 0.5, 0, 0, 0 | 5/5 |
| PCA-PLS + covariates | selected on val demeaned r | 0, 0.5, 0.3, 0.3, 0 | 0.5, 0, 0, 0.1, 0.5 | 0.3, 0.3, 0.3, 0.1, 0.5 | 5/5 |
| PCA-PLS + covariates | selected on val avg rank | 0, 0.3, 0.1, 0.3, 0.3 | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.3, 0.3, 0.3, 0.1, 0.1 | 5/5 |
| Sarwar MLP | selected on val demeaned r | 0.1, 0.1, 0.1, 0.3, 0.3 | 0.5, 0, 0, 0, 0.1 | 0.1, 0.3, 0.1, 0, 0 | 5/5 |
| Sarwar MLP | selected on val avg rank | 0.1, 0.3, 0.1, 0.3, 0.3 | 0, 0.1, 0.1, 0.1, 0.5 | 0.1, 0.3, 0.1, 0, 0.3 | 5/5 |
