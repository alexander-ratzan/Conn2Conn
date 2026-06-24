# Main tables

_Generated from the reproduction-grid CSVs; see `make_tables.py`._


## Table 1 · Dataset and evaluation

| Field | Value |
|---|---|
| Cohort | HCP Young Adults (FC + SC + demographics + brain-volume) |
| Subjects (per split) | ~878 (683 train / 195 test, family-aware) |
| Splits | 10 frozen, family-aware (siblings/twins kept on one side) |
| Parcellation A | Glasser — 360 regions, 64,620 edges |
| Parcellation B | 4S456Parcels — 456 regions, 103,740 edges |
| Connectome rep. | upper-triangle edge vector |
| Primary recon metric | demeaned_pearson (per-subject, train-mean removed) |
| Secondary metrics | avg_rank, top1_acc (fingerprint), r2/mse |
| Recon estimator | PCA(256)→PLS(64)→inverse-PCA (+ BayesianRidge, KernelRidge controls) |
| Downstream estimator | BayesianRidge (reportable for scalar targets) |


## Table 2 · Task grid and claims

| Task | Input | Target | Estimator | Metric | Claim tested |
|---|---|---|---|---|---|
| Reconstruction | FC | SC | PCA→PLS | demeaned_pearson | cross-modal FC→SC |
| Reconstruction | SC | FC | PCA→PLS | demeaned_pearson | cross-modal SC→FC |
| Oracle | FC | FC | BayesianRidge | demeaned_pearson | within-modality ceiling |
| Oracle | SC | SC | BayesianRidge | demeaned_pearson | within-modality ceiling |
| Baseline | bv+demo | SC/FC | PCA→PLS | demeaned_pearson | subject-info → connectome |
| Downstream | bv+demo | Cognition | BayesianRidge | pearson / lift | checkpoint baseline |
| Downstream | obs_FC | Cognition | BayesianRidge | lift_over_bvdemo | does FC beat baseline? |
| Downstream | obs_SC | Cognition | BayesianRidge | lift_over_bvdemo | does SC beat baseline? |
| Downstream | pred_SC / pred_FC | Cognition | BayesianRidge | lift_over_bvdemo | does imputation transfer? |
| Leak check | any | sex / age | BayesianRidge | bal_acc / pearson | diagnostic, not outcome |
| Family | pred_SC variants | MZ/DZ/sib | — | AUC | identity signal survives? |


## Table 3 · Headline findings

| Finding | Glasser | 4S456 | Estimator | Interpretation |
|---|---|---|---|---|
| FC→SC reconstruction | 0.136 | 0.146 | PCA→PLS | FC recovers SC deviations |
| SC→FC reconstruction | 0.084 | 0.087 | PCA→PLS | weaker reverse direction |
| FC→SC / SC→FC ratio | 1.615× | 1.679× | PCA→PLS | directional asymmetry |
| FC→FC oracle ceiling | 0.673 | 0.632 | BayesianRidge | self-predictability ~equal |
| SC→SC oracle ceiling | 0.648 | 0.622 | BayesianRidge | self-predictability ~equal |
| bv+demo → CogCryst | 0.354 | 0.354 | BayesianRidge | the checkpoint to beat |
| obs FC CogCryst lift | +0.133 | +0.122 | BayesianRidge | FC clears the baseline |
| obs SC CogCryst lift | -0.101 | -0.095 | BayesianRidge | SC below baseline |
| pred FC CogCryst lift | -0.135 | -0.109 | BayesianRidge | imputation is harmful |
