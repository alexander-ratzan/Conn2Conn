Test demeaned r by condition (mean over 5 seeds):

| label | E0.1 Glasser fc2sc | E0.2 Glasser sc2fc | E0.3 4S456Parcels fc2sc | E0.4 4S456Parcels sc2fc |
|---|---|---|---|---|
| Rest (4 runs) | 0.1483 | 0.0848 | 0.1379 | 0.0548 |
| Rest S1 (2 runs) | 0.1280 | 0.0525 | 0.1204 | 0.0423 |
| Emotion | 0.0565 | 0.0127 | 0.0421 | 0.0122 |
| Gambling | 0.0684 | 0.0187 | 0.0715 | 0.0248 |
| Language | 0.0577 | 0.0374 | 0.0901 | 0.0290 |
| Motor | 0.0610 | 0.0206 | 0.0687 | 0.0201 |
| Relational | 0.0462 | 0.0162 | 0.0601 | 0.0152 |
| Social | 0.0723 | 0.0199 | 0.0804 | 0.0286 |
| Working memory | 0.1078 | 0.0299 | 0.1005 | 0.0386 |

Condition-order agreement (Spearman over condition means):

| metric | step_a | step_b | spearman_all | spearman_tasks |
|---|---|---|---|---|
| pearson | E0.1 Glasser fc2sc | E0.2 Glasser sc2fc | 0.88 | 0.79 |
| pearson | E0.1 Glasser fc2sc | E0.3 4S456Parcels fc2sc | 0.47 | -0.07 |
| pearson | E0.1 Glasser fc2sc | E0.4 4S456Parcels sc2fc | 0.87 | 0.75 |
| pearson | E0.2 Glasser sc2fc | E0.3 4S456Parcels fc2sc | 0.42 | -0.11 |
| pearson | E0.2 Glasser sc2fc | E0.4 4S456Parcels sc2fc | 0.98 | 0.96 |
| pearson | E0.3 4S456Parcels fc2sc | E0.4 4S456Parcels sc2fc | 0.33 | -0.25 |
| demeaned_pearson | E0.1 Glasser fc2sc | E0.2 Glasser sc2fc | 0.77 | 0.50 |
| demeaned_pearson | E0.1 Glasser fc2sc | E0.3 4S456Parcels fc2sc | 0.88 | 0.75 |
| demeaned_pearson | E0.1 Glasser fc2sc | E0.4 4S456Parcels sc2fc | 0.88 | 0.75 |
| demeaned_pearson | E0.2 Glasser sc2fc | E0.3 4S456Parcels fc2sc | 0.93 | 0.86 |
| demeaned_pearson | E0.2 Glasser sc2fc | E0.4 4S456Parcels sc2fc | 0.93 | 0.86 |
| demeaned_pearson | E0.3 4S456Parcels fc2sc | E0.4 4S456Parcels sc2fc | 1.00 | 1.00 |
| avg_rank | E0.1 Glasser fc2sc | E0.2 Glasser sc2fc | 0.55 | 0.04 |
| avg_rank | E0.1 Glasser fc2sc | E0.3 4S456Parcels fc2sc | 0.78 | 0.54 |
| avg_rank | E0.1 Glasser fc2sc | E0.4 4S456Parcels sc2fc | 0.60 | 0.46 |
| avg_rank | E0.2 Glasser sc2fc | E0.3 4S456Parcels fc2sc | 0.80 | 0.57 |
| avg_rank | E0.2 Glasser sc2fc | E0.4 4S456Parcels sc2fc | 0.67 | 0.50 |
| avg_rank | E0.3 4S456Parcels fc2sc | E0.4 4S456Parcels sc2fc | 0.90 | 0.96 |
| top1_acc | E0.1 Glasser fc2sc | E0.2 Glasser sc2fc | 0.28 | -0.29 |
| top1_acc | E0.1 Glasser fc2sc | E0.3 4S456Parcels fc2sc | 0.48 | -0.07 |
| top1_acc | E0.1 Glasser fc2sc | E0.4 4S456Parcels sc2fc | 0.68 | 0.57 |
| top1_acc | E0.2 Glasser sc2fc | E0.3 4S456Parcels fc2sc | 0.52 | 0.14 |
| top1_acc | E0.2 Glasser sc2fc | E0.4 4S456Parcels sc2fc | 0.17 | -0.18 |
| top1_acc | E0.3 4S456Parcels fc2sc | E0.4 4S456Parcels sc2fc | 0.72 | 0.64 |
