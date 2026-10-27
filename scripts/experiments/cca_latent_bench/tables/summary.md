# E1 summary (seeds 0, 1, 2, 3, 4; test split; headline readout ridge)

| H | comparison | metric | Δ ± SE | seeds > 0 | supported | note |
|---|---|---|---|---|---|---|
| H1 | rest CCA encoding − rest PCA codes (FC → SC) | demeaned_pearson | -0.0095 ± 0.0020 | 0 / 5 | False |  |
| H1 | rest CCA encoding − rest PCA codes (FC → SC) | avg_rank | -0.0336 ± 0.0070 | 0 / 5 | False |  |
| H1 | rest CCA encoding − rest PCA codes (FC → SC) | top1_acc | -0.0333 ± 0.0168 | 1 / 5 | False |  |
| H2 | 8-condition consensus − rest encoding (FC → SC) | demeaned_pearson | +0.0075 ± 0.0017 | 5 / 5 | True |  |
| H2 (reported) | 7-task consensus − rest encoding | demeaned_pearson | +0.0045 ± 0.0018 | 4 / 5 | True |  |
| H2 (reported) | concatenated encodings − consensus | demeaned_pearson | -0.0012 ± 0.0006 | 1 / 5 | False |  |
| H3 | 8-condition consensus − v3:E0 rest (PCA-PLS learnable) | demeaned_pearson | -0.0094 ± 0.0051 | 1 / 5 | False | non-inferior (margin 0.01): False |
| H4 | SC → consensus latent → each FC condition − v3:E0 SC → FC (mean over 8 conditions) | demeaned_pearson | +0.0202 ± 0.0025 | 5 / 5 | True |  |
| H4 | SC → consensus latent → condition − direct SC → condition ridge (mean over 8) | demeaned_pearson | +0.0062 ± 0.0015 | 5 / 5 | True |  |

