# windowed_fc_gain G7 — does training on part samples help?

Test split, mean over the 8 conditions, mean ± SE over seeds (n = seeds). Energy: lower is better.

| route | model | arm | n | demeaned r | avg rank | top-1 | energy | set avg rank | spread ratio |
|---|---|---|---|---|---|---|---|---|---|
| cca | gauss_fullcov | none | 5 | 0.0490 ± 0.0023 | 0.6758 ± 0.0048 | 0.0208 ± 0.0039 | 0.0462 ± 0.0005 | 0.8437 ± 0.0134 | 2.4701 ± 0.0323 |
| cca | gauss_fullcov | aug_halves | 5 | 0.0488 ± 0.0025 | 0.6753 ± 0.0054 | 0.0205 ± 0.0038 | 0.0464 ± 0.0006 | 0.8387 ± 0.0109 | 2.4697 ± 0.0327 |
| cca | gauss_fullcov | token_halves | 5 | 0.0488 ± 0.0025 | 0.6753 ± 0.0054 | 0.0205 ± 0.0038 | 0.0464 ± 0.0006 | 0.8387 ± 0.0109 | 2.4697 ± 0.0327 |
| cca | gauss_fullcov | aug_quarters | 5 | 0.0488 ± 0.0024 | 0.6756 ± 0.0053 | 0.0213 ± 0.0037 | 0.0462 ± 0.0005 | 0.8465 ± 0.0137 | 2.4730 ± 0.0339 |
| cca | gauss_fullcov | token_quarters | 5 | 0.0488 ± 0.0024 | 0.6756 ± 0.0053 | 0.0213 ± 0.0037 | 0.0462 ± 0.0005 | 0.8465 ± 0.0137 | 2.4730 ± 0.0339 |
| cca | gauss_own | none | 5 | 0.0490 ± 0.0023 | 0.6758 ± 0.0048 | 0.0208 ± 0.0039 | 0.0462 ± 0.0005 | 0.8437 ± 0.0134 | 2.4701 ± 0.0323 |
| cca | gauss_own | aug_halves | 5 | 0.0488 ± 0.0025 | 0.6753 ± 0.0054 | 0.0205 ± 0.0038 | 0.0467 ± 0.0006 | 0.8441 ± 0.0119 | 2.5553 ± 0.0335 |
| cca | gauss_own | token_halves | 5 | 0.0488 ± 0.0025 | 0.6753 ± 0.0054 | 0.0205 ± 0.0038 | 0.0467 ± 0.0006 | 0.8441 ± 0.0119 | 2.5553 ± 0.0335 |
| cca | gauss_own | parts_halves | 5 | 0.0487 ± 0.0025 | 0.6750 ± 0.0053 | 0.0203 ± 0.0037 | 0.0468 ± 0.0004 | 0.8474 ± 0.0079 | 2.6009 ± 0.0341 |
| cca | gauss_own | aug_quarters | 5 | 0.0488 ± 0.0024 | 0.6756 ± 0.0053 | 0.0213 ± 0.0037 | 0.0477 ± 0.0004 | 0.8463 ± 0.0103 | 2.7541 ± 0.0370 |
| cca | gauss_own | token_quarters | 5 | 0.0488 ± 0.0024 | 0.6756 ± 0.0053 | 0.0213 ± 0.0037 | 0.0477 ± 0.0004 | 0.8463 ± 0.0103 | 2.7541 ± 0.0370 |
| cca | gauss_own | parts_quarters | 5 | 0.0487 ± 0.0025 | 0.6760 ± 0.0056 | 0.0209 ± 0.0035 | 0.0480 ± 0.0004 | 0.8496 ± 0.0058 | 2.8247 ± 0.0365 |
| cca | ridge | none | 5 | 0.0490 ± 0.0023 | 0.6758 ± 0.0048 | 0.0208 ± 0.0039 | – | – | – |
| cca | ridge | aug_halves | 5 | 0.0488 ± 0.0025 | 0.6753 ± 0.0054 | 0.0205 ± 0.0038 | – | – | – |
| cca | ridge | token_halves | 5 | 0.0488 ± 0.0025 | 0.6753 ± 0.0054 | 0.0205 ± 0.0038 | – | – | – |
| cca | ridge | parts_halves | 5 | 0.0487 ± 0.0025 | 0.6750 ± 0.0053 | 0.0203 ± 0.0037 | – | – | – |
| cca | ridge | aug_quarters | 5 | 0.0488 ± 0.0024 | 0.6756 ± 0.0053 | 0.0213 ± 0.0037 | – | – | – |
| cca | ridge | token_quarters | 5 | 0.0488 ± 0.0024 | 0.6756 ± 0.0053 | 0.0213 ± 0.0037 | – | – | – |
| cca | ridge | parts_quarters | 5 | 0.0487 ± 0.0025 | 0.6760 ± 0.0056 | 0.0209 ± 0.0035 | – | – | – |
| pca | gauss_fullcov | none | 5 | 0.0441 ± 0.0015 | 0.6621 ± 0.0044 | 0.0176 ± 0.0016 | 0.0028 ± 0.0000 | 0.6386 ± 0.0105 | 1.5085 ± 0.0105 |
| pca | gauss_fullcov | aug_halves | 5 | 0.0439 ± 0.0015 | 0.6629 ± 0.0046 | 0.0180 ± 0.0013 | 0.0028 ± 0.0000 | 0.6460 ± 0.0147 | 1.5084 ± 0.0104 |
| pca | gauss_fullcov | token_halves | 5 | 0.0439 ± 0.0015 | 0.6629 ± 0.0046 | 0.0180 ± 0.0013 | 0.0028 ± 0.0000 | 0.6460 ± 0.0147 | 1.5084 ± 0.0104 |
| pca | gauss_fullcov | aug_quarters | 5 | 0.0434 ± 0.0015 | 0.6641 ± 0.0044 | 0.0190 ± 0.0019 | 0.0028 ± 0.0000 | 0.6600 ± 0.0136 | 1.5098 ± 0.0102 |
| pca | gauss_fullcov | token_quarters | 5 | 0.0434 ± 0.0015 | 0.6641 ± 0.0044 | 0.0190 ± 0.0019 | 0.0028 ± 0.0000 | 0.6600 ± 0.0136 | 1.5098 ± 0.0102 |
| pca | gauss_own | none | 5 | 0.0441 ± 0.0015 | 0.6621 ± 0.0044 | 0.0176 ± 0.0016 | 0.0028 ± 0.0000 | 0.6386 ± 0.0105 | 1.5085 ± 0.0105 |
| pca | gauss_own | aug_halves | 5 | 0.0439 ± 0.0015 | 0.6629 ± 0.0046 | 0.0180 ± 0.0013 | 0.0029 ± 0.0000 | 0.6294 ± 0.0089 | 1.6098 ± 0.0111 |
| pca | gauss_own | token_halves | 5 | 0.0439 ± 0.0015 | 0.6629 ± 0.0046 | 0.0180 ± 0.0013 | 0.0029 ± 0.0000 | 0.6294 ± 0.0089 | 1.6098 ± 0.0111 |
| pca | gauss_own | parts_halves | 5 | 0.0437 ± 0.0015 | 0.6633 ± 0.0047 | 0.0184 ± 0.0015 | 0.0029 ± 0.0000 | 0.6307 ± 0.0104 | 1.6613 ± 0.0115 |
| pca | gauss_own | aug_quarters | 5 | 0.0434 ± 0.0015 | 0.6641 ± 0.0044 | 0.0190 ± 0.0019 | 0.0031 ± 0.0000 | 0.6489 ± 0.0069 | 1.8330 ± 0.0125 |
| pca | gauss_own | token_quarters | 5 | 0.0434 ± 0.0015 | 0.6641 ± 0.0044 | 0.0190 ± 0.0019 | 0.0031 ± 0.0000 | 0.6489 ± 0.0069 | 1.8330 ± 0.0125 |
| pca | gauss_own | parts_quarters | 5 | 0.0427 ± 0.0015 | 0.6636 ± 0.0045 | 0.0196 ± 0.0018 | 0.0032 ± 0.0000 | 0.6383 ± 0.0098 | 1.9145 ± 0.0133 |
| pca | ridge | none | 5 | 0.0441 ± 0.0015 | 0.6621 ± 0.0044 | 0.0176 ± 0.0016 | – | – | – |
| pca | ridge | aug_halves | 5 | 0.0439 ± 0.0015 | 0.6629 ± 0.0046 | 0.0180 ± 0.0013 | – | – | – |
| pca | ridge | token_halves | 5 | 0.0439 ± 0.0015 | 0.6629 ± 0.0046 | 0.0180 ± 0.0013 | – | – | – |
| pca | ridge | parts_halves | 5 | 0.0437 ± 0.0015 | 0.6633 ± 0.0047 | 0.0184 ± 0.0015 | – | – | – |
| pca | ridge | aug_quarters | 5 | 0.0434 ± 0.0015 | 0.6641 ± 0.0044 | 0.0190 ± 0.0019 | – | – | – |
| pca | ridge | token_quarters | 5 | 0.0434 ± 0.0015 | 0.6641 ± 0.0044 | 0.0190 ± 0.0019 | – | – | – |
| pca | ridge | parts_quarters | 5 | 0.0427 ± 0.0015 | 0.6636 ± 0.0045 | 0.0196 ± 0.0018 | – | – | – |

## G7 verdicts (supported = seed-mean beyond 2 SE, same sign in ≥ 4 / 5 seeds; signs: + = better)

| model | route | arm | demeaned_pearson vs none | demeaned_pearson vs linear | avg_rank vs none | avg_rank vs linear | energy vs none | energy vs linear | supported gains (vs none and linear) | supported losses vs none | G7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss_fullcov | cca | aug_halves | -0.0002 | -0.0002 | -0.0005 | -0.0005 | 0.0001 | 0.0001 |  |  | False |
| gauss_fullcov | cca | aug_quarters | -0.0003 | -0.0003 | -0.0002 | -0.0002 | -0.0000 | -0.0000 |  |  | False |
| gauss_fullcov | cca | token_halves | -0.0002 | -0.0002 | -0.0005 | -0.0005 | 0.0001 | 0.0001 |  |  | False |
| gauss_fullcov | cca | token_quarters | -0.0003 | -0.0003 | -0.0002 | -0.0002 | -0.0000 | -0.0000 |  |  | False |
| gauss_fullcov | pca | aug_halves | -0.0002 | -0.0002 | 0.0008 | 0.0008 | -0.0000 | -0.0000 | avg_rank |  | True |
| gauss_fullcov | pca | aug_quarters | -0.0007 | -0.0007 | 0.0020 | 0.0020 | -0.0000 | -0.0000 |  |  | False |
| gauss_fullcov | pca | token_halves | -0.0002 | -0.0002 | 0.0008 | 0.0008 | -0.0000 | -0.0000 | avg_rank |  | True |
| gauss_fullcov | pca | token_quarters | -0.0007 | -0.0007 | 0.0020 | 0.0020 | -0.0000 | -0.0000 |  |  | False |
| gauss_own | cca | aug_halves | -0.0002 | -0.0002 | -0.0005 | -0.0005 | 0.0004 | 0.0004 |  | energy | False |
| gauss_own | cca | aug_quarters | -0.0003 | -0.0003 | -0.0002 | -0.0002 | 0.0014 | 0.0014 |  | energy | False |
| gauss_own | cca | parts_halves | -0.0003 | -0.0003 | -0.0008 | -0.0008 | 0.0005 | 0.0005 |  | energy | False |
| gauss_own | cca | parts_quarters | -0.0004 | -0.0004 | 0.0002 | 0.0002 | 0.0018 | 0.0018 |  | energy | False |
| gauss_own | cca | token_halves | -0.0002 | -0.0002 | -0.0005 | -0.0005 | 0.0004 | 0.0004 |  | energy | False |
| gauss_own | cca | token_quarters | -0.0003 | -0.0003 | -0.0002 | -0.0002 | 0.0014 | 0.0014 |  | energy | False |
| gauss_own | pca | aug_halves | -0.0002 | -0.0002 | 0.0008 | 0.0008 | 0.0001 | 0.0001 | avg_rank | energy | False |
| gauss_own | pca | aug_quarters | -0.0007 | -0.0007 | 0.0020 | 0.0020 | 0.0003 | 0.0003 |  | energy | False |
| gauss_own | pca | parts_halves | -0.0004 | -0.0004 | 0.0012 | 0.0012 | 0.0001 | 0.0001 | avg_rank | demeaned_pearson, energy | False |
| gauss_own | pca | parts_quarters | -0.0014 | -0.0014 | 0.0016 | 0.0016 | 0.0004 | 0.0004 |  | energy | False |
| gauss_own | pca | token_halves | -0.0002 | -0.0002 | 0.0008 | 0.0008 | 0.0001 | 0.0001 | avg_rank | energy | False |
| gauss_own | pca | token_quarters | -0.0007 | -0.0007 | 0.0020 | 0.0020 | 0.0003 | 0.0003 |  | energy | False |
| ridge | cca | aug_halves | -0.0002 | -0.0002 | -0.0005 | -0.0005 | – | – |  |  | False |
| ridge | cca | aug_quarters | -0.0003 | -0.0003 | -0.0002 | -0.0002 | – | – |  |  | False |
| ridge | cca | parts_halves | -0.0003 | -0.0003 | -0.0008 | -0.0008 | – | – |  |  | False |
| ridge | cca | parts_quarters | -0.0004 | -0.0004 | 0.0002 | 0.0002 | – | – |  |  | False |
| ridge | cca | token_halves | -0.0002 | -0.0002 | -0.0005 | -0.0005 | – | – |  |  | False |
| ridge | cca | token_quarters | -0.0003 | -0.0003 | -0.0002 | -0.0002 | – | – |  |  | False |
| ridge | pca | aug_halves | -0.0002 | -0.0002 | 0.0008 | 0.0008 | – | – | avg_rank |  | True |
| ridge | pca | aug_quarters | -0.0007 | -0.0007 | 0.0020 | 0.0020 | – | – |  |  | False |
| ridge | pca | parts_halves | -0.0004 | -0.0004 | 0.0012 | 0.0012 | – | – | avg_rank | demeaned_pearson | False |
| ridge | pca | parts_quarters | -0.0014 | -0.0014 | 0.0016 | 0.0016 | – | – |  |  | False |
| ridge | pca | token_halves | -0.0002 | -0.0002 | 0.0008 | 0.0008 | – | – | avg_rank |  | True |
| ridge | pca | token_quarters | -0.0007 | -0.0007 | 0.0020 | 0.0020 | – | – |  |  | False |
