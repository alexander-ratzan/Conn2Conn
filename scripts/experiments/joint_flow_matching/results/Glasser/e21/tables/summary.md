# E2.1 summary (test, 5 seeds; point metrics averaged over the 8 conditions within a seed)

| H | comparison | metric | Δ ± SE | seeds > 0 | supported | note |
|---|---|---|---|---|---|---|
| H1 | ridge + MLP corrector − ridge (SC → task latent → task FC) | demeaned r | +0.0003 ± 0.0001 | 4 / 5 | True |  |
| H2 | corrector + identity loss − corrector | average rank | -0.0053 ± 0.0023 | 1 / 5 | False | demeaned r Δ -0.0024 ± 0.0006; non-inferior (> −0.005): True |
| H3 (primary) | linear-Gaussian − flow (energy distance to each subject's 8 encodings; > 0 = flow better) | energy distance | -0.0007 ± 0.0002 | 0 / 5 | False |  |
| H4 | flow − linear-Gaussian (set-to-set identification) | set average rank | +0.0046 ± 0.0035 | 4 / 5 | False | shuffled-SC flow set average rank 0.500 (chance 0.5) |
| H5 | flow spread correlation with each subject's observed spread (vs 0) | correlation | +0.0001 ± 0.0272 | 2 / 5 | False |  |
| H5 (diagnostic) | flow − spread-only null (SC shuffled for the residual only): spread correlation | correlation | +0.0004 ± 0.0003 | 3 / 5 | False |  |
| H5 (diagnostic) | rest test–retest: flow's predicted rest spread vs each subject's session 1 − session 2 difference | correlation | -0.0095 ± 0.0110 | 2 / 5 | False | spread ratio flow 7.45, Gaussian 7.02 (1 = calibrated) |
| H6 | flow sample mean − corrector (point accuracy) | demeaned r | -0.0021 ± 0.0003 | 0 / 5 | False | non-inferior (mean − 2 SE > −0.005): True |
| H7 | latent → SC: ridge + MLP corrector − ridge (mean over consensus and 7 task inputs) | SC demeaned r | +0.0041 ± 0.0010 | 5 / 5 | True |  |
