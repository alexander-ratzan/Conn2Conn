# E4.4 — flows vs baselines (test, seeds 0–4, k = 256)

Setting means (mean ± SE over seeds):

| setting | model | demeaned r | average rank | top-1 | energy score | coverage 90 | spread-error r |
|---|---|---|---|---|---|---|---|
| single source | B0 | 0.1071 ± 0.0007 | 0.8445 ± 0.0027 | 0.2493 ± 0.0037 | – | – | – |
| single source | B0-codes | 0.0952 ± 0.0008 | 0.7795 ± 0.0025 | 0.1301 ± 0.0013 | – | – | – |
| single source | B1 | 0.0986 ± 0.0014 | 0.7837 ± 0.0018 | 0.1077 ± 0.0028 | – | – | – |
| single source | B1-pooled | 0.1238 ± 0.0008 | 0.9161 ± 0.0014 | 0.3243 ± 0.0032 | – | – | – |
| single source | B1-pooled-tgt | 0.1187 ± 0.0012 | 0.8427 ± 0.0016 | 0.1787 ± 0.0028 | – | – | – |
| single source | B2 | 0.1047 ± 0.0021 | 0.8146 ± 0.0024 | 0.1189 ± 0.0033 | – | – | – |
| single source | B3 | – | – | – | 0.0154 ± 0.0001 | 0.9302 ± 0.0027 | -0.0082 ± 0.0116 |
| single source | FM-nl | 0.0970 ± 0.0019 | 0.7954 ± 0.0013 | 0.1144 ± 0.0025 | 0.0155 ± 0.0001 | 0.7830 ± 0.0053 | 0.0993 ± 0.0194 |
| single source | FM-lin shared | 0.1194 ± 0.0008 | 0.9105 ± 0.0009 | 0.3053 ± 0.0036 | 0.0152 ± 0.0001 | 0.8791 ± 0.0033 | 0.0043 ± 0.0085 |
| single source | FM-nl @T* | – | – | – | 0.0190 ± 0.0001 | 0.0000 ± 0.0000 | 0.1363 ± 0.0098 |
| single source | FM-lin shared @T* | – | – | – | 0.0179 ± 0.0001 | 0.0000 ± 0.0000 | -0.0018 ± 0.0168 |
| single source + SC | B1 | 0.0914 ± 0.0009 | 0.7726 ± 0.0022 | 0.0887 ± 0.0038 | – | – | – |
| single source + SC | B2 | 0.1005 ± 0.0018 | 0.8144 ± 0.0029 | 0.1067 ± 0.0029 | – | – | – |
| single source + SC | B3 | – | – | – | 0.0155 ± 0.0000 | 0.9332 ± 0.0029 | -0.0098 ± 0.0114 |
| single source + SC | FM-nl | 0.0986 ± 0.0016 | 0.8049 ± 0.0018 | 0.1188 ± 0.0030 | 0.0156 ± 0.0001 | 0.7335 ± 0.0058 | 0.0995 ± 0.0204 |
| single source + SC | FM-lin shared | 0.1182 ± 0.0011 | 0.9094 ± 0.0015 | 0.2929 ± 0.0059 | 0.0153 ± 0.0001 | 0.8778 ± 0.0033 | 0.0042 ± 0.0071 |
| single source + SC | FM-nl @T* | – | – | – | 0.0191 ± 0.0001 | 0.0000 ± 0.0000 | 0.1238 ± 0.0147 |
| single source + SC | FM-lin shared @T* | – | – | – | 0.0180 ± 0.0001 | 0.0000 ± 0.0000 | -0.0009 ± 0.0154 |
| SC only | B1 | 0.0437 ± 0.0014 | 0.6610 ± 0.0039 | 0.0180 ± 0.0013 | – | – | – |
| SC only | B2 | 0.0458 ± 0.0017 | 0.6609 ± 0.0045 | 0.0288 ± 0.0020 | – | – | – |
| SC only | B3 | – | – | – | 0.0155 ± 0.0000 | 0.9242 ± 0.0053 | -0.0115 ± 0.0125 |
| SC only | FM-nl | 0.0317 ± 0.0018 | 0.6275 ± 0.0032 | 0.0157 ± 0.0024 | 0.0155 ± 0.0001 | 0.8185 ± 0.0048 | 0.0265 ± 0.0161 |
| SC only | FM-lin shared | 0.0361 ± 0.0014 | 0.6726 ± 0.0052 | 0.0239 ± 0.0020 | 0.0155 ± 0.0001 | 0.8528 ± 0.0045 | -0.0016 ± 0.0099 |
| SC only | FM-nl @T* | – | – | – | 0.0190 ± 0.0001 | 0.0000 ± 0.0000 | 0.0109 ± 0.0203 |
| SC only | FM-lin shared @T* | – | – | – | 0.0184 ± 0.0001 | 0.0000 ± 0.0000 | -0.0035 ± 0.0136 |

Hypotheses (support rule: seed-mean Δ > 2 SE, same sign in ≥ 4 / 5 seeds):

| H | comparison | metric | setting | Δ ± SE | seeds > 0 | supported | note |
|---|---|---|---|---|---|---|---|
| H1 | FM-nl − B1 | demeaned_pearson | single source | -0.0016 ± 0.0010 | 1 / 5 | False |  |
| H1 | FM-nl − B1 | demeaned_pearson | single source + SC | +0.0072 ± 0.0012 | 5 / 5 | True |  |
| H1 | FM-nl − B1 | demeaned_pearson | SC only | -0.0120 ± 0.0019 | 0 / 5 | False |  |
| H1 | FM-nl − B1 | avg_rank | single source | +0.0117 ± 0.0015 | 5 / 5 | True |  |
| H1 | FM-nl − B1 | avg_rank | single source + SC | +0.0323 ± 0.0011 | 5 / 5 | True |  |
| H1 | FM-nl − B1 | avg_rank | SC only | -0.0335 ± 0.0030 | 0 / 5 | False |  |
| H1 | FM-nl − B1 | top1_acc | single source | +0.0067 ± 0.0019 | 5 / 5 | True |  |
| H1 | FM-nl − B1 | top1_acc | single source + SC | +0.0300 ± 0.0026 | 5 / 5 | True |  |
| H1 | FM-nl − B1 | top1_acc | SC only | -0.0023 ± 0.0033 | 2 / 5 | False |  |
| H1 | FM-nl − B2 | demeaned_pearson | single source | -0.0077 ± 0.0014 | 0 / 5 | False |  |
| H1 | FM-nl − B2 | demeaned_pearson | single source + SC | -0.0019 ± 0.0017 | 1 / 5 | False |  |
| H1 | FM-nl − B2 | demeaned_pearson | SC only | -0.0141 ± 0.0025 | 0 / 5 | False |  |
| H1 | FM-nl − B2 | avg_rank | single source | -0.0192 ± 0.0019 | 0 / 5 | False |  |
| H1 | FM-nl − B2 | avg_rank | single source + SC | -0.0095 ± 0.0024 | 0 / 5 | False |  |
| H1 | FM-nl − B2 | avg_rank | SC only | -0.0335 ± 0.0028 | 0 / 5 | False |  |
| H1 | FM-nl − B2 | top1_acc | single source | -0.0045 ± 0.0012 | 0 / 5 | False |  |
| H1 | FM-nl − B2 | top1_acc | single source + SC | +0.0121 ± 0.0028 | 5 / 5 | True |  |
| H1 | FM-nl − B2 | top1_acc | SC only | -0.0131 ± 0.0033 | 0 / 5 | False |  |
| H2 | B3 − FM-nl | energy_score | single source | -0.0000 ± 0.0000 | 1 / 5 | False | > 0: flow better |
| H2 | B3 − FM-nl | energy_score | single source + SC | -0.0001 ± 0.0000 | 2 / 5 | False | > 0: flow better |
| H2 | B3 − FM-nl | energy_score | SC only | -0.0000 ± 0.0000 | 1 / 5 | False | > 0: flow better |
| H1 | FM-lin shared − B1 | demeaned_pearson | single source | +0.0208 ± 0.0007 | 5 / 5 | True |  |
| H1 | FM-lin shared − B1 | demeaned_pearson | single source + SC | +0.0268 ± 0.0005 | 5 / 5 | True |  |
| H1 | FM-lin shared − B1 | demeaned_pearson | SC only | -0.0076 ± 0.0014 | 0 / 5 | False |  |
| H1 | FM-lin shared − B1 | avg_rank | single source | +0.1268 ± 0.0011 | 5 / 5 | True |  |
| H1 | FM-lin shared − B1 | avg_rank | single source + SC | +0.1368 ± 0.0019 | 5 / 5 | True |  |
| H1 | FM-lin shared − B1 | avg_rank | SC only | +0.0117 ± 0.0053 | 4 / 5 | True |  |
| H1 | FM-lin shared − B1 | top1_acc | single source | +0.1977 ± 0.0017 | 5 / 5 | True |  |
| H1 | FM-lin shared − B1 | top1_acc | single source + SC | +0.2042 ± 0.0042 | 5 / 5 | True |  |
| H1 | FM-lin shared − B1 | top1_acc | SC only | +0.0059 ± 0.0016 | 5 / 5 | True |  |
| H1 | FM-lin shared − B2 | demeaned_pearson | single source | +0.0147 ± 0.0013 | 5 / 5 | True |  |
| H1 | FM-lin shared − B2 | demeaned_pearson | single source + SC | +0.0177 ± 0.0007 | 5 / 5 | True |  |
| H1 | FM-lin shared − B2 | demeaned_pearson | SC only | -0.0097 ± 0.0012 | 0 / 5 | False |  |
| H1 | FM-lin shared − B2 | avg_rank | single source | +0.0959 ± 0.0016 | 5 / 5 | True |  |
| H1 | FM-lin shared − B2 | avg_rank | single source + SC | +0.0950 ± 0.0023 | 5 / 5 | True |  |
| H1 | FM-lin shared − B2 | avg_rank | SC only | +0.0117 ± 0.0050 | 4 / 5 | True |  |
| H1 | FM-lin shared − B2 | top1_acc | single source | +0.1864 ± 0.0042 | 5 / 5 | True |  |
| H1 | FM-lin shared − B2 | top1_acc | single source + SC | +0.1862 ± 0.0037 | 5 / 5 | True |  |
| H1 | FM-lin shared − B2 | top1_acc | SC only | -0.0048 ± 0.0015 | 0 / 5 | False |  |
| H1c | FM-lin shared − B1-pooled | demeaned_pearson | single source | -0.0044 ± 0.0002 | 0 / 5 | False | > 0: flow beats the closed-form fit with the same sharing |
| H1c | FM-lin shared − B1-pooled | avg_rank | single source | -0.0056 ± 0.0006 | 0 / 5 | False | > 0: flow beats the closed-form fit with the same sharing |
| H1c | FM-lin shared − B1-pooled | top1_acc | single source | -0.0190 ± 0.0010 | 0 / 5 | False | > 0: flow beats the closed-form fit with the same sharing |
| H1c | FM-lin shared − B1-pooled-tgt | demeaned_pearson | single source | +0.0007 ± 0.0005 | 3 / 5 | False | > 0: flow beats the closed-form fit with the same sharing |
| H1c | FM-lin shared − B1-pooled-tgt | avg_rank | single source | +0.0678 ± 0.0007 | 5 / 5 | True | > 0: flow beats the closed-form fit with the same sharing |
| H1c | FM-lin shared − B1-pooled-tgt | top1_acc | single source | +0.1266 ± 0.0028 | 5 / 5 | True | > 0: flow beats the closed-form fit with the same sharing |
| H2 | B3 − FM-lin shared | energy_score | single source | +0.0002 ± 0.0000 | 5 / 5 | True | > 0: flow better |
| H2 | B3 − FM-lin shared | energy_score | single source + SC | +0.0002 ± 0.0000 | 5 / 5 | True | > 0: flow better |
| H2 | B3 − FM-lin shared | energy_score | SC only | -0.0000 ± 0.0000 | 1 / 5 | False | > 0: flow better |
| H4 | FM-nl spread-error r vs 0 | spread_error_r | single source | +0.0993 ± 0.0194 | 5 / 5 | True |  |
| H4 | FM-nl spread-error r vs 0 | spread_error_r | single source + SC | +0.0995 ± 0.0204 | 5 / 5 | True |  |
| H4 | FM-nl spread-error r vs 0 | spread_error_r | SC only | +0.0265 ± 0.0161 | 4 / 5 | False |  |
| H4 | FM-lin shared spread-error r vs 0 | spread_error_r | single source | +0.0043 ± 0.0085 | 4 / 5 | False |  |
| H4 | FM-lin shared spread-error r vs 0 | spread_error_r | single source + SC | +0.0042 ± 0.0071 | 3 / 5 | False |  |
| H4 | FM-lin shared spread-error r vs 0 | spread_error_r | SC only | -0.0016 ± 0.0099 | 2 / 5 | False |  |
| H3 | FM-nl − FM-lin shared | demeaned_pearson | single source | -0.0224 ± 0.0016 | 0 / 5 | False | nonlinearity |
| H3 | FM-nl − FM-lin shared | avg_rank | single source | -0.1151 ± 0.0011 | 0 / 5 | False | nonlinearity |
| H3 | FM-nl − FM-lin shared | top1_acc | single source | -0.1909 ± 0.0035 | 0 / 5 | False | nonlinearity |
| H3 | FM-lin shared − FM-nl | energy_score | single source | -0.0003 ± 0.0000 | 0 / 5 | False | > 0: FM-nl better |
| H3 | FM-nl − FM-lin shared | demeaned_pearson | single source + SC | -0.0196 ± 0.0014 | 0 / 5 | False | nonlinearity |
| H3 | FM-nl − FM-lin shared | avg_rank | single source + SC | -0.1045 ± 0.0021 | 0 / 5 | False | nonlinearity |
| H3 | FM-nl − FM-lin shared | top1_acc | single source + SC | -0.1741 ± 0.0040 | 0 / 5 | False | nonlinearity |
| H3 | FM-lin shared − FM-nl | energy_score | single source + SC | -0.0003 ± 0.0000 | 0 / 5 | False | > 0: FM-nl better |
| H3 | FM-nl − FM-lin shared | demeaned_pearson | SC only | -0.0044 ± 0.0017 | 0 / 5 | False | nonlinearity |
| H3 | FM-nl − FM-lin shared | avg_rank | SC only | -0.0452 ± 0.0057 | 0 / 5 | False | nonlinearity |
| H3 | FM-nl − FM-lin shared | top1_acc | SC only | -0.0083 ± 0.0029 | 0 / 5 | False | nonlinearity |
| H3 | FM-lin shared − FM-nl | energy_score | SC only | -0.0000 ± 0.0000 | 1 / 5 | False | > 0: FM-nl better |
| H3 | FM-lin shared − B1 | demeaned_pearson | single source | +0.0208 ± 0.0007 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | avg_rank | single source | +0.1268 ± 0.0011 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | top1_acc | single source | +0.1977 ± 0.0017 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | B3 − FM-lin shared | energy_score | single source | +0.0002 ± 0.0000 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | demeaned_pearson | single source + SC | +0.0268 ± 0.0005 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | avg_rank | single source + SC | +0.1368 ± 0.0019 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | top1_acc | single source + SC | +0.2042 ± 0.0042 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | B3 − FM-lin shared | energy_score | single source + SC | +0.0002 ± 0.0000 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | demeaned_pearson | SC only | -0.0076 ± 0.0014 | 0 / 5 | False | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | avg_rank | SC only | +0.0117 ± 0.0053 | 4 / 5 | True | fitting by flow matching vs closed form |
| H3 | FM-lin shared − B1 | top1_acc | SC only | +0.0059 ± 0.0016 | 5 / 5 | True | fitting by flow matching vs closed form |
| H3 | B3 − FM-lin shared | energy_score | SC only | -0.0000 ± 0.0000 | 1 / 5 | False | fitting by flow matching vs closed form |
| H5 | B1: (+SC) − (no SC) | demeaned_pearson | single source | -0.0072 ± 0.0010 | 0 / 5 | False |  |
| H5 | B1: (+SC) − (no SC) | avg_rank | single source | -0.0111 ± 0.0023 | 0 / 5 | False |  |
| H5 | B1: (+SC) − (no SC) | top1_acc | single source | -0.0189 ± 0.0024 | 0 / 5 | False |  |
| H5 | B2: (+SC) − (no SC) | demeaned_pearson | single source | -0.0041 ± 0.0011 | 0 / 5 | False |  |
| H5 | B2: (+SC) − (no SC) | avg_rank | single source | -0.0002 ± 0.0022 | 2 / 5 | False |  |
| H5 | B2: (+SC) − (no SC) | top1_acc | single source | -0.0122 ± 0.0038 | 0 / 5 | False |  |
| H5 | FM-nl: (+SC) − (no SC) | demeaned_pearson | single source | +0.0017 ± 0.0006 | 4 / 5 | True |  |
| H5 | FM-nl: (+SC) − (no SC) | avg_rank | single source | +0.0095 ± 0.0011 | 5 / 5 | True |  |
| H5 | FM-nl: (+SC) − (no SC) | top1_acc | single source | +0.0044 ± 0.0017 | 4 / 5 | True |  |
| H5 | FM-lin shared: (+SC) − (no SC) | demeaned_pearson | single source | -0.0012 ± 0.0005 | 1 / 5 | False |  |
| H5 | FM-lin shared: (+SC) − (no SC) | avg_rank | single source | -0.0011 ± 0.0008 | 1 / 5 | False |  |
| H5 | FM-lin shared: (+SC) − (no SC) | top1_acc | single source | -0.0124 ± 0.0034 | 0 / 5 | False |  |

H6 — share of the 56 single-source cells beating the copy (mean ± SE over seeds):

| model | vs | demeaned r | average rank |
|---|---|---|---|
| B1 | B0 | 0.26 ± 0.03 | 0.13 ± 0.01 |
| B1 | B0-codes | 0.45 ± 0.03 | 0.49 ± 0.02 |
| B1-pooled | B0 | 0.85 ± 0.00 | 0.92 ± 0.01 |
| B1-pooled | B0-codes | 1.00 ± 0.00 | 1.00 ± 0.00 |
| B1-pooled-tgt | B0 | 0.69 ± 0.02 | 0.47 ± 0.02 |
| B1-pooled-tgt | B0-codes | 0.85 ± 0.02 | 0.98 ± 0.01 |
| B2 | B0 | 0.52 ± 0.04 | 0.27 ± 0.03 |
| B2 | B0-codes | 0.61 ± 0.03 | 0.85 ± 0.02 |
| FM-nl | B0 | 0.33 ± 0.04 | 0.12 ± 0.01 |
| FM-nl | B0-codes | 0.47 ± 0.03 | 0.64 ± 0.04 |
| FM-lin shared | B0 | 0.83 ± 0.01 | 0.90 ± 0.02 |
| FM-lin shared | B0-codes | 0.99 ± 0.01 | 1.00 ± 0.00 |
