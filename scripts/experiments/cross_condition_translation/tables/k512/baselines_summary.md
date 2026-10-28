# E4.0 baselines — summary (k = 512; test, mean ± SE over seeds)

G0 (B1 / B2 without SC reproduce E2.0 L1 / L4 at k = 512): PASS (max |Δ| L1 1.1e-16, L4 1.1e-16; 5 seeds).

Ceilings: own 512-code reconstruction demeaned r 0.6124, average rank 1.0000; rest session 1 ↔ 2 copy demeaned r 0.4943, average rank 0.9889.

| model | setting | demeaned r | average rank | top-1 |
|---|---|---|---|---|
| B1+SC | SC only | 0.0441 ± 0.0011 | 0.6732 ± 0.0035 | 0.0193 ± 0.0009 |
| B2+SC | SC only | 0.0495 ± 0.0024 | 0.6708 ± 0.0067 | 0.0278 ± 0.0028 |
| B0 | other 7 | 0.2084 ± 0.0013 | 0.9956 ± 0.0004 | 0.8895 ± 0.0048 |
| B0-codes | other 7 | 0.1907 ± 0.0010 | 0.9792 ± 0.0007 | 0.6915 ± 0.0067 |
| B1 | other 7 | 0.1176 ± 0.0010 | 0.8365 ± 0.0012 | 0.2004 ± 0.0060 |
| B1+SC | other 7 | 0.1186 ± 0.0012 | 0.8409 ± 0.0026 | 0.2055 ± 0.0050 |
| B2 | other 7 | 0.1559 ± 0.0028 | 0.9049 ± 0.0026 | 0.2425 ± 0.0084 |
| B2+SC | other 7 | 0.1549 ± 0.0027 | 0.9051 ± 0.0023 | 0.2385 ± 0.0082 |
| B0 | single source | 0.1071 ± 0.0007 | 0.8445 ± 0.0027 | 0.2493 ± 0.0037 |
| B0-codes | single source | 0.1014 ± 0.0007 | 0.8082 ± 0.0022 | 0.1728 ± 0.0016 |
| B1 | single source | 0.0939 ± 0.0014 | 0.7718 ± 0.0017 | 0.0982 ± 0.0027 |
| B1+SC | single source | 0.0894 ± 0.0011 | 0.7781 ± 0.0023 | 0.0837 ± 0.0034 |
| B2 | single source | 0.1074 ± 0.0013 | 0.8029 ± 0.0021 | 0.1045 ± 0.0024 |
| B2+SC | single source | 0.1052 ± 0.0017 | 0.8120 ± 0.0033 | 0.0982 ± 0.0031 |
| ceiling | single source | 0.6124 ± 0.0012 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

Distribution (B3, code space; means over cells):

| model | setting | energy score | 90 % coverage |
|---|---|---|---|
| B3 | other 7 | 0.0188 | 0.949 |
| B3 | single source | 0.0169 | 0.945 |
| B3+SC | SC only | 0.0171 | 0.942 |
| B3+SC | other 7 | 0.0184 | 0.946 |
| B3+SC | single source | 0.0169 | 0.945 |

B2 selected (d, lambda) per seed: (32, 1), (32, 1), (32, 1), (64, 1), (32, 1).

Single-source cells where ridge beats the copy on demeaned r: 10 / 56 (edges); 12 / 56 (codes).

Best single source per target (B1, demeaned r):

- rest: wm (0.226)
- emotion: rest (0.070)
- gambling: wm (0.136)
- language: rest (0.149)
- motor: rest (0.116)
- relational: wm (0.099)
- social: rest (0.114)
- wm: rest (0.161)
