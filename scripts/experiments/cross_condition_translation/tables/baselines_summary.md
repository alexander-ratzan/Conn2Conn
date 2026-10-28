# E4.0 baselines — summary (test, mean ± SE over seeds)

G0 (B1 / B2 without SC reproduce E2.0 L1 / L4 at k = 256): PASS (max |Δ| L1 1.1e-16, L4 1.1e-16; 5 seeds).

Ceilings: own 256-code reconstruction demeaned r 0.5560, average rank 1.0000; rest session 1 ↔ 2 copy demeaned r 0.4943, average rank 0.9889.

| model | setting | demeaned r | average rank | top-1 |
|---|---|---|---|---|
| B1+SC | SC only | 0.0441 ± 0.0015 | 0.6621 ± 0.0044 | 0.0176 ± 0.0016 |
| B2+SC | SC only | 0.0465 ± 0.0016 | 0.6581 ± 0.0035 | 0.0244 ± 0.0039 |
| B0 | other 7 | 0.2084 ± 0.0013 | 0.9956 ± 0.0004 | 0.8895 ± 0.0048 |
| B0-codes | other 7 | 0.1772 ± 0.0012 | 0.9641 ± 0.0010 | 0.5553 ± 0.0063 |
| B1 | other 7 | 0.1005 ± 0.0011 | 0.7653 ± 0.0028 | 0.1174 ± 0.0051 |
| B1+SC | other 7 | 0.1102 ± 0.0012 | 0.7974 ± 0.0030 | 0.1475 ± 0.0062 |
| B2 | other 7 | 0.1537 ± 0.0042 | 0.9001 ± 0.0065 | 0.2368 ± 0.0250 |
| B2+SC | other 7 | 0.1529 ± 0.0042 | 0.8998 ± 0.0062 | 0.2313 ± 0.0243 |
| B0 | single source | 0.1071 ± 0.0007 | 0.8445 ± 0.0027 | 0.2493 ± 0.0037 |
| B0-codes | single source | 0.0952 ± 0.0008 | 0.7795 ± 0.0025 | 0.1301 ± 0.0013 |
| B1 | single source | 0.0973 ± 0.0015 | 0.7674 ± 0.0022 | 0.0971 ± 0.0030 |
| B1+SC | single source | 0.0914 ± 0.0011 | 0.7673 ± 0.0024 | 0.0769 ± 0.0029 |
| B2 | single source | 0.1066 ± 0.0026 | 0.7991 ± 0.0066 | 0.1048 ± 0.0096 |
| B2+SC | single source | 0.1041 ± 0.0024 | 0.8055 ± 0.0051 | 0.0956 ± 0.0072 |
| ceiling | single source | 0.5560 ± 0.0012 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

Distribution (B3, code space; means over cells):

| model | setting | energy score | 90 % coverage |
|---|---|---|---|
| B3 | other 7 | 0.0189 | 0.915 |
| B3 | single source | 0.0152 | 0.923 |
| B3+SC | SC only | 0.0155 | 0.921 |
| B3+SC | other 7 | 0.0174 | 0.923 |
| B3+SC | single source | 0.0153 | 0.924 |

Single-source cells where ridge beats identity on demeaned r: 13 / 56 (edges); 22 / 56 (identity in codes).

Best single source per target (B1, demeaned r):

- rest: wm (0.229)
- emotion: rest (0.071)
- gambling: wm (0.141)
- language: rest (0.147)
- motor: rest (0.123)
- relational: wm (0.100)
- social: rest (0.117)
- wm: rest (0.162)
