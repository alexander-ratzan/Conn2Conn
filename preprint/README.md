# Conn2Conn Preprint Workspace

This directory is the working manuscript area.

| File | Purpose |
|---|---|
| `STRUCTURE.md` | Paper map: section arc, main figures, main tables, supplement sections, and style rules |
| `manuscript.md` | First narrative manuscript draft with figure/table callouts and headline numbers |
| `supplement.md` | First supplementary draft focused on sanity checks, reproducibility, and reviewer defenses |

Current drafting stance:

- Main text stays claim-driven and compact.
- Supplement carries detailed sanity checks and operational reproducibility.
- All headline cognition/oracle numbers should name the estimator.
- Reconstruction leads with `demeaned_pearson`; downstream leads with BayesianRidge lift over
  `bv+demo`.
- F8 is exploratory and should be described as a property-selected mechanism mode, not as a fixed
  PC3 result.

Next useful steps:

1. Generate the main figures from the CSVs.
2. Generate supplementary tables mechanically from source outputs.
3. Convert Markdown to LaTeX only after the argument and figure order stabilize.
