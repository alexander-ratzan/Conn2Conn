# Imputation Objective Functions — Phase 1 (1A + 2C)

Test whether changing the imputer's **objective** moves it along the F7 reconstruct↔identify↔cognition
frontier. Plan: [`dev-notes/objective-functions/PLAN.md`](../../dev-notes/objective-functions/PLAN.md).
Glasser × 5 seeds (0–4), BR backbone, 5-fold OOF, CogCryst-resid supervision.

## Estimators (`_obj_estimators.py`)
- `BR`, `PLS` — references (reconstruction / covariance).
- `obj1a` — BR + reliability-gated per-PC **amplitude restoration** (identity objective).
- `obj2c` — BR + **cognition-weighted** reconstruction, supervised on CogCryst-resid.
- `obj2c_raw` — `obj2c` on RAW CogCryst (side-check: was residualizing needed?).

## Run
```bash
cd .../reproduction/obj_functions && bash submit_obj.sh     # array 0-4 + afterok finalize
# watch: ls sentinels/DONE_obj_*.sentinel | grep -v finalize | wc -l   # /5
```

## Output — `outputs/scorecard.csv` (the 3-axis result)
Per estimator: reconstruction `demeaned_r` / `avg_rank` / `top1`, identity `sib_AUC`, cognition lift
(`pred_SC` + `pred_SC+bv+demo` on Cog{Cryst,Total,Fluid}).
**Diagonal hypothesis:** `obj1a` wins sib_AUC, `obj2c` wins cogCryst, `BR` wins recon dr.
