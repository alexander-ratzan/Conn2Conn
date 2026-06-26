# BR-only Imputation + Downstream Run

Isolated "Bayesian-ridge SOTA" run. **BayesianRidge in both estimator slots** (impute the
connectomes with BR, predict cognition with BR), over an expanded **18-input** set, **Glasser only**.
Writes only under this folder — the spine grid's outputs/artifacts are never touched.

- **Plan / status:** [`dev-notes/BR-only/PLAN.md`](../../dev-notes/BR-only/PLAN.md) ·
  [`dev-notes/BR-only/TODO.md`](../../dev-notes/BR-only/TODO.md)
- **Why:** steelman of F5 — does the *strongest* imputer change the "imputation doesn't transfer"
  result? (No hypothesis forced; both outcomes informative.)

## Files
- `run_br_unit.py`  — per (parc, seed): BR-impute pred_* (save) → BR-only downstream over 18 inputs.
- `finalize_br.py`  — merge per-seed parts → `outputs/downstream_br.csv`, leak verdict, completeness.
- `run_br_unit.sbatch` / `finalize_br.sbatch` / `submit_br.sh` — Torch launch (16G/2h, array 0-9%10).
- `outputs/downstream_br.csv` — **the result file** (source of truth).
- `outputs/leak_verdict_br.csv`, `configs/expected_cells_br.csv`.
- `outputs/artifacts/Glasser/seed{N}/` — BR-imputed `pred_{SC,FC}_{train,test}.npy` (+ subject_ids).

## Run
```bash
# local pilot (one seed):  python run_br_unit.py --parc Glasser --seed 0
# Torch full run:          cd .../br_imputation && bash submit_br.sh
# watch (no squeue):       ls sentinels/DONE_br_*.sentinel | grep -v finalize | wc -l   # /10
```

## Known, deliberate choice (NOT a leak)
Train-side `pred_*` are made **in-sample** (`pred_SC_train = BR(FC_tr, FC_tr, SC_tr)`). The
train/test split is seed-frozen and the imputer is fit on **train only** — `pred_*_test` is genuine
out-of-sample, no test leak. The in-sample train imputation is a mild train/test input-distribution
mismatch kept on purpose (apples-to-apples vs the PLS spine). OOF is the orthogonal future fix.
See PLAN.md §6.
