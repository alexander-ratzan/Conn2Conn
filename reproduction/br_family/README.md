# BR-family (F6/F7 heritability) Run

Does the **stronger BR reconstructor** carry more heritable family signal than PLS — and does it
change the F7 predictor/identifier tradeoff? Glasser × 10 seeds. Faithful port of
[`family_mechanism/`](../family_mechanism/), with **one change**: the four imputed variants
(`pred_SC_raw`, `pred_SC_resid_bvdemo`, `pred_FC_raw`, `pred_FC_resid_bvdemo`) are built with
**capped BayesianRidge** instead of PLS. Everything else (pairing, demeaned-cosine sims, AUC,
bootstrap, perm, FDR) is reused **verbatim** from `_fm_common.py`.

- Motivation: cognition transfer showed no real boost from imputed connectomes (see
  [`br_imputation/FINDINGS.md`](../br_imputation/FINDINGS.md)); **heritability (F6) is the one place
  predicted connectomes have shown value** (PLS `pred_SC_resid_bvdemo` sibling AUC ≈ 0.81). This run
  tests whether BR pushes that higher.

## Variants (8) and what's BR vs unchanged
- **BR-imputed (the swap):** `pred_SC_raw`, `pred_SC_resid_bvdemo`, `pred_FC_raw`, `pred_FC_resid_bvdemo`.
- **Unchanged (built-in sanity vs spine PLS):** `obs_SC`, `obs_FC`, `combined_pred_SC` (already BR),
  `bvdemo_to_SC` (OLS). Their AUCs MUST match the spine PLS family run (`finalize` guards this).

## Run
```bash
cd .../reproduction/br_family && bash submit_br_family.sh   # array 0-9 + afterok finalize
# watch: ls sentinels/DONE_brfam_*.sentinel | grep -v finalize | wc -l   # /10
```

## Output
- `outputs/family_auc_br.csv` — (variant, relation) × {auc, auc_lo, auc_hi, p_perm, p_fdr, sig_fdr}.
- finalize prints **BR vs PLS sibling AUC** for the imputed variants (the result) + the guard.
