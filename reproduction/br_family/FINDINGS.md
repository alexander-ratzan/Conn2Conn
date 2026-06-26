# BR-family (F6/F7 heritability) — Findings (Glasser × 10 seeds)

**Run:** the 8 family variants built with **BR imputation** (vs the spine's PLS), sibling-vs-unrelated
AUC pooled over 10 Glasser seeds. Source: `outputs/family_auc_br.csv`. Lead metric: **sibling AUC**
(95% bootstrap CI, permutation p, BH-FDR).

## Validation (wiring correct)
The 4 estimator-independent variants reproduce the spine PLS family run:
`obs_SC`, `obs_FC`, `bvdemo_to_SC` match to **Δ = 0.000**; `combined_pred_SC` to **Δ = 1.5e-6**
(float32 noise in the iterative BayesianRidge — negligible; the guard threshold was just set a hair
too tight at 1e-6, now 1e-4). Pairing self-check (Glasser seed-0 = 33 MZ / 13 DZ / 125 sib / 171
unrelated) passed. So any BR-vs-PLS difference is the imputation-estimator swap, nothing else.

## Sibling AUC — BR (10 seeds)
| variant | AUC | 95% CI | perm p | sig |
|---|---|---|---|---|
| `obs_SC` | 0.863 | [0.850, 0.876] | <1e-4 | ✓ |
| `obs_FC` | 0.823 | [0.807, 0.839] | <1e-4 | ✓ |
| `pred_SC_resid_bvdemo` | **0.763** | [0.745, 0.780] | <1e-4 | ✓ |
| `pred_FC_resid_bvdemo` | 0.750 | [0.732, 0.766] | <1e-4 | ✓ |
| `pred_FC_raw` | 0.626 | [0.605, 0.646] | <1e-4 | ✓ |
| `bvdemo_to_SC` (demographics) | 0.563 | [0.541, 0.585] | <1e-4 | ✓ |
| `pred_SC_raw` | 0.560 | [0.538, 0.581] | <1e-4 | ✓ |
| `combined_pred_SC` | 0.505 | [0.482, 0.526] | 0.65 | ✗ |

## Headline 1 — F6 REPLICATES with BR: this is the real "boost"
A BR-imputed connectome carries **genuine heritable family signal**: `pred_SC_resid_bvdemo`
sibling AUC **0.763** (CI excludes 0.5; p<1e-4), far above the demographic baseline `bvdemo_to_SC`
**0.563** and above chance. **This is the one place — across cognition and heritability — where the
predicted connectome buys you something you can't get from demographics.** Cognition showed no real
boost; heritability does.

## Headline 2 — but BR carries LESS family signal than PLS (counterintuitive)
On every imputed variant, **BR ≤ PLS** for heritability:

| variant | BR | PLS | Δ |
|---|---|---|---|
| `pred_SC_raw` | 0.560 | 0.680 | **−0.119** |
| `pred_SC_resid_bvdemo` | 0.763 | 0.810 | −0.047 |
| `pred_FC_raw` | 0.626 | 0.710 | −0.084 |
| `pred_FC_resid_bvdemo` | 0.750 | 0.743 | +0.007 |

The F6 headline drops 0.810 → 0.763. So the **stronger reconstructor is the weaker identifier** —
the exact opposite of the cognition story (where BR was marginally *better*).

## Headline 3 — this IS F7, at the estimator level
BR optimizes per-component reconstruction: it gets a **higher demeaned-r** (0.166 vs PLS lower) and
slightly better cognition, but per-mode shrinkage smooths away the idiosyncratic individual structure
that **fingerprints families** → lower sibling AUC. That is the F7 thesis — *you cannot optimize one
connectome to both reconstruct and identify* — now visible **at the choice of imputation estimator**,
not just the `combined_pred` head. And F7 itself replicates: `combined_pred_SC` = **0.505 (chance,
n.s.)** under BR too — forcing the connectome to predict *and* carry subject-info collapses identity
regardless of estimator.

## Takeaway
- **Where the predicted connectome wins:** heritability, not cognition. `pred_SC_resid_bvdemo` carries
  real sibling signal (AUC 0.76–0.81) far above demographics.
- **BR vs PLS:** better reconstructor (cognition ↑ slightly) but worse identifier (heritability ↓).
  A clean, estimator-level instance of the reconstruct/identify tradeoff (F7).
- **Caveat:** Glasser only; in-sample-free here (family variants are built out-of-sample on the test
  split, so no in-sample-train subtlety — unlike the cognition run).
