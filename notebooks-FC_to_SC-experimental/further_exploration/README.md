# Further exploration — mechanism, robustness, and the recorded null

Companion directory to the validated work in
[`../model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb`](../model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb)
and the open-work notebook [`../tier_extensions.ipynb`](../tier_extensions.ipynb).

By the end of the 2026-06-01 expansion, the main notebook had established three
findings:

1. **FC → SC > SC → FC asymmetry** holds at 10-seed precision across every basis
   (`none`, `bv`, `demo`, `bv+demo`), ratios 1.30–1.67×, all `p ≤ 0.01` vs the 1.15
   threshold.
2. **Cognition prediction** survives demographic residualization for FC-derived signal
   (~73% of crystallized variance retained) and essentially collapses for SC.
3. **Family structure**: `pred_SC_raw` separates siblings at AUC = 0.680, but
   `combined_pred_SC` collapses to AUC = 0.505 (under diagnostic — see STEP 8.4 of
   the main notebook). MZ stays at 0.897, DZ at 0.701.

This directory drills into **why** those effects exist (Depth 1), **how robust** they
are to choices we made (Depth 2), and **why we are not chasing** the cognition-ceiling
trap (Depth 3, recorded as a null).

## Depths

| File | Depth | What it answers |
|---|---|---|
| `depth1_spectral_mechanism.ipynb` | **Mechanism** | Is the FC→SC asymmetry concentrated in 1–3 SC components (PC2 was the standout earlier)? Is that same component the heritable one? If yes, the asymmetry, modality dissociation, and family-structure findings collapse into one story: FC predicts a low-dim heritable structural backbone. |
| `depth2_robustness_sensitivity.ipynb` | **Robustness** | Does the asymmetry ratio swing with K_PCA / K_PLS choices? Does a non-PLS estimator give the same answer? Does the `combined_pred_SC` sibling-AUC collapse survive the fixes the STEP 8.4 diagnostic surfaced? |
| `depth3_cognition_ceiling.ipynb` | **Recorded null** | Why we are *not* trying to squeeze more cognition signal out of either modality. No runnable cells — purely a recorded "we considered, here's why no." |

## Why no Depth 3 runnable

Drilling into cognition prediction is the trap we already killed four times:
- Task-aware objectives, fancier regressors, BV-optimization — all fight an information
  ceiling that's structural to the dataset, not algorithmic.
- More cognition R² would not change the headline finding (asymmetry exists) and would
  not produce a citable mechanism.
- BV alone is already a strong cognition predictor (Phase 2 results); marginal returns
  from method changes are small.

The recorded null exists so future-you doesn't relitigate this decision. If a
collaborator asks "did you try X for cognition?", point them at that notebook.

## Shared infrastructure

`_setup.py` contains:
- `load_seed_split(seed=0)` — returns `(base, train_idx, test_idx, FC_tr, FC_te, SC_tr, SC_te, bv_tr, bv_te, demo_tr, demo_te)` exactly as the main notebook constructs them.
- Closed-form helpers (`pca_pls_predict`, `combined_predict`, `br_per_component_predict`, `fit_basis_ols`) — identical to the STEP 6.0 helpers in the main notebook.
- Phase 2 Analysis 1 helpers (`demeaned_cosine_pair_sim`, `extract_pair_sims`,
  `pair_indices_by_relation`, `auc_vs_unrelated`) — identical to STEP 8.1.

Each notebook does `from _setup import *` in its setup cell so we keep the helpers in
one place. If a helper diverges from the main notebook, the main notebook is canonical
— update `_setup.py` to match, then re-run the notebook here.

## Results layout

Per-notebook outputs land under
`../model_overviews/results/local_results/further_exploration/<notebook_name>/`
so they sit next to the rest of the project's local results (e.g.
`diagnostic_combined_pred_SC_sibling_collapse/`).
