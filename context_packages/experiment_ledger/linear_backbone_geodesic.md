# Linear Backbone Geodesic Metric Test — Ledger Entry

**Code:** `scripts/experiments/linear_backbone_geodesic/linear_backbone_geodesic.ipynb`
**Outputs:** inline in the notebook (figures + printed metrics); future bulky outputs → `results/experiments/linear_backbone_geodesic/`
**Status:** exploratory · no written conclusion in the notebook · saved outputs not re-run since the April 2026 `models/` refactor
**W&B / Ray:** none — run in `dev` mode with `save_checkpoint=False` (no W&B run, no `ray_tune_id`)

---

## 1. Question

How do geodesic FC-distance identifiability metrics (log-Euclidean, Frobenius) behave for the pure
linear `LatentAttnMasked` backbone, compared with the default correlation-based identifiability?

`residual_mode="none"` disables the attention residual branch, so the model is a learned linear map in
PCA latent space: SC edges → SC PCA scores → linear map → FC PCA scores → FC edges.

## 2. Setup (as recorded)

| | |
|---|---|
| Model | `LatentAttnMasked`, `residual_mode="none"`, `residual_gain_init=0.0` |
| Data | SC → FC, Glasser, `shuffle_seed=0`, `data_load_mode="precomputed"` |
| Model config in effect | `n_components_pca=256`, `reg=1e-6` (model init printout confirms `k=256`) |
| Trainer | 100 epochs, batch 128, `lr=5e-4`, `loss_type="balanced_composite"` over `[mse, correye, neidist]`, EMA decay 0.95, 20 warmup steps |
| Entry point | `Sim(...)._run_learned_single(mode="dev", save_checkpoint=False, run_eval=True)` |
| Evaluation | test split, n = 195 subjects |

The notebook constructs `Sim` with a first override (`k=128`, `lr=1e-4`, `reg=1e-7`), then trains with a
second override passed to `_run_learned_single`; the printed model init shows the second one was used.

## 3. How to run

Open the notebook in `kraken_env` on a GPU compute node (it trains for 100 epochs; the recorded run used an
A100) — not on a login node. The first cell's bootstrap resolves `REPO_ROOT`, so it runs from its current
folder.

## 4. Recorded results (test split, n = 195)

| Metric | Top-1 acc. | Avg. rank %ile | Notes |
|---|---|---|---|
| Correlation identifiability (default evaluator) | 0.103 (20/195; chance 0.005) | 0.846 (chance 0.5) | pFC r_intra − r_inter = 0.0673, t = 20.41, p = 3.4e-50, Cohen's d = 1.46; null d ≈ 0 |
| Log-Euclidean, raw (not demeaned) | 0.005 | 0.515 | every target's nearest prediction is the same subject (index 173); 100% of predictions needed SPD projection, ~10% of eigenvalues clipped |
| Frobenius, demeaned | 0.056 | 0.782 | no SPD projection needed |
| Log-Euclidean, demeaned, SPD eps sweep 1e-8 → 1e-2 | 0.041 → 0.056 | 0.721 → 0.754 | mean self-distance falls 96.3 → 29.0 as eps grows |

## 5. Observations (not conclusions)

- Geodesic identifiability is weaker than correlation-based identifiability for this model in every recorded variant.
- Raw log-Euclidean is degenerate (chance-level, single-subject collapse) alongside heavy SPD projection of the
  predicted matrices; whether projection causes the collapse was not tested in the notebook.
- Demeaning restores above-chance geodesic ranking; results are mildly sensitive to the SPD eps.

## 6. Caveats

- Saved outputs predate the April 2026 `models/` refactor (their warnings reference the old `models/eval.py`)
  and the 2026-09-23 import fixes; they have not been reproduced with current code.
- The Frobenius cell's nearest-neighbour sanity print reuses the non-demeaned log-Euclidean reconstruction
  from the previous cell, so its printed `winners` line repeats that result rather than describing Frobenius.
- Single seed (0), single configuration; no hyperparameter search.

Last updated at: 2026-09-23 EDT
