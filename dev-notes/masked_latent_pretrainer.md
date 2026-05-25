# MaskedLatentPretrainer — deep dive

Source: [`models/architectures/latent_attention/masked_latent_pretrainer.py`](../models/architectures/latent_attention/masked_latent_pretrainer.py)
Attention primitives: [`models/architectures/latent_attention/latent_attn_masked.py`](../models/architectures/latent_attention/latent_attn_masked.py)
Loss: [`models/train/loss.py`](../models/train/loss.py)

## What it is in one sentence

`MaskedLatentPretrainer` is a **self-supervised joint SC+FC masked autoencoder that operates in PCA-latent space** — essentially a BERT/MAE for connectome components: randomly hide some SC and FC PCA coefficients, then reconstruct them from the ones left visible.

## The "demean and add back" idea is baked in — receipts

The model trains and tests entirely on **individual deviations from the group average**, then restores the group mean at decode. This is exactly the "subtract the group avg, learn the individual biomarkers, add it back" idea.

**Encode** (`masked_latent_pretrainer.py:334-346`):
```python
c = (x − source_mean) @ source_loadings_k        # subtract GROUP MEAN (edge space), project to PCA
c = (c − source_score_mean) / source_score_std   # z-score each component
```

**Decode** (`:348-351`):
```python
c = c * target_score_std + target_score_mean     # un-z-score
y = c @ target_loadings_k.T + target_mean        # ADD THE GROUP MEAN BACK
```

`− source_mean` / `− target_mean` is "subtract the group average," and `+ target_mean` at decode is "add it back at the end." The model never spends a parameter reproducing the group mean. The extra z-scoring normalizes each PCA component to unit variance so the loss weights all components equally instead of letting the top high-variance components dominate.

**Implication:** the demeaning trick is already implemented here, and results still land in the same band as everything else (~0.08–0.10 demeaned). Demeaning *reorganizes effort, it does not manufacture signal.* See [benchmarks_and_model_scores.md](benchmarks_and_model_scores.md).

## How the data becomes tokens

Per subject, the model builds a token sequence (`:375-399`):
```
[ CLS? , SC₀, SC₁, …, SC_{k-1} , FC₀, FC₁, …, FC_{k-1} ]
```
Each token is `[scalar_slot ‖ component_embedding]`:
- **scalar_slot** (1 number) = that component's z-scored PCA coefficient — the actual value.
- **component_embedding** (learned, `token_embedding_dim`=16) = a per-component, per-modality identity vector ("I am SC component 3" vs "I am FC component 3"). There are **separate embedding tables for SC and FC** (`:166-167`), so the model does not assume SC↔FC symmetry.
- Optional **CLS token** carries covariates (projected demographics / FreeSurfer).

## The masking (the "MAE" part)

`:358-373`: independently Bernoulli-mask SC components (`sc_mask_ratio`) and FC components (`fc_mask_ratio`), guaranteeing at least `min_masked_components_per_modality` per modality. A masked token's scalar slot is replaced by a **learned constant** (`sc_mask_value` / `fc_mask_value`) — its embedding stays, so the model still knows *which* component is missing, just not its value.

## The attention (what does the work)

A self-attention layer (`latent_attn_masked.py:56-87`) mixes the tokens: `Q=W_Q·token`, `K=W_K·token`, `attn = softmax(QKᵀ/√d)`, output `= attn·V`. The key reconstruction mechanism is **`visible_only_attention`** (`:401-409`): masked tokens are removed from the *keys*, so a masked component can only gather information from the **visible** components — forcing genuine inference rather than copying.

Default is one raw attention layer (`transformer_layers=0`); set `>0` for a stack of pre-LN transformer blocks instead.

## Readout + loss

Two separate heads (`:441-442`) read each token's post-attention vector → a scalar prediction of its coefficient: `sc_readout_head` for SC, `fc_readout_head` for FC. The loss (`:455-472` + `loss.py:396-406`) is **MSE on masked positions only**, averaged across both modalities (`0.5·(sc_loss + fc_loss)` under `per_modality_mean`). No downstream supervision — pure reconstruction.

## Three clever bits

1. **Joint masking forces a bidirectional kernel.** Because both SC and FC are masked in the same pass, the *same* attention kernel must learn to fill masked FC from visible SC (SC→FC), masked SC from visible FC (FC→SC), and within-modality structure. One model, all directions — hence "pretrainer."
2. **Prior warm-start** (`learned_cov_init`, `:304-332`): initializes W_Q/W_K so the *initial* attention kernel ≈ the empirical SC↔FC cross-component correlation matrix `C` (factorized via SVD). Attention starts at the statistically-known coupling, then refines. `prior_qk_init_rel_error` reports how close the init got.
3. **Transfer learning** (`export_to_latent_attn_masked`, `:575`): copies pretrained embeddings/attention/readout into a `LatentAttnMasked` model for supervised fine-tuning.

## How it is actually used for SC→FC

`forward(x)` (`:474-499`) runs the **"downstream" pattern**: all SC visible, all FC masked → reconstruct FC purely from SC → decode to edges. That is a **zero-shot SC→FC predictor** that falls out of the SSL objective for free. The notebook reported this probe hitting ~0.83 Pearson — a competent SC→FC predictor without ever being trained directly for it.

## Honest assessment

The companion `masked_mlp_pretraining` diagnostic showed the masked-reconstruction plateau is **structural** — per-component Pearson collapses from ~0.49 (train) to ~0.1 (val), meaning the model largely copies visible within-modality context rather than learning strong cross-modal coupling. So even this sophisticated, demeaning-baked-in, prior-warm-started MAE hits the same individual-signal ceiling. **The bottleneck is information, not architecture.**

## Why this matters for Phase 1

This MAE framing is the ideal scaffold for the multi-task FC idea. It is already an "any-subset-of-components → any-other-subset" reconstruction model with separate per-modality embedding tables. To do Phase 1, add more token blocks — rest-FC, task₁-FC, …, task₇-FC, SC — and mask/reconstruct jointly. The architecture barely changes; you are just adding modalities to the token sequence. The substrate Phase 1 needs already exists.

## Open questions for the model author

**Q1 — What does the SC↔FC cross-component correlation matrix `C` actually look like?**
The prior-init factorizes the empirical SC–FC cross-component correlation `C` (`_compute_cross_component_correlation`). Since PCA decorrelates components *within* a modality, that cross-block `C` is the only exploitable signal for SC→FC. How many of the k components carry non-trivial cross-correlation, and how big? Is the plateau basically "C is near-zero past the first few components"?
*Why it matters:* the ceiling diagnostic. If `C` is near-zero for most components, no architecture can do better — a data limit, not a model limit. Also the most Phase-1-relevant question: if rest-FC's `C` is weak, that is the gap multi-task FC must fill.

**Q2 — Is there a train/inference masking mismatch?**
Pretraining uses `fc_mask_ratio ≈ 0.5` (half of FC visible, so a masked FC component can attend to other visible FC tokens), but `forward()` masks *all* FC. Does that inflate the pretraining metric relative to the real use case? Would annealing `fc_mask_ratio → 1`, or adding "all-FC-masked" episodes, close the train/val gap?
*Why it matters:* the MLP diagnostic showed per-component Pearson collapsing 0.49 → 0.1. Pretraining may be getting "help" from partially-visible FC that the downstream task never has — a concrete, fixable methodological issue.

**Q3 — Why z-score the components (equal weighting) instead of variance-weighting?**
`zscore_pca_scores=True` makes the latent MSE weight all k components equally, including high-index low-variance ones that are mostly noise. Was `latent_mse` (z-scored, equal weight) compared against `latent_weighted_mse` (variance-proportional)? Is chasing near-unpredictable tail components dragging down demeaned correlation?
*Why it matters:* a real loss-design tradeoff that shapes how Phase 1 runs should be configured.

**Q4 (strategic) — Does it beat the closed-form baselines on demeaned correlation, or is the value-add the transfer/fine-tuning and covariate/multimodal extensibility?**
Given everything plateaus in the same band, what is the intended contribution of the SSL pretrainer specifically?
