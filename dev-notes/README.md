# dev-notes

Working notes for the Conn2Conn project (SC↔FC connectome prediction/translation on HCP data).

## Model deep-dives
- [masked_latent_pretrainer.md](masked_latent_pretrainer.md) — `MaskedLatentPretrainer` explained: a joint SC+FC masked autoencoder in PCA-latent space (BERT/MAE for connectome components). Covers the baked-in "demean and add back" design, tokenization/masking/attention, zero-shot SC→FC use, and open questions for the author.

## Benchmarks & results
- [benchmarks_and_model_scores.md](benchmarks_and_model_scores.md) — reference for every eval metric (raw/demeaned Pearson, R², avg rank, top-1, identifiability/Cohen's d, Hungarian matching + sample-size sweep, null/oracle bounds) and the authoritative **10-seed** score tables for all models. Headline: everything plateaus at demeaned r ≈ 0.07–0.10; the bottleneck is information, not architecture.

## Data / preprocessing
- [sc_reconstruction.md](sc_reconstruction.md) — what SC reconstruction the project uses (SIFT2 + inverse-node-volume + log1p, verified on disk), the recommended pipeline (Zalesky/Smolders), the four edge metrics in one `connectivity.mat`, and why a multi-metric SC ablation isn't worth it (redundant transforms of one tractogram; multi-view budget belongs on the functional/task side).

## Plans
- [plans/Quick Meeting Notes-2.pdf](plans/) — colleague's phased **FC→SC** research plan (swap direction, demeaned-correlation headline, multi-task FC concatenation, time-series transformer, JEPA future). Endorsed priorities: prove task FC injects individual signal (#3) and the task-to-region specificity map (#4) over chasing a SOTA model.

## Literature
- [research-papers/](research-papers/) — 8 background papers on structure–function connectome prediction:
  - **Finn 2015** — functional connectome fingerprinting (individual identifiability).
  - **Kim 2021 (STAGIN)** — spatio-temporal attention GNN on dynamic FC.
  - **Benkarim 2022** — Riemannian/diffusion SC→FC prediction.
  - **Smolders 2023** — critique: SC→FC models may not beat group-average FC.
  - **Zalesky 2024** — rebuttal + standardized individual-effect benchmarks (two copies; same Network Neuroscience paper).
  - **Jamison 2025 (Krakencoder)** — unified multimodal connectome translation/fusion (the key prior art).
  - **Lu 2025 (NetFormer)** — interpretable time-varying connectivity from neural dynamics.

## Infrastructure
- [torch-jupyter-tmux-skill.md](torch-jupyter-tmux-skill.md) — SOP for running Jupyter on NYU Torch HPC (tmux + SSH tunnel + `$SCRATCH`).
