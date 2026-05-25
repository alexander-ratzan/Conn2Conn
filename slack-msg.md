# Slack messages — Alex (Conn2Conn)

_updated 2026-05-25 18:45 EDT_

## Sent

**1 — Pivot: SC→FC ceiling → FC→SC + multi-task**

I think you exhausted SC→FC-from-rest and proved its ceiling, which justifies pivoting to FC→SC + multi-task. The dead end is the motivation for the next direction. As you know, I believe FC carries more individual biomarkers/dynamics than SC. That, plus SC being a reliable prediction target, is why I think FC→SC has real headroom, although it may be less useful in a clinical setting. I honestly think your results here are already publishable and would be super useful to anyone working on SC→FC. You should be proud of yourself.

**2 — MaskedLatentPretrainer question**

Also I noticed that the `MaskedLatentPretrainer` does its masking/reconstruction in PCA-latent space, so it skips the shared group-average connectome and focuses on individual variation (something we talked about briefly on the whiteboard). Curious whether you think this needs to be run on the other large non-linear models (GNNs etc.) to check every stone, or if this sanity check is enough to show the hack doesn't drastically improve the metrics for the larger models.

**3 — SC data processing (resolved)**

> nvm — the data processing technique matches Zalesky and Smolders, so all good there.

Context behind the "nvm" (for our records):
- Verified on disk that SC = **SIFT2 + inverse-node-volume + log1p** (QSIPrep `sift_invnodevol_radius2_count_connectivity`) — the recommended recipe, not raw streamline counts. Matches Zalesky (2024) / Smolders (2023).
- All four edge metrics live in one `connectivity.mat` per subject (confirmed on sub-100206), so a 4-way ablation is *free* — but **not worth it**: three of the four (count / sift / sift+invnodevol) are redundant transforms of one tractogram; only `meanlength` is orthogonal. The multi-view budget belongs on the **functional/task** side, not the structural side.
- Full write-up: [dev-notes/sc_reconstruction.md](dev-notes/sc_reconstruction.md).

## Suggested follow-ups (optional, not yet sent)

**Meeting (recommended):**
> This is a bigger direction call though — would be great to talk the pivot through whenever you have 20 min.

**wandb access:**
> Could you add me to the wandb project when you get a sec? Want to be able to pull the result tables.
