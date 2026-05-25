# SC reconstruction — what the project uses, and why it's already the recommended recipe

_Verified on disk 2026-05-25._

## TL;DR
The project's structural connectome is **SIFT2-weighted + inverse-node-volume normalized + log1p** (QSIPrep `sift_invnodevol_radius2_count_connectivity`), **not raw streamline counts**. This is exactly the recipe Zalesky (2024) and Smolders (2023) recommend, so SC reconstruction is **not** the bottleneck for the ~0.10 demeaned-correlation ceiling — the easy preprocessing lever is already pulled.

## What the loader uses
- Default metric: `sc_metric_type='sift_invnodevol_radius2_count_connectivity'`, `sc_apply_log1p=True` ([data/hcp_dataset.py:64-66](../data/hcp_dataset.py#L64), [data/dataset_utils.py:386-389](../data/dataset_utils.py#L386)).
- Raw data root: `/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/` (QSIPrep/QSIRecon outputs).
- Per-subject file: `HCP1200_DTI/qsirecon/sub-<id>/anat/sub-<id>_space-T1w_connectivity.mat` ([data/dataset_utils.py:344](../data/dataset_utils.py#L344)).
- `_load_single_sc_file` reads the field `atlas_<parcellation>_<metric_type>` from that one `.mat`.
- Precompute cache (`.npy`, one metric): `/scratch/asr655/neuroinformatics/Conn2Conn_data/{sc,fc,parcel_node_features}` — distinct from the raw `.mat` tree.

## The recommended pipeline (data → transform → transform)
```
1. Raw dMRI + T1w
      ▼  denoise (MP-PCA), de-Gibbs, motion/eddy/distortion correct, bias-field
2. Cleaned DWI
      ▼  response fn → Constrained Spherical Deconvolution (CSD)
3. Fiber Orientation Distributions (FODs)
      ▼  probabilistic tractography (iFOD2) + Anatomically-Constrained (ACT)
4. Whole-brain tractogram (millions of streamlines)
      ▼  ★ SIFT2 (or COMMIT) ★  ← the Zalesky/Smolders recommendation
5. Quantitative, bias-corrected per-streamline weights
      ▼  parcellate (FreeSurfer → Glasser/4S456Parcels), assign endpoints (radius2)
6. Node definitions + endpoint assignment
      ▼  tck2connectome: sum SIFT2 weights per edge, ÷ node volume (invnodevol)
7. N×N structural connectome
      ▼  symmetrize, zero diagonal, log1p, vectorize upper triangle
8. SC feature vector → model
```
The papers only really prescribe **step 5**: use SIFT2/COMMIT instead of raw counts, because raw streamline counts aren't quantitative (they depend on the tractography algorithm, seeding, tract length/curvature). Step 7's `invnodevol` (ROI-size correction) is the secondary recommendation. The project does steps 2–7 via QSIPrep and step 8 in the loader.

## The four edge metrics (all in one `connectivity.mat`)
Verified on `sub-100206` (13 connectivity fields total). All four present for both project parcellations (Glasser, 4S456Parcels):

| Field | Edge = | Bias it fixes / flaw |
|---|---|---|
| `radius2_count_connectivity` | raw streamline count | not quantitative; ROI-size biased (the one Zalesky/Smolders warn against) |
| `sift_radius2_count_connectivity` | SIFT2-weighted count | fixes density bias; still ROI-size biased |
| **`sift_invnodevol_radius2_count_connectivity`** ← used | SIFT2 count ÷ node volume | fixes **both** density + ROI-size bias |
| `radius2_meanlength_connectivity` | mean streamline length | orthogonal *geometric* signal (length, not density) |

### Verify on disk
```bash
python - <<'PY'
import scipy.io, re
f='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_DTI/qsirecon/sub-100206/anat/sub-100206_space-T1w_connectivity.mat'
m=scipy.io.loadmat(f, simplify_cells=True); keys=[k for k in m if not k.startswith('__')]
for w in ['radius2_count_connectivity','sift_radius2_count_connectivity',
          'sift_invnodevol_radius2_count_connectivity','radius2_meanlength_connectivity']:
    print(f"[{'x' if any(k.endswith(w) for k in keys) else ' '}] {w}")
PY
```

## Why we are NOT doing the 4-way ablation
Switching `sc_metric_type` is a one-line, no-reprocessing change, so the ablation is nearly free — but it's not worth the cycles:
- `count`, `sift_count`, `sift_invnodevol` are **transforms of the same tractogram** (one diffusion scan, reweighted/normalized) → strongly correlated, near-redundant. Stacking them adds ~no individual signal.
- Only `meanlength` is genuinely orthogonal (geometry vs density) — the lone non-redundant channel, and a long shot.
- Corroboration: `SC_r2t` (region-to-tract profiles, an alternate SC *view*) was already tried and scored **worse** than SC — piling on structural representations doesn't help.
- The individual info in `invnodevol` (parcel volumes) is already exploited via the FreeSurfer-volume covariates, which gave the best demeaned r (0.103).

**Key distinction:** multi-task FC = different *recordings* (different brain states) → genuinely new information. Multi-metric SC = different *summaries of one recording* → redundant. So the multi-view budget belongs on the **functional/task side**, not the structural side.

## The only real SC-side upgrade
**COMMIT** (a microstructure-informed alternative to SIFT2 at step 5). It is **not** in the current `.mat` and would require reprocessing from the tractogram — not low-effort, out of scope for now.

## Related
- [benchmarks_and_model_scores.md](benchmarks_and_model_scores.md) — the ceiling this SC feeds.
- [masked_latent_pretrainer.md](masked_latent_pretrainer.md) — the demeaning/PCA-latent model.
- Literature: Zalesky (2024), Smolders (2023) in [research-papers/](research-papers/); SIFT2 method = Smith et al. (2015).
