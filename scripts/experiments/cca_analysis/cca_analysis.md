# cca_analysis — multiview CCA vs PCA representations for cross-view connectome translation (spec v4 E5)

**Question.** Does a multiview CCA co-embedding of the nine views (SC, rest, 7 tasks) give more accurate and more
identifiable translations than PCA representations, pairwise (H1) and when pooling the other views (H2)? And does a
composite (identity-aware) loss on the CCA decoders help (H3)? Spec: `context_packages/repo_spec_docs/spec_doc_v4.md` E5
(decision D11). Formal walk-through, figures and maths: `scripts/notebooks/model_overviews/cca_analysis_overview.ipynb`
(a renderer: it loads this experiment's outputs and computes nothing).

**Setup.**
- **Data:** matched cohort of 917 Glasser subjects; seeds 0–4; E2.1's splits (659 / 74 / 184 at seed 0); 72 ordered
  view pairs.
- **Metrics:** test demeaned r, average rank and top-1, computed on decoded edges. Selection uses validation average
  rank (D9).
- **Representations** (each selected on validation):
  - R0: nodal PCA of the mean train connectome, $q$ ∈ {2, 4, 8, 16}.
  - R1: condition-specific PCA.
  - R2: block PCA.
  - R3: variance-adjusted block PCA (D2).
  - R1–R3 use $k$ ∈ {64, 128, 256, 512}.
- **Maps:** ridge and the conditional Gaussian (Ledoit–Wolf) on R0–R3. Multiview CCA (MAXVAR) at full rank (equal to
  regularised CCA on the edges) with ridge decoders to native edges, plus CCA-256 (on R3 codes) as the E1 / E4
  reference. Copy of the source residual as a reference.
- **H3:** every CCA decoder retrained from the ridge solution with grid v3's composite factorial (8 combinations).

**Code.**
- Settings and shared code: `config.yml`; `e5_common.py` (representations, code-space scorer, ridge, Ledoit–Wolf
  conditional mean).
- `run_seed.py`: E5.1, covering H1, H2 and G0 on CPU, one job per seed.
- `run_composite.py`: E5.2, covering H3 and G1 on GPU; `--part` / `--nparts` split it into jobs.
- `run_descriptives.py`: the notebook's section-1 inputs.
- `analyze.py`: tables and figures.
- `launch/`: `run_seeds.sh`, `run_composite.sh`, `run_composite_parts.sh`, `run_analysis.sh`, `render_notebook.sh`.
- Results: `results/cache/experiments/cca_analysis/` (`results/seed<S>.json`, `h3/`, `render/`).
- Tables: `tables/`. Figures: `figures/`.

## Gates

- **G0** (5 / 5 seeds pass):
  - The loaded edges equal E2's cache exactly.
  - R3 ridge at $k$ = 256 reproduces E4.0c's B1 rows (max |Δ| ≤ 2e-9).
  - Full-rank CCA with E4.0c's selection and MAXVAR decoders reproduces E4.0c's B2 full-rank rows (≤ 1e-15).
  - The code-space scorer equals explicit decoding.
- **G1:** see H3.

## Descriptive statistics (`tables/descriptives_pairs.csv`)

Residuals (each connectome minus its view's train mean), test split, mean of 5 seeds:
- **FC ↔ FC (56 pairs):**
  - Within-subject residual r is 0.107 (range 0.06–0.19).
  - Average rank is 0.94 and top-1 0.46.
  - Small shared variance, highly subject-specific.
- **SC ↔ FC:** r ≈ 0.001, average rank 0.55, top-1 < 0.01. Nothing is shared edge-wise.

## H1 — CCA vs PCA representations, every pair (`tables/h1_*.csv`, `figures/h1_pairs_*.png`)

Test, per-seed mean over the pairs in the block, mean ± SE over 5 seeds:

| Model | All 72: demeaned r / average rank / top-1 | FC → FC (56) | SC → FC (8) | FC → SC (8) |
|---|---|---|---|---|
| CCA, full rank + ridge decoder | 0.091 / 0.793 / 0.093 | 0.098 / 0.812 / 0.108 | 0.042 / 0.668 / 0.025 | 0.096 / 0.788 / 0.058 |
| CCA-256 (on R3) | 0.089 / 0.811 / 0.106 | 0.094 / 0.831 / 0.125 | 0.043 / 0.686 / 0.028 | 0.094 / 0.794 / 0.053 |
| R3 block PCA (var.-adj.), ridge | 0.087 / 0.766 / 0.096 | 0.096 / 0.785 / 0.114 | 0.045 / 0.675 / 0.021 | 0.068 / 0.729 / 0.050 |
| R1 condition PCA, ridge | 0.076 / 0.766 / 0.098 | 0.082 / 0.780 / 0.113 | 0.041 / 0.688 / 0.025 | 0.070 / 0.740 / 0.061 |
| R2 block PCA, ridge | 0.084 / 0.758 / 0.087 | 0.092 / 0.776 / 0.102 | 0.044 / 0.670 / 0.019 | 0.065 / 0.722 / 0.047 |
| R0 nodal PCA, ridge | 0.072 / 0.749 / 0.087 | 0.080 / 0.766 / 0.102 | 0.041 / 0.672 / 0.024 | 0.047 / 0.707 / 0.042 |
| Copy (FC → FC only) | – | 0.107 / 0.845 / 0.249 | – | – |

Contrast CCA − best PCA (R3 on every seed), average rank:

| Block | Δ average rank | Seeds > 0 | Δ demeaned r |
|---|---|---|---|
| All | +0.027 ± 0.001 | 5 / 5 | +0.004 ± 0.001 |
| FC → FC | +0.027 ± 0.001 | 5 / 5 | +0.002 ± 0.001 |
| SC → FC | −0.007 ± 0.007 | 2 / 5 | −0.003 ± 0.001 |
| FC → SC | +0.058 ± 0.004 | 5 / 5 | +0.028 ± 0.002 |

**H1 is supported over all pairs.** It is supported for FC → FC and for FC → SC, where it is strongest. It is not
supported for SC → FC, where every representation sits at E2.1's floor.

Secondary findings:
- **Truncation helps identification.** CCA-256 identifies better than full-rank CCA (+0.018 average rank, 5 / 5) and
  is slightly less accurate.
- **Ordering of the PCA representations:** R3 > R1 ≈ R2 > R0. R0 has the highest own-reconstruction ceiling but is
  the worst transfer space.
- **The conditional Gaussian** trails validation-tuned ridge in every representation.
- **Copying beats every learned FC → FC map** on identification.

## H2 — other views → held-out view (`tables/h2_*.csv`)

| Model (mean over the 9 held-out views) | other8: demeaned r / average rank / top-1 | other7 (FC only) |
|---|---|---|
| CCA-agg (mean encoding) + ridge decoder | 0.143 / 0.912 / 0.253 | 0.145 / 0.916 / 0.284 |
| CCA-cat (concatenated encodings) | 0.142 / 0.914 / 0.257 | 0.145 / 0.917 / 0.282 |
| Best single source (CCA) | 0.122 / 0.880 / 0.191 | 0.120 / 0.878 / 0.195 |
| Ridge, concatenated R3 codes | 0.140 / 0.889 / 0.244 | 0.146 / 0.897 / 0.278 |
| Copy (mean of the FC sources' residuals) | 0.208 / 0.996 / 0.890 | 0.208 / 0.996 / 0.890 |

**H2 is supported.**
- **vs the best single source:** CCA-agg and CCA-cat lead by +0.031 to +0.039 average rank and +0.020 to +0.026
  demeaned r (5 / 5).
- **vs the best concatenated PCA ridge:** they lead by +0.018 to +0.026 average rank (5 / 5), with demeaned r level.
- **Averaging vs concatenating:** they tie, so the latent already pools the views.
- **SC as a source:** adding SC changes nothing.
- **SC as the target:** the pooled FC encodings predict SC (0.90 average rank) far better than concatenated PCA codes
  (0.84).

## H3 — composite loss on the CCA decoders

Every CCA decoder (72 pairs; H2's CCA-agg and CCA-cat) was retrained from the ridge solution on ridge objective + grid v3's composite factorial at 0.5. Each combination kept its best validation-average-rank weights (epoch 0 = ridge). **G1 passed on 5 / 5 seeds:** MSE-only decoders reproduce ridge to ≤ 2e-5 demeaned r and ≤ 2e-4 average rank. Seeds 1 and 4 timed out on A100s (2 h limit) and were rerun as 2 parts each on H100 / H200 (`launch/run_composite_parts.sh`); compute was about 10.4 GPU-h, cap raised 10 → 13 by the user.

Δ vs the ridge decoder, mean over all 106 decoders (`tables/h3_summary.csv`):

| Combination | Δ average rank | Δ top-1 | Δ demeaned r | Seeds > 0 | Supported |
|---|---|---|---|---|---|
| + Demeaned corr-eye | +0.041 ± 0.001 | +0.060 | −0.011 ± 0.001 | 5 / 5 | no (accuracy) |
| + VM + CEdm | +0.035 | +0.048 | −0.010 | 5 / 5 | no |
| + CEdm + ND | +0.022 | +0.029 | −0.006 | 5 / 5 | no (marginal) |
| + VM + CEdm + ND | +0.021 | +0.028 | −0.006 | 5 / 5 | no (marginal) |
| + Neighbour dist | +0.010 ± 0.001 | +0.018 | −0.003 | 5 / 5 | **yes** |
| + VM + ND | +0.009 | +0.017 | −0.003 | 5 / 5 | **yes** |
| + Var-match | −0.000 | +0.002 | −0.000 | 1 / 5 | no (inert) |

By decoder group:
- **SC → FC:** CEdm raises average rank from 0.668 to 0.729 (−0.005 demeaned r, supported).
- **FC → SC:** every term is inert.
- **H2 other 7 → held-out:** CEdm raises average rank from 0.916 to 0.950 and top-1 from 0.28 to 0.39 (−0.011 demeaned r).

In the seed-0 rest examples the predicted residuals are 3–6 times smaller than the true residual whatever the loss (notebook section 5).

## Conclusions

1. **The CCA co-embedding is a better translation space than any PCA representation, for identification** (H1). Full-rank CCA beats the best PCA representation (R3, variance-adjusted block PCA, the best on every seed) by +0.027 average rank over the 72 pairs, at no cost in accuracy. The gain is largest for FC → SC (+0.058 average rank, +0.028 demeaned r) and absent for SC → FC.
2. **Representation matters within PCA as well:** R3 > R1 ≈ R2 > R0. Equal-variance pooling across conditions (D2) is the best linear transfer space. A basis built from the group-mean connectome (R0) reconstructs well but transfers worst. The Ledoit–Wolf conditional Gaussian never beats validation-tuned ridge.
3. **Truncation is a useful regulariser for identification.** CCA on 256 block-PCA codes identifies better than full-rank CCA (+0.018 average rank), which justifies E1 / E4's truncated latent when identification is the goal. Full rank is slightly more accurate.
4. **The latent pools views well** (H2). Averaging the encodings of the other views equals a decoder that weights them, beats the best single source by +0.03–0.04 average rank, and beats concatenated PCA codes by +0.02. SC adds nothing as an extra source, but it is best predicted from the pooled FC latent.
5. **Identity-aware decoder losses trade accuracy for identification** (H3). Neighbour distance passes the support rule; demeaned corr-eye gives four times the identification gain at about twice the allowed accuracy loss. The loss should follow the metric the application needs.
6. **Learned linear maps remain below the copy reference for FC → FC identification.** Copy reaches average rank 0.845 (single source) and 0.996 (other 7). Every learned map trails it. The subject-specific signal shared across conditions is small in variance (within-subject residual r ≈ 0.11) and largely outside what a linear map fit on 659 subjects transfers. Closing that gap is spec v4 E4's question.

Last updated: 2026-10-28
