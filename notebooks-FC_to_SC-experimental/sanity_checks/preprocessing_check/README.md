# Reduction-axis robustness check for the FC↔SC asymmetry

**Question**: The headline result of this project — FC→SC predicts ~1.5× better than
SC→FC at 10-seed precision across every basis — was computed with a specific
preprocessing pipeline: **PCA(256) on the source, PLS(64) on the latents, inverse
PCA on the target**. Reviewers will ask whether the asymmetry is a property of the
data or an artifact of that learned linear reduction.

This directory tests the asymmetry across a three-point ladder of reduction
methods. The compute is cheap; the implication is strong: if the asymmetry holds
across no reduction / learned linear / data-blind random, it's not a reduction
artifact.

## The three-point ladder

| Method | What it tests |
|---|---|
| **A. PCA→PLS→PCA** (main model) | Learned linear reduction on both ends. |
| **B. Full PLS** (no reduction) | PLSRegression directly on the raw 64,620-edge vectors, both ends. If A and B agree, the reduction isn't hiding signal *or* injecting it. |
| **C. JL→PLS→PCA** (data-blind random) | Replace input PCA with random projection. If A and C agree, the learned PCA basis on FC isn't doing anything privileged — any random projection captures the cross-modal signal equally well. |

For Method C, we test three Johnson-Lindenstrauss variants to rule out density
sensitivity:

| JL variant | Sklearn class | Notes |
|---|---|---|
| **Gaussian dense** | `GaussianRandomProjection(n_components=256)` | Dense N(0, 1/k) entries |
| **Sparse Achlioptas (default)** | `SparseRandomProjection(n_components=256, density='auto')` | density ≈ 1/√p ≈ 0.004 for p=64620; values ∈ {−√(1/d), 0, √(1/d)} |
| **Sparse 1/3** | `SparseRandomProjection(n_components=256, density=0.333)` | Original Achlioptas 1/3 density |

All three JL variants use the same output PCA(256) as Method A — we're isolating
the *input* reduction question. Each variant gets one new random matrix per seed.

## Scripts

| Script | Compute |
|---|---|
| `method_a_pca_pls_pca.py` | 10 seeds × 2 directions = 20 fits. ~3 min. |
| `method_b_full_pls.py` | 10 seeds × 2 directions = 20 fits on raw 64620-dim NIPALS. ~15–25 min. |
| `method_c_jl_pls_pca.py` | 10 seeds × 2 directions × 3 JL variants = 60 fits. ~5–8 min. |
| `synthesize_reduction_axis.py` | Loads all 3 method CSVs, computes per-seed asymmetry ratios, prints comparison table + automated verdict. ~10 s. |

All four are wrapped in `run_all.sbatch` (cpu_short, 16 CPU, 64GB RAM, 1h walltime).

## Outputs

- `method_<x>_results.csv` per method — columns: `method, jl_variant, seed, direction, demeaned_pearson, top1_acc, avg_rank`
- `reduction_axis_synthesis.csv` — pivoted per-method ratios across seeds
- `<script>_output.txt` — captured stdout
- `findings.md` — written after the job by the driver, with the headline verdict

## Verdict template (decided in `synthesize_reduction_axis.py`)

For each method, compute median (and [min, max]) asymmetry ratio across 10 seeds.
Main-model baseline is the published 10-seed ratio (~1.55× for PCA→PLS→PCA at
K_PCA=256 / K_PLS=64).

- **CLEAN** — all three methods give median ratio within ~0.15× of the main model
  (i.e. 1.4–1.7×). Headline survives.
- **PARTIAL** — methods agree on direction (all > 1.15× one-tailed) but magnitudes
  differ noticeably. Asymmetry is robust to reduction choice in *sign* but
  reduction does shift effect size.
- **ARTIFACT** — methods disagree on direction or magnitude (e.g., full PLS or JL
  flat at ratio 1.0 while learned PCA shows 1.55×). The asymmetry is a reduction
  property, not a data property. (Unlikely given the K-sweep already showed
  stability.)

## How it ran

Single SLURM batch on `cpu_short`. Scripts run sequentially, write per-method
CSVs, then the synthesis. `DONE.sentinel` touched at the end. No `squeue`
polling. Inputs from `_setup.load_seed_split` per seed.
