# Tractography reliability sanity check for PC3 localization

**Question**: Depth 1.1 found that SC-PC3 is overwhelmingly concentrated in visual ||
visual (11.7×), dorsal-attention || dorsal-attention (8.2×), and DAN || visual (4.9×)
network pairs, with 62% of L2 energy in 1% of edges and a strong rich-club signature.
Before writing this up as a biological finding, we need to rule out the methodological
alternative: **PC3 might just be living in the edges that tractography reconstructs
most reliably**, in which case "PC3 is visual/DAN" really means "PC3 is wherever the
diffusion scanner is reliable."

Posterior visual cortex has short, dense, well-myelinated connections — exactly the
ones tractography reconstructs cleanest. So the worry is real.

The rigorous version of this check requires HCP test-retest data (~45 subjects scanned
twice) — we don't have that wired into this pipeline. The scripts here run the **proxy
version**: edge weight (streamlines reliably reconstructed → higher weight) and
anatomical distance (longer connections → more tracking error → less reliable). These
are the standard fallbacks when retest is unavailable.

## Scripts

| Script | Test |
|---|---|
| `proxy1_strength.py` | Spearman(\|PC3\|, edge_strength) + OLS R². Does PC3 just track edge weight? |
| `proxy2_distance.py` | Spearman(\|PC3\|, edge_distance) + OLS R². Distance = euclidean(centroid_i, centroid_j) from Glasser MNI coords. Does PC3 favor short connections? |
| `decisive_partialled_enrichment.py` | **THE LOAD-BEARING TEST.** Residualize \|PC3\| on [strength, distance] via OLS, re-rank edges on residual, re-run network enrichment on top-200 residual edges. If visual\|\|visual + DAN survive at high enrichment, the biological claim stands. If they collapse to ~1×, PC3 was tracking reliability. |
| `retest_check_note.py` | Documentation-only. Explains the rigorous version (ICC over HCP retest scans), flags that we don't have retest data wired into this pipeline, lists what would be needed. |

## Verdict template

After `decisive_partialled_enrichment.py` runs, three outcomes are possible:

1. **Visual/DAN survives partialling (residual enrichment > 5×)** — biological claim clean.
   Headline: "PC3's visual/DAN localization is not explained by edge reliability
   (survives partialling streamline density and inter-region distance; residual
   enrichment X×)."

2. **Visual/DAN partly survives (residual enrichment 2–5×)** — honest hedge.
   "Concentrates in visual/DAN cortex, partially but not fully attributable to higher
   reconstruction reliability in these short posterior connections."

3. **Visual/DAN collapses (residual enrichment ~1×)** — methodological artifact.
   "The FC-predictable mode concentrates in the most reliably-measured connections,
   consistent with prior reports that prediction accuracy tracks tractography
   reliability." PC3-localization comes out of the paper or becomes a one-liner.

## Inputs

All scripts read the saved Depth 1 outputs:
- `…/results/local_results/further_exploration/depth1_spectral_mechanism/sc_pc_loadings.npy`
  (10 components × 64,620 edges) — PC3 is row index 2.
- `data/atlas_info/Glasser_dseg_reformatted.csv` — 360 regions × {Yeo7 network,
  hemisphere, MNI x/y/z}.
- Seed-0 SC_train (via `_setup.load_seed_split`) for edge_strength.

## Outputs

- `findings.md` — synthesis written after job completes, summarizing all 4 outputs into
  a verdict.
- `<script_name>_output.txt` — captured stdout of each Python script.
- `enrichment_residual_top200.csv` — full residualized enrichment table from the decisive test.

## How it ran

Single SLURM batch job on `cpu_short`, 16 CPUs / 32G / 30 min walltime. All 4 scripts
run sequentially (each is <1 min — these are 64k-element numpy/scipy ops, not heavy).
Job auto-exits when scripts complete; a `DONE.sentinel` file is touched at the end so
the driver knows when to read results back. No `squeue` polling per project convention.
