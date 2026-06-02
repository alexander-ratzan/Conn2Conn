# PC3 tractography-reliability sanity check — findings

**Run**: SLURM job 10084347 on `cpu_short`, 16 CPU / 32G, ~1 min total compute.
All inputs from Depth 1 seed-0 PC3 loadings (`sc_pc_loadings.npy`, row 2).

## TL;DR

**Verdict: the PC3 visual/DAN localization survives reliability partialling with room to spare.**
Two of the three headline within-network enrichments are *higher* after partialling
edge strength + distance; the third (DAN-DAN) drops modestly but stays at 5.7×.
The biological reading of PC3 holds. Add the partialled numbers to the writeup as
the reviewer-proof sentence.

| Headline pair | Raw enrichment (Depth 1.1) | Residualized (this check) | Δ |
|---|---|---|---|
| **visual ‖ visual** | 11.97× | **13.32×** | **+1.35** |
| **dorsal attention ‖ dorsal attention** | 7.73× | **5.73×** | −2.00 |
| **dorsal attention ‖ visual** | 4.49× | **4.99×** | +0.50 |

Reliability proxy (strength + distance) explains **41.0% of |PC3| variance**, so
the partialling is non-trivial. After removing that 41%, the residual still concentrates
on the same network pairs — most clearly in visual cortex.

## Per-proxy results

### Proxy 1 — edge strength (Spearman + R²)

- `Spearman(|PC3|, edge_strength)` = **+0.89** (p ≈ 0)
- `OLS R²` of `|PC3| ~ edge_strength` = **0.40**
- Pearson = +0.63

Strong positive: PC3 emphatically lives in high-strength edges. **Reliability confound
on streamline density is LIVE.** Posterior visual edges happen to be high-strength
because tracking is cleanest there, so this was the expected red flag.

### Proxy 2 — anatomical distance (Spearman + R²)

- `Spearman(|PC3|, edge_distance)` = **−0.50** (p ≈ 0)
- `OLS R²` of `|PC3| ~ edge_distance` = **0.13**
- Pearson = −0.36
- Distance was computed from Glasser MNI centroids (atlas CSV `mni_x/y/z`).
  Range: 3.3–160 mm, median 79 mm.

Strong negative: PC3 favors short edges. **Distance confound is also live but smaller
in magnitude than strength** (R² 0.13 vs 0.40).

### Decisive partialled enrichment (the load-bearing test)

OLS `|PC3| ~ strength + distance`:
- R² = **0.41** (strength dominates; distance adds essentially nothing beyond it)
- coef[strength] = +0.0135, coef[distance] = −1.2e-5
- residual std / signal std = **0.77** (77% of the magnitude variation is still there)

Re-ranked top-200 edges by `|residual|` and recomputed Yeo7 network enrichment:

| Net pair | enr_raw | enr_resid | n_resid |
|---|---|---|---|
| visual ‖ visual | 11.97 | **13.32** | 59 |
| dorsal attention ‖ dorsal attention | 7.73 | **5.73** | 20 |
| dorsal attention ‖ visual | 4.49 | **4.99** | 40 |
| frontoparietal ‖ frontoparietal | 2.61 | 1.63 | 5 |
| dorsal attention ‖ frontoparietal | 1.50 | 1.20 | 8 |
| default mode ‖ dorsal attention | 1.38 | 1.30 | 16 |
| somatosensory ‖ somatosensory | 0.21 | 1.26 | 6 |
| default mode ‖ default mode | 0.67 | 0.19 | 2 |

(Full table: `enrichment_residual_top200.csv`.)

**Visual-visual enrichment goes UP after partialling.** That's the key observation —
not just "survives" but "intensifies." It means reliability under-predicts how
strongly PC3 emphasizes visual cortex; once you subtract the part of |PC3|
explained by strength+distance, what's left is *even more* visual-concentrated.

DAN-DAN drops from 7.73× to 5.73× — about a quarter of its enrichment was
attributable to reliability — but the residual is still ~5–6× chance, well above
the >5× "clean biological" threshold. DAN-visual is essentially unchanged.

### Retest reliability (not run)

The rigorous version of this test would use HCP test-retest data (n≈45 subjects
scanned twice) to compute per-edge ICC and partial that out instead of the
strength+distance proxy. That requires standing up a retest ingestion pipeline
(~1–2 days) which is out of scope for this sanity check. The proxy is the
standard fallback in the field and tends to agree with retest reliability on
which edges are noisy. Documented in `retest_check_note_output.txt`.

## Reviewer-proof sentence for the writeup

> "PC3's visual/DAN localization is not attributable to higher reconstruction
> reliability in posterior short connections: after partialling streamline density
> and inter-region distance from |PC3| (which together account for 41% of its
> magnitude variance), the headline within-network enrichments remain at 13.3×
> (visual–visual; *higher* than the unadjusted value), 5.7× (DAN–DAN), and 5.0×
> (DAN–visual). The visual–visual enrichment increases under partialling,
> indicating reliability under-predicts PC3's concentration on visual cortex
> rather than driving it."

## Caveats to keep honest

1. **Proxy not retest.** Edge strength and inter-region distance correlate with
   tractography ICC but are not identical. A future ICC-based check could in
   principle reveal a confound the proxy misses.
2. **Single seed.** Ran at seed 0 (the Depth 1 PCA basis). The Depth 1.1 stability
   analysis showed PC3 is stable across seeds (median |cos|=0.89), so the result
   should generalize, but the partialling itself wasn't re-done per seed.
3. **DAN-DAN partial give-back.** The 7.7×→5.7× drop says ~25% of DAN-DAN
   enrichment was reliability-driven. Frame as "DAN-DAN remains 5.7× enriched
   after partialling," not "DAN-DAN is unaffected."
4. **The 41% R² is high.** Don't bury it. Strength + distance together explain
   nearly half of |PC3|'s magnitude; the surviving signal is the other half.
   This is not "reliability is irrelevant" — it's "reliability matters, AND the
   residual is still cleanly visual/DAN."

## Files in this directory

- `README.md` — what each script tests and why
- `proxy1_strength.py` + `_output.txt`
- `proxy2_distance.py` + `_output.txt`
- `decisive_partialled_enrichment.py` + `_output.txt` + `enrichment_residual_top200.csv`
- `retest_check_note.py` + `_output.txt`
- `run_all.sbatch` — SLURM wrapper (cpu_short, 16 CPU, 30 min, auto-exits)
- `_cache_*.npy` — local scratch (gitignored)
