# FC noise sanity check

Measures how much of the **FC** connectome is reproducible signal vs measurement noise vs
day-to-day state — the physical counterpart to MASTER_FINDINGS F10 (statistical
saturation). Scoped to HCP-YA (current cache). SC noise is **not** measurable here (no
test-retest dMRI, no tractograms) — see the roadmap `planning/roadmap/noise-sanity-check.md`.

## Data model
Per subject, a 2×2 FC design (both Glasser + 4S456): `run-1/run-2` (session, ~1 day) ×
`LR/RL` (direction, minutes). The standard loader averages LR+RL away; `build_fc_cells.py`
re-caches the 4 cells separately. **No BOLD timeseries** → no half-split / bootstrap.

## Scripts (run in order; `run_all.sbatch` does this)
| File | Check | What it produces |
|---|---|---|
| `build_fc_cells.py` | PREP | 4-cell FC cache at `/scratch/ans9868/noise_cache/fc_cells/parc-*/` (Torch; reads per-direction relmat TSVs) |
| `a_reliability_ceiling.py` | A1+A2 (+F1) | reliability ceiling in native metric: within-session (LR↔RL) + between-session (REST1↔REST2); top1_acc = fingerprinting |
| `b_variance_decomposition.py` | B1 | G-theory 2×2 variance components → trait / state / within-session / **noise** fractions + averaged-connectome reliability G |
| `f_discriminability.py` | F2 | whole-connectome fingerprint top1 + discriminability (the distributed-signal reconciliation) |
| `e_crossmodal_disattenuation.py` | E1 | SC→FC achieved ÷ FC reliability ceiling = "% of reproducible FC captured" (+ bv+demo→FC reference) |
| `synthesize_noise.py` | — | one summary + headline numbers |

Outputs land in `outputs/` (CSVs) + the per-edge variance components `.npz`.

## What this can and cannot answer
- ✅ FC: reliability ceiling, trait/state/noise decomposition, fingerprinting, and the
  SC→FC fraction-of-ceiling — all from data we have.
- ❌ SC noise (FC→SC disattenuation, SC reliability): blocked — needs test-retest dMRI.
- ⚠️ within-session (LR↔RL) carries a phase-encode distortion confound (slightly inflates
  the "within-session/noise" component).
- Deferred: analytic sampling-noise floor (D1) — needs a confirmed per-run TR count.

## Compute
All vectorized linear algebra (no model fitting, no seeds for the reliability parts).
One-time PREP re-cache ~10–20 min (I/O); analyses ~15–25 min total for both parcellations.
`run_all.sbatch`: cpu_short, 16 CPU, 64 GB, `:ro` overlay.
