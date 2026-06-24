# Why does `bv+demo` not perfectly predict its own sex/age? — RESOLVED

## TL;DR (resolution)
**Root cause:** a **float32 numerical bug**. `bv+demo` contains exactly-collinear one-hot columns
(sex/race each sum to 1 → rank-deficient); PCA's SVD in **float32** is ill-conditioned on that and
dropped the demographic directions by a **split-dependent** amount. `bv+demo→age` per seed was
`[0.99,1.0,0.90,0.94,0.77,0.77,0.86,0.46,1.0,0.72]` (float32) vs **1.000 on every seed (float64)**.
The reconstruction path was protected by `PLSRegression(scale=True)`; the scalar `PCA→{BR,PLS,KR}`
path had neither scaling nor float64.

**Fix:** cast `X` to float64 before PCA in the scalar estimators (`_grid_common.py`, commit `69add40`).

**Impact on the science — NONE.** Full 20-unit downstream re-run (commit `f8d84d1`):
- **Cognition (F4/F5): byte-identical.** `max |lift Δ|` across ALL cognition cells = **0.0**;
  `bv+demo→Cog{Total,Fluid,Cryst}` baseline unchanged (0.359/0.283/0.354 both before & after).
- **Leak diagnostics (sex/age): fixed.** `bv+demo→sex` 0.945→**1.000**, `→age` 0.840→**1.000**
  (both parcs) — now correctly perfect since the labels are in the features.
- **0 LEAK_FAIL** maintained (ok 3326 / EXEMPT_FLAGGED 1070 / EXPECTED_SIGNAL 4).

Why cognition was untouched: float32 only mangled the *collinear demographic directions* (what you
need to predict sex/age), while cognition prediction rides on the well-conditioned brain-volume
directions. So the bug's entire blast radius was the **sex/age leak-check panel** (never a finding) —
e.g. the S4 figure that flagged this. F1/F2/Ceiling-B/F6/F7/F8 never used this path and are unaffected.

**Reconstruction confirmed unaffected (no re-run needed).** `bv+demo→SC` (the only recon pair with a
collinear *source*) is float32 ≡ float64 to 4 decimals via the actual capped estimators
(BR 0.1864=0.1864, 0.1753=0.1753; PLS within 5e-4). The bug needs the *target* to BE the collinear
column (the leak case); when the target is a connectome, the bvdemo-source prediction is robust. So
the reconstruction grid (`reconstruction.csv`) is untouched; only `downstream.csv`'s sex/age rows
changed.

---

## Original investigation (kept for the record)


**Observed (downstream.csv, BR, Glasser):** `bv+demo → sex` = 0.945 (bal acc), `→ age` = 0.840 (r),
even though `bv+demo` contains `sex_oh` and `age_z`. `pred_X+bv+demo` → ~1.0. Flagged as a possible
baseline bug (would inflate cognition lifts if the baseline under-uses demographics).

## What the sanity check (`diag_bvdemo_baseline.py`) found

**Hypothesis tested:** "PCA compresses the demographic info to nothing." → **Refuted for `bv+demo`.**

- `bv+demo` is ~22-dim; the scalar path uses `PCA(k=min(256,22)=22)` — a **lossless rotation, no
  truncation**. On a faithful synthetic (correlated 16-col brain-vol block + independent age_z +
  sex/race one-hots), `PCA(22)→BayesianRidge` recovers age **1.000** and sex **1.000**, even though
  age sits in a low-variance PC (#4, 5.5% var). BayesianRidge shrinkage does **not** erase it at full
  rank.
- Compression only bites under **truncation**: `PCA(k=2)→BR` → age 0.066, sex 0.485. So the intuition
  is correct *in general* — and it **does** apply to the **connectome** inputs (`obs_FC` is 64,620-wide
  → `k=256` → large truncation) — but **not** to the 22-dim `bv+demo` baseline.

**Implication:** the `bv+demo` cognition baseline is **not** weakened by PCA compression (it's full
rank), so the earlier worry that lifts are inflated by an under-powered baseline is **largely
allayed** — pending the real check.

## Leading explanations for the sub-1.0 numbers (to confirm on Torch)

1. **`age` (0.84) — likely BENIGN.** If the `age_z` *feature* and the `age` *target* use different
   encodings (HCP continuous restricted age vs binned public age), recovery caps at their
   correlation even with no truncation (synthetic B: a binned target caps PCA→BR at the cont↔binned
   correlation). Then 0.84 is the true ceiling, not a bug.
2. **`sex` (0.945) — the real puzzle.** `sex_oh` should be an exact one-hot of the `sex` target, so a
   gap is more suspicious: label mismatch on some subjects, NaN handling, or a `pd.get_dummies`
   column-order/centering quirk. Synthetic sex = 1.000, so the real gap is not explained by the method.

## Decider: `--real` on Torch (ready, not yet run; VPN down)

```bash
python reproduction/exploration/diag_bvdemo_baseline.py --real --parc Glasser --seed 0
```
Prints the real `bvdemo` width, where `age` lives in the PCA spectrum, and the method ladder incl.
**raw-OLS vs PCA→BR** on the actual features:
- raw-OLS ≈ 0.84 too → **encoding mismatch (benign)**, no code change needed.
- raw-OLS ≈ 1.0 but PCA→BR ≈ 0.84 → **a real path bug** → fix (whiten PCA scores / route bv+demo
  through `_block_latents`) and re-run downstream, then re-check F4/F5.

**Status:** do NOT apply the whitening "fix" yet — the synthetic shows whitening does not change the
full-rank case, so it is only warranted if the real run shows a path bug. The connectome-input
truncation (k=256) is a separate, intentional design parameter, not a bug.
