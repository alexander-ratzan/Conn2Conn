# Grid Exploration — Do the Findings Hold, and What Else Is in There?

Deep double-check of the 13,640-cell reproduction grid, **straight from the merged CSVs**
(`outputs/{reconstruction,downstream,leak_verdict}.csv`). Regenerate with
`python exploration/explore.py` (→ `digest.txt`) and `python exploration/make_figures.py`
(→ `figures/`). Everything below is computed, not remembered.

**Bottom line:** every headline claim (F1, F2, F4, F5, Ceiling B) replicates on *both*
parcellations and is stable across seeds. The interesting part is in the margins — eight
things the summary report doesn't say, three of which actually *strengthen* the story and two
of which are methodological flags. Verdicts at the end.

---

## Part 1 — Do the confirmatory findings line up? (Yes.)

| Claim | Result (both parcs) | Status |
|---|---|---|
| **F1 asymmetry** FC→SC > SC→FC | 1.68× (4S456), 1.61× (Glasser), pca_pls | ✅ replicates |
| **F2 dissociation** anatomy vs demographics | clean double dissociation (see below) | ✅ replicates, **stronger than stated** |
| **F4 FC→cognition** real over baseline | obs_FC lifts CogCryst 60% of seeds, p≈0.03–0.04 | ✅ replicates |
| **F5 SC underperforms / pred adds nothing** | obs_SC lift < 0; pred_FC *actively harmful* | ✅ replicates, **stronger** |
| **Ceiling B** within-modal oracle | FC→FC 0.63/0.67, SC→SC 0.62/0.65 (BR) | ✅ replicates |

All match `reports/reproduction_findings.md` to the third decimal. No cell is non-finite; the
only NaN `pearson`s are the 2,200 `sex` rows (classification → uses `balanced_acc`, by design).

---

## Part 2 — Eight things worth knowing (the "anything unusual" hunt)

### ① The cross-modal asymmetry is NOT a reliability artifact ⭐ (strengthens F1)
The obvious skeptic's objection to "FC→SC beats SC→FC" is "SC is just noisier, so it's a worse
*target*." The oracle kills that: at the within-modal ceiling, **FC and SC are essentially
equally self-predictable** — FC→FC / SC→SC = **1.02× (4S456), 1.04× (Glasser)**. Both modalities
are ~equally reliable, yet FC→SC beats SC→FC by **1.6–1.7×**. So the asymmetry is a genuine
*directional information* effect (FC contains more about SC than vice-versa), not a target-noise
artifact. This is a defensible, reviewer-proof framing of F1.

### ② Clean double dissociation: anatomy→structure, demographics→function ⭐ (strengthens F2)
Not just "bv predicts SC." It's a crossed dissociation, on both parcellations
(`figures/f2_dissociation.png`):

| | →SC (structure) | →FC (function) |
|---|---|---|
| **bv** (anatomy proxies) | **0.185 / 0.162** (wins) | 0.045 / 0.049 |
| **demo** (demographics) | 0.133 / 0.130 | **0.105 / 0.111** (wins) |
| ratio bv/demo | 1.39 / 1.24 | **0.43 / 0.44** |

Anatomy is the better predictor *of structure*; demographics is >2× better *of function*. A clean
2×2 crossover is a much stronger statement than a single main effect.

### ③ pred_FC is *actively harmful* downstream, not merely useless ⭐ (sharpens F5)
`figures/f3_downstream_lift.png`. The imputed connectomes don't just fail to help cognition —
**pred_FC drives prediction below the bv+demo baseline by −0.11 to −0.135** (the most negative
cell in the whole downstream grid, both parcs). pred_SC is roughly neutral (≈0). So "imputation
doesn't transfer to cognition" understates it: imputed FC injects structured noise that *beats the
baseline down*. obs_SC is also net-negative for cognition. Only **obs_FC** (and obs_FC+bv+demo)
lifts cognition.

### ④ The kernel_ridge 3×3 sweep is degenerate — 9 variants ≈ 1 (compute flag)
Across the entire 9-cell gamma×alpha grid, the within-seed max−min spread of `demeaned_pearson`
is **0.004–0.005** — i.e. the HPs do essentially nothing (`digest.txt §2`). 9 of the 11
estimator-variants carry near-identical information. **Implication:** a future re-run could collapse
KR to a single HP and shed ~⅔ of the estimator axis (and a big chunk of the downstream blow-up the
PCA-cache note targets) with no loss of signal. Worth noting in the methods as "HP-insensitive."

### ⑤ Estimator choice moves every headline magnitude (reporting flag)
The report mixes estimators: F1 cross-modal uses **pca_pls**, Ceiling B uses **bayesian_ridge**.
But BR is the stronger estimator *everywhere*:
- Oracle: BR FC→FC = 0.63–0.67 vs **pca_pls only 0.42–0.45** — PLS reaches ~65% of BR's ceiling.
- Cross-modal: BR FC→SC = 0.171/0.166 vs pca_pls 0.146/0.136 — **BR is ~17% higher.**

The asymmetry *ratio* is estimator-robust (good), but the **absolute headline numbers depend on
which estimator you quote.** Recommend stating the estimator next to every number, and consider
leading F1 with BR (the best estimator) rather than PLS, for consistency with Ceiling B.
Downstream is the same story inverted: **pca_pls gives ~1.5× larger lifts** than BR
(obs_FC→CogCryst: pca ≈ 0.18–0.20 vs BR ≈ 0.12–0.13) — but BR is the conservative choice and is
correctly what the report leads with.

### ⑥ The finer parcellation specifically boosts structure prediction (structured, not noise)
4S456 vs Glasser is *not* a wash — the difference is signed and consistent (`figures/f4_cross_parcellation.png`):
**every →SC pair is higher on 4S456 (+8 to +14.5%), every →FC pair is slightly lower (−5 to −7%).**
The +60% finer parcellation adds resolvable structural detail (bv→SC +14.5%, bv+demo→SC +12.9%) but
marginally dilutes functional prediction. This is the genuinely-new cross-parcellation evidence and
it has a clean interpretation, not just "numbers track."

### ⑦ Cross-modal prediction captures only ~20–24% of the achievable ceiling (honest framing)
FC→SC (pca_pls) / SC→SC oracle (BR) = **23.6% (4S456), 21.0% (Glasser)** (`figures/f5_ceiling_gap.png`).
The cross-modal map is real but recovers only ~⅕–¼ of what's in principle recoverable about the
target connectome. Good honest ceiling language for the paper: "well above chance, far below the
within-modal oracle."

### ⑧ Only the *combined* FC+SC crosses the sex-leak threshold (mild curiosity)
All 4 EXPECTED_SIGNAL rows are **obs_FC+obs_SC → sex** at score 0.991–0.995 (threshold 0.99).
Neither FC nor SC alone crosses; combining the two modalities pushes sex-decodability just over the
line. Real biology (sex is strongly encoded in connectomes), correctly *not* flagged as a leak — but
a nice illustration that FC and SC carry partly *complementary* demographic information. 0 LEAK_FAIL.

---

## Part 3 — Smaller observations
- **CogCryst is the only reliably FC-predictable cognition.** CogCryst lifts hit 60–80% of seeds
  significant; CogFluid **never** reaches significance from any connectome input (0% seeds for obs_FC).
  Crystallized > fluid intelligence in connectome-predictability — consistent with the literature.
- **Identifiability ladder** (top1 fingerprint accuracy, pca_pls): FC→FC oracle ≈ **0.997–0.999**
  (near-perfect), FC→SC cross-modal only 0.12–0.15, SC→FC worst (0.049). Adding bv+demo *raises*
  identifiability (FC+bv+demo→SC top1 0.17–0.23 > FC→SC alone) — subject-info sharpens the fingerprint.
- **Connectome adds ~nothing over bv+demo for reconstruction.** FC+bv+demo→SC vs bv+demo→SC is
  −0.007 (4S456) / +0.002 (Glasser): once you have anatomy+demographics, the *opposite* connectome is
  redundant for predicting SC. SC+bv+demo→FC adds a little more (+0.005 to +0.012). Consistent with ②.
- **Seed stability is excellent.** Headline CVs: FC→SC just **1.8% (4S456) / 5.0% (Glasser)**; the
  noisiest headline is SC→FC at ~7–9%. Nothing is seed-fragile.
- **age "prediction" from bv+demo (0.83–0.84) is near-trivial** — age is *in* demo. The real signal is
  connectomes predicting age much more weakly (obs_FC ≈ 0.43–0.47), and pred_* degrade it (0.25–0.32).

---

## Part 4 — Verdicts

**Confirmatory:** ✅ All of F1, F2, F4, F5, Ceiling B replicate on both parcellations, stable across
10 seeds, no data-integrity issues. The grid is sound.

**Net-new / actionable, in priority order:**
1. **Use ①, ②, ③ in the paper — they strengthen the story** (asymmetry isn't a reliability artifact;
   F2 is a true double dissociation; pred_FC is harmful, not just useless).
2. **Reporting hygiene (⑤):** state the estimator next to every headline number; the absolute values
   differ by ~17% (recon) to ~1.5× (downstream) by estimator even though directions are robust.
3. **Compute (④):** the KR 9-HP sweep is degenerate — collapse to 1 HP on any re-run.
4. **Framing (⑥, ⑦):** the 4S456→SC boost is a real interpretable effect; lead the ceiling discussion
   with "~⅕–¼ of the within-modal oracle."

**Nothing alarming surfaced** — no leak failures, no implausible cells, no seed-fragile headline, no
sign flips across parcellations.
