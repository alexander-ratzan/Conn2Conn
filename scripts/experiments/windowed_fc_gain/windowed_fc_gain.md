# windowed_fc_gain — does FC from parts of the scan add usable supervision?

**Spec:** v4 E3 pre-check (E3.A: more samples per subject), run 2026-10-27. A controlled test before joint conditional
cross-modality / cross-task translation: if FC built from parts of a subject's scan behaves like extra noisy draws of
that subject's FC, the linear baselines (ridge / conditional Gaussian / CCA) and the flow can train on 2–4 samples per
subject instead of one. The linear baselines are kept throughout, so any flow gain has to beat them.

**Code:** `run_gates.py` (gates 1–6, one condition per call; `--report` writes tables, figures, summary),
`config.yml` (schemes, estimator, gate thresholds fixed before results), `launch/run_gates.sh` (CPU array over the 8
conditions) + `launch/report.sh`. Estimator: `data/timeseries.py` `segment_fc` / `part_bounds`.
**Data:** `Conn2Conn_data/timeseries` (Glasser), training target = `Conn2Conn_data/fc` combined matrix (xcp-d: Pearson
of the concatenated runs). HCP session summaries (`context_packages/HCP_context/sessionSummaryCSV_1200Release.zip`,
untracked) → `results/scan_sessions_fmri.csv` (tracked; run, phase-encoding run order, E-Prime script, completeness).
**Outputs:** `results/<condition>.json`, `<condition>_inclusion.csv`, `<condition>_subjects.csv`;
`tables/summary.md`, `gate_verdicts.csv`, `gate1_estimator.csv`, `gates2to5_parts.csv`, `gate5_timing_groups.csv`,
`gate6_inclusion.csv`; `figures/fig_identification_vs_length.png`, `fig_part_shift.png`.

```bash
jid=$(sbatch --parsable scripts/experiments/windowed_fc_gain/launch/run_gates.sh)
sbatch --dependency=afterok:$jid scripts/experiments/windowed_fc_gain/launch/report.sh
```

## Background (2026-10-27)

- LR and RL are separate, sequential runs (HCP S1200 manual; session summaries: tasks run RL first with E-Prime
  script 1, then LR with script 2; rest session 1 RL → LR, session 2 LR → RL since Oct 2012). The two task runs repeat
  the paradigm with different stimuli; social differs in block composition by design.
- LR vs RL block timing, from group-mean timecourses (subjects' averages keep only design-locked activity): matched for
  gambling, relational, emotion (LR vs RL 0.84–0.91 vs RL split-half ceiling 0.88–0.93), mostly for social (0.75,
  0.87 at a 1-TR shift), not for wm, motor, language (0.27–0.43 vs ceilings 0.97–0.99, no single shift aligns them).
- The target concatenates the runs; correlating a concatenation lets segment means and scales enter, so samples use
  per-segment demeaning (`demean_concat`) and G1 checks that this estimator matches the target at full length.

## Design

Samples take the same part of **every** run (both phase-encoding directions; for rest both sessions), so they mirror
the target's composition: `halves` (2 samples), `quarters` (4). References of the same kind of length: `direction`
(all LR runs vs all RL runs) and, for rest, `session` (day 1 vs day 2). Identification follows
`models/eval/metrics` (avg rank 1 = own sample first, chance 0.5); "dm" = demeaned (each sample type's mean over the
subjects removed), the subject-specific part our models are scored on. All included subjects of a condition (no split).

| Gate | Question | Threshold (fixed before results) |
|---|---|---|
| G1 | Does the estimator reproduce the target at full length? | median edge r ≥ 0.98 |
| G2 | Do a subject's part samples identify each other? | halves, demeaned avg rank ≥ 0.90 |
| G3 | Does each part sample point at the subject's full target? | halves, demeaned avg rank ≥ 0.95 |
| G4 | Is there a systematic difference between parts (time on task, block mix)? | ‖mean A − mean B‖ / median ‖T_i − mean T‖ ≤ 0.25 |
| G5 | Do tasks with unmatched LR / RL timing behave worse? | descriptive |
| G6 | How many subjects are usable (full-length runs; tasks: standard acquisition pattern)? | ≥ 85 % per condition |

**Caveat built into G2 / G3.** Parts of the same runs share run-level effects (distortion pattern, that run's arousal
and motion, denoising), and every part is a subset of the target's data. G2 therefore overstates retest agreement and
G3 is partly guaranteed; the `direction` and `session` splits (different runs) are the honest retest comparison.

## Results (gates 1–6, 2026-10-27; jobs 21166851 / 21166852, 1.5–5 min per condition)

Full tables: `tables/summary.md`; figures `figures/fig_identification_vs_length.png` (own-subject demeaned r vs sample
length) and `fig_part_shift.png`.

| Gate | Outcome |
|---|---|
| G1 | **pass, all conditions.** `demean_concat` at full length reproduces the target (median edge r ≥ 0.9996, worst subject 0.994; rest exact). Plain concat is exact (sanity). Equal-weight estimators do not: `zscore_concat` / `fisher_mean` 0.96–0.99 (median max edge error 0.17–0.67): segment variance weighting, not segment means, is what averaging changes. |
| G2 | **pass but uninformative.** Halves and quarters identify each other perfectly in every condition (avg rank 1.000, even 63-s emotion quarters). Own-subject demeaned r: halves 0.61–0.75, quarters 0.44–0.61. The same-length LR-vs-RL split (different runs) gives 0.04–0.15 for tasks (avg rank 0.79–0.98, top-1 4–55 %) and 0.52 for rest (session split 0.49). Within-run parts share that run's state, distortion and noise, so their agreement is mostly run-level, not retest. |
| G3 | **pass (partly by construction).** Each part sample is a subset of the target's data: own demeaned r to the target 0.90–0.93 (halves), 0.76–0.84 (quarters); relative error ‖S − T‖ / ‖T − mean T‖ 0.38–0.50 (halves), 0.64–0.85 (quarters); the mean of a subject's parts is within 3–7 % (halves) of T. |
| G4 | **pass.** Systematic part shift / subject spread: tasks 0.03–0.09 (halves), 0.04–0.13 (quarters); rest 0.14 (halves), up to 0.21 for first vs last quarter, rising with distance in time (0.11, 0.17, 0.21 from quarter 1): a time-in-run drift in rest (64 % of edges FDR-significant, median per-edge effect d 0.10). The LR vs RL split shifts more (0.12–0.24): a direction bias, which samples that keep both directions avoid. |
| G5 | Unmatched-timing tasks (wm, language, motor) have larger part shifts (halves 0.06 vs 0.03 matched; wm 0.09–0.13) and lower part agreement (wm halves 0.61 vs 0.72–0.75), consistent with different block mixes per part; all still pass. |
| G6 | **pass except relational (0.846 < 0.85).** Usable subjects: emotion 99.6 %, motor 99.8 %, gambling 99.0 %, social 98.7 %, language 96.0 %, rest 92.0 % (87 without 4 full runs), wm 91.6 % (91 non-standard acquisition), relational 84.6 % (159 non-standard: E-Prime scripts 3 / 4 or LR acquired first). |

**Reading.** Part samples are close to the target plus within-run sampling noise (halves ~0.4, quarters ~0.7 of a
subject's deviation from the mean), with negligible systematic shift for tasks and a modest time-in-run drift for rest.
They are not independent retests: across runs, the subject-specific part of task FC agrees only weakly (LR vs RL
demeaned r 0.04–0.15 at 2–5 min per run; rest 0.52 at 29 min), so the target's own retest noise is far larger than
the variation between halves. Doubling therefore mainly adds noise-perturbed copies of each subject's target (an
augmentation), not new information about the subject; whether that helps a model is G7's question. Side finding:
single-run task FC carries little reproducible subject-specific signal, in line with tasks sitting near the null for
SC → FC in v3:E0.

**Open decision (G6, relational).** 159 relational subjects (15 %) follow a non-standard acquisition pattern and are
excluded from part samples by the fixed rule; they keep their full target. Relaxing the rule (the halves shift is small
for this matched-timing task) needs the user.

## G7 — does training on part samples help a model? (2026-10-27)

**Setup.** E2.1's setup unchanged (matched cohort, 917 Glasser subjects; seeds 0–4; code space and E1 CCA latent fit on
full train targets): part samples of train subjects only, encoded into the same space; val / test on full targets,
scored as E2.1. Routes: CCA latent (T 1) and block FC PCA (T 0.25). Arms: `none`, `aug_*` (full + parts), `token_*`
(flows: sample-type token, sampled as "full"; closed form: Gaussian covariance from full-target residuals),
`parts_*` (parts only), for halves and quarters. Closed form (G7a, CPU): ridge on each subject's mean over its units,
linear-Gaussian with covariance from all unit residuals (`own`) or full residuals (`fullcov`). Flows (G7b, 4.5 GPU-h):
E2.1's residual flow (CCA), linear and nonlinear un-anchored flows (both routes), retrained at E2.1's selected settings,
equal optimizer steps per epoch. Code: `build_parts.py`, `g7_common.py`, `run_g7a.py`, `run_g7b.py`, `analyze_g7.py`
(launchers in `launch/`); tables `tables/g7_summary.md`, `g7_arms.csv`, `g7_paired_vs_none.csv`, `g7_verdicts.csv`,
`g7_regression_check.csv`; figures `figures/fig_g7_deltas.png` (change vs full targets only), `fig_g7_absolute.png`
(absolute test values, seeds as dots, linear baseline line), `fig_g7_conditions.png` (per-condition change in demeaned r); part cache `results/cache/experiments/windowed_fc_gain/`.

**Checks (G7.0).** Rebuilt full targets align with the cached ones (every subject's own best match; median r 0.99999,
worst 0.991; the max gap is G1's demean-vs-plain-concatenation difference). The `none` ridge reproduces E2.1's cached
ridge exactly; the un-anchored flows' `none` arms reproduce E2.1's records (e.g. linear flow CCA demeaned r 0.0442 vs
0.0442, energy 0.0616 vs 0.0615). E2.1's residual-flow record predates the 3000-epoch cap; under current settings its
`none` arm is worse (energy 0.076 vs 0.047), in line with E2.1 §8 (longer training hurts CCA flows); all its arms use the
current settings, so the comparison within G7 is fair.

**Result: augmentation does not help.** Test, paired by seed against each model's `none` arm (support rule as E2.1):

| Model | Effect of part samples |
|---|---|
| Ridge (both routes) | none in practice: Δ demeaned r −0.0002 to −0.0014, Δ average rank ≤ ±0.002 (block PCA halves +0.0008, formally supported) |
| Linear-Gaussian, `own` covariance | means as ridge; spread inflates (CCA ratio 2.47 → 2.56 halves, 2.75 quarters, 2.82 parts-only quarters), supported energy losses |
| Linear-Gaussian, `fullcov` (token analogue) | unchanged (energy identical to `none`) |
| Residual flow (CCA) | **worse in every arm**, more with quarters: demeaned r −0.002 to −0.006, average rank −0.007 to −0.024, energy +0.009 to +0.025 (all supported); the token does not prevent it |
| Linear flow (both routes) | unchanged on point metrics; block PCA energy slightly better (−0.0002 to −0.0007, supported) but still ~3x the Gaussian's 0.0028 |
| Nonlinear flow | CCA: neutral to slightly worse (halves −0.006 average rank); block PCA: demeaned r −0.002 to −0.008 (supported), token-quarters average rank +0.014 (supported) but with a demeaned-r loss |

G7 (gain over both the model's `none` and the linear baseline, no supported loss): formally met only by ridge /
`fullcov` Gaussian on block PCA halves, by +0.0008 average rank, which is negligible. **No flow arm passes.** The linear
baselines remain the best models (CCA: ridge demeaned r 0.0490, Gaussian energy 0.0462; best flow arm 0.0450 / 0.0616).

Per condition (`fig_g7_conditions.png`): ridge and the linear flow move by at most ±0.003 in any condition. The flows'
losses are largest for rest (residual flow, full + 4 quarters −0.012; nonlinear block-PCA flow, 4 quarters only −0.013).
The nonlinear CCA flow gains in some tasks (gambling, motor, social up to +0.003) while losing rest and emotion; these
gains are not consistent across arms.

**Reading.** As gates 2–4 predicted, part samples are the target plus within-run noise: for linear models they average
back to the full target (no gain); for the flows they add noisy targets that mainly widen or blur what is learned, and
the anchored residual flow, which models the residual distribution directly, is hurt most. The sample-type token keeps
calibration only where the model is linear. More samples per subject from the same runs do not provide the independent
subject variability that E3.A was after; that would need genuinely independent acquisitions (other runs / sessions),
which across runs carry little reproducible subject-specific task FC (gate 2 reference).

## Next

E3.A as "more samples per subject from parts of the same scan" is not supported. Options for E3: independent samples
across runs / sessions (rest sessions as repeats, E3 outline), or other uses of the timeseries (e.g. dynamics as a
target in its own right).

Last updated: 2026-10-27
