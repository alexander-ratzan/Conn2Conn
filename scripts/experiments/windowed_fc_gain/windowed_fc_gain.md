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

## Next

Gate 7 (does training on part samples help a model; closed-form SC → latent in E2.1's setup) waits for the user.

Last updated: 2026-10-27
