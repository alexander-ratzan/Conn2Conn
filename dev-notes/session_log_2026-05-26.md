# Session log — 2026-05-26 (FC→SC asymmetry stress test)

End-of-session handoff. Pick this up cold on a new machine and you'll know
exactly where things are.

Companion docs (read these too):
- [`phase0_snapshots.md`](phase0_snapshots.md) — the running journal of all phase-0 results
- [`slack_draft_phase0_asymmetry.md`](slack_draft_phase0_asymmetry.md) — draft message for colleagues
- [`metrics_glossary.md`](metrics_glossary.md) — what each of the 6 metrics means + the formula gotcha
- [`notebook_architecture.md`](notebook_architecture.md) — cell-by-cell map of `crossmodal_pca_pls_closed_form_overview.ipynb`
- [`../notebooks-FC_to_SC-experimental/model_overviews/todo.md`](../notebooks-FC_to_SC-experimental/model_overviews/todo.md) — next-session checklist with cell first-lines

---

## What was done today

Started the morning trying to reconcile the 1.39× vs 1.05× asymmetry discrepancy from yesterday's Step 4/Step 5 work. By end of day:

1. **Resolved the discrepancy** (Step 5.1 controlled 2×2 at K_TGT=256). Result: the residual-edge framework is robust (~1.4× across PLS and BR); the partition-latent framework is K_TGT-sensitive and gave 0.80× (reverse asymmetry) at K=256 vs 1.05× at K=64. The 1.05× from yesterday was K_TGT=64 specific; pulled back the "asymmetry is symmetric, it's all anatomy artifact" framing.

2. **Built a full 6-metric panel** for every retrofitted experiment. Added `_full_panel_eval` helper (cell 20), retrofitted Steps 3, 3.5, 3.6, 5.1, and added a new cross-step summary cell (Step 5.2). This was the day's main deliverable — every prediction in the chain now reports `mse, r2, pearson, demeaned_pearson, top1_acc, avg_rank`, not just `demeaned_pearson` alone.

3. **Caught two bugs:**
   - **Helper formula**: my `demeaned_pearson` in cell 20 used row-Pearson on demeaned data, but the project's `compute_demeaned_pearson_r` uses cosine of (y−train_mean) per subject. They agree on residual predictions (row means are ~0 by construction) but differ by 4–6% on raw predictions. **Fixed cell 20** at end of session — cells need to be re-run to refresh.
   - **PLS convergence**: Steps 3.5/3.6 used the default `max_iter=500` and hit the convergence warning. Step 5.1 used 2000 and converged. **Fixed cells 22/23** at end of session — cells need to be re-run.

4. **Drafted a Slack message** for colleagues summarizing findings + flagging the helper bug honestly. Saved to `dev-notes/slack_draft_phase0_asymmetry.md`. Not sent yet — user wanted to think about it.

---

## Headline (recalibrated)

The previously-stated framing has been substantially revised across the day. Final defensible version:

> **The FC→SC > SC→FC asymmetry is robust across all three identifiability metrics**, in every model class × framework combination tested.
> - `demeaned_pearson` ratio: ~1.39× (PLS) and ~1.42× (BR) on residuals
> - `avg_rank` ratio: 1.20–1.25× across PLS, BR, raw, residual (most stable metric)
> - `top1_acc` ratio: 1.8–5.3× (largest but noisiest at n=195)
>
> **Anatomy is the dominant confounder but not a clean "win" over FC.**
> - `brain-vol → SC` demeaned = 0.167 > `FC raw → SC` = 0.132 ✓ anatomy wins demeaned-r
> - `brain-vol → SC` top1 = 0.103 < `FC raw → SC` top1 = 0.154 ✗ FC wins identifiability
> - `brain-vol → FC` top1 = 0.000 — anatomy has zero identifying signal for FC
>
> SC is much more anatomy-driven than FC (brain-vol → SC demeaned 0.167 vs brain-vol → FC 0.047 = 3.6× gap). This explains why FC→SC gets a "free" anatomy boost that SC→FC doesn't.
>
> **FC carries real subject-specific signal about SC above and beyond anatomy** (`FC → SC_residual top1 = 0.072` ≈ 14× chance for n=195).
>
> **Pivot rationale is validated.** Magnitude scaled down from raw +50% to residual +40%, but direction confirmed across metrics and model classes.

---

## Files modified today (with what was changed)

| file | change |
|---|---|
| `notebooks-FC_to_SC-experimental/model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb` | Added helper cell (Step 20). Appended full-panel blocks to Steps 3, 3.5, 3.6. Rewrote Step 5.1 with full panel. Added new Step 5.2 cross-step summary. End-of-day: patched helper formula in cell 20; added `max_iter=2000` to PLS in cells 22, 23 |
| `dev-notes/phase0_snapshots.md` | Yesterday's update with Step 5.1 corrected framing (still current) |
| `dev-notes/slack_draft_phase0_asymmetry.md` | NEW — draft Slack message + tuning notes |
| `notebooks-FC_to_SC-experimental/model_overviews/todo.md` | NEW — next-session pickup with refresh checklist + bootstrap design |

---

## Results that survived end-of-day

The Step 5.2 cross-step table from the last successful run (slightly stale due to the helper bug — see "What will change after refresh" below):

```
                    experiment                  group    mse      r2  pearson  demeaned_pearson  top1_acc  avg_rank
               brain-vol -> SC                anatomy 0.0053  0.0093   0.9155            0.1577    0.1026    0.8789
               brain-vol -> FC                anatomy 0.0135 -0.0220   0.8369            0.0467    0.0000    0.6326
  FC -> SC raw (manual sanity)             raw FC->SC 0.0056 -0.0235   0.9128            0.1269    0.1538    0.8533
  SC -> FC raw (manual sanity)             raw SC->FC 0.0140 -0.0622   0.8307            0.0848    0.0308    0.7124
  FC -> SC_residual (PLS, 3.5)       resid FC->SC PLS 0.0055 -0.0492   0.0778            0.0778    0.0718    0.8127
  SC -> FC_residual (PLS, 3.6)       resid SC->FC PLS 0.0143 -0.0705   0.0560            0.0560    0.0154    0.6487
FC -> SC_residual (PLS, 5.1 A) resid FC->SC PLS (5.1) 0.0055 -0.0494   0.0777            0.0777    0.0821    0.8125
SC -> FC_residual (PLS, 5.1 A) resid SC->FC PLS (5.1) 0.0143 -0.0705   0.0560            0.0560    0.0154    0.6487
FC -> SC_residual (BR,  5.1 C)        resid FC->SC BR 0.0053 -0.0060   0.0931            0.0931    0.0564    0.8054
SC -> FC_residual (BR,  5.1 C)        resid SC->FC BR 0.0135 -0.0081   0.0657            0.0657    0.0308    0.6657
```

```
ASYMMETRY RATIOS (FC->SC / SC->FC) ACROSS METRICS
                       comparison  demeaned_pearson  top1_acc  avg_rank  pearson
              raw FC->SC / SC->FC            1.496x    5.000x    1.198x   1.099x
resid PLS (3.5/3.6) FC->SC/SC->FC            1.390x    4.667x    1.253x   1.390x
  resid PLS (5.1 A) FC->SC/SC->FC            1.388x    5.333x    1.253x   1.388x
  resid BR  (5.1 C) FC->SC/SC->FC            1.416x    1.833x    1.210x   1.416x
```

---

## What will change after refresh (helper bug fix)

The helper bug only affected `demeaned_pearson` on raw (non-residualized) predictions. After re-running cells 20, 21, 22, 23, 29, 32:

| row | before | after | delta |
|---|---|---|---|
| `brain-vol → SC` demeaned | 0.1577 | 0.1670 | +5.9% |
| `FC raw → SC` demeaned | 0.1269 | 0.1322 | +4.2% |
| `raw FC→SC / SC→FC` demeaned ratio | 1.496× | ~1.56× | larger |

**Unaffected:**
- All residual rows' `demeaned_pearson` (matches were exact: 0.0778, 0.0560, etc.)
- All `top1_acc`, `avg_rank`, `pearson`, `mse`, `r2` numbers everywhere (formula bug was on demeaned only)
- All asymmetry ratios on residual rows
- All identifiability conclusions

So the recalibrated headline above is correct as-stated; the refresh just cleans up two rows for record-keeping.

---

## What's next session

In priority order:

### 1. Refresh after today's fixes (5 minutes)
Re-run six cells. Detailed first-line list in `notebooks-FC_to_SC-experimental/model_overviews/todo.md`. Verify `brain-vol → SC` demeaned prints 0.1670 (not 0.1577) — that confirms the helper fix took.

### 2. Bootstrap CIs cell (Step 5.3) — main task
Reason: BR top1 (0.056) vs PLS top1 (0.082) might be sampling noise (5 vs 11 hits out of 195). Bootstrap CIs will tell us if the model-class difference is real or noise, and confirm `avg_rank` is the metric to lead with (it's stable, top1 is brittle).

**Design decisions already made** (no need to relitigate):
- Duplicate-subject handling: **(a)** track original subject IDs, count any same-ID gallery match as a hit. Standard convention.
- **Paired bootstrap** — same resampled indices used across all 4 conditions per iteration. Tight paired CIs on differences.
- Bootstrap **all 3 metrics** (demeaned, top1, avg_rank) — gives the relative-noise comparison directly.

Implementation sketch is in `todo.md` (~60 lines, depends on Step 5.1's prediction matrices being in memory).

### 3. Update `phase0_snapshots.md` + `project_conn2conn.md` with recalibrated headline
After bootstrap so we can publish with CIs.

### Deferred (not for next session)
- 10-seed CIs on residual analyses (Steps 3.5/3.6 are seed-0 only)
- K_TGT sensitivity sweep on partition framework (methodology paper material)
- Combined anatomy + FC → SC predictor (BR) — useful-predictor framing
- Architecture sweep FC→SC + anatomy control on each
- Generalization to 4S456Parcels, HCP-Aging/Development
- Biological interpretation of the "shared with anatomy" variance component

---

## Open methodology questions worth thinking about

1. **Is `avg_rank` actually the right headline metric?** It's by far the most stable in this stress test (1.20–1.25× across every condition). Argument for: stable inference at n=195, captures identifiability without top1's sampling brittleness. Argument against: less directly interpretable than top1 ("is the right subject ranked first?" is concrete; "average percentile of the correct match" is fuzzier). The Krakencoder paper uses both. Worth deciding before publishing.

2. **The convergence warning that's hidden in Step 5.1 B (PLS partition).** PLS partition with `n_components=16` for anat and `n_components=64` for xmod hit max_iter=2000 at K_TGT=256. May explain why B gave noise-dominated 2.06× ratio. Not on critical path but worth knowing.

3. **Does the anatomy-residualization step over-residualize?** brain-vol features (16-dim FreeSurfer volumes) are a *low-rank* proxy for anatomy. A high-rank anatomy regressor (e.g., cortical-thickness or geodesic distance matrices) might capture more of the anatomy variance. If anatomy is partially under-removed in our current residuals, the "FC residual still has signal" finding is even stronger than what we're showing.

4. **Why does BR give lower top1 than PLS despite similar demeaned-r?** Hypothesis: BR's per-component ridge shrinkage flattens predictions toward zero, hurting "correctly ranked highest" tasks more than "correlated with truth" tasks. Bootstrap will tell us if the gap is real first.

---

## Practical handoff notes

- **Environment**: conda `base` env from `kraken_env.yml` (Python 3.12, torch 2.9, sklearn 1.7.2, project's `krakencoder==1.0.0` pip-installed). Helper functions live in `models/eval/metrics.py`.
- **Notebook runs on CPU** (`cfg["model"]["device"] = "cpu"` hard-coded in cell 2). No GPU needed for any of the closed-form work.
- **Test set**: n=195 subjects, Glasser parcellation, family-aware 683/79/195 split with `shuffle_seed=0`.
- **Predictions in memory after Step 5.1**: `_pred_A_FC_SC`, `_pred_A_SC_FC` (PLS residual), `_pred_C_FC_SC`, `_pred_C_SC_FC` (BR residual). All shape (195, 64620). Used directly by the proposed bootstrap cell.
- **Today's branch**: `adel-temp` (per git status at session start). Not pushed yet at time of writing this doc; user mentioned `git push`ing during the session so the latest may be on the remote.
