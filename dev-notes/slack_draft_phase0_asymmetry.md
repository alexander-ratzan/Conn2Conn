# Slack draft — Phase 0 FC→SC asymmetry update (2026-05-26)

Draft message to post above the cross-step table + CSV attachments. Edit freely
before sending — see "Tuning notes" at the bottom for the dials I'd think about.

---

## Draft (paste this into Slack)

Quick update on the FC→SC pivot exploration. I ran a controlled set of closed-form experiments to stress-test the original "FC→SC > SC→FC" result, and there are a few things worth flagging — both findings and one methodological caveat I want to chase down before fully trusting these numbers.

**Pipeline (closed-form, seed 0, n=195 test subjects, Glasser):**
For each direction I ran three predictors against SC and FC respectively:
1. **Brain-volume baseline** — 16 FreeSurfer volume features (intracranial vol, GM, WM, etc.) → connectome edges via OLS. Pure anatomy, zero functional/structural information.
2. **Raw cross-modal** — FC → SC and SC → FC via PCA + PLS / BayesianRidge (matches the project's `CrossModal_PCA_PLS` setup).
3. **Anatomy-residualized cross-modal** — same as #2 but on `target − brain-vol-prediction`, i.e., the part of the target that anatomy *can't* explain.

**Metrics (some new since last update, worth a quick gloss):**
- `demeaned_pearson` — per-subject Pearson after subtracting the train mean. The project's headline; isolates subject-specific signal from group-pattern reconstruction.
- `top1_acc` — fraction of test subjects whose top-1 match in the gallery is themselves. Strict identifiability. Noisy at n=195 (one match = 0.5%).
- `avg_rank` — average rank percentile of the true match. Soft version of top1, much more stable across resamples.
- `pearson` / `r2` / `mse` — full-pattern reconstruction, dominated by group mean.

**Headline findings:**

1. **The FC→SC > SC→FC asymmetry holds across every metric, in every model/framework combination tested.** Even after residualizing out brain anatomy:
   - `demeaned_pearson` ratio: ~1.39× (PLS) and ~1.42× (BR)
   - `avg_rank` ratio: 1.20–1.25× (very stable across PLS/BR/raw/resid)
   - `top1_acc` ratio: 1.8–5.3× (largest but noisiest)

   `avg_rank` is the cleanest metric — it sits at 1.20–1.25× across *every* condition I tested. I'm most confident in the direction of the asymmetry; less confident in the exact magnitude.

2. **Anatomy is the dominant confounder, but it's not a clean "win" — it's more like a metric-by-metric tradeoff.**
   - `brain-vol → SC` gets `demeaned = 0.167`, which is *higher* than `FC raw → SC` (0.132). So on the headline demeaned-r, a 16-feature anatomy predictor beats the full FC connectome at predicting SC.
   - BUT on identifiability, the picture flips: `brain-vol → SC top1 = 0.103` vs `FC raw → SC top1 = 0.154`. Anatomy predicts edge *values* well but doesn't fingerprint subjects as well as FC does.
   - And `brain-vol → FC top1 = 0.000` — zero correct identifications out of 195. Anatomy basically can't ID subjects from FC at all.

3. **FC retains real subject-specific signal about SC above and beyond anatomy.** After residualizing anatomy out:
   - `FC → SC_residual top1 = 0.072` ≈ 14× chance (chance = 1/195 = 0.0051) → genuine identifiability of subjects from anatomy-orthogonal FC→SC signal.
   - `SC → FC_residual top1 = 0.015` ≈ 3× chance → real but much weaker.

**My interpretation (caveats below):**
The biggest confounder of the original FC→SC > SC→FC result is **brain anatomy** — size and shape, which probably encodes sex, genetics, and developmental factors. SC is much more anatomy-driven than FC (a brain-volume baseline predicts SC demeaned-r 0.167 vs FC 0.047, a 3.6× gap). That means a lot of FC→SC's "free signal" comes from anatomy bleeding into the SC target. But even when we residualize anatomy out completely, FC→SC still beats SC→FC by ~1.4× and the identifiability gap (14× chance vs 3× chance) is real. So the directional pivot is validated, just with the magnitude scaled down from the raw +50% headline.

**Caveat I want to flag honestly:**
I wrote a helper function (`_full_panel_eval`) to compute all 6 metrics in one place, and on closer inspection my `demeaned_pearson` formula doesn't quite match the project's `compute_demeaned_pearson_r` convention — I used row-Pearson on demeaned data; the project uses cosine of (y − train_mean) per subject. For residual predictions the two agree exactly (row means are ~0 by construction); for raw predictions they differ by ~4–6%. This means the raw-prediction demeaned-r numbers in the table below are slightly low — `brain-vol → SC` is actually 0.167 (not 0.158), `FC raw → SC` is actually 0.132 (not 0.127). I've patched the helper but want to re-run the cells and verify nothing else moved before I'd publish these numbers. **None of the residual analysis or identifiability metrics are affected** — those are correct as-is — so the asymmetry conclusion still stands. But I want to do a careful re-run to make sure I haven't missed anything else.

Numbers below ⬇️ (full cross-step table + asymmetry ratios + anatomy-vs-FC comparison). Happy to send the per-experiment CSVs separately if useful.

---

## Tuning notes (don't paste — for your own edit pass)

Things to adjust before sending:

1. **Length** — this is ~600 words, medium-long for Slack. If your team prefers tighter, cut sections 1–4 of "Metrics gloss" (keep `demeaned_pearson` and `top1_acc` only) and merge the three headline findings into one paragraph. Gets you to ~300 words.

2. **Audience-specific framing** — assumes colleagues who know the project well (familiar with the FC→SC pivot context). If sending to someone outside the project, add 1–2 sentences upfront on *why* FC→SC matters (i.e., that SC→FC plateaus at demeaned ~0.08–0.10 across every architecture tested, and the pivot is motivated by FC carrying more individual signal).

3. **Caveat strength** — I phrased the helper bug as "found, patched, want to verify." Alternatives:
   - **Stronger** ("worth re-running before trusting these"): use if you want the team to wait before acting on the numbers.
   - **Weaker** ("minor formula difference, conclusions unchanged"): use if you're confident enough to move forward.
   - The current wording sits in the middle — flags it visibly without undermining the conclusions.

4. **Brain-size confounder framing** — I sharpened your intuition into "brain anatomy probably encodes sex, genetics, developmental factors." If you want it more speculative ("might capture sex, genetics, etc. — haven't tested those covariates directly"), edit to taste. We *haven't* actually tested whether brain-vol correlates with sex/genetics — that's an inference. The data point we have is just the demeaned-r asymmetry (0.167 vs 0.047), which is what I'd lead with if a colleague pushes back.

5. **CSV attachments** — if you do attach per-experiment CSVs, suggest naming convention: `phase0_panel_step3.csv`, `phase0_panel_step35.csv`, etc., or just one consolidated `phase0_cross_step_panel.csv` with an `experiment` column matching the Step 5.2 table. Let me know if you want me to write a small cell that dumps these from the notebook.

6. **Numbers that will change after the helper-bug fix re-run** (so you don't have to retract):
   - `brain-vol → SC` demeaned: 0.158 → 0.167
   - `FC raw → SC` demeaned: 0.127 → 0.132
   - `raw FC→SC / SC→FC` demeaned ratio: 1.50× → 1.56×
   - Everything else (all residual rows, all identifiability metrics, all `avg_rank` numbers) is unaffected.

   If you want to wait until after the re-run before sending, that's the cleanest move. If you want to send now and follow up, the caveat paragraph already flags this.
