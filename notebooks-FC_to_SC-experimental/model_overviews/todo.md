# todo — `crossmodal_pca_pls_closed_form_overview.ipynb`

State as of end-of-session 2026-05-26. Pick up here next time.

---

## Where we are right now

Phase 0 + Step 5.1 + full-panel retrofit is done. The recalibrated headline:

> The FC→SC > SC→FC asymmetry is **robust across `demeaned_pearson`, `avg_rank`, `top1_acc`** — every metric points the same direction in every model/framework combination (raw, residual, PLS, BR). `avg_rank` is the cleanest at 1.20–1.25× across all conditions; `demeaned_pearson` at 1.39×; `top1_acc` 1.8–5.3× (largest but noisiest at n=195). Anatomy's apparent demeaned-r dominance from Step 3 (0.167 brain-vol > 0.134 FC) **does not extend to identifiability** — `brain-vol → SC` top1=0.103 vs `FC raw → SC` top1=0.154; `brain-vol → FC` top1=**0.000**. Anatomy predicts edge values well but doesn't fingerprint subjects. FC carries genuine subject-specific signal about SC above anatomy (residual top1 ≈ 14× chance). **Pivot rationale validated. The "anatomy artifact" framing from Step 4 needs to be substantially walked back.**

Three open methodology questions, in priority order below.

---

## 1. Refresh after today's two fixes (5 min, just re-runs)

Two cells were edited at the very end of today's session — outputs are stale:

- **Cell 20 (`# ============= HELPER: FULL 6-METRIC PANEL =============`)** — `demeaned_pearson` formula now matches the project's `compute_demeaned_pearson_r` (cosine of (y − train_mean) per subject), not row-Pearson on demeaned data. Residual numbers won't change; raw-prediction numbers shift slightly:
  - Step 3 `brain-vol → SC`: 0.158 → 0.167
  - Step 3.5 sanity `FC → SC raw`: 0.127 → 0.132
  - Step 3.6 sanity `SC → FC raw`: 0.085 → 0.090
- **Cells 22, 23 (Steps 3.5 / 3.6)** — `PLSRegression(..., max_iter=2000)` (was default 500, which hit the convergence warning). Demeaned numbers won't move meaningfully but `top1_acc` for Step 3.5 will likely shift from 0.072 → ~0.082 to match Step 5.1 A.

**To refresh:** re-run in order:
1. `# ============= HELPER: FULL 6-METRIC PANEL =============`
2. `# ============= STEP 3: BRAIN-VOLUME-ONLY BASELINE (confound check) =============`  (only if you want refreshed `_panel_step3` for 5.2)
3. `# ============= STEP 3.5: Does FC predict SC ABOVE brain anatomy? =============`
4. `# ============= STEP 3.6: SYMMETRIC -- SC -> FC after removing anatomy =============`
5. `# ============= STEP 5.1: CONTROLLED 2x2 AT K_TGT=256 (FULL-PANEL RETROFIT) =============`  (only if you want refreshed `panel_A_*`, `panel_C_*`)
6. `# ============= STEP 5.2: CROSS-STEP FULL-PANEL SUMMARY =============`

Confirm: Step 5.2's `brain-vol → SC` row should print `demeaned_pearson = 0.1670` (not 0.1577) after refresh.

---

## 2. Bootstrap top1_acc / avg_rank / demeaned_pearson CIs  ← **next session's main task**

**Motivation:** the BR top1=0.056 vs PLS top1=0.082 difference is 5 vs 11 hits out of 195 — could be sampling noise. Bootstrap CIs will tell us if the cross-model top1 difference (and the 1.8× vs 5.3× ratio spread) is real or noise. Expected result: top1 CIs are wide (±~0.03), demeaned_pearson and avg_rank CIs are tighter — which would justify dropping top1 as the headline identifiability metric in favor of `avg_rank`.

**Design decisions already made:**
- Duplicate-subject handling: option **(a)** — track original subject IDs, count any same-ID gallery match as a hit. Standard convention.
- **Paired bootstrap** — same resampled indices used for all 4 conditions per iteration, so paired CIs on `PLS top1 − BR top1` and on the directional ratio are tight (correlated noise cancels).
- Bootstrap **all 3 metrics** (`demeaned_pearson`, `avg_rank`, `top1_acc`), not just top1 — gives the relative-noise comparison directly.

**Implementation sketch** (one new cell, ~60 lines, depends on Step 5.1 variables `_pred_A_FC_SC`, `_pred_A_SC_FC`, `_pred_C_FC_SC`, `_pred_C_SC_FC` and the residual truth matrices being in memory):

```python
N = _SC_resid_test.shape[0]   # 195
B = 2000
rng = np.random.default_rng(0)

# Precompute full corr matrices once
cc = {
    "A_FC_SC": compute_corr_matrix(_SC_resid_test, _pred_A_FC_SC),
    "A_SC_FC": compute_corr_matrix(_FC_resid_test, _pred_A_SC_FC),
    "C_FC_SC": compute_corr_matrix(_SC_resid_test, _pred_C_FC_SC),
    "C_SC_FC": compute_corr_matrix(_FC_resid_test, _pred_C_SC_FC),
}
# similar dicts for demeaned-vector cosines and avg_rank-from-cc

def top1_with_dupes(cc_full, ids):
    sub = cc_full[ids][:, ids]
    top1_idx = sub.argmax(axis=1)
    return float(np.mean(ids[top1_idx] == ids))   # same-original-ID counts as hit

for b in range(B):
    ids = rng.integers(0, N, size=N)
    for k, mat in cc.items():
        top1_dist[k][b] = top1_with_dupes(mat, ids)

# 95% CIs on marginals, paired differences (PLS-BR), paired ratios (FC->SC / SC->FC)
```

**Expected runtime:** ~3-5 minutes for 2000 iters.

**Decision flowchart after running:**
- If `avg_rank` CIs are ~3× tighter than `top1_acc` CIs → switch headline identifiability metric to `avg_rank`; quote `top1_acc` only as a directional confirmation.
- If `PLS top1 − BR top1` paired CI includes 0 → "model class doesn't meaningfully change top1; the 1.8× vs 5.3× ratio is sampling noise."
- If it excludes 0 → "BR predictions are systematically less identifiable than PLS despite similar demeaned-r — interesting, worth understanding."

**Caveat to write in cell:** this is bootstrap over a *single test split (seed 0)*, captures sampling variability within this test set only. 10-seed CIs (Step 2 already has them for raw FC→SC) capture across-split variability separately. Both matter.

---

## 3. Snapshot doc update

After the refresh (#1) numbers settle, update:

- `dev-notes/phase0_snapshots.md` — rewrite the "MAJOR Step 4 update" + "CORRECTED Step 5.1" paragraphs to add the identifiability story. Key new sentences:
  - Brain-vol → SC wins demeaned-r but loses top1_acc to FC → SC (0.103 vs 0.154).
  - Brain-vol → FC top1_acc = 0.000.
  - Asymmetry confirmed across all three identifiability metrics; avg_rank cleanest at 1.20–1.25×.
  - The "anatomy artifact" interpretation needs walking back: anatomy is good at edge-value reconstruction, not at subject identification.
- `~/.claude/projects/-Volumes-CrucialX6-Home-projects-Conn2Conn/memory/project_conn2conn.md` — same recalibration in the long paragraph about Phase 0 results.

Do this **after** #2 so the snapshot has bootstrap CIs and updated language all in one pass.

---

## 4. Deferred items (not for next session)

In priority order:

- **10-seed CIs on residual analyses** — Steps 3.5/3.6 are single-seed (seed 0). Re-run across seeds 0..9 to get tight per-direction CIs on the residual demeaned-r. Expected: similar ~±0.005 std as Step 2's raw 10-seed.
- **K_TGT sensitivity sweep on partition framework** — sweep K_TGT ∈ {32, 64, 128, 256} on BR partition to understand why it gave 0.80× (reverse asymmetry) at K=256 vs 1.05× at K=64. Methodology paper material; not on the critical path for the pivot decision.
- **Combined anatomy + FC → SC predictor (BR)** — does adding FC to brain-vol features beat brain-vol alone in unresidualized space? This is the "useful predictor" framing rather than "unique information." Validates the pivot from a more applied angle.
- **Architecture sweep FC → SC + anatomy control on each** — once the closed-form story is locked, the colleague's nonlinear models (VAE, MLP, latent attention) should be re-evaluated FC→SC with anatomy-residual targets to see if any architecture pulls ahead.
- **Generalization** — 4S456Parcels (we have the data), and HCP-Aging / HCP-Development (data prep needed). Tests whether the FC→SC asymmetry is a Glasser/HCP-YA artifact.
- **"Shared with anatomy" biological interpretation** — what does the ~0.09 of cross-modal signal that is *shared* between FC and brain-vol represent? Brain size? Hemispheric asymmetry? Worth a paragraph in the eventual writeup.

---

## Quick reference: cell first-lines (VS Code doesn't show indices)

| order | first line of cell |
|---|---|
| helper | `# ============= HELPER: FULL 6-METRIC PANEL =============` |
| Step 3 | `# ============= STEP 3: BRAIN-VOLUME-ONLY BASELINE (confound check) =============` |
| Step 3.5 | `# ============= STEP 3.5: Does FC predict SC ABOVE brain anatomy? =============` |
| Step 3.6 | `# ============= STEP 3.6: SYMMETRIC -- SC -> FC after removing anatomy =============` |
| Step 5.1 | `# ============= STEP 5.1: CONTROLLED 2x2 AT K_TGT=256 (FULL-PANEL RETROFIT) =============` |
| Step 5.2 | `# ============= STEP 5.2: CROSS-STEP FULL-PANEL SUMMARY =============` |
| (next) Step 5.3 | bootstrap CIs (not written yet — see #2 above) |
