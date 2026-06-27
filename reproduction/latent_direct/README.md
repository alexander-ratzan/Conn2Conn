# latent_direct — classify on the task-tuned latent (no inverse-PCA round-trip)

**Question:** does the `inverse-PCA → re-PCA` "glow-up" in the imputation→downstream pipeline
attenuate the objective's signal? Test: run the **same objectives as `obj_functions`** but stop at
the **latent** and classify there directly.

```
obj_functions:  src → PCA → [obj] → Ŵ → INVERSE-PCA → connectome → re-PCA → BR → task
latent_direct:  src → PCA → [obj] → Ŵ ─────────────────────────────────────▶ BR → task
```
Only the boxed round-trip is removed, so `Δ = latent_direct − obj_functions` on the matched
(FC2SC) objective isolates exactly what the upscale costs.

## Design (Glasser × 3 seeds: 0,1,2)
- **Arms:** `FC2SC`, `SC2FC` (cross-modal, 5 objectives BR/PLS/obj1a/obj2c/obj2c_raw),
  `FCSC` (observed-both concat latents; plain/obj2c/obj2c_raw; cognition only).
- **Objectives:** reused verbatim from `obj_functions/_obj_estimators.py` internals, returning the
  task-tuned **latent** instead of the connectome (`_latent_estimators.py`).
- **Tasks:** cognition = BayesianRidge **directly on the latent** (no PCA) → lift over bv+demo
  (+ paired-perm p), 3 targets; identity = sibling AUC from demeaned-cosine pairs of the
  residualized latent (cross-modal arms).
- **Discipline:** 5-fold OOF for supervised objectives; frozen splits; float64; leak-free targets.

## Run (Torch)
```bash
cd /scratch/ans9868/Conn2Conn/reproduction/latent_direct && bash submit_latent.sh
# progress: ls sentinels/DONE_lat_*.sentinel | grep -v finalize | wc -l   # /3
#           test -f sentinels/DONE_lat_finalize.sentinel && echo DONE
```
Outputs: `outputs/scorecard_latent.csv` + the **Δ-vs-round-trip** print in `logs/lat_finalize.txt`.

## Expected, from theory (the experiment checks these)
- **identity & reconstruction: Δ≈0 by construction** — PCA components are orthonormal, so
  `inverse_transform` is an isometry on centered latents; cosine similarities (hence sibling AUC)
  are unchanged. The round-trip cannot hurt identity.
- **cognition: the live axis** — the round-trip's re-PCA re-orders the basis and BR re-shrinks in
  it, which *could* attenuate the objective. If `ΔCogCryst_lift > 0`, the upscale was eating signal.
