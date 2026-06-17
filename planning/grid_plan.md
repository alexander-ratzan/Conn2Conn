# Grid Plan

This file is intentionally a short pointer.

The active grid plan lives in:

[reproducibility_and_grid_plan.md](reproducibility_and_grid_plan.md)

That document keeps the locked design clear:

- `bv+demo` is an input set, not a model.
- Estimators are PCA -> PLS, BayesianRidge, and KernelRidge (RBF).
- Reconstruction and downstream are separate claim-driven tables.
- No Krakencoder, learnable PLS, MLP, or CovProjector in the reproducibility grid.

