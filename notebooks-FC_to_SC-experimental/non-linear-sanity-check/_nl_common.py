"""Common imports for the non-linear-sanity-check experiments.

Reuses the shared tractography helpers (data loading, reps, linear + nonlinear
predictors) from ../tractography_predict/_tract_setup.py so the linear baselines
are byte-identical to the linear runs.
"""
from pathlib import Path
import sys

for _cand in [Path(__file__).resolve().parent.parent / "tractography_predict",
              Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/tractography_predict"),
              Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/tractography_predict")]:
    if (_cand / "_tract_setup.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError("could not locate tractography_predict/_tract_setup.py")

from _tract_setup import (  # noqa: F401
    load_seed_split_with_r2t, source_train_test, source_blocks, target_train_test,
    pca_pls_predict, block_pca_pls_predict, full_panel_eval,
    kernelridge_predict, kernelridge_blocks_predict,
    hgb_scalar_predict, kr_scalar_predict,
    PCA, BayesianRidge, LinearRegression,
)
