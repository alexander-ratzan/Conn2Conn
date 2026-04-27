import numpy as np
import torch.nn as nn


class TestRetestPrecomputed(nn.Module):
    """
    Test-retest noise-floor baseline. Predicts the FC of one resting-state session
    from the FC of the other within the same subject. No learning, no inference —
    purely data wiring through the standard precomputed-evaluation path.

    Targets  = base.fc_session{target_session}_upper_triangles
    Preds    = base.fc_session{3 - target_session}_upper_triangles

    By rebinding `base.fc_matrices` / `base.fc_upper_triangles` / `base.fc_train_avg`
    to the target session's arrays at construction time, the standard `Evaluator`
    transparently uses the target session as "the FC" — the population mean comes
    from the training-set mean of `target_session`, exactly as requested.

    Args:
        base: HCP_Base instance built with `expose_fc_sessions=True`, `target='FC'`.
        target_session: 1 or 2. Default 1. The other session becomes the prediction.
    """

    is_precomputed = True

    def __init__(self, base, target_session: int = 1, **kwargs):
        super().__init__()

        if base.target != "FC":
            raise ValueError(
                f"TestRetestPrecomputed requires base.target == 'FC'; got {base.target!r}."
            )
        if not getattr(base, "expose_fc_sessions", False):
            raise ValueError(
                "TestRetestPrecomputed requires HCP_Base built with expose_fc_sessions=True."
            )
        if target_session not in (1, 2):
            raise ValueError(f"target_session must be 1 or 2; got {target_session}.")

        pred_session = 3 - target_session

        targets = getattr(base, f"fc_session{target_session}_upper_triangles").astype(np.float32)
        preds   = getattr(base, f"fc_session{pred_session}_upper_triangles").astype(np.float32)
        train_avg = getattr(base, f"fc_session{target_session}_train_avg")
        target_matrices = getattr(base, f"fc_session{target_session}_matrices")

        if preds.shape != targets.shape:
            raise ValueError(
                f"Session shape mismatch: preds {preds.shape} vs targets {targets.shape}."
            )

        # Rebind the canonical FC slots on `base` to the target-session arrays so
        # downstream Evaluator code (which reads dataset.fc_train_avg and
        # dataset.fc_upper_triangles[train_indices]) computes its train_mean / train_std
        # from session {target_session} — i.e., the population mean is the training-set
        # mean of the chosen target session.
        base.fc_matrices = target_matrices
        base.fc_upper_triangles = targets
        base.fc_train_avg = train_avg

        self._preds_all     = preds
        self._targets_all   = targets
        self._split_indices = base.trainvaltest_partition_indices
        self.target_session = target_session
        self.pred_session   = pred_session

        n_subj = self._preds_all.shape[0]
        n_base = base.fc_matrices.shape[0]
        if n_subj != n_base:
            raise ValueError(
                f"TestRetestPrecomputed: session arrays have {n_subj} subjects but "
                f"base.fc_matrices has {n_base}."
            )

        print(
            f"TestRetestPrecomputed: target=session{target_session}, pred=session{pred_session}, "
            f"shape={preds.shape}"
        )

    def predict_split(self, split: str):
        """Return (preds, targets) numpy float32 arrays for the requested split."""
        idx = self._split_indices[split]
        return self._preds_all[idx], self._targets_all[idx]

    def forward(self, x):
        raise RuntimeError(
            "TestRetestPrecomputed.forward() should never be called directly. "
            "Use predict_split() or run through Sim._evaluate_model."
        )
