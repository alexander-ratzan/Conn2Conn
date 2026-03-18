"""Node-feature builders for graph baselines."""

import torch


def build_node_features(sc_dense: torch.Tensor, feature_type: str = "identity") -> torch.Tensor:
    """
    Build node features for batched dense SC adjacency.

    Args:
        sc_dense: [B, N, N] dense SC adjacency.
        feature_type: "identity" or "sc_row".

    Returns:
        Node features [B, N, F].
    """
    if sc_dense.ndim != 3:
        raise ValueError(f"Expected sc_dense [B, N, N], got shape={tuple(sc_dense.shape)}")

    bsz, n, _ = sc_dense.shape

    if feature_type == "identity":
        eye = torch.eye(n, device=sc_dense.device, dtype=sc_dense.dtype)
        return eye.unsqueeze(0).expand(bsz, -1, -1)

    if feature_type == "sc_row":
        return sc_dense

    raise ValueError(f"Unknown feature_type='{feature_type}'. Choose from ['identity', 'sc_row']")
