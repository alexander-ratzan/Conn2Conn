"""Lightweight utilities to bridge upper-triangle vectors to graph-style tensors."""

import math
import torch


def infer_num_nodes_from_upper_triangle_dim(num_edges: int) -> int:
    """Solve n(n-1)/2 = num_edges for n."""
    if num_edges <= 0:
        raise ValueError(f"num_edges must be positive, got {num_edges}.")
    n = int((1 + math.sqrt(1 + 8 * num_edges)) / 2)
    if (n * (n - 1)) // 2 != int(num_edges):
        raise ValueError(
            f"Cannot infer a valid node count from upper-triangle dim={num_edges}."
        )
    return n


def upper_triangle_to_symmetric(x_ut: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Convert batched upper-triangle vectors [B, E] to dense symmetric matrices [B, N, N].
    Diagonal is zeroed.
    """
    if x_ut.ndim != 2:
        raise ValueError(f"Expected [B, E] upper-triangle tensor, got shape={tuple(x_ut.shape)}")

    bsz = x_ut.shape[0]
    device = x_ut.device
    dtype = x_ut.dtype
    tri = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    if tri.shape[1] != x_ut.shape[1]:
        raise ValueError(
            f"Upper-triangle dim mismatch: expected {tri.shape[1]} for N={num_nodes}, got {x_ut.shape[1]}."
        )

    out = torch.zeros((bsz, num_nodes, num_nodes), device=device, dtype=dtype)
    out[:, tri[0], tri[1]] = x_ut
    out[:, tri[1], tri[0]] = x_ut
    return out


def get_label_edge_index(num_nodes: int, device=None) -> torch.Tensor:
    """Return fixed upper-triangle edge indices [2, E] used as prediction targets."""
    return torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)


def build_message_passing_adjacency(sc_dense: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Build normalized adjacency for dense GCN propagation.

    Returns A_norm = D^{-1/2} A_hat D^{-1/2}, where A_hat = A + I when add_self_loops=True.
    """
    if sc_dense.ndim != 3:
        raise ValueError(f"Expected [B, N, N] dense adjacency, got shape={tuple(sc_dense.shape)}")

    A = sc_dense
    if add_self_loops:
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)
        A = A + eye

    deg = A.sum(dim=-1).clamp(min=1e-8)
    deg_inv_sqrt = deg.pow(-0.5)
    A_norm = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A_norm
