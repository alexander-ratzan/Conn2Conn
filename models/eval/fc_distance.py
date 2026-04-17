"""
Distance helpers for reconstructed FC matrices.

Paper alignment:
- `affine_invariant` is the closest match to the geodesic distance used in
  Venkatesh et al., "Comparing functional connectivity matrices: A
  geometry-aware approach applied to participant identification" (NeuroImage,
  2020), and to the legacy `GeneEx2Conn/models/metrics/distance_FC.py`
  implementation derived from that code release.
- `log_euclidean` is retained as a faster SPD-aware alternative.
- `frobenius` is a non-Riemannian baseline on reconstructed matrices.
"""

import numpy as np

from data.data_utils import tri2square


def ensure_symmetric(mat):
    """Return the symmetric part of a square matrix."""
    arr = np.asarray(mat, dtype=np.float64)
    return 0.5 * (arr + arr.T)


def project_to_spd(mat, eps=1e-6, return_metadata=False):
    """
    Project a symmetric matrix onto the SPD cone by flooring eigenvalues.

    Args:
        mat: square matrix
        eps: minimum eigenvalue after projection
        return_metadata: whether to return projection diagnostics

    Returns:
        spd_mat, metadata (optional)
    """
    sym = ensure_symmetric(mat)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, eps)
    spd = eigvecs @ np.diag(clipped) @ eigvecs.T
    spd = ensure_symmetric(spd)

    if not return_metadata:
        return spd

    metadata = {
        "min_eig_before": float(np.min(eigvals)),
        "min_eig_after": float(np.min(clipped)),
        "n_clipped": int(np.sum(eigvals < eps)),
        "n_negative": int(np.sum(eigvals < 0.0)),
        "clip_mass": float(np.sum(np.maximum(eps - eigvals, 0.0))),
        "negative_mass": float(np.sum(np.maximum(-eigvals, 0.0))),
        "was_projected": bool(np.any(eigvals < eps)),
    }
    return spd, metadata


def spd_matrix_log(mat, eps=1e-6):
    """Matrix logarithm for an SPD matrix using eigendecomposition."""
    spd = project_to_spd(mat, eps=eps)
    eigvals, eigvecs = np.linalg.eigh(spd)
    log_eigvals = np.log(np.maximum(eigvals, eps))
    logm = eigvecs @ np.diag(log_eigvals) @ eigvecs.T
    return ensure_symmetric(logm)


def spd_inverse_sqrt(mat, eps=1e-6):
    """Inverse square root of an SPD matrix."""
    spd = project_to_spd(mat, eps=eps)
    eigvals, eigvecs = np.linalg.eigh(spd)
    inv_sqrt = eigvecs @ np.diag(np.maximum(eigvals, eps) ** -0.5) @ eigvecs.T
    return ensure_symmetric(inv_sqrt)


def log_euclidean_distance(mat_a, mat_b, eps=1e-6):
    """Log-Euclidean geodesic distance between SPD matrices."""
    log_a = spd_matrix_log(mat_a, eps=eps)
    log_b = spd_matrix_log(mat_b, eps=eps)
    return float(np.linalg.norm(log_a - log_b, ord="fro"))


def affine_invariant_geodesic_distance(mat_a, mat_b, eps=1e-6):
    """
    Affine-invariant Riemannian distance between SPD matrices.

    This is the metric most directly aligned with the geometry-aware FC paper
    and with the legacy `distance_FC.geodesic()` implementation used in the
    older GeneEx2Conn codebase.
    """
    a_spd = project_to_spd(mat_a, eps=eps)
    b_spd = project_to_spd(mat_b, eps=eps)
    a_inv_sqrt = spd_inverse_sqrt(a_spd, eps=eps)
    mid = ensure_symmetric(a_inv_sqrt @ b_spd @ a_inv_sqrt)
    eigvals = np.linalg.eigvalsh(mid)
    eigvals = np.maximum(eigvals, eps)
    return float(np.sqrt(np.sum(np.log(eigvals) ** 2)))


def reconstruct_fc_matrices(vectors, numrois, diag_value=1.0):
    """Convert an array of upper-triangle FC vectors into square symmetric matrices."""
    arr = np.asarray(vectors, dtype=np.float64)
    mats = np.stack(
        [tri2square(arr[i], numroi=numrois, diagval=diag_value) for i in range(arr.shape[0])],
        axis=0,
    )
    return np.asarray(mats, dtype=np.float64)


def prepare_fc_matrices(vectors, numrois, diag_value=1.0, eps=1e-6):
    """
    Reconstruct FC matrices and project each one to SPD.

    Returns:
        spd_mats: (n_subjects, n_roi, n_roi)
        metadata: projection summary
    """
    mats = reconstruct_fc_matrices(vectors, numrois=numrois, diag_value=diag_value)
    spd_mats = np.empty_like(mats)
    min_eigs_before = np.empty(mats.shape[0], dtype=np.float64)
    min_eigs_after = np.empty(mats.shape[0], dtype=np.float64)
    clipped_counts = np.empty(mats.shape[0], dtype=np.int64)
    negative_counts = np.empty(mats.shape[0], dtype=np.int64)
    clip_mass = np.empty(mats.shape[0], dtype=np.float64)
    negative_mass = np.empty(mats.shape[0], dtype=np.float64)
    projected_flags = np.zeros(mats.shape[0], dtype=bool)
    n_eigs = mats.shape[1]

    for i in range(mats.shape[0]):
        spd_mats[i], meta = project_to_spd(mats[i], eps=eps, return_metadata=True)
        min_eigs_before[i] = meta["min_eig_before"]
        min_eigs_after[i] = meta["min_eig_after"]
        clipped_counts[i] = meta["n_clipped"]
        negative_counts[i] = meta["n_negative"]
        clip_mass[i] = meta["clip_mass"]
        negative_mass[i] = meta["negative_mass"]
        projected_flags[i] = meta["was_projected"]

    metadata = {
        "n_matrices": int(mats.shape[0]),
        "n_eigs_per_matrix": int(n_eigs),
        "n_projected": int(np.sum(projected_flags)),
        "fraction_projected": float(np.mean(projected_flags)) if mats.shape[0] > 0 else 0.0,
        "min_eig_before": min_eigs_before,
        "min_eig_after": min_eigs_after,
        "n_clipped_eigs": clipped_counts,
        "n_negative_eigs": negative_counts,
        "clip_mass": clip_mass,
        "negative_mass": negative_mass,
        "clipped_fraction_per_matrix": clipped_counts / n_eigs,
        "negative_fraction_per_matrix": negative_counts / n_eigs,
        "mean_clipped_fraction": float(np.mean(clipped_counts / n_eigs)),
        "mean_negative_fraction": float(np.mean(negative_counts / n_eigs)),
        "mean_clip_mass": float(np.mean(clip_mass)),
        "median_clip_mass": float(np.median(clip_mass)),
        "mean_negative_mass": float(np.mean(negative_mass)),
        "median_negative_mass": float(np.median(negative_mass)),
        "projected_flags": projected_flags,
    }
    return spd_mats, metadata


def pairwise_log_euclidean_distance(mats_a, mats_b, eps=1e-6):
    """Efficient pairwise Log-Euclidean distances between two SPD matrix sets."""
    mats_a = np.asarray(mats_a, dtype=np.float64)
    mats_b = np.asarray(mats_b, dtype=np.float64)
    logs_a = np.stack([spd_matrix_log(m, eps=eps) for m in mats_a], axis=0)
    logs_b = np.stack([spd_matrix_log(m, eps=eps) for m in mats_b], axis=0)
    flat_a = logs_a.reshape(logs_a.shape[0], -1)
    flat_b = logs_b.reshape(logs_b.shape[0], -1)
    sq_a = np.sum(flat_a ** 2, axis=1, keepdims=True)
    sq_b = np.sum(flat_b ** 2, axis=1, keepdims=True).T
    cross = flat_a @ flat_b.T
    sq_dist = np.maximum(sq_a + sq_b - 2.0 * cross, 0.0)
    return np.sqrt(sq_dist)


def pairwise_affine_invariant_distance(mats_a, mats_b, eps=1e-6):
    """Pairwise affine-invariant distances. Accurate but expensive for large cohorts."""
    mats_a = np.asarray(mats_a, dtype=np.float64)
    mats_b = np.asarray(mats_b, dtype=np.float64)
    dist = np.empty((mats_a.shape[0], mats_b.shape[0]), dtype=np.float64)
    for i, mat_a in enumerate(mats_a):
        for j, mat_b in enumerate(mats_b):
            dist[i, j] = affine_invariant_geodesic_distance(mat_a, mat_b, eps=eps)
    return dist


def pairwise_frobenius_distance(mats_a, mats_b):
    """Pairwise Frobenius distances between two matrix sets (no SPD assumptions)."""
    mats_a = np.asarray(mats_a, dtype=np.float64)
    mats_b = np.asarray(mats_b, dtype=np.float64)
    flat_a = mats_a.reshape(mats_a.shape[0], -1)
    flat_b = mats_b.reshape(mats_b.shape[0], -1)
    sq_a = np.sum(flat_a ** 2, axis=1, keepdims=True)
    sq_b = np.sum(flat_b ** 2, axis=1, keepdims=True).T
    cross = flat_a @ flat_b.T
    sq_dist = np.maximum(sq_a + sq_b - 2.0 * cross, 0.0)
    return np.sqrt(sq_dist)


def pairwise_fc_distance(
    mats_a,
    mats_b,
    method="log_euclidean",
    eps=1e-6,
):
    """
    Pairwise distance between two sets of FC matrices.

    Args:
        mats_a, mats_b: arrays with shape (n, roi, roi)
        method:
            - 'affine_invariant': paper-faithful geometry-aware FC distance
            - 'log_euclidean': faster SPD-aware approximation/surrogate
            - 'frobenius': non-Riemannian baseline
    """
    if method == "log_euclidean":
        return pairwise_log_euclidean_distance(mats_a, mats_b, eps=eps)
    if method == "affine_invariant":
        return pairwise_affine_invariant_distance(mats_a, mats_b, eps=eps)
    if method == "frobenius":
        return pairwise_frobenius_distance(mats_a, mats_b)
    raise ValueError(f"Unknown SPD distance method: {method}")
