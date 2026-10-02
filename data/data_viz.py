import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from scipy.stats import t as student_t

from data.data_utils import square2tri


__all__ = [
    "distance_matrix_from_centroids",
    "compute_overview_ranges",
    "compute_demeaned_overview_ranges",
    "get_data_overview_matrices",
    "get_demeaned_data_overview_matrices",
    "plot_data_matrix_overview",
    "plot_demeaned_data_matrix_overview",
    "get_edge_feature_table",
    "get_partition_edge_feature_table",
    "plot_connectivity_vs_distance",
    "compute_edgewise_distance_regression",
    "plot_edgewise_regression_summary",
    "plot_single_edge_distance_scatter",
    "compute_edgewise_multifeature_regression",
    "compare_edgewise_models",
    "plot_multifeature_regression_summary",
    "plot_single_edge_distance_scatter_across_partitions",
    "make_data_matrix_gif",
    "load_subject_session_fc",
    "plot_subject_session_overview",
    "plot_subject_session_overview_demeaned",
    "session_diagnostics",
    "condition_upper_triangles",
    "get_condition_connectomes",
    "plot_condition_connectomes",
    "compute_condition_similarity",
    "plot_condition_similarity",
    "compute_subject_condition_correlations",
    "plot_subject_condition_corrmap",
]


_DEFAULT_HCP_DIR = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/"
_SESSION_KEYS = ("S1_LR", "S1_RL", "S2_LR", "S2_RL")
_SESSION_TO_DIR_RUN = {
    "S1_LR": ("LR", 1),
    "S1_RL": ("RL", 1),
    "S2_LR": ("LR", 2),
    "S2_RL": ("RL", 2),
}


_VALID_PARTITIONS = ("train", "val", "test")
_PARTITION_DISPLAY = {"train": "Train", "val": "Validation", "test": "Test"}
# Panels that are correlation-valued — Fisher-z only applies to these.
_CORRELATION_KEYS = ("FC", "SC_r2t_corr")
_PANEL_KEYS = ("SC", "SC_r2t_corr", "distance", "FC")
_PANEL_CMAPS = {
    "SC": "viridis",
    "SC_r2t_corr": "viridis",
    "distance": "viridis_r",
    "FC": "RdBu_r",
}
# Panels whose raw-view range should be symmetric about 0.
_SYMMETRIC_PANELS = ("FC",)


def distance_matrix_from_centroids(centroids):
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)


def _validate_partition(partition):
    if partition not in _VALID_PARTITIONS:
        raise ValueError(f"partition must be one of {_VALID_PARTITIONS}; got {partition!r}")
    return partition


def _partition_indices(base, partition):
    _validate_partition(partition)
    return np.asarray(base.trainvaltest_partition_indices[partition])


def _resolve_partition_global_index(base, partition="val", position=None, subject_id=None):
    part_indices = _partition_indices(base, partition)
    if subject_id is not None:
        matches = np.where(base.metadata_df["subject"].astype(str).values == str(subject_id))[0]
        if len(matches) == 0:
            raise ValueError(f"subject_id {subject_id} not found in metadata_df")
        global_idx = int(matches[0])
        if global_idx not in set(part_indices.tolist()):
            raise ValueError(f"subject_id {subject_id} is not part of the {partition!r} set")
        return global_idx
    if position is None:
        position = 0
    if position < 0 or position >= len(part_indices):
        raise IndexError(f"position must be in [0, {len(part_indices) - 1}] for partition={partition!r}")
    return int(part_indices[position])


def _aggregate_stack(arr, agg="mean"):
    if agg == "mean":
        return np.mean(arr, axis=0)
    if agg == "median":
        return np.median(arr, axis=0)
    raise ValueError(f"Unsupported aggregate={agg!r}; use None, 'mean', or 'median'")


def _fisher_aggregate_stack(arr, agg="mean", clip_eps=1e-6):
    """Aggregate correlation-valued matrices in Fisher-z space, then back-transform.

    Clips to (-1+eps, 1-eps) to keep atanh finite at the diagonal / near-perfect edges.
    """
    clipped = np.clip(arr, -1.0 + clip_eps, 1.0 - clip_eps)
    z = np.arctanh(clipped)
    if agg == "mean":
        z_agg = np.mean(z, axis=0)
    elif agg == "median":
        z_agg = np.median(z, axis=0)
    else:
        raise ValueError(f"Unsupported aggregate={agg!r}; use 'mean' or 'median'")
    return np.tanh(z_agg).astype(arr.dtype, copy=False)


def _distance_stack(base, indices):
    return np.stack(
        [distance_matrix_from_centroids(base.parcel_centroids[i]) for i in indices], axis=0
    )


def compute_overview_ranges(
    base,
    partition="val",
    indices=None,
    distance_quantile=0.99,
    sc_quantile=0.99,
):
    if indices is None:
        indices = _partition_indices(base, partition)
    else:
        indices = np.asarray(indices)

    sc_subset = base.sc_matrices[indices]
    fc_subset = base.fc_matrices[indices]
    dist_subset = _distance_stack(base, indices)

    sc_vmax = float(np.quantile(sc_subset, sc_quantile))
    dist_vmax = float(np.quantile(dist_subset, distance_quantile))
    fc_abs = float(np.quantile(np.abs(fc_subset), 0.995))

    return {
        "SC": {"cmap": "viridis", "vmin": 0.0, "vmax": sc_vmax},
        "SC_r2t_corr": {"cmap": "viridis", "vmin": -1.0, "vmax": 1.0},
        "distance": {"cmap": "viridis_r", "vmin": 0.0, "vmax": dist_vmax},
        "FC": {"cmap": "RdBu_r", "vmin": -fc_abs, "vmax": fc_abs},
    }


def compute_demeaned_overview_ranges(subject_residuals, lo_q=0.005, hi_q=0.995):
    """Per-subject ranges for a demeaned plot.

    - FC (diverging cmap) → symmetric about 0 based on abs-quantile.
    - Other panels → (lo_q, hi_q) quantiles of the residual values.
    """
    ranges = {}
    for key in _PANEL_KEYS:
        resid = subject_residuals[key]
        cmap = _PANEL_CMAPS[key]
        if key in _SYMMETRIC_PANELS or key in _CORRELATION_KEYS:
            vmax = float(np.quantile(np.abs(resid), hi_q))
            vmax = vmax if vmax > 0 else 1e-6
            ranges[key] = {"cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}
        else:
            vmin = float(np.quantile(resid, lo_q))
            vmax = float(np.quantile(resid, hi_q))
            if vmin == vmax:
                vmax = vmin + 1e-6
            ranges[key] = {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    return ranges


def get_data_overview_matrices(
    base,
    partition="val",
    position=None,
    subject_id=None,
    aggregate=None,
    apply_fisher_z=False,
):
    _validate_partition(partition)
    part_label = _PARTITION_DISPLAY[partition]

    if aggregate is None:
        global_idx = _resolve_partition_global_index(
            base, partition=partition, position=position, subject_id=subject_id
        )
        subject_id_out = str(base.metadata_df.iloc[global_idx]["subject"])
        return {
            "SC": base.sc_matrices[global_idx],
            "SC_r2t_corr": base.sc_r2t_corr_matrices[global_idx],
            "distance": distance_matrix_from_centroids(base.parcel_centroids[global_idx]),
            "FC": base.fc_matrices[global_idx],
            "subject_id": subject_id_out,
            "partition": partition,
            "label": f"{part_label} subject {subject_id_out}",
        }

    aggregate = str(aggregate).lower()
    if aggregate not in {"mean", "median"}:
        raise ValueError("aggregate must be one of {None, 'mean', 'median'}")

    part_indices = _partition_indices(base, partition)
    dist_stack = _distance_stack(base, part_indices)

    sc_agg = _aggregate_stack(base.sc_matrices[part_indices], aggregate)
    dist_agg = _aggregate_stack(dist_stack, aggregate)

    corr_agg_fn = _fisher_aggregate_stack if apply_fisher_z else _aggregate_stack
    fc_agg = corr_agg_fn(base.fc_matrices[part_indices], aggregate)
    r2t_agg = corr_agg_fn(base.sc_r2t_corr_matrices[part_indices], aggregate)

    label = f"{part_label} {aggregate}"
    if apply_fisher_z:
        label += " (Fisher-z)"

    return {
        "SC": sc_agg,
        "SC_r2t_corr": r2t_agg,
        "distance": dist_agg,
        "FC": fc_agg,
        "subject_id": None,
        "partition": partition,
        "label": label,
    }


def get_demeaned_data_overview_matrices(
    base,
    partition="val",
    position=None,
    subject_id=None,
    demean_partition="train",
    demean_aggregation="mean",
    demean_apply_fisher_z=False,
):
    """Single-subject matrices minus a reference aggregate from `demean_partition`."""
    subject_data = get_data_overview_matrices(
        base,
        partition=partition,
        position=position,
        subject_id=subject_id,
        aggregate=None,
    )
    reference = get_data_overview_matrices(
        base,
        partition=demean_partition,
        aggregate=demean_aggregation,
        apply_fisher_z=demean_apply_fisher_z,
    )

    residuals = {key: (subject_data[key] - reference[key]).astype(np.float32) for key in _PANEL_KEYS}

    ref_tag = _PARTITION_DISPLAY[demean_partition].lower()
    label_bits = [f"demeaned vs {ref_tag} {demean_aggregation}"]
    if demean_apply_fisher_z:
        label_bits.append("Fisher-z")
    label = f"{subject_data['label']} — {' '.join(label_bits)}"

    return {
        **residuals,
        "subject_id": subject_data["subject_id"],
        "partition": partition,
        "demean_partition": demean_partition,
        "demean_aggregation": demean_aggregation,
        "demean_apply_fisher_z": demean_apply_fisher_z,
        "reference": reference,
        "label": label,
    }


def _add_small_colorbar(fig, ax, im, ticks=None):
    cax = inset_axes(
        ax,
        width="3.5%",
        height="34%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.06, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(im, cax=cax)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=8, length=2)
    return cb


def _format_subject_header(base, data, header_metadata=None):
    partition = data.get("partition", "val")
    part_label = _PARTITION_DISPLAY.get(partition, partition.capitalize())

    if data.get("subject_id") is None:
        return f"{base.parcellation} | {data['label']}"

    header_metadata = None if header_metadata is None else str(header_metadata).lower()
    if header_metadata not in {None, "age_sex", "age_sex_ethnicity"}:
        raise ValueError("header_metadata must be one of {None, 'age_sex', 'age_sex_ethnicity'}")

    subject_id = str(data["subject_id"])
    # Prefer the full label (covers demean suffix) over a bare "partition subject id".
    base_title = data.get("label") or f"{part_label} subject {subject_id}"
    parts = [f"{base.parcellation} | {base_title}"]
    if header_metadata is None:
        return parts[0]

    meta = base.metadata_df
    match = meta[meta["subject"].astype(str) == subject_id]
    if match.empty:
        return parts[0]
    row = match.iloc[0]

    age = row.get("age")
    sex = row.get("sex")
    race_eth = row.get("race_eth")

    meta_parts = []
    if age is not None and not (isinstance(age, float) and np.isnan(age)):
        meta_parts.append(f"age={age}")
    if sex is not None and not (isinstance(sex, float) and np.isnan(sex)):
        meta_parts.append(f"sex={sex}")
    if header_metadata == "age_sex_ethnicity" and race_eth is not None and not (isinstance(race_eth, float) and np.isnan(race_eth)):
        meta_parts.append(f"ethnicity={race_eth}")

    return parts[0] if not meta_parts else parts[0] + " | " + ", ".join(meta_parts)


def _render_panels(fig, axes, data, ranges):
    panel_titles = {
        "SC": "SC",
        "SC_r2t_corr": "SC_r2t corr",
        "distance": "Distance",
        "FC": "FC",
    }
    for ax, key in zip(axes, _PANEL_KEYS):
        mat = data[key]
        spec = ranges[key]
        im = ax.imshow(
            mat,
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(panel_titles[key], fontsize=12, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        if key == "SC":
            ax.set_xlabel("region", fontsize=10)
            ax.set_ylabel("region", fontsize=10)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

        if spec["vmin"] < 0 < spec["vmax"]:
            ticks = [spec["vmin"], 0.0, spec["vmax"]]
        else:
            ticks = [spec["vmin"], spec["vmax"]]
        _add_small_colorbar(fig, ax, im, ticks=ticks)


def plot_data_matrix_overview(
    base,
    partition="val",
    position=None,
    subject_id=None,
    aggregate=None,
    apply_fisher_z=False,
    ranges=None,
    figsize=(14, 3.8),
    dpi=180,
    show=True,
    suptitle=None,
    header_metadata=None,
):
    data = get_data_overview_matrices(
        base,
        partition=partition,
        position=position,
        subject_id=subject_id,
        aggregate=aggregate,
        apply_fisher_z=apply_fisher_z,
    )
    ranges = compute_overview_ranges(base, partition=partition) if ranges is None else ranges

    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.34, top=0.86, bottom=0.12)
    _render_panels(fig, axes, data, ranges)

    final_title = suptitle or _format_subject_header(base, data, header_metadata=header_metadata)
    fig.suptitle(final_title, fontsize=14, y=0.92)

    if show:
        plt.show()
    return fig, axes, data


def plot_demeaned_data_matrix_overview(
    base,
    partition="val",
    position=None,
    subject_id=None,
    demean_partition="train",
    demean_aggregation="mean",
    demean_apply_fisher_z=False,
    ranges=None,
    figsize=(14, 3.8),
    dpi=180,
    show=True,
    suptitle=None,
    header_metadata=None,
):
    data = get_demeaned_data_overview_matrices(
        base,
        partition=partition,
        position=position,
        subject_id=subject_id,
        demean_partition=demean_partition,
        demean_aggregation=demean_aggregation,
        demean_apply_fisher_z=demean_apply_fisher_z,
    )
    ranges = compute_demeaned_overview_ranges(data) if ranges is None else ranges

    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.34, top=0.86, bottom=0.12)
    _render_panels(fig, axes, data, ranges)

    final_title = suptitle or _format_subject_header(base, data, header_metadata=header_metadata)
    fig.suptitle(final_title, fontsize=14, y=0.92)

    if show:
        plt.show()
    return fig, axes, data


def _pairwise_mean_volume_matrix(parcel_volume):
    vol = np.asarray(parcel_volume, dtype=np.float32)
    return (0.5 * (vol[:, None] + vol[None, :])).astype(np.float32)


def _pairwise_volume_mismatch_matrix(parcel_volume):
    vol = np.asarray(parcel_volume, dtype=np.float32)
    return np.abs(vol[:, None] - vol[None, :]).astype(np.float32)


def _atlas_roi_df(parcellation, hemi):
    roi_df = pd.read_csv(
        f"/scratch/asr655/neuroinformatics/Conn2Conn/data/atlas_info/{parcellation}_dseg_reformatted.csv"
    ).copy()
    roi_df["roi_idx"] = np.arange(len(roi_df), dtype=np.int64)
    roi_df["hemisphere"] = roi_df["hemisphere"].astype(str)
    if hemi == "left":
        return roi_df[roi_df["hemisphere"].str.contains("L")].reset_index(drop=True)
    if hemi == "right":
        return roi_df[roi_df["hemisphere"].str.contains("R")].reset_index(drop=True)
    return roi_df.reset_index(drop=True)


def get_edge_feature_table(
    base,
    partition="val",
    position=None,
    subject_id=None,
    aggregate=None,
    apply_fisher_z=False,
    demean_partition=None,
    demean_aggregation="mean",
    demean_apply_fisher_z=False,
):
    """Return an edge-level DataFrame for FC/SC analyses.

    The table is built from upper-triangle entries only and includes:
    `distance`, `SC`, `FC`, `SC_r2t_corr`, plus pairwise parcel-volume summaries.
    If `demean_partition` is provided, SC / FC / SC_r2t_corr / distance are
    replaced with subject-minus-reference residuals using the same semantics as
    `plot_demeaned_data_matrix_overview`.
    """
    data = get_data_overview_matrices(
        base,
        partition=partition,
        position=position,
        subject_id=subject_id,
        aggregate=aggregate,
        apply_fisher_z=apply_fisher_z,
    )

    if aggregate is None:
        global_idx = _resolve_partition_global_index(
            base, partition=partition, position=position, subject_id=subject_id
        )
        parcel_volume = np.asarray(base.parcel_volume[global_idx], dtype=np.float32)
        subject_id_out = str(base.metadata_df.iloc[global_idx]["subject"])
    else:
        part_indices = _partition_indices(base, partition)
        parcel_volume = _aggregate_stack(base.parcel_volume[part_indices], str(aggregate).lower())
        subject_id_out = None

    if demean_partition is not None:
        reference = get_data_overview_matrices(
            base,
            partition=demean_partition,
            aggregate=demean_aggregation,
            apply_fisher_z=demean_apply_fisher_z,
        )
        ref_part_indices = _partition_indices(base, demean_partition)
        ref_parcel_volume = _aggregate_stack(
            base.parcel_volume[ref_part_indices], str(demean_aggregation).lower()
        ).astype(np.float32)

        for key in _PANEL_KEYS:
            data[key] = (np.asarray(data[key], dtype=np.float32) - np.asarray(reference[key], dtype=np.float32)).astype(np.float32)
        parcel_volume = (parcel_volume - ref_parcel_volume).astype(np.float32)
        label_bits = [f"demeaned vs {_PARTITION_DISPLAY[demean_partition].lower()} {demean_aggregation}"]
        if demean_apply_fisher_z:
            label_bits.append("Fisher-z")
        data["label"] = f"{data['label']} — {' '.join(label_bits)}"

    mean_volume = _pairwise_mean_volume_matrix(parcel_volume)
    sum_volume = (parcel_volume[:, None] + parcel_volume[None, :]).astype(np.float32)

    edge_df = pd.DataFrame(
        {
            "distance": square2tri(data["distance"]),
            "SC": square2tri(data["SC"]),
            "FC": square2tri(data["FC"]),
            "SC_r2t_corr": square2tri(data["SC_r2t_corr"]),
            "mean_volume": square2tri(mean_volume),
            "sum_volume": square2tri(sum_volume),
        }
    )
    edge_df["partition"] = partition
    edge_df["subject_id"] = subject_id_out
    edge_df["label"] = data["label"]
    if demean_partition is not None:
        edge_df["demeaned_vs"] = f"{demean_partition}:{demean_aggregation}"
        edge_df["demean_apply_fisher_z"] = bool(demean_apply_fisher_z)
    return edge_df


def get_partition_edge_feature_table(
    base,
    partition="val",
    apply_fisher_z=False,
    demean_partition=None,
    demean_aggregation="mean",
    demean_apply_fisher_z=False,
):
    """Return a pooled edge-level DataFrame across every subject in `partition`."""
    part_indices = _partition_indices(base, partition)
    subject_ids = base.metadata_df.iloc[part_indices]["subject"].astype(str).tolist()

    frames = []
    for subject_id in subject_ids:
        subject_df = get_edge_feature_table(
            base,
            partition=partition,
            subject_id=subject_id,
            aggregate=None,
            apply_fisher_z=apply_fisher_z,
            demean_partition=demean_partition,
            demean_aggregation=demean_aggregation,
            demean_apply_fisher_z=demean_apply_fisher_z,
        ).copy()
        frames.append(subject_df)

    pooled_df = pd.concat(frames, ignore_index=True)
    pooled_label = f"{_PARTITION_DISPLAY[partition]} pooled subjects"
    if demean_partition is not None:
        pooled_label += f" — demeaned vs {_PARTITION_DISPLAY[demean_partition].lower()} {demean_aggregation}"
        if demean_apply_fisher_z:
            pooled_label += " Fisher-z"
    pooled_df["label"] = pooled_label
    return pooled_df


def compute_edgewise_distance_regression(
    base,
    partition="val",
    y_key="FC",
    fdr_alpha=0.05,
):
    """Fit a subject-level regression for each edge: y_edge ~ distance_edge."""
    if y_key not in {"FC", "SC", "SC_r2t_corr"}:
        raise ValueError("y_key must be one of {'FC', 'SC', 'SC_r2t_corr'}")

    part_indices = _partition_indices(base, partition)
    if y_key == "FC":
        y = np.asarray(base.fc_upper_triangles[part_indices], dtype=np.float32)
    elif y_key == "SC":
        y = np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float32)
    else:
        y = np.asarray(base.sc_r2t_corr_upper_triangles[part_indices], dtype=np.float32)

    distance_stack = _distance_stack(base, part_indices)
    x = np.stack([square2tri(mat) for mat in distance_stack], axis=0).astype(np.float32)

    x_mean = x.mean(axis=0, dtype=np.float64)
    y_mean = y.mean(axis=0, dtype=np.float64)
    x_centered = x - x_mean
    y_centered = y - y_mean

    cov = np.sum(x_centered * y_centered, axis=0, dtype=np.float64)
    var_x = np.sum(x_centered * x_centered, axis=0, dtype=np.float64)
    var_y = np.sum(y_centered * y_centered, axis=0, dtype=np.float64)

    slope = np.divide(cov, var_x, out=np.zeros_like(cov), where=var_x > 0)
    intercept = y_mean - slope * x_mean
    corr = np.divide(
        cov,
        np.sqrt(var_x * var_y),
        out=np.full_like(cov, np.nan),
        where=(var_x > 0) & (var_y > 0),
    )
    r2 = corr ** 2
    n_subjects = int(len(part_indices))
    dof = max(n_subjects - 2, 1)
    denom = np.clip(1.0 - (corr ** 2), 1e-12, None)
    t_stat = corr * np.sqrt(dof / denom)
    p_value = 2.0 * student_t.sf(np.abs(t_stat), df=dof)
    y_hat = intercept[None, :] + slope[None, :] * x
    resid = y - y_hat
    sse = np.sum(resid ** 2, axis=0, dtype=np.float64)

    finite_mask = np.isfinite(p_value)
    q_value = np.full_like(p_value, np.nan, dtype=np.float64)
    if finite_mask.any():
        p_finite = p_value[finite_mask]
        order = np.argsort(p_finite)
        ranked = p_finite[order]
        m = len(ranked)
        q_ranked = ranked * m / np.arange(1, m + 1)
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_ranked = np.clip(q_ranked, 0.0, 1.0)
        q_tmp = np.empty_like(p_finite)
        q_tmp[order] = q_ranked
        q_value[finite_mask] = q_tmp

    roi_df = _atlas_roi_df(base.parcellation, base.hemi)
    tri_i, tri_j = np.triu_indices(len(roi_df), k=1)
    roi_i = roi_df.iloc[tri_i].reset_index(drop=True)
    roi_j = roi_df.iloc[tri_j].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "edge_idx": np.arange(len(tri_i), dtype=np.int64),
            "roi_i": tri_i,
            "roi_j": tri_j,
            "subject_count": int(len(part_indices)),
            "partition": partition,
            "y_key": y_key,
            "slope": slope.astype(np.float32),
            "intercept": intercept.astype(np.float32),
            "corr": corr.astype(np.float32),
            "r2": r2.astype(np.float32),
            "t_stat": t_stat.astype(np.float32),
            "p_value": p_value.astype(np.float64),
            "q_value": q_value.astype(np.float64),
            "distance_beta": slope.astype(np.float32),
            "distance_delta_r2": r2.astype(np.float32),
            "distance_mean": x_mean.astype(np.float32),
            "distance_std": x.std(axis=0, dtype=np.float64).astype(np.float32),
            f"{y_key.lower()}_mean": y_mean.astype(np.float32),
            f"{y_key.lower()}_std": y.std(axis=0, dtype=np.float64).astype(np.float32),
            "label_i": roi_i["label"].to_numpy(),
            "label_j": roi_j["label"].to_numpy(),
            "network_label_i": roi_i["network_label"].to_numpy(),
            "network_label_j": roi_j["network_label"].to_numpy(),
            "network_label_17network_i": roi_i["network_label_17network"].to_numpy(),
            "network_label_17network_j": roi_j["network_label_17network"].to_numpy(),
            "atlas_name_i": roi_i["atlas_name"].to_numpy(),
            "atlas_name_j": roi_j["atlas_name"].to_numpy(),
            "atlas_id_i": roi_i["id"].to_numpy(),
            "atlas_id_j": roi_j["id"].to_numpy(),
            "hemisphere_i": roi_i["hemisphere"].to_numpy(),
            "hemisphere_j": roi_j["hemisphere"].to_numpy(),
            "structure_i": roi_i["structure"].to_numpy(),
            "structure_j": roi_j["structure"].to_numpy(),
            "mni_x_i": roi_i["mni_x"].to_numpy(dtype=np.float32),
            "mni_y_i": roi_i["mni_y"].to_numpy(dtype=np.float32),
            "mni_z_i": roi_i["mni_z"].to_numpy(dtype=np.float32),
            "mni_x_j": roi_j["mni_x"].to_numpy(dtype=np.float32),
            "mni_y_j": roi_j["mni_y"].to_numpy(dtype=np.float32),
            "mni_z_j": roi_j["mni_z"].to_numpy(dtype=np.float32),
        }
    )
    out["same_hemisphere"] = out["hemisphere_i"] == out["hemisphere_j"]
    out["same_network"] = out["network_label_i"] == out["network_label_j"]
    out["same_network_17"] = out["network_label_17network_i"] == out["network_label_17network_j"]
    out["fdr_alpha"] = float(fdr_alpha)
    out["significant_fdr"] = out["q_value"] <= float(fdr_alpha)
    out["sse"] = sse.astype(np.float64)
    out["n_subjects"] = int(n_subjects)
    out["model_df"] = 2
    out["model_name"] = "distance"
    out["label"] = f"{_PARTITION_DISPLAY[partition]} edgewise {y_key} ~ distance"
    return out


def _benjamini_hochberg_qvalues(p_value):
    finite_mask = np.isfinite(p_value)
    q_value = np.full_like(p_value, np.nan, dtype=np.float64)
    if finite_mask.any():
        p_finite = p_value[finite_mask]
        order = np.argsort(p_finite)
        ranked = p_finite[order]
        m = len(ranked)
        q_ranked = ranked * m / np.arange(1, m + 1)
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_ranked = np.clip(q_ranked, 0.0, 1.0)
        q_tmp = np.empty_like(p_finite)
        q_tmp[order] = q_ranked
        q_value[finite_mask] = q_tmp
    return q_value


def compute_edgewise_multifeature_regression(
    base,
    partition="val",
    y_key="FC",
    model_name="distance_mean_volume_mismatch",
    fdr_alpha=0.05,
):
    """Fit an edgewise multivariate regression across subjects for each edge."""
    if y_key not in {"FC", "SC", "SC_r2t_corr"}:
        raise ValueError("y_key must be one of {'FC', 'SC', 'SC_r2t_corr'}")

    part_indices = _partition_indices(base, partition)
    n_subjects = int(len(part_indices))
    if y_key == "FC":
        y = np.asarray(base.fc_upper_triangles[part_indices], dtype=np.float64)
    elif y_key == "SC":
        y = np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float64)
    else:
        y = np.asarray(base.sc_r2t_corr_upper_triangles[part_indices], dtype=np.float64)

    distance_stack = _distance_stack(base, part_indices)
    distance_ut = np.stack([square2tri(mat) for mat in distance_stack], axis=0).astype(np.float64)

    mean_volume_ut = []
    mismatch_ut = []
    for idx in part_indices:
        vol = np.asarray(base.parcel_volume[idx], dtype=np.float64)
        mean_volume_ut.append(square2tri(_pairwise_mean_volume_matrix(vol)))
        mismatch_ut.append(square2tri(_pairwise_volume_mismatch_matrix(vol)))
    mean_volume_ut = np.stack(mean_volume_ut, axis=0).astype(np.float64)
    mismatch_ut = np.stack(mismatch_ut, axis=0).astype(np.float64)

    eps = 1e-8
    if model_name == "distance_mean_volume_mismatch":
        feature_arrays = {
            "distance": distance_ut,
            "mean_volume": mean_volume_ut,
            "volume_mismatch": mismatch_ut,
        }
    elif model_name == "distance_log_product_volume_log_volume_mismatch":
        feature_arrays = {
            "distance": distance_ut,
            "log_product_volume": np.log(np.clip(4.0 * (mean_volume_ut ** 2) - (mismatch_ut ** 2), eps, None)),
            "log_volume_mismatch": np.log1p(np.clip(mismatch_ut, 0.0, None)),
        }
    elif model_name == "distance_log_product_volume_log_volume_mismatch_sc":
        feature_arrays = {
            "distance": distance_ut,
            "log_product_volume": np.log(np.clip(4.0 * (mean_volume_ut ** 2) - (mismatch_ut ** 2), eps, None)),
            "log_volume_mismatch": np.log1p(np.clip(mismatch_ut, 0.0, None)),
            "SC": np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float64),
        }
    elif model_name == "distance_log_product_volume_log_volume_mismatch_sc_subject_z":
        sc = np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float64)
        sc_mean = sc.mean(axis=1, keepdims=True)
        sc_std = sc.std(axis=1, keepdims=True)
        sc_subject_z = np.divide(sc - sc_mean, sc_std, out=np.zeros_like(sc), where=sc_std > 0)
        feature_arrays = {
            "distance": distance_ut,
            "log_product_volume": np.log(np.clip(4.0 * (mean_volume_ut ** 2) - (mismatch_ut ** 2), eps, None)),
            "log_volume_mismatch": np.log1p(np.clip(mismatch_ut, 0.0, None)),
            "SC_subject_z": sc_subject_z,
        }
    elif model_name == "sc":
        feature_arrays = {
            "SC": np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float64),
        }
    elif model_name == "distance_sc":
        feature_arrays = {
            "distance": distance_ut,
            "SC": np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float64),
        }
    else:
        raise ValueError(
            "model_name must be one of "
            "{'distance_mean_volume_mismatch', "
            "'distance_log_product_volume_log_volume_mismatch', "
            "'distance_log_product_volume_log_volume_mismatch_sc', "
            "'distance_log_product_volume_log_volume_mismatch_sc_subject_z', "
            "'sc', 'distance_sc'}"
        )

    feature_names = list(feature_arrays.keys())
    p = len(feature_names)
    n_edges = distance_ut.shape[1]

    X = np.empty((n_subjects, n_edges, p + 1), dtype=np.float64)
    X[:, :, 0] = 1.0
    for j, name in enumerate(feature_names, start=1):
        X[:, :, j] = feature_arrays[name]

    XtX = np.einsum("sep,seq->epq", X, X)
    XtY = np.einsum("sep,se->ep", X, y)
    XtX_inv = np.linalg.pinv(XtX)
    beta = np.einsum("epq,eq->ep", XtX_inv, XtY)
    y_hat = np.einsum("sep,ep->se", X, beta)
    resid = y - y_hat

    dof = max(n_subjects - (p + 1), 1)
    sse = np.sum(resid ** 2, axis=0, dtype=np.float64)
    sigma2 = sse / dof
    cov_beta = XtX_inv * sigma2[:, None, None]
    se_beta = np.sqrt(np.clip(np.diagonal(cov_beta, axis1=1, axis2=2), 0.0, None))
    t_stat = np.divide(beta, se_beta, out=np.full_like(beta, np.nan), where=se_beta > 0)
    p_value = 2.0 * student_t.sf(np.abs(t_stat), df=dof)

    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2, axis=0, dtype=np.float64)
    r2 = 1.0 - np.divide(sse, ss_tot, out=np.full_like(sse, np.nan), where=ss_tot > 0)
    delta_r2 = {}
    for j, name in enumerate(feature_names, start=1):
        keep_idx = [k for k in range(p + 1) if k != j]
        X_red = X[:, :, keep_idx]
        XtX_red = np.einsum("sep,seq->epq", X_red, X_red)
        XtY_red = np.einsum("sep,se->ep", X_red, y)
        XtX_red_inv = np.linalg.pinv(XtX_red)
        beta_red = np.einsum("epq,eq->ep", XtX_red_inv, XtY_red)
        y_hat_red = np.einsum("sep,ep->se", X_red, beta_red)
        resid_red = y - y_hat_red
        sse_red = np.sum(resid_red ** 2, axis=0, dtype=np.float64)
        r2_red = 1.0 - np.divide(sse_red, ss_tot, out=np.full_like(sse_red, np.nan), where=ss_tot > 0)
        delta_r2[name] = (r2 - r2_red).astype(np.float32)

    roi_df = _atlas_roi_df(base.parcellation, base.hemi)
    tri_i, tri_j = np.triu_indices(len(roi_df), k=1)
    roi_i = roi_df.iloc[tri_i].reset_index(drop=True)
    roi_j = roi_df.iloc[tri_j].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "edge_idx": np.arange(n_edges, dtype=np.int64),
            "roi_i": tri_i,
            "roi_j": tri_j,
            "subject_count": n_subjects,
            "partition": partition,
            "y_key": y_key,
            "model_name": model_name,
            "intercept": beta[:, 0].astype(np.float32),
            "r2": r2.astype(np.float32),
            f"{y_key.lower()}_mean": y.mean(axis=0).astype(np.float32),
            f"{y_key.lower()}_std": y.std(axis=0).astype(np.float64).astype(np.float32),
            "distance_mean": distance_ut.mean(axis=0).astype(np.float32),
            "distance_std": distance_ut.std(axis=0).astype(np.float64).astype(np.float32),
            "mean_volume_mean": mean_volume_ut.mean(axis=0).astype(np.float32),
            "mean_volume_std": mean_volume_ut.std(axis=0).astype(np.float64).astype(np.float32),
            "volume_mismatch_mean": mismatch_ut.mean(axis=0).astype(np.float32),
            "volume_mismatch_std": mismatch_ut.std(axis=0).astype(np.float64).astype(np.float32),
            "label_i": roi_i["label"].to_numpy(),
            "label_j": roi_j["label"].to_numpy(),
            "network_label_i": roi_i["network_label"].to_numpy(),
            "network_label_j": roi_j["network_label"].to_numpy(),
            "network_label_17network_i": roi_i["network_label_17network"].to_numpy(),
            "network_label_17network_j": roi_j["network_label_17network"].to_numpy(),
            "atlas_name_i": roi_i["atlas_name"].to_numpy(),
            "atlas_name_j": roi_j["atlas_name"].to_numpy(),
            "atlas_id_i": roi_i["id"].to_numpy(),
            "atlas_id_j": roi_j["id"].to_numpy(),
            "hemisphere_i": roi_i["hemisphere"].to_numpy(),
            "hemisphere_j": roi_j["hemisphere"].to_numpy(),
            "structure_i": roi_i["structure"].to_numpy(),
            "structure_j": roi_j["structure"].to_numpy(),
        }
    )

    for j, name in enumerate(feature_names, start=1):
        out[f"{name}_beta"] = beta[:, j].astype(np.float32)
        out[f"{name}_se"] = se_beta[:, j].astype(np.float32)
        out[f"{name}_t_stat"] = t_stat[:, j].astype(np.float32)
        out[f"{name}_p_value"] = p_value[:, j].astype(np.float64)
        out[f"{name}_delta_r2"] = delta_r2[name]
        q_val = _benjamini_hochberg_qvalues(p_value[:, j])
        out[f"{name}_q_value"] = q_val.astype(np.float64)
        out[f"{name}_significant_fdr"] = q_val <= float(fdr_alpha)

    out["same_hemisphere"] = out["hemisphere_i"] == out["hemisphere_j"]
    out["same_network"] = out["network_label_i"] == out["network_label_j"]
    out["same_network_17"] = out["network_label_17network_i"] == out["network_label_17network_j"]
    out["fdr_alpha"] = float(fdr_alpha)
    out["sse"] = sse.astype(np.float64)
    out["n_subjects"] = int(n_subjects)
    out["model_df"] = int(p + 1)
    out["label"] = f"{_PARTITION_DISPLAY[partition]} edgewise {y_key} ~ " + " + ".join(feature_names)
    return out


def compare_edgewise_models(
    reduced_df,
    full_df,
    added_predictor_names,
    fdr_alpha=0.05,
):
    """Nested per-edge F-test comparing a reduced and full regression table."""
    key_cols = [
        "edge_idx",
        "label_i",
        "label_j",
        "network_label_i",
        "network_label_j",
        "network_label_17network_i",
        "network_label_17network_j",
        "hemisphere_i",
        "hemisphere_j",
        "same_hemisphere",
        "same_network",
        "same_network_17",
        "partition",
        "y_key",
    ]
    missing = [c for c in key_cols + ["sse", "r2", "n_subjects", "model_df"] if c not in reduced_df.columns or c not in full_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for nested comparison: {sorted(set(missing))}")

    reduced = reduced_df[key_cols + ["sse", "r2", "n_subjects", "model_df"]].copy()
    full = full_df[key_cols + ["sse", "r2", "n_subjects", "model_df"]].copy()
    merged = reduced.merge(full, on=key_cols, suffixes=("_reduced", "_full"))

    n = merged["n_subjects_full"].to_numpy(dtype=np.int64, copy=False)
    df_reduced = merged["model_df_reduced"].to_numpy(dtype=np.int64, copy=False)
    df_full = merged["model_df_full"].to_numpy(dtype=np.int64, copy=False)
    sse_reduced = merged["sse_reduced"].to_numpy(dtype=np.float64, copy=False)
    sse_full = merged["sse_full"].to_numpy(dtype=np.float64, copy=False)
    q = np.maximum(df_full - df_reduced, 1)
    df2 = np.maximum(n - df_full, 1)
    num = np.maximum(sse_reduced - sse_full, 0.0) / q
    den = np.maximum(sse_full, 1e-12) / df2
    f_stat = num / den
    p_value = student_t.sf(np.sqrt(np.maximum(f_stat, 0.0)), df=df2) * 2.0
    # Use F distribution via beta relation avoided; scipy.stats.f not imported. Approx via t only exact for q=1.
    # For q>1, fall back to scipy's F distribution would be better, but q here is small and fixed by nested block.
    # Import lazily to keep local surface simple.
    from scipy.stats import f as fisher_f
    p_value = fisher_f.sf(np.maximum(f_stat, 0.0), q, df2)
    q_value = _benjamini_hochberg_qvalues(p_value)

    out = merged[key_cols].copy()
    out["comparison"] = f"{full_df['model_name'].iloc[0]} vs {reduced_df['model_name'].iloc[0]}"
    out["added_predictors"] = ", ".join(added_predictor_names)
    out["r2_reduced"] = merged["r2_reduced"].to_numpy(dtype=np.float32, copy=False)
    out["r2_full"] = merged["r2_full"].to_numpy(dtype=np.float32, copy=False)
    out["delta_r2"] = (merged["r2_full"] - merged["r2_reduced"]).to_numpy(dtype=np.float32, copy=False)
    out["f_stat"] = f_stat.astype(np.float32)
    out["block_p_value"] = p_value.astype(np.float64)
    out["block_q_value"] = q_value.astype(np.float64)
    out["block_significant_fdr"] = q_value <= float(fdr_alpha)
    out["fdr_alpha"] = float(fdr_alpha)
    coef_suffixes = ("beta", "delta_r2", "q_value", "significant_fdr", "p_value", "t_stat", "se")
    passthrough_cols = []
    for col in full_df.columns:
        if any(col.endswith(f"_{suffix}") for suffix in coef_suffixes):
            passthrough_cols.append(col)
    for col in passthrough_cols:
        out[col] = full_df[col].to_numpy(copy=False)
    return out


def plot_connectivity_vs_distance(
    edge_df,
    y_keys=("FC", "SC"),
    figsize=(11.5, 4.2),
    dpi=180,
    show=True,
    suptitle=None,
    alpha=0.25,
    s=8,
    rasterized=True,
):
    """Scatter distance vs connectivity using a shared x-axis layout."""
    y_keys = tuple(y_keys)
    title_map = {
        "FC": "Distance vs FC",
        "SC": "Distance vs SC",
        "SC_r2t_corr": "Distance vs SC_r2t corr",
    }
    color_map = {
        "FC": "#b2182b",
        "SC": "#2166ac",
        "SC_r2t_corr": "#4d9221",
    }

    fig, axes = plt.subplots(1, len(y_keys), figsize=figsize, dpi=dpi, sharex=True)
    if len(y_keys) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.18)

    x = edge_df["distance"].to_numpy(dtype=np.float32, copy=False)
    for ax, key in zip(axes, y_keys):
        y = edge_df[key].to_numpy(dtype=np.float32, copy=False)
        ax.scatter(
            x,
            y,
            s=s,
            alpha=alpha,
            linewidths=0,
            color=color_map.get(key, "#444444"),
            rasterized=rasterized,
        )
        ax.set_title(title_map.get(key, f"Distance vs {key}"), fontsize=12, pad=6)
        ax.set_xlabel("Distance", fontsize=10)
        ax.set_ylabel(key, fontsize=10)
        ax.grid(alpha=0.15, linewidth=0.6)

    final_title = suptitle or str(edge_df["label"].iloc[0])
    fig.suptitle(final_title, fontsize=14, y=0.95)
    if show:
        plt.show()
    return fig, axes


def plot_edgewise_regression_summary(
    reg_df,
    figsize=(12.5, 4.2),
    dpi=180,
    show=True,
    suptitle=None,
    bins=60,
):
    """Summarize per-edge distance regressions with slope distribution and distance trend."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.16)

    slope = reg_df["slope"].to_numpy(dtype=np.float32, copy=False)
    mean_distance = reg_df["distance_mean"].to_numpy(dtype=np.float32, copy=False)
    corr = reg_df["corr"].to_numpy(dtype=np.float32, copy=False)

    axes[0].hist(slope, bins=bins, color="#b2182b", alpha=0.82, edgecolor="none")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Edgewise Slope Distribution", fontsize=12, pad=6)
    axes[0].set_xlabel("Slope of connectivity ~ distance", fontsize=10)
    axes[0].set_ylabel("Edge count", fontsize=10)
    axes[0].grid(alpha=0.15, linewidth=0.6)

    axes[1].scatter(mean_distance, slope, s=7, alpha=0.18, linewidths=0, color="#2166ac", rasterized=True)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Slope vs Mean Distance", fontsize=12, pad=6)
    axes[1].set_xlabel("Mean distance across subjects", fontsize=10)
    axes[1].set_ylabel("Slope", fontsize=10)
    axes[1].grid(alpha=0.15, linewidth=0.6)

    if np.isfinite(corr).any():
        finite_corr = corr[np.isfinite(corr)]
        text = (
            f"median slope = {np.median(slope):.4g}\n"
            f"mean slope = {np.mean(slope):.4g}\n"
            f"frac slope < 0 = {np.mean(slope < 0):.3f}\n"
            f"median corr = {np.median(finite_corr):.4g}"
        )
        axes[1].text(
            0.03,
            0.97,
            text,
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    final_title = suptitle or str(reg_df["label"].iloc[0])
    fig.suptitle(final_title, fontsize=14, y=0.95)
    if show:
        plt.show()
    return fig, axes


def plot_single_edge_distance_scatter(
    base,
    edge_idx,
    partition="val",
    y_key="FC",
    figsize=(5.2, 4.2),
    dpi=180,
    show=True,
    suptitle=None,
    alpha=0.65,
    s=22,
):
    """Plot the subject-level distance vs connectivity scatter for one edge."""
    reg_df = compute_edgewise_distance_regression(base, partition=partition, y_key=y_key)
    row = reg_df.iloc[int(edge_idx)]

    part_indices = _partition_indices(base, partition)
    if y_key == "FC":
        y_all = np.asarray(base.fc_upper_triangles[part_indices], dtype=np.float32)
    elif y_key == "SC":
        y_all = np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float32)
    else:
        y_all = np.asarray(base.sc_r2t_corr_upper_triangles[part_indices], dtype=np.float32)

    distance_stack = _distance_stack(base, part_indices)
    x_all = np.stack([square2tri(mat) for mat in distance_stack], axis=0).astype(np.float32)

    x = x_all[:, int(edge_idx)]
    y = y_all[:, int(edge_idx)]
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100, dtype=np.float32)
    y_line = row["intercept"] + row["slope"] * x_line

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.scatter(x, y, s=s, alpha=alpha, linewidths=0, color="#b2182b")
    ax.plot(x_line, y_line, color="black", linewidth=1.4, alpha=0.9)
    ax.set_xlabel("Distance", fontsize=10)
    ax.set_ylabel(y_key, fontsize=10)
    ax.grid(alpha=0.15, linewidth=0.6)

    title = suptitle or (
        f"{_PARTITION_DISPLAY[partition]} edge {int(edge_idx)} | "
        f"{row['label_i']} - {row['label_j']} | "
        f"{y_key} ~ distance\n"
        f"slope={row['slope']:.4g}, corr={row['corr']:.4g}"
    )
    ax.set_title(title, fontsize=11, pad=8)

    if show:
        plt.show()
    return fig, ax, row


def plot_single_edge_distance_scatter_across_partitions(
    base,
    edge_idx,
    partitions=("train", "val", "test"),
    y_key="FC",
    figsize=(14.0, 4.0),
    dpi=180,
    show=True,
    suptitle=None,
    alpha=0.65,
    s=18,
):
    """Plot one edge across multiple partitions using the same distance-vs-connectivity scatter layout."""
    fig, axes = plt.subplots(1, len(partitions), figsize=figsize, dpi=dpi, sharey=True)
    if len(partitions) == 1:
        axes = [axes]

    rows = []
    for ax, partition in zip(axes, partitions):
        reg_df = compute_edgewise_distance_regression(base, partition=partition, y_key=y_key)
        row = reg_df.iloc[int(edge_idx)]
        rows.append(row)
        part_indices = _partition_indices(base, partition)
        if y_key == "FC":
            y_all = np.asarray(base.fc_upper_triangles[part_indices], dtype=np.float32)
        elif y_key == "SC":
            y_all = np.asarray(base.sc_upper_triangles[part_indices], dtype=np.float32)
        else:
            y_all = np.asarray(base.sc_r2t_corr_upper_triangles[part_indices], dtype=np.float32)
        distance_stack = _distance_stack(base, part_indices)
        x_all = np.stack([square2tri(mat) for mat in distance_stack], axis=0).astype(np.float32)
        x = x_all[:, int(edge_idx)]
        y = y_all[:, int(edge_idx)]
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100, dtype=np.float32)
        y_line = row["intercept"] + row["slope"] * x_line
        ax.scatter(x, y, s=s, alpha=alpha, linewidths=0, color="#b2182b")
        ax.plot(x_line, y_line, color="black", linewidth=1.2, alpha=0.9)
        ax.set_title(
            f"{_PARTITION_DISPLAY[partition]}\n"
            f"r={row['corr']:.3f}, q={row['q_value']:.3g}",
            fontsize=10,
            pad=6,
        )
        ax.set_xlabel("Distance", fontsize=10)
        ax.grid(alpha=0.15, linewidth=0.6)
    axes[0].set_ylabel(y_key, fontsize=10)
    fig.subplots_adjust(wspace=0.24, top=0.76, bottom=0.18)
    title = suptitle or (
        f"Edge {int(edge_idx)} | {rows[0]['label_i']} - {rows[0]['label_j']} | {y_key} ~ distance"
    )
    fig.suptitle(title, fontsize=13, y=0.95)
    if show:
        plt.show()
    return fig, axes, rows


def plot_multifeature_regression_summary(
    reg_df,
    coef_name,
    figsize=(12.5, 4.2),
    dpi=180,
    show=True,
    suptitle=None,
    bins=60,
):
    """Summarize one coefficient from an edgewise multifeature regression table."""
    beta_col = f"{coef_name}_beta"
    q_col = f"{coef_name}_q_value"
    sig_col = f"{coef_name}_significant_fdr"
    delta_r2_col = f"{coef_name}_delta_r2"
    if beta_col not in reg_df.columns:
        raise ValueError(f"Coefficient {coef_name!r} not found in regression table")
    if delta_r2_col not in reg_df.columns:
        raise ValueError(f"Delta R2 column for coefficient {coef_name!r} not found in regression table")

    beta = reg_df[beta_col].to_numpy(dtype=np.float32, copy=False)
    delta_r2 = reg_df[delta_r2_col].to_numpy(dtype=np.float32, copy=False)
    q_value = reg_df[q_col].to_numpy(dtype=np.float64, copy=False)
    sig_mask = reg_df[sig_col].to_numpy(dtype=bool, copy=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.16)

    axes[0].hist(beta, bins=bins, color="#4d9221", alpha=0.82, edgecolor="none")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{coef_name} Coefficient Distribution", fontsize=12, pad=6)
    axes[0].set_xlabel(f"{coef_name} beta", fontsize=10)
    axes[0].set_ylabel("Edge count", fontsize=10)
    axes[0].grid(alpha=0.15, linewidth=0.6)

    axes[1].scatter(delta_r2[~sig_mask], beta[~sig_mask], s=7, alpha=0.12, linewidths=0, color="#2166ac", rasterized=True)
    if sig_mask.any():
        axes[1].scatter(delta_r2[sig_mask], beta[sig_mask], s=12, alpha=0.55, linewidths=0, color="#b2182b", rasterized=True)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].axvline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_title(f"{coef_name} Beta vs Delta R2", fontsize=12, pad=6)
    axes[1].set_xlabel(f"Delta R2 from adding {coef_name}", fontsize=10)
    axes[1].set_ylabel(f"{coef_name} beta", fontsize=10)
    axes[1].grid(alpha=0.15, linewidth=0.6)

    finite_q = q_value[np.isfinite(q_value)]
    text = (
        f"median beta = {np.median(beta):.4g}\n"
        f"mean beta = {np.mean(beta):.4g}\n"
            f"FDR-significant = {int(sig_mask.sum())}\n"
            f"median delta R2 = {np.median(delta_r2):.4g}\n"
            f"best q = {np.min(finite_q):.4g}" if finite_q.size else
        f"median beta = {np.median(beta):.4g}\nmean beta = {np.mean(beta):.4g}\nFDR-significant = {int(sig_mask.sum())}\nmedian delta R2 = {np.median(delta_r2):.4g}"
    )
    axes[1].text(
        0.03,
        0.97,
        text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    final_title = suptitle or f"{reg_df['label'].iloc[0]} | coefficient={coef_name}"
    fig.suptitle(final_title, fontsize=14, y=0.95)
    if show:
        plt.show()
    return fig, axes


def _hemi_roi_indices(parcellation, hemi):
    roi_df = pd.read_csv(
        f"/scratch/asr655/neuroinformatics/Conn2Conn/data/atlas_info/{parcellation}_dseg_reformatted.csv"
    )
    if hemi == "left":
        roi_mask = roi_df["hemisphere"].str.contains("L")
    elif hemi == "right":
        roi_mask = roi_df["hemisphere"].str.contains("R")
    else:
        roi_mask = np.ones(len(roi_df), dtype=bool)
    return np.where(roi_mask)[0]


def _session_tsv_path(HCP_dir, subject_id, parcellation, direction, run):
    sub_folder = f"sub-{subject_id}"
    fname = (
        f"{sub_folder}_task-rest_dir-{direction}_run-{run}"
        f"_space-fsLR_seg-{parcellation}_stat-pearsoncorrelation_relmat.tsv"
    )
    return os.path.join(HCP_dir, "HCP1200_fMRI/xcpd-0-9-1", sub_folder, "func", fname)


def _read_fc_tsv(path, roi_indices):
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    mat = df.values.astype(np.float32)
    return mat[np.ix_(roi_indices, roi_indices)]


def load_subject_session_fc(base, subject_id, HCP_dir=_DEFAULT_HCP_DIR):
    """Load the four per-direction/run FC matrices plus the cached concat for one subject.

    Returns dict with keys 'S1_LR', 'S1_RL', 'S2_LR', 'S2_RL', 'concat', 'subject_id'.
    Hemisphere subsetting follows base.parcellation / base.hemi.
    """
    parcellation = base.parcellation
    roi_indices = _hemi_roi_indices(parcellation, base.hemi)

    out = {"subject_id": str(subject_id)}
    for key in _SESSION_KEYS:
        direction, run = _SESSION_TO_DIR_RUN[key]
        path = _session_tsv_path(HCP_dir, subject_id, parcellation, direction, run)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing session FC file: {path}")
        out[key] = _read_fc_tsv(path, roi_indices)

    # Reuse the cached concat from base (already hemi-subset). Use metadata_df
    # to find the row, since it is canonicalized to align with base.fc_matrices
    # (base.fc_subject_ids is the pre-canonicalization list and would mis-map).
    sids = base.metadata_df["subject"].astype(int).to_numpy()
    matches = np.where(sids == int(subject_id))[0]
    if len(matches) == 0:
        raise ValueError(f"subject_id {subject_id} not found in base.metadata_df")
    out["concat"] = np.asarray(base.fc_matrices[int(matches[0])], dtype=np.float32)
    return out


def _fisher_mean(stack, clip_eps=1e-6):
    clipped = np.clip(stack, -1.0 + clip_eps, 1.0 - clip_eps)
    return np.tanh(np.mean(np.arctanh(clipped), axis=0)).astype(stack.dtype, copy=False)


def _fc_panel(ax, mat, vmax, title):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_subject_session_overview(
    base,
    subject_id,
    HCP_dir=_DEFAULT_HCP_DIR,
    figsize_per_panel=(2.6, 2.6),
    dpi=150,
    show=True,
):
    """Three figures for the test-retest EDA of one subject.

    Returns (fig_raw, fig_arith, fig_fisher, data) where data is the per-session dict
    plus the derived session means under both averaging schemes.
    """
    sessions = load_subject_session_fc(base, subject_id, HCP_dir=HCP_dir)
    s1_stack = np.stack([sessions["S1_LR"], sessions["S1_RL"]], axis=0)
    s2_stack = np.stack([sessions["S2_LR"], sessions["S2_RL"]], axis=0)

    arith = {
        "S1_mean": np.mean(s1_stack, axis=0),
        "S2_mean": np.mean(s2_stack, axis=0),
    }
    fisher = {
        "S1_mean_z": _fisher_mean(s1_stack),
        "S2_mean_z": _fisher_mean(s2_stack),
    }

    # Shared symmetric color scale across every panel for visual comparability.
    all_mats = np.stack(
        [
            sessions["S1_LR"], sessions["S1_RL"], sessions["S2_LR"], sessions["S2_RL"],
            sessions["concat"],
            arith["S1_mean"], arith["S2_mean"],
            fisher["S1_mean_z"], fisher["S2_mean_z"],
        ],
        axis=0,
    )
    vmax = float(np.quantile(np.abs(all_mats), 0.995))

    sid = sessions["subject_id"]
    parc = base.parcellation
    base_title = f"{parc} | sub-{sid}"

    # --- Figure 1: 4 raw scans + concat ---
    fig_raw, axes_raw = plt.subplots(
        1, 5, figsize=(figsize_per_panel[0] * 5, figsize_per_panel[1]), dpi=dpi
    )
    fig_raw.subplots_adjust(wspace=0.18, top=0.82, right=0.93)
    panels_raw = [
        (sessions["S1_LR"], "S1 LR"),
        (sessions["S1_RL"], "S1 RL"),
        (sessions["S2_LR"], "S2 LR"),
        (sessions["S2_RL"], "S2 RL"),
        (sessions["concat"], "Concat (pipeline FC)"),
    ]
    last_im = None
    for ax, (mat, title) in zip(axes_raw, panels_raw):
        last_im = _fc_panel(ax, mat, vmax, title)
    fig_raw.suptitle(f"{base_title} — raw per-direction scans + concat", fontsize=12, y=0.96)
    fig_raw.colorbar(last_im, ax=axes_raw, fraction=0.012, pad=0.02, ticks=[-vmax, 0.0, vmax])

    # --- Figure 2: arithmetic per-session means + concat ---
    fig_arith, axes_arith = plt.subplots(
        1, 3, figsize=(figsize_per_panel[0] * 3, figsize_per_panel[1]), dpi=dpi
    )
    fig_arith.subplots_adjust(wspace=0.2, top=0.82, right=0.92)
    panels_arith = [
        (arith["S1_mean"], "S1 mean (LR+RL)/2"),
        (arith["S2_mean"], "S2 mean (LR+RL)/2"),
        (sessions["concat"], "Concat"),
    ]
    last_im = None
    for ax, (mat, title) in zip(axes_arith, panels_arith):
        last_im = _fc_panel(ax, mat, vmax, title)
    fig_arith.suptitle(f"{base_title} — arithmetic LR/RL averaging", fontsize=12, y=0.96)
    fig_arith.colorbar(last_im, ax=axes_arith, fraction=0.018, pad=0.02, ticks=[-vmax, 0.0, vmax])

    # --- Figure 3: Fisher-z per-session means + concat ---
    fig_fisher, axes_fisher = plt.subplots(
        1, 3, figsize=(figsize_per_panel[0] * 3, figsize_per_panel[1]), dpi=dpi
    )
    fig_fisher.subplots_adjust(wspace=0.2, top=0.82, right=0.92)
    panels_fisher = [
        (fisher["S1_mean_z"], "S1 Fisher-z mean"),
        (fisher["S2_mean_z"], "S2 Fisher-z mean"),
        (sessions["concat"], "Concat"),
    ]
    last_im = None
    for ax, (mat, title) in zip(axes_fisher, panels_fisher):
        last_im = _fc_panel(ax, mat, vmax, title)
    fig_fisher.suptitle(f"{base_title} — Fisher-z LR/RL averaging", fontsize=12, y=0.96)
    fig_fisher.colorbar(last_im, ax=axes_fisher, fraction=0.018, pad=0.02, ticks=[-vmax, 0.0, vmax])

    if show:
        plt.show()

    data = {
        **sessions,
        **arith,
        **fisher,
        "vmax": vmax,
    }
    return fig_raw, fig_arith, fig_fisher, data


_VALID_DEMEAN_SESSIONS = ("concat", "session1", "session2")


def _load_partition_session_stack(base, partition, session, apply_fisher_z, HCP_dir):
    """Per-subject session-level FC for every subject in `partition`, hemi-subset.

    For session='concat' uses `base.fc_matrices` directly. For 'session1'/'session2'
    reads the two per-direction TSVs and averages them within subject (arith or Fisher).
    """
    part_indices = _partition_indices(base, partition)
    # `metadata_df` and `fc_matrices` are aligned to the canonicalized subject set,
    # so partition indices index both directly. (`base.fc_subject_ids` is the
    # pre-canonicalization list and must NOT be used for row lookup here.)
    subject_ids = base.metadata_df.iloc[part_indices]["subject"].astype(int).tolist()

    if session == "concat":
        return np.asarray(base.fc_matrices[part_indices], dtype=np.float32)

    if session not in ("session1", "session2"):
        raise ValueError(f"demean_session must be one of {_VALID_DEMEAN_SESSIONS}")

    keys = ("S1_LR", "S1_RL") if session == "session1" else ("S2_LR", "S2_RL")
    roi_indices = _hemi_roi_indices(base.parcellation, base.hemi)

    within_mean_fn = _fisher_mean if apply_fisher_z else (lambda s: np.mean(s, axis=0))

    def _load_one(sid):
        scans = []
        for k in keys:
            direction, run = _SESSION_TO_DIR_RUN[k]
            path = _session_tsv_path(HCP_dir, sid, base.parcellation, direction, run)
            if not os.path.exists(path):
                return None
            scans.append(_read_fc_tsv(path, roi_indices))
        return within_mean_fn(np.stack(scans, axis=0))

    from concurrent.futures import ThreadPoolExecutor
    n_jobs = min(len(subject_ids), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_load_one, subject_ids))

    stack = [m for m in results if m is not None]
    missing = len(results) - len(stack)
    if missing:
        print(f"[session reference] skipped {missing}/{len(results)} subjects missing {session} TSVs")
    if not stack:
        raise RuntimeError(f"No {session} FC files found for partition {partition!r}")
    return np.stack(stack, axis=0).astype(np.float32)


def _partition_session_reference(
    base, partition, session, aggregation, apply_fisher_z, HCP_dir
):
    cache = getattr(base, "_session_reference_cache", None)
    if cache is None:
        cache = {}
        base._session_reference_cache = cache
    key = (partition, session, aggregation, bool(apply_fisher_z))
    if key in cache:
        return cache[key]

    stack = _load_partition_session_stack(base, partition, session, apply_fisher_z, HCP_dir)
    if apply_fisher_z:
        ref = _fisher_aggregate_stack(stack, aggregation)
    else:
        ref = _aggregate_stack(stack, aggregation)
    cache[key] = ref.astype(np.float32)
    return cache[key]


def plot_subject_session_overview_demeaned(
    base,
    subject_id,
    demean_partition="train",
    demean_session="concat",
    demean_aggregation="mean",
    demean_apply_fisher_z=False,
    HCP_dir=_DEFAULT_HCP_DIR,
    figsize_per_panel=(2.6, 2.6),
    dpi=150,
    show=True,
):
    """Demeaned version of `plot_subject_session_overview`.

    Each panel shows `subject_FC - reference`, where `reference` is aggregated across
    `demean_partition` using `demean_session` ('concat' | 'session1' | 'session2')
    with `demean_aggregation` ('mean' | 'median') and optional Fisher-z.
    Color scale is symmetric about 0 and shared across all panels.
    """
    if demean_session not in _VALID_DEMEAN_SESSIONS:
        raise ValueError(f"demean_session must be one of {_VALID_DEMEAN_SESSIONS}")

    sessions = load_subject_session_fc(base, subject_id, HCP_dir=HCP_dir)
    s1_stack = np.stack([sessions["S1_LR"], sessions["S1_RL"]], axis=0)
    s2_stack = np.stack([sessions["S2_LR"], sessions["S2_RL"]], axis=0)

    arith = {"S1_mean": np.mean(s1_stack, axis=0), "S2_mean": np.mean(s2_stack, axis=0)}
    fisher = {"S1_mean_z": _fisher_mean(s1_stack), "S2_mean_z": _fisher_mean(s2_stack)}

    reference = _partition_session_reference(
        base, demean_partition, demean_session, demean_aggregation,
        demean_apply_fisher_z, HCP_dir,
    )

    def _resid(mat):
        return (mat - reference).astype(np.float32)

    resid_scans = {k: _resid(sessions[k]) for k in _SESSION_KEYS}
    resid_concat = _resid(sessions["concat"])
    resid_arith = {k: _resid(v) for k, v in arith.items()}
    resid_fisher = {k: _resid(v) for k, v in fisher.items()}

    all_resid = np.stack(
        list(resid_scans.values()) + [resid_concat]
        + list(resid_arith.values()) + list(resid_fisher.values()),
        axis=0,
    )
    vmax = float(np.quantile(np.abs(all_resid), 0.995))
    vmax = vmax if vmax > 0 else 1e-6

    sid = sessions["subject_id"]
    ref_tag = f"{_PARTITION_DISPLAY[demean_partition].lower()} {demean_session} {demean_aggregation}"
    if demean_apply_fisher_z:
        ref_tag += " (Fisher-z)"
    base_title = f"{base.parcellation} | sub-{sid} — demeaned vs {ref_tag}"

    def _render(panels, row_label):
        n = len(panels)
        fig, axes = plt.subplots(
            1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]), dpi=dpi
        )
        fig.subplots_adjust(wspace=0.2, top=0.82, right=0.93)
        last_im = None
        for ax, (mat, title) in zip(axes, panels):
            last_im = _fc_panel(ax, mat, vmax, title)
        fig.suptitle(f"{base_title} — {row_label}", fontsize=12, y=0.96)
        fig.colorbar(last_im, ax=axes, fraction=0.012, pad=0.02, ticks=[-vmax, 0.0, vmax])
        return fig

    fig_raw = _render(
        [
            (resid_scans["S1_LR"], "S1 LR"),
            (resid_scans["S1_RL"], "S1 RL"),
            (resid_scans["S2_LR"], "S2 LR"),
            (resid_scans["S2_RL"], "S2 RL"),
            (resid_concat, "Concat"),
        ],
        "raw per-direction + concat",
    )
    fig_arith = _render(
        [
            (resid_arith["S1_mean"], "S1 mean (LR+RL)/2"),
            (resid_arith["S2_mean"], "S2 mean (LR+RL)/2"),
            (resid_concat, "Concat"),
        ],
        "arithmetic LR/RL averaging",
    )
    fig_fisher = _render(
        [
            (resid_fisher["S1_mean_z"], "S1 Fisher-z mean"),
            (resid_fisher["S2_mean_z"], "S2 Fisher-z mean"),
            (resid_concat, "Concat"),
        ],
        "Fisher-z LR/RL averaging",
    )

    if show:
        plt.show()

    data = {
        **{k: resid_scans[k] for k in _SESSION_KEYS},
        "concat": resid_concat,
        **resid_arith,
        **resid_fisher,
        "reference": reference,
        "demean_partition": demean_partition,
        "demean_session": demean_session,
        "demean_aggregation": demean_aggregation,
        "demean_apply_fisher_z": demean_apply_fisher_z,
        "subject_id": sid,
        "vmax": vmax,
    }
    return fig_raw, fig_arith, fig_fisher, data


def _upper_tri(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _edge_corr(a, b):
    av = _upper_tri(a)
    bv = _upper_tri(b)
    return float(np.corrcoef(av, bv)[0, 1])


def _l2_diff(a, b):
    return float(np.mean((_upper_tri(a) - _upper_tri(b)) ** 2))


def _max_abs_diff(a, b):
    return float(np.max(np.abs(_upper_tri(a) - _upper_tri(b))))


def session_diagnostics(retest_data):
    """Numeric checks on a retest_data dict from `plot_subject_session_overview`.

    Returns a pandas DataFrame summarizing:
      - arith vs Fisher-z agreement on per-session means
      - within-subject S1 vs S2 similarity (under each averaging scheme)
      - agreement of (S1+S2)/2 with the full concat
      - per-direction noise floor: each scan vs concat
    """
    s1_lr, s1_rl = retest_data["S1_LR"], retest_data["S1_RL"]
    s2_lr, s2_rl = retest_data["S2_LR"], retest_data["S2_RL"]
    concat = retest_data["concat"]

    s1_arith, s2_arith = retest_data["S1_mean"], retest_data["S2_mean"]
    s1_z, s2_z = retest_data["S1_mean_z"], retest_data["S2_mean_z"]

    overall_arith = 0.5 * (s1_arith + s2_arith)
    overall_z = _fisher_mean(np.stack([s1_z, s2_z], axis=0))

    rows = [
        # arith vs fisher-z agreement on the same session
        ("S1: arith vs Fisher-z mean", _edge_corr(s1_arith, s1_z), _l2_diff(s1_arith, s1_z), _max_abs_diff(s1_arith, s1_z)),
        ("S2: arith vs Fisher-z mean", _edge_corr(s2_arith, s2_z), _l2_diff(s2_arith, s2_z), _max_abs_diff(s2_arith, s2_z)),
        # test-retest similarity under each scheme
        ("Test-retest (arith): S1_mean vs S2_mean", _edge_corr(s1_arith, s2_arith), _l2_diff(s1_arith, s2_arith), _max_abs_diff(s1_arith, s2_arith)),
        ("Test-retest (Fisher): S1_mean_z vs S2_mean_z", _edge_corr(s1_z, s2_z), _l2_diff(s1_z, s2_z), _max_abs_diff(s1_z, s2_z)),
        # session-mean vs concat
        ("Concat vs (S1+S2)/2 arith", _edge_corr(concat, overall_arith), _l2_diff(concat, overall_arith), _max_abs_diff(concat, overall_arith)),
        ("Concat vs (S1+S2)/2 Fisher", _edge_corr(concat, overall_z), _l2_diff(concat, overall_z), _max_abs_diff(concat, overall_z)),
        # per-direction noise floor against concat
        ("Concat vs S1_LR", _edge_corr(concat, s1_lr), _l2_diff(concat, s1_lr), _max_abs_diff(concat, s1_lr)),
        ("Concat vs S1_RL", _edge_corr(concat, s1_rl), _l2_diff(concat, s1_rl), _max_abs_diff(concat, s1_rl)),
        ("Concat vs S2_LR", _edge_corr(concat, s2_lr), _l2_diff(concat, s2_lr), _max_abs_diff(concat, s2_lr)),
        ("Concat vs S2_RL", _edge_corr(concat, s2_rl), _l2_diff(concat, s2_rl), _max_abs_diff(concat, s2_rl)),
        # within-session LR vs RL noise floor
        ("S1 LR vs RL", _edge_corr(s1_lr, s1_rl), _l2_diff(s1_lr, s1_rl), _max_abs_diff(s1_lr, s1_rl)),
        ("S2 LR vs RL", _edge_corr(s2_lr, s2_rl), _l2_diff(s2_lr, s2_rl), _max_abs_diff(s2_lr, s2_rl)),
    ]
    df = pd.DataFrame(rows, columns=["comparison", "edge_corr", "mse_offdiag", "max_abs_diff"])
    df["comparison"] = pd.Categorical(df["comparison"], categories=[r[0] for r in rows], ordered=True)
    return df


def make_data_matrix_gif(
    base,
    gif_path="data_matrix_overview.gif",
    partition="val",
    n_subjects=10,
    start_position=0,
    duration_ms=600,
    tmpdir="tmp_data_matrix_frames",
    figsize=(14, 3.8),
    dpi=180,
    cleanup_tmp=True,
    header_metadata=None,
):
    part_indices = _partition_indices(base, partition)
    start_position = int(start_position)
    end_pos = min(start_position + int(n_subjects), len(part_indices))
    chosen_positions = list(range(start_position, end_pos))
    if not chosen_positions:
        raise ValueError(f"No {partition!r} subjects selected for GIF generation.")

    ranges = compute_overview_ranges(base, partition=partition, indices=part_indices)
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    frames = []

    for frame_idx, pos in enumerate(chosen_positions):
        subject_data = get_data_overview_matrices(base, partition=partition, position=pos)
        fig, _, _ = plot_data_matrix_overview(
            base,
            partition=partition,
            position=pos,
            aggregate=None,
            ranges=ranges,
            figsize=figsize,
            dpi=dpi,
            show=False,
            suptitle=_format_subject_header(base, subject_data, header_metadata=header_metadata),
            header_metadata=header_metadata,
        )
        frame_path = tmpdir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frames.append(Image.open(frame_path))

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration_ms),
            loop=0,
            optimize=True,
        )
        print(f"GIF saved to {gif_path}")

    if cleanup_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return Path(gif_path)


# ---------------------------------------------------------------------------
# Condition connectomes (rest, rest sessions, HCP tasks)
# ---------------------------------------------------------------------------
_CONDITION_DISPLAY = {
    "rest": "Rest",
    "rest_S1": "Rest S1",
    "rest_S2": "Rest S2",
    "emotion": "Emotion",
    "gambling": "Gambling",
    "language": "Language",
    "motor": "Motor",
    "relational": "Relational",
    "social": "Social",
    "wm": "WM",
}
_SIMILARITY_METRICS = ("pearson", "euclidean", "geodesic")
_SIMILARITY_TITLES = {"pearson": "Edge correlation", "euclidean": "Euclidean distance", "geodesic": "Geodesic distance"}
_SIMILARITY_CMAPS = {"pearson": "viridis", "euclidean": "viridis_r", "geodesic": "viridis_r"}


def _condition_label(condition):
    return _CONDITION_DISPLAY.get(condition, condition)


def _n_rois_from_edges(n_edges):
    n_rois = int(round((1 + np.sqrt(1 + 8 * n_edges)) / 2))
    if n_rois * (n_rois - 1) // 2 != n_edges:
        raise ValueError(f"{n_edges} is not an upper-triangle edge count")
    return n_rois


def _edges_to_square(vec, diag_value):
    n_rois = _n_rois_from_edges(vec.shape[-1])
    iu = np.triu_indices(n_rois, k=1)
    mat = np.zeros((n_rois, n_rois), dtype=np.float64)
    mat[iu] = vec
    mat.T[iu] = vec
    np.fill_diagonal(mat, diag_value)
    return mat


def condition_upper_triangles(base, conditions):
    """Map each condition to its (n_canonical_subjects, n_edges) FC array, aligned to `base` subject order.

    'rest' is the main FC; 'rest_S1' / 'rest_S2' need `HCP_Base(expose_fc_sessions=True)`; task names need
    `HCP_Base(fc_conditions=[...])`.
    """
    out = {}
    for condition in conditions:
        if condition == "rest":
            arr = base.fc_upper_triangles
        elif condition in ("rest_S1", "rest_S2"):
            if not getattr(base, "expose_fc_sessions", False):
                raise ValueError(f"{condition!r} needs HCP_Base(expose_fc_sessions=True)")
            arr = base.fc_session1_upper_triangles if condition == "rest_S1" else base.fc_session2_upper_triangles
        else:
            tri = getattr(base, "fc_condition_upper_triangles", {})
            if condition not in tri:
                raise ValueError(f"{condition!r} needs HCP_Base(fc_conditions=[..., {condition!r}])")
            arr = tri[condition]
        out[condition] = np.asarray(arr)
    n_subjects = {arr.shape[0] for arr in out.values()}
    if len(n_subjects) != 1:
        raise ValueError(f"conditions are not subject-aligned: {n_subjects}")
    return out


def _aggregate_edges(edges, aggregate="mean", apply_fisher_z=False):
    if apply_fisher_z:
        return _fisher_aggregate_stack(edges, agg=aggregate)
    return _aggregate_stack(edges, agg=aggregate)


def _condition_selection(base, partition, position, subject_id):
    """(global indices, subject_id or None) for a whole partition or one subject in it."""
    if position is None and subject_id is None:
        return _partition_indices(base, partition), None
    global_idx = _resolve_partition_global_index(base, partition=partition, position=position, subject_id=subject_id)
    return np.array([global_idx]), base.metadata_df["subject"].iloc[global_idx]


# --- Figure 1: condition connectome grid -------------------------------------------------------------
def get_condition_connectomes(
    base,
    conditions,
    partition="val",
    position=None,
    subject_id=None,
    aggregate="mean",
    apply_fisher_z=False,
    demean_partition=None,
):
    """Square FC matrix per condition for a partition aggregate or one subject.

    With `demean_partition`, each matrix has that partition's mean of the *same condition* subtracted
    (subject residual or partition-mean offset vs the reference mean).
    """
    edges = condition_upper_triangles(base, conditions)
    indices, sid = _condition_selection(base, partition, position, subject_id)
    ref_idx = _partition_indices(base, demean_partition) if demean_partition is not None else None

    matrices = {}
    for condition in conditions:
        if sid is None:
            vec = _aggregate_edges(edges[condition][indices], aggregate, apply_fisher_z)
        else:
            vec = np.asarray(edges[condition][indices[0]], dtype=np.float64)
        if ref_idx is not None:
            vec = vec - _aggregate_edges(edges[condition][ref_idx], aggregate, apply_fisher_z)
        matrices[condition] = _edges_to_square(vec, diag_value=0.0 if ref_idx is not None else 1.0)

    part_label = _PARTITION_DISPLAY.get(partition, partition)
    if sid is None:
        agg_label = ("Fisher-z " if apply_fisher_z else "") + aggregate
        label = f"{part_label} population {agg_label} (n={len(indices)})"
    else:
        label = f"{part_label} subject {sid}"
    if ref_idx is not None:
        label += f" - {_PARTITION_DISPLAY.get(demean_partition, demean_partition)} mean"
    return {
        "matrices": matrices,
        "conditions": list(conditions),
        "partition": partition,
        "subject_id": sid,
        "demeaned": ref_idx is not None,
        "label": label,
    }


def plot_condition_connectomes(
    base,
    conditions,
    partition="val",
    position=None,
    subject_id=None,
    aggregate="mean",
    apply_fisher_z=False,
    demean_partition=None,
    vmax=None,
    ncols=None,
    figsize_per_panel=(2.8, 2.8),
    dpi=150,
    header_metadata=None,
    show=True,
):
    """Grid of one FC matrix per condition (population aggregate or one subject), shared color scale.

    Raw matrices default to [-1, 1]; demeaned ones to the shared 99.5th percentile of |residual|.
    """
    data = get_condition_connectomes(
        base, conditions, partition=partition, position=position, subject_id=subject_id,
        aggregate=aggregate, apply_fisher_z=apply_fisher_z, demean_partition=demean_partition,
    )
    if vmax is None:
        if data["demeaned"]:
            off = np.concatenate([np.abs(m[np.triu_indices(m.shape[0], 1)]) for m in data["matrices"].values()])
            vmax = float(np.quantile(off, 0.995)) or 1.0
        else:
            vmax = 1.0
    ncols = len(conditions) if ncols is None else int(ncols)
    nrows = int(np.ceil(len(conditions) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows + 0.5),
        dpi=dpi, squeeze=False,
    )
    for ax in axes.ravel()[len(conditions):]:
        ax.axis("off")
    for ax, condition in zip(axes.ravel(), conditions):
        im = _fc_panel(ax, data["matrices"][condition], vmax, _condition_label(condition))
        _add_small_colorbar(fig, ax, im, ticks=[-vmax, 0.0, vmax])
    data["vmax"] = vmax
    fig.suptitle(_format_subject_header(base, data, header_metadata=header_metadata), fontsize=12)
    # leave room right of each panel for its inset colorbar, and between rows for titles
    height = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.01, right=0.95, bottom=0.02, top=1 - 0.55 / height, wspace=0.32, hspace=0.22)
    if show:
        plt.show()
    return fig, axes, data


# --- Figure 2: condition x condition similarity --------------------------------------------------------
def _shrink(mat, shrinkage):
    if not shrinkage:
        return mat
    return (1.0 - shrinkage) * mat + shrinkage * np.eye(mat.shape[0])


def _condition_similarity_one(edge_rows, metrics, geodesic, shrinkage, eps):
    """K x K similarity/distance matrices for one set of K condition edge vectors (K, E)."""
    from models.eval.fc_distance import pairwise_affine_invariant_distance_within, pairwise_log_euclidean_distance

    edge_rows = np.asarray(edge_rows, dtype=np.float64)
    out = {}
    if "pearson" in metrics:
        out["pearson"] = np.corrcoef(edge_rows)
    if "euclidean" in metrics:  # direct differences (K is small): exact 0 for identical conditions
        out["euclidean"] = np.linalg.norm(edge_rows[:, None, :] - edge_rows[None, :, :], axis=-1)
    if "geodesic" in metrics:
        mats = np.stack([_shrink(_edges_to_square(v, 1.0), shrinkage) for v in edge_rows])
        if geodesic == "affine_invariant":
            out["geodesic"] = pairwise_affine_invariant_distance_within(mats, eps=eps)
        elif geodesic == "log_euclidean":
            out["geodesic"] = pairwise_log_euclidean_distance(mats, mats, eps=eps)
            np.fill_diagonal(out["geodesic"], 0.0)
        else:
            raise ValueError(f"geodesic must be 'affine_invariant' or 'log_euclidean'; got {geodesic!r}")
    return out


def compute_condition_similarity(
    base,
    conditions,
    partition="val",
    mode="population",
    position=None,
    subject_id=None,
    aggregate="mean",
    apply_fisher_z=False,
    metrics=_SIMILARITY_METRICS,
    geodesic="affine_invariant",
    shrinkage=None,
    max_subjects=None,
    eps=1e-6,
):
    """Condition x condition similarity (edge Pearson r) and distances (Euclidean on edges, geodesic on matrices).

    mode:
        'population': between the partition's condition aggregates (mean / Fisher-z mean)
        'subject':    between one subject's conditions (`position` / `subject_id`)
        'subjects':   per subject in the partition (first `max_subjects`), returned as mean and std
    shrinkage: λ in (1-λ)C + λI before the geodesic (task scans have fewer TRs than parcels, so single-subject
        task FC is singular). None -> 0 for 'population', 0.1 otherwise.
    """
    metrics = tuple(metrics)
    unknown = set(metrics) - set(_SIMILARITY_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {_SIMILARITY_METRICS}")
    if mode not in ("population", "subject", "subjects"):
        raise ValueError("mode must be 'population', 'subject', or 'subjects'")
    if shrinkage is None:
        shrinkage = 0.0 if mode == "population" else 0.1
    edges = condition_upper_triangles(base, conditions)
    part_label = _PARTITION_DISPLAY.get(partition, partition)

    if mode == "population":
        indices = _partition_indices(base, partition)
        rows = np.stack([_aggregate_edges(edges[c][indices], aggregate, apply_fisher_z) for c in conditions])
        sims = _condition_similarity_one(rows, metrics, geodesic, shrinkage, eps)
        result = {m: {"mean": v, "std": None} for m, v in sims.items()}
        agg_label = ("Fisher-z " if apply_fisher_z else "") + aggregate
        label, n_subjects, subject = f"{part_label} population {agg_label} (n={len(indices)})", len(indices), None
    elif mode == "subject":
        gidx = _resolve_partition_global_index(base, partition=partition, position=position, subject_id=subject_id)
        rows = np.stack([edges[c][gidx] for c in conditions])
        sims = _condition_similarity_one(rows, metrics, geodesic, shrinkage, eps)
        result = {m: {"mean": v, "std": None} for m, v in sims.items()}
        subject = base.metadata_df["subject"].iloc[gidx]
        label, n_subjects = f"{part_label} subject {subject}", 1
    else:
        indices = _partition_indices(base, partition)
        if max_subjects is not None:
            indices = indices[: int(max_subjects)]
        per_subject = [
            _condition_similarity_one(np.stack([edges[c][i] for c in conditions]), metrics, geodesic, shrinkage, eps)
            for i in indices
        ]
        result = {
            m: {
                "mean": np.mean([s[m] for s in per_subject], axis=0),
                "std": np.std([s[m] for s in per_subject], axis=0),
                "per_subject": np.stack([s[m] for s in per_subject]),
            }
            for m in metrics
        }
        label, n_subjects, subject = f"{part_label} per-subject mean (n={len(indices)})", len(indices), None

    return {
        "metrics": result,
        "conditions": list(conditions),
        "mode": mode,
        "partition": partition,
        "subject_id": subject,
        "n_subjects": n_subjects,
        "geodesic": geodesic,
        "shrinkage": float(shrinkage),
        "label": label,
    }


def plot_condition_similarity(
    base,
    similarity,
    annotate=True,
    show_std=False,
    figsize_per_panel=(4.4, 3.9),
    dpi=150,
    show=True,
):
    """Heatmap per metric from `compute_condition_similarity` (values annotated; ± std in 'subjects' mode)."""
    metrics = list(similarity["metrics"])
    labels = [_condition_label(c) for c in similarity["conditions"]]
    k = len(labels)
    fig, axes = plt.subplots(1, len(metrics), figsize=(figsize_per_panel[0] * len(metrics), figsize_per_panel[1]),
                             dpi=dpi, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        mat = similarity["metrics"][metric]["mean"]
        std = similarity["metrics"][metric].get("std")
        off = mat[~np.eye(k, dtype=bool)]
        vmin, vmax = float(np.nanmin(off)), float(np.nanmax(off))
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
        shown = mat.astype(float).copy()
        np.fill_diagonal(shown, np.nan)  # self-similarity is 1 / 0 by construction; keep the color range for the rest
        cmap = plt.get_cmap(_SIMILARITY_CMAPS[metric]).copy()
        cmap.set_bad("0.85")
        im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        title = _SIMILARITY_TITLES[metric]
        if metric == "geodesic":
            title += f"\n({similarity['geodesic'].replace('_', '-')}, λ={similarity['shrinkage']:g})"
        ax.set_title(title, fontsize=10, pad=6)
        if annotate:
            norm = (mat - vmin) / (vmax - vmin)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    txt = f"{mat[i, j]:.2f}" if metric == "pearson" else f"{mat[i, j]:.3g}"
                    if show_std and std is not None:
                        txt += f"\n±{std[i, j]:.2f}"
                    dark = norm[i, j] > 0.5 if _SIMILARITY_CMAPS[metric] == "viridis" else norm[i, j] < 0.5
                    ax.text(j, i, txt, ha="center", va="center", fontsize=5.5 if k > 8 else 7,
                            color="black" if dark else "white")
        _add_small_colorbar(fig, ax, im, ticks=[vmin, vmax])
    fig.suptitle(f"{base.parcellation} | condition similarity | {similarity['label']}", fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


# --- Figure 3: subject x condition correlation structure --------------------------------------------------
def _row_normalize(x):
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def compute_subject_condition_correlations(
    base,
    conditions,
    partition="val",
    demean=None,
    demean_partition="train",
    order_by="original",
    block_by="subject",
    max_subjects=None,
):
    """Pearson r between every (subject, condition) edge vector in a partition.

    demean: None (raw), 'condition' (subtract each condition's `demean_partition` mean), or 'pooled' (subtract
        one mean over all conditions, keeping condition offsets).
    order_by: subject ordering (`HCP_Base.subject_order`); block_by: 'subject' (rows = subject-major blocks of
        conditions) or 'condition' (condition-major blocks of subjects).
    Returns corr (SK x SK in display order), the within/between-subject K x K condition-pair means, and a summary.
    """
    if demean not in (None, "condition", "pooled"):
        raise ValueError("demean must be None, 'condition', or 'pooled'")
    if block_by not in ("subject", "condition"):
        raise ValueError("block_by must be 'subject' or 'condition'")
    edges = condition_upper_triangles(base, conditions)
    indices = _partition_indices(base, partition)
    if max_subjects is not None:
        indices = indices[: int(max_subjects)]
    indices = indices[base.subject_order(indices, order_by)]
    n_s, n_k = len(indices), len(conditions)

    stack = np.stack([np.asarray(edges[c][indices], dtype=np.float32) for c in conditions], axis=1)  # (S, K, E)
    if demean == "condition":
        ref = _partition_indices(base, demean_partition)
        stack = stack - np.stack([edges[c][ref].mean(axis=0) for c in conditions])[None].astype(np.float32)
    elif demean == "pooled":
        ref = _partition_indices(base, demean_partition)
        stack = stack - np.mean([edges[c][ref].mean(axis=0) for c in conditions], axis=0)[None, None].astype(np.float32)

    flat = _row_normalize(stack.reshape(n_s * n_k, -1).astype(np.float64))
    corr_sk = (flat @ flat.T).reshape(n_s, n_k, n_s, n_k)
    within = np.einsum("ikil->kl", corr_sk) / n_s
    total = corr_sk.sum(axis=(0, 2))
    between = (total - np.einsum("ikil->kl", corr_sk)) / (n_s * (n_s - 1))
    off = ~np.eye(n_k, dtype=bool)
    summary = pd.Series({
        "within-subject, cross-condition": float(within[off].mean()) if n_k > 1 else np.nan,
        "between-subject, same condition": float(np.diag(between).mean()),
        "between-subject, cross-condition": float(between[off].mean()) if n_k > 1 else np.nan,
    })
    summary["within - between (cross-condition)"] = summary.iloc[0] - summary.iloc[2]

    if block_by == "subject":
        corr = corr_sk.reshape(n_s * n_k, n_s * n_k)
    else:
        corr = corr_sk.transpose(1, 0, 3, 2).reshape(n_k * n_s, n_k * n_s)
    return {
        "corr": corr,
        "within": within,
        "between": between,
        "summary": summary,
        "conditions": list(conditions),
        "subject_ids": base.metadata_df["subject"].iloc[indices].tolist(),
        "partition": partition,
        "order_by": order_by,
        "block_by": block_by,
        "demean": demean,
        "demean_partition": demean_partition if demean else None,
    }


def _corrmap_limits(corr):
    vals = corr[~np.eye(corr.shape[0], dtype=bool)]
    absmax = float(np.nanmax(np.abs(vals)))
    return (-absmax, absmax) if np.isfinite(absmax) and absmax > 0 else (-1.0, 1.0)


def _draw_corrmap(fig, ax, res, scale_mode):
    corr = res["corr"]
    vmin, vmax = (-1.0, 1.0) if scale_mode == "fixed" else _corrmap_limits(corr)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    n_s, n_k = len(res["subject_ids"]), len(res["conditions"])
    block, n_blocks = (n_k, n_s) if res["block_by"] == "subject" else (n_s, n_k)
    if n_blocks <= 60:
        for b in range(1, n_blocks):
            ax.axhline(b * block - 0.5, color="k", lw=0.3, alpha=0.5)
            ax.axvline(b * block - 0.5, color="k", lw=0.3, alpha=0.5)
    if res["block_by"] == "condition":
        ticks = [(b + 0.5) * n_s - 0.5 for b in range(n_k)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([_condition_label(c) for c in res["conditions"]], rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels([_condition_label(c) for c in res["conditions"]], fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"subject × condition ({n_s} × {n_k})", fontsize=9)
    s = res["summary"]
    demean_label = "raw" if res["demean"] is None else f"demeaned ({res['demean']}, {res['demean_partition']} mean)"
    ax.set_title(
        f"{demean_label}\nwithin-subj x-cond r={s.iloc[0]:.3f} | between-subj same-cond r={s.iloc[1]:.3f}\n"
        f"between-subj x-cond r={s.iloc[2]:.3f}",
        fontsize=8.5, pad=5,
    )
    _add_small_colorbar(fig, ax, im, ticks=[vmin, 0.0, vmax])


def plot_subject_condition_corrmap(
    base,
    conditions,
    partition="val",
    order_by="original",
    block_by="subject",
    demean="condition",
    demean_partition="train",
    max_subjects=None,
    scale_mode="per_plot",
    show_pair_matrices=True,
    figsize=(13, 6.5),
    dpi=150,
    show=True,
):
    """Raw and demeaned subject x condition correlation maps side by side, plus within/between-subject
    condition-pair matrices (second figure) when `show_pair_matrices`.

    scale_mode: 'per_plot' (symmetric off-diagonal max per panel) or 'fixed' ([-1, 1]).
    """
    common = dict(partition=partition, order_by=order_by, block_by=block_by, max_subjects=max_subjects,
                  demean_partition=demean_partition)
    raw = compute_subject_condition_correlations(base, conditions, demean=None, **common)
    dem = compute_subject_condition_correlations(base, conditions, demean=demean, **common)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for ax, res in zip(axes, (raw, dem)):
        _draw_corrmap(fig, ax, res, scale_mode)
    part_label = _PARTITION_DISPLAY.get(partition, partition)
    fig.suptitle(
        f"{base.parcellation} | {part_label} (n={len(raw['subject_ids'])}) | subject × condition FC correlation | "
        f"order={order_by}, blocks={block_by}",
        fontsize=11,
    )
    fig.tight_layout()

    pair_fig = None
    if show_pair_matrices:
        labels = [_condition_label(c) for c in conditions]
        k = len(labels)
        pair_fig, pair_axes = plt.subplots(1, 4, figsize=(18, 4.4), dpi=dpi)
        panels = [
            (raw["within"], "within-subject (raw)"), (raw["between"], "between-subject (raw)"),
            (dem["within"], "within-subject (demeaned)"), (dem["between"], "between-subject (demeaned)"),
        ]
        for ax, (mat, title) in zip(pair_axes, panels):
            m = mat.copy()
            if title.startswith("within"):
                np.fill_diagonal(m, np.nan)  # same subject, same condition = 1 by construction
            vals = m[np.isfinite(m)]
            vmin, vmax = float(vals.min()), float(vals.max())
            im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks(range(k))
            ax.set_yticks(range(k))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(f"mean r, {title}", fontsize=9)
            for i in range(k):
                for j in range(k):
                    if np.isfinite(m[i, j]):
                        dark = (m[i, j] - vmin) / (vmax - vmin + 1e-12) > 0.5
                        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5 if k > 8 else 6.5,
                                color="black" if dark else "white")
            _add_small_colorbar(pair_fig, ax, im, ticks=[vmin, vmax])
        pair_fig.suptitle(f"{base.parcellation} | {part_label} | condition-pair mean correlations", fontsize=11)
        pair_fig.tight_layout()

    if show:
        plt.show()
    summary = pd.DataFrame({"raw": raw["summary"], f"demeaned ({demean})": dem["summary"]})
    return fig, pair_fig, {"raw": raw, "demeaned": dem, "summary": summary}
