from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image


__all__ = [
    "distance_matrix_from_centroids",
    "compute_overview_ranges",
    "get_validation_overview_matrices",
    "plot_validation_matrix_overview",
    "make_validation_matrix_gif",
]


def distance_matrix_from_centroids(centroids):
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)


def _resolve_val_global_index(base, val_position=None, subject_id=None):
    val_indices = np.asarray(base.trainvaltest_partition_indices["val"])
    if subject_id is not None:
        matches = np.where(base.metadata_df["subject"].astype(str).values == str(subject_id))[0]
        if len(matches) == 0:
            raise ValueError(f"subject_id {subject_id} not found in metadata_df")
        global_idx = int(matches[0])
        if global_idx not in set(val_indices.tolist()):
            raise ValueError(f"subject_id {subject_id} is not part of the validation set")
        return global_idx
    if val_position is None:
        val_position = 0
    if val_position < 0 or val_position >= len(val_indices):
        raise IndexError(f"val_position must be in [0, {len(val_indices) - 1}]")
    return int(val_indices[val_position])


def _aggregate_stack(arr, agg="mean"):
    if agg == "mean":
        return np.mean(arr, axis=0)
    if agg == "median":
        return np.median(arr, axis=0)
    raise ValueError(f"Unsupported aggregate={agg!r}; use None, 'mean', or 'median'")


def compute_overview_ranges(base, val_indices=None, distance_quantile=0.99, sc_quantile=0.99):
    if val_indices is None:
        val_indices = np.asarray(base.trainvaltest_partition_indices["val"])
    sc_subset = base.sc_matrices[val_indices]
    fc_subset = base.fc_matrices[val_indices]
    dist_subset = np.stack([distance_matrix_from_centroids(base.parcel_centroids[i]) for i in val_indices], axis=0)

    sc_vmax = float(np.quantile(sc_subset, sc_quantile))
    dist_vmax = float(np.quantile(dist_subset, distance_quantile))
    fc_abs = float(np.quantile(np.abs(fc_subset), 0.995))

    return {
        "SC": {"cmap": "viridis", "vmin": 0.0, "vmax": sc_vmax},
        "SC_r2t_corr": {"cmap": "viridis", "vmin": -1.0, "vmax": 1.0},
        "distance": {"cmap": "viridis_r", "vmin": 0.0, "vmax": dist_vmax},
        "FC": {"cmap": "RdBu_r", "vmin": -fc_abs, "vmax": fc_abs},
    }


def get_validation_overview_matrices(base, val_position=None, subject_id=None, aggregate=None):
    val_indices = np.asarray(base.trainvaltest_partition_indices["val"])
    if aggregate is None:
        global_idx = _resolve_val_global_index(base, val_position=val_position, subject_id=subject_id)
        subject_id_out = str(base.metadata_df.iloc[global_idx]["subject"])
        return {
            "SC": base.sc_matrices[global_idx],
            "SC_r2t_corr": base.sc_r2t_corr_matrices[global_idx],
            "distance": distance_matrix_from_centroids(base.parcel_centroids[global_idx]),
            "FC": base.fc_matrices[global_idx],
            "subject_id": subject_id_out,
            "label": f"Validation subject {subject_id_out}",
        }

    aggregate = str(aggregate).lower()
    if aggregate not in {"mean", "median"}:
        raise ValueError("aggregate must be one of {None, 'mean', 'median'}")

    return {
        "SC": _aggregate_stack(base.sc_matrices[val_indices], aggregate),
        "SC_r2t_corr": _aggregate_stack(base.sc_r2t_corr_matrices[val_indices], aggregate),
        "distance": _aggregate_stack(
            np.stack([distance_matrix_from_centroids(base.parcel_centroids[i]) for i in val_indices], axis=0),
            aggregate,
        ),
        "FC": _aggregate_stack(base.fc_matrices[val_indices], aggregate),
        "subject_id": None,
        "label": f"Validation {aggregate}",
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
    if data.get("subject_id") is None:
        return f"{base.parcellation} | {data['label']}"

    header_metadata = None if header_metadata is None else str(header_metadata).lower()
    if header_metadata not in {None, "age_sex", "age_sex_ethnicity"}:
        raise ValueError("header_metadata must be one of {None, 'age_sex', 'age_sex_ethnicity'}")

    subject_id = str(data["subject_id"])
    parts = [f"{base.parcellation} | Validation subject {subject_id}"]
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


def plot_validation_matrix_overview(
    base,
    val_position=None,
    subject_id=None,
    aggregate=None,
    ranges=None,
    figsize=(14, 3.8),
    dpi=180,
    show=True,
    suptitle=None,
    header_metadata=None,
):
    data = get_validation_overview_matrices(base, val_position=val_position, subject_id=subject_id, aggregate=aggregate)
    ranges = compute_overview_ranges(base) if ranges is None else ranges

    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.34, top=0.86, bottom=0.12)

    panel_order = [
        ("SC", "SC"),
        ("SC_r2t_corr", "SC_r2t corr"),
        ("distance", "Distance"),
        ("FC", "FC"),
    ]

    for ax, (key, title) in zip(axes, panel_order):
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
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        if key == "SC":
            ax.set_xlabel("region", fontsize=10)
            ax.set_ylabel("region", fontsize=10)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

        ticks = None
        if key in {"SC_r2t_corr", "FC"}:
            ticks = [spec["vmin"], 0.0, spec["vmax"]]
        elif key in {"SC", "distance"}:
            ticks = [spec["vmin"], spec["vmax"]]
        _add_small_colorbar(fig, ax, im, ticks=ticks)

    final_title = suptitle or _format_subject_header(base, data, header_metadata=header_metadata)
    fig.suptitle(final_title, fontsize=14, y=0.92)

    if show:
        plt.show()
    return fig, axes, data


def make_validation_matrix_gif(
    base,
    gif_path="validation_matrix_overview.gif",
    n_subjects=10,
    start_val_position=0,
    duration_ms=600,
    tmpdir="tmp_validation_matrix_frames",
    figsize=(14, 3.8),
    dpi=180,
    cleanup_tmp=True,
    header_metadata=None,
):
    val_indices = np.asarray(base.trainvaltest_partition_indices["val"])
    start_val_position = int(start_val_position)
    end_pos = min(start_val_position + int(n_subjects), len(val_indices))
    chosen_positions = list(range(start_val_position, end_pos))
    if not chosen_positions:
        raise ValueError("No validation subjects selected for GIF generation.")

    ranges = compute_overview_ranges(base, val_indices=val_indices)
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    frames = []

    for frame_idx, val_pos in enumerate(chosen_positions):
        subject_data = get_validation_overview_matrices(base, val_position=val_pos)
        fig, axes, data = plot_validation_matrix_overview(
            base,
            val_position=val_pos,
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
