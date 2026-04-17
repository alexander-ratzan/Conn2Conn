import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch

from data.data_utils import tri2square
from models.architectures.crossmodal_pca_pls import CrossModalPCA
from models.architectures.krakencoder_precomputed import KrakencoderPrecomputed

__all__ = [
    "resolve_subject_id_and_indices",
    "fit_pca_models",
    "predict_single_subject",
    "plot_ground_truth_demeaning",
    "plot_demeaned_predictions",
    "demeaned_pearson_r",
    "vector_to_fc_square",
]


def resolve_subject_id_and_indices(base_sc, base_fc, val_position=None, subject_id=None):
    val_indices_sc = np.asarray(base_sc.trainvaltest_partition_indices["val"])
    val_subject_ids = base_sc.metadata_df.iloc[val_indices_sc]["subject"].astype(str).tolist()

    if subject_id is None:
        if val_position is None:
            val_position = 0
        subject_id = val_subject_ids[int(val_position)]
    else:
        subject_id = str(subject_id)
        if subject_id not in val_subject_ids:
            raise ValueError(f"subject_id {subject_id} is not in the validation set")

    global_idx_sc = int(np.where(base_sc.metadata_df["subject"].astype(str).values == subject_id)[0][0])
    global_idx_fc = int(np.where(base_fc.metadata_df["subject"].astype(str).values == subject_id)[0][0])
    val_pos = val_subject_ids.index(subject_id)
    return subject_id, val_pos, global_idx_sc, global_idx_fc


def _small_cbar(fig, ax, im, ticks=None):
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


def demeaned_pearson_r(y_true, y_pred, mean_vec):
    yt = np.asarray(y_true, dtype=np.float64) - np.asarray(mean_vec, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64) - np.asarray(mean_vec, dtype=np.float64)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return float(np.corrcoef(yt, yp)[0, 1])


def vector_to_fc_square(vec, numroi, diagval=1.0):
    return tri2square(np.asarray(vec, dtype=np.float32), numroi=numroi, diagval=diagval)


def fit_pca_models(base_sc, base_fc, num_components=256, kraken_predictions_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    null_model = CrossModalPCA(base_sc, num_components=num_components, device=device).eval()
    oracle_model = CrossModalPCA(base_fc, num_components=num_components, device=device).eval()
    kraken_model = KrakencoderPrecomputed(base_sc, kraken_predictions_dir=kraken_predictions_dir)
    return null_model, oracle_model, kraken_model, device


def predict_single_subject(null_model, oracle_model, kraken_model, base_sc, base_fc, global_idx_sc, global_idx_fc, device):
    x_sc = torch.tensor(base_sc.sc_upper_triangles[global_idx_sc:global_idx_sc + 1], dtype=torch.float32, device=device)
    x_fc = torch.tensor(base_fc.fc_upper_triangles[global_idx_fc:global_idx_fc + 1], dtype=torch.float32, device=device)
    with torch.no_grad():
        null_pred = null_model(x_sc).detach().cpu().numpy()[0]
        oracle_pred = oracle_model(x_fc).detach().cpu().numpy()[0]
    kraken_pred = np.asarray(kraken_model._preds_all[global_idx_sc], dtype=np.float32)
    y_true = np.asarray(base_sc.fc_upper_triangles[global_idx_sc], dtype=np.float32)
    return y_true, null_pred, oracle_pred, kraken_pred


def plot_ground_truth_demeaning(base_sc, subject_id, global_idx_sc, figsize=(11.5, 3.8), dpi=180, title_fontsize=15):
    numroi = base_sc.fc_matrices.shape[1]
    subj_fc = np.asarray(base_sc.fc_matrices[global_idx_sc], dtype=np.float32)
    pop_mean_vec = np.asarray(base_sc.fc_train_avg, dtype=np.float32)
    pop_mean_fc = vector_to_fc_square(pop_mean_vec, numroi=numroi, diagval=1.0)
    demeaned_fc = subj_fc - pop_mean_fc
    demeaned_abs = float(np.max(np.abs(demeaned_fc)))
    if demeaned_abs == 0:
        demeaned_abs = 1e-6

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.34, top=0.90, bottom=0.10)

    specs = [
        (f"Subject FC ({subject_id})", subj_fc, "RdBu_r", -1.0, 1.0, [-1.0, 0.0, 1.0]),
        ("Population Mean FC", pop_mean_fc, "RdBu_r", -1.0, 1.0, [-1.0, 0.0, 1.0]),
        ("Demeaned FC", demeaned_fc, "RdBu_r", -demeaned_abs, demeaned_abs, [-demeaned_abs, 0.0, demeaned_abs]),
    ]

    for ax, (title, mat, cmap, vmin, vmax, ticks) in zip(axes, specs):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=title_fontsize, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        _small_cbar(fig, ax, im, ticks=ticks)

    plt.show()
    return fig, axes


def plot_demeaned_predictions(
    base_sc,
    subject_id,
    y_true,
    kraken_pred,
    null_pred,
    oracle_pred,
    include_ground_truth=False,
    figsize=None,
    dpi=180,
    shared_range=False,
    show_suptitle=False,
    title_fontsize=15,
    suptitle_fontsize=14,
):
    numroi = base_sc.fc_matrices.shape[1]
    mean_vec = np.asarray(base_sc.fc_train_avg, dtype=np.float32)

    gt_dm = vector_to_fc_square(y_true - mean_vec, numroi=numroi, diagval=0.0)
    kraken_dm = vector_to_fc_square(kraken_pred - mean_vec, numroi=numroi, diagval=0.0)
    null_dm = vector_to_fc_square(null_pred - mean_vec, numroi=numroi, diagval=0.0)
    oracle_dm = vector_to_fc_square(oracle_pred - mean_vec, numroi=numroi, diagval=0.0)

    panel_data = [
        ("Krakencoder", kraken_dm, demeaned_pearson_r(y_true, kraken_pred, mean_vec)),
        ("Null", null_dm, demeaned_pearson_r(y_true, null_pred, mean_vec)),
        ("Oracle", oracle_dm, demeaned_pearson_r(y_true, oracle_pred, mean_vec)),
    ]
    if include_ground_truth:
        panel_data.append(("Ground Truth", gt_dm, None))

    if figsize is None:
        figsize = (4.0 * len(panel_data), 3.8)

    shared_vmax = float(max(np.max(np.abs(mat)) for _, mat, _ in panel_data))
    if shared_vmax == 0:
        shared_vmax = 1e-6

    fig, axes = plt.subplots(1, len(panel_data), figsize=figsize, dpi=dpi)
    if len(panel_data) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.34, top=0.88, bottom=0.20)

    for ax, (label, mat, r_val) in zip(axes, panel_data):
        panel_vmax = shared_vmax if shared_range else float(np.max(np.abs(mat)))
        if panel_vmax == 0:
            panel_vmax = 1e-6
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-panel_vmax, vmax=panel_vmax, aspect="equal", interpolation="nearest")
        label_text = label if r_val is None else f"{label}\nDemeaned r = {r_val:.3f}"
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label_text, fontsize=title_fontsize, labelpad=10)
        _small_cbar(fig, ax, im, ticks=[-panel_vmax, 0.0, panel_vmax])

    if show_suptitle:
        fig.suptitle(
            f"{base_sc.parcellation} | Validation subject {subject_id} demeaned predictions",
            fontsize=suptitle_fontsize,
            y=0.94,
        )
    plt.show()
    return fig, axes
