"""Evaluation support helpers.

Contains non-metric helpers used by evaluation analyses.
"""

import numpy as np
import torch
import io
import warnings
import matplotlib.pyplot as plt
from PIL import Image


def generate_mean_baseline(train_mean, n_subjects):
    """Generate a baseline predictor that repeats the training-set mean."""
    train_mean_np = train_mean.detach().cpu().numpy() if isinstance(train_mean, torch.Tensor) else np.asarray(train_mean)
    return np.tile(train_mean_np, (n_subjects, 1))


def generate_noise_baseline(train_mean, train_std, n_subjects, seed=None):
    """Generate a training-mean plus independent Gaussian noise baseline."""
    train_mean_np = train_mean.detach().cpu().numpy() if isinstance(train_mean, torch.Tensor) else np.asarray(train_mean)
    train_std_np = train_std.detach().cpu().numpy() if isinstance(train_std, torch.Tensor) else np.asarray(train_std)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(n_subjects, train_mean_np.shape[0])) * train_std_np
    return train_mean_np + noise


def extract_violin_metrics(results):
    """Extract wandb-friendly metrics from violin plot results."""
    extracted = {}
    for condition, stats in results.items():
        prefix = condition.lower().replace(" ", "_").replace("(", "").replace(")", "")
        extracted[f"{prefix}_mean_r_intra"] = float(stats["mean_r_intra"])
        extracted[f"{prefix}_mean_r_inter"] = float(stats["mean_r_inter"])
        extracted[f"{prefix}_mean_d"] = float(stats["mean_d"])
        extracted[f"{prefix}_t_stat"] = float(stats["t_stat"])
        extracted[f"{prefix}_p_value"] = float(stats["p_value"])
        extracted[f"{prefix}_cohen_d"] = float(stats["cohen_d"])
    return extracted


def extract_hungarian_metrics(results):
    """Extract wandb-friendly metrics from Hungarian matching results."""
    extracted = {}
    for condition, hung in results.items():
        prefix = condition.lower().replace(" ", "_").replace("(", "").replace(")", "")
        extracted[f"{prefix}_accuracy"] = float(hung["accuracy"])
        extracted[f"{prefix}_n_correct"] = int(hung["n_correct"])
    return extracted


def extract_sample_size_metrics(results):
    """Extract wandb-friendly metrics from sample size analysis results."""
    return {
        "sample_sizes": results["sample_sizes"].tolist(),
        "pfc_mean_accuracy": results["pFC"]["mean"].tolist(),
        "null_noise_mean_accuracy": results["Null (noise)"]["mean"].tolist(),
        "null_permute_mean_accuracy": results["Null (permute)"]["mean"].tolist(),
        "n_significant_vs_permute": int(np.sum(results["significant_permute"])),
        "n_significant_vs_noise": int(np.sum(results["significant_noise"])),
    }


def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_to_height(img, target_height):
    """Resize image to target height while preserving aspect ratio."""
    if img.height == target_height:
        return img
    ratio = target_height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, target_height), Image.Resampling.LANCZOS)


def resize_to_width(img, target_width):
    """Resize image to target width while preserving aspect ratio."""
    if img.width == target_width:
        return img
    ratio = target_width / img.width
    new_height = int(img.height * ratio)
    return img.resize((target_width, new_height), Image.Resampling.LANCZOS)


def create_title_banner(title, width, height, bg_color, text_color):
    """Create a title banner image using matplotlib for reliable font rendering."""
    bg_color_mpl = tuple(c / 255 for c in bg_color)
    text_color_mpl = tuple(c / 255 for c in text_color)

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=bg_color_mpl)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg_color_mpl)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        title,
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        color=text_color_mpl,
        ha="center",
        va="center",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=bg_color_mpl, edgecolor="none", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.width != width or img.height != height:
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img


def generate_report(figures_to_combine, figure_labels, filepath):
    """Generate a combined JPEG report from collected figures."""
    section_spacing = 120
    title_height = 100
    title_bg_color = (230, 230, 235)
    title_text_color = (40, 40, 50)

    images = []
    for layout_type, figs in figures_to_combine:
        if layout_type == "single":
            images.append(fig_to_image(figs))
        elif layout_type == "side_by_side":
            left_img = fig_to_image(figs[0]) if figs[0] is not None else None
            right_img = fig_to_image(figs[1]) if figs[1] is not None else None
            if left_img is not None and right_img is not None:
                max_height = max(left_img.height, right_img.height)
                left_img = resize_to_height(left_img, max_height)
                right_img = resize_to_height(right_img, max_height)
                gap = 20
                combined = Image.new("RGB", (left_img.width + gap + right_img.width, max_height), (255, 255, 255))
                combined.paste(left_img, (0, 0))
                combined.paste(right_img, (left_img.width + gap, 0))
                images.append(combined)
            elif left_img is not None:
                images.append(left_img)
            elif right_img is not None:
                images.append(right_img)
        elif layout_type == "stacked":
            top_img = fig_to_image(figs[0]) if figs[0] is not None else None
            bottom_img = fig_to_image(figs[1]) if figs[1] is not None else None
            if top_img is not None and bottom_img is not None:
                max_width = max(top_img.width, bottom_img.width)
                top_img = resize_to_width(top_img, max_width)
                bottom_img = resize_to_width(bottom_img, max_width)
                gap = 15
                combined = Image.new("RGB", (max_width, top_img.height + gap + bottom_img.height), (255, 255, 255))
                combined.paste(top_img, (0, 0))
                combined.paste(bottom_img, (0, top_img.height + gap))
                images.append(combined)
            elif top_img is not None:
                images.append(top_img)
            elif bottom_img is not None:
                images.append(bottom_img)

    if not images:
        warnings.warn("No images to combine for report")
        return

    max_width = max(img.width for img in images)
    resized_images = [resize_to_width(img, max_width) for img in images]
    title_images = [
        create_title_banner(
            figure_labels[idx] if idx < len(figure_labels) else f"Section {idx + 1}",
            max_width,
            title_height,
            title_bg_color,
            title_text_color,
        )
        for idx in range(len(resized_images))
    ]

    n_sections = len(resized_images)
    total_height = (
        sum(img.height for img in resized_images)
        + sum(img.height for img in title_images)
        + (n_sections - 1) * section_spacing
    )

    final_report = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for idx, img in enumerate(resized_images):
        final_report.paste(title_images[idx], (0, y_offset))
        y_offset += title_images[idx].height
        final_report.paste(img, (0, y_offset))
        y_offset += img.height
        if idx < n_sections - 1:
            y_offset += section_spacing

    if not filepath.lower().endswith(".jpg") and not filepath.lower().endswith(".jpeg"):
        filepath = filepath + ".jpg"

    final_report.save(filepath, "JPEG", quality=95)
    print(f"Report saved to: {filepath}")


def format_metrics_table(metrics_dict, columns, headers=None):
    """Format selected metrics as a markdown table."""
    if headers is None:
        headers = columns

    rows = ["| Metric | " + " | ".join(headers) + " |", "|--------|" + "|".join(["-------"] * len(columns)) + "|"]
    metric_names = {
        "mean_corr": "Mean Corr",
        "top1_acc": "Top-1 Accuracy",
        "avg_rank_percentile": "Avg Rank %ile",
        "pfc_mean_r_intra": "pFC r_intra",
        "pfc_mean_r_inter": "pFC r_inter",
        "pfc_cohen_d": "pFC Cohen's d",
        "pfc_p_value": "pFC p-value",
        "null_mean_r_intra": "Null r_intra",
        "null_mean_r_inter": "Null r_inter",
        "pfc_accuracy": "pFC Accuracy",
        "null_noise_accuracy": "Null (noise) Accuracy",
        "null_permute_accuracy": "Null (permute) Accuracy",
    }

    important_metrics = ["mean_corr", "top1_acc", "avg_rank_percentile"]
    for metric in important_metrics:
        if any(metric in metrics_dict.get(col, {}) for col in columns if isinstance(metrics_dict.get(col), dict)):
            display_name = metric_names.get(metric, metric)
            values = []
            for col in columns:
                if col in metrics_dict and isinstance(metrics_dict[col], dict):
                    val = metrics_dict[col].get(metric, "-")
                    if isinstance(val, float):
                        val = f"{val:.3f}"
                    values.append(str(val))
                else:
                    values.append("-")
            rows.append(f"| {display_name} | " + " | ".join(values) + " |")

    return "\n".join(rows)
