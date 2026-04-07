"""Plot helpers for scenario scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_learning_curve(
    losses: Sequence[float],
    out_path: str | Path,
    *,
    title: str,
    ylabel: str = "Train Loss",
) -> Path:
    """Save a simple train-loss curve and return the output path."""
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_metric_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    out_path: str | Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = list(range(len(labels)))
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_difference_histogram(
    global_diff,
    local_diff,
    out_path: str | Path,
    *,
    title: str,
    bins: int = 40,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(global_diff, bins=bins, alpha=0.6, label="global_swap - original")
    ax.hist(local_diff, bins=bins, alpha=0.6, label="local_swap - original")
    ax.set_title(title)
    ax.set_xlabel("Difference (denormalized sale)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_sample_series_plot(
    original,
    global_swap,
    local_swap,
    out_path: str | Path,
    *,
    title: str,
    max_samples: int = 12,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    num_samples = min(len(original), max_samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(9, 2.2 * num_samples), sharex=True)
    if num_samples == 1:
        axes = [axes]

    for idx in range(num_samples):
        axes[idx].plot(original[idx], label="original", linewidth=2)
        axes[idx].plot(global_swap[idx], label="global_swap", linewidth=1.5)
        axes[idx].plot(local_swap[idx], label="local_swap", linewidth=1.5)
        axes[idx].set_ylabel(f"s{idx}")
        axes[idx].grid(alpha=0.25)

    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Time step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

def save_condition_series_examples(
    y_true: np.ndarray,
    pred_map: dict[str, np.ndarray],
    segment_ranges: Sequence[tuple[int, int]],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(segment_ranges), 1, figsize=(10, 3.2 * len(segment_ranges)), sharex=False)
    if len(segment_ranges) == 1:
        axes = [axes]

    for ax, (start, end) in zip(axes, segment_ranges):
        x = np.arange(start, end)
        ax.plot(x, y_true[start:end], label="y_true", linewidth=2.2)
        for name, pred in pred_map.items():
            ax.plot(x, pred[start:end], label=name, linewidth=1.5)
        ax.set_ylabel(f"[{start}:{end})")
        ax.grid(alpha=0.25)

    axes[0].legend(ncol=min(4, len(pred_map) + 1), loc="upper right")
    axes[-1].set_xlabel("Sample index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

def save_residual_histogram(
    residual_map: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: str,
    bins: int = 50,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, residual in residual_map.items():
        ax.hist(residual, bins=bins, alpha=0.45, label=name)
    ax.set_title(title)
    ax.set_xlabel("Residual (y_pred - y_true)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

def save_residual_boxplot(
    residual_map: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = list(residual_map.keys())
    values = [residual_map[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel("Residual (y_pred - y_true)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_residual_scatter(
    y_true: np.ndarray,
    pred_map: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: str,
    max_points: int = 3000,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    n = y_true.shape[0]
    if n > max_points:
        indices = np.linspace(0, n - 1, num=max_points, dtype=int)
    else:
        indices = np.arange(n)

    fig, axes = plt.subplots(1, len(pred_map), figsize=(5 * len(pred_map), 4), sharey=True)
    if len(pred_map) == 1:
        axes = [axes]

    for ax, (name, pred) in zip(axes, pred_map.items()):
        ax.scatter(y_true[indices], (pred - y_true)[indices], s=6, alpha=0.35)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("y_true")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Residual (y_pred - y_true)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

def save_probe_heatmap(
    matrix,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.9), max(4, len(row_labels) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

def save_probe_heatmap(
    matrix,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.9), max(4, len(row_labels) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

def save_swap_direction_plot(
    probe_names: Sequence[str],
    exp1_gap: Sequence[float],
    exp2_gap: Sequence[float],
    out_path: str | Path,
    *,
    title: str,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(len(probe_names)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(probe_names) * 0.9), 4.5))
    ax.bar([i - width / 2 for i in x], exp1_gap, width=width, label="exp1 (common-specific)")
    ax.bar([i + width / 2 for i in x], exp2_gap, width=width, label="exp2 (common-specific)")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=30, ha="right")
    ax.set_ylabel("Gap")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output