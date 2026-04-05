"""Plot helpers for scenario scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
