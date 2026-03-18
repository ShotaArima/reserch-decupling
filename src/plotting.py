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
