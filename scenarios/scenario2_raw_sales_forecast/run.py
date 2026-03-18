"""Scenario 2: raw daily sales 7-day forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent

import numpy as np
import torch
from torch import nn

from src.data import FreshRetailConfig, load_freshretail_dataframe, normalize_columns, coerce_numeric_columns
from src.metrics import wape, wpe
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead
from src.plotting import save_learning_curve

INPUT_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
TARGET = "sale_amount"
HORIZON = 7


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = coerce_numeric_columns(df, INPUT_FEATURES)
    df = df.dropna(subset=INPUT_FEATURES)
    arr = normalize_columns(df, INPUT_FEATURES)[INPUT_FEATURES].values

    x = torch.tensor(arr[:-HORIZON], dtype=torch.float32)
    y = torch.tensor(df[TARGET].values[HORIZON:], dtype=torch.float32).unsqueeze(-1)

    body = DecouplingAutoEncoder(DecouplingConfig(input_dim=len(INPUT_FEATURES)))
    head = ForecastHead(local_dim=16, global_dim=16, horizon=1)
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.L1Loss()
    losses: list[float] = []

    for _ in range(100):
        print(f"step {_}")
        _, local, global_latent = body(x)
        pred = head(local, global_latent)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    y_pred = pred.detach().cpu().numpy().reshape(-1)
    y_true = y.detach().cpu().numpy().reshape(-1)
    print(f"WAPE={wape(y_true, y_pred):.4f}")
    print(f"WPE={wpe(y_true, y_pred):.4f}")

    curve_path = save_learning_curve(
        losses,
        SCENARIO_DIR / "train_loss_curve.png",
        title="Scenario 2 Train Loss Curve",
    )
    print(f"Saved train loss curve to: {curve_path}")


if __name__ == "__main__":
    main()
