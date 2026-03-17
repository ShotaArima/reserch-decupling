"""Scenario 2: raw daily sales 7-day forecasting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data import FreshRetailConfig, load_freshretail_dataframe, normalize_columns
from src.metrics import wape, wpe
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead

INPUT_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
TARGET = "sale_amount"
HORIZON = 7


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = df.dropna(subset=INPUT_FEATURES)
    arr = normalize_columns(df, INPUT_FEATURES)[INPUT_FEATURES].values

    x = torch.tensor(arr[:-HORIZON], dtype=torch.float32)
    y = torch.tensor(df[TARGET].values[HORIZON:], dtype=torch.float32).unsqueeze(-1)

    body = DecouplingAutoEncoder(DecouplingConfig(input_dim=len(INPUT_FEATURES)))
    head = ForecastHead(local_dim=16, global_dim=16, horizon=1)
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.L1Loss()

    for _ in range(5):
        _, local, global_latent = body(x)
        pred = head(local, global_latent)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    y_pred = pred.detach().cpu().numpy().reshape(-1)
    y_true = y.detach().cpu().numpy().reshape(-1)
    print(f"WAPE={wape(y_true, y_pred):.4f}")
    print(f"WPE={wpe(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()
