"""Scenario 4: two-stage pipeline (recovery -> forecasting)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from src.data import FreshRetailConfig, coerce_numeric_columns, load_freshretail_dataframe, normalize_columns
from src.metrics import wape
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead

FEATURES = ["sale_amount", "hours_stock_status", "discount", "holiday_flag", "activity_flag"]


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = coerce_numeric_columns(df, FEATURES)
    df = df.dropna(subset=FEATURES)
    x = torch.tensor(normalize_columns(df, FEATURES)[FEATURES].values, dtype=torch.float32)

    # Stage 1: latent demand recovery
    recovery = DecouplingAutoEncoder(DecouplingConfig(input_dim=len(FEATURES)))
    rec_opt = torch.optim.Adam(recovery.parameters(), lr=1e-3)

    for _ in range(5):
        rec, local, global_latent = recovery(x)
        loss = nn.functional.mse_loss(rec, x)
        rec_opt.zero_grad()
        loss.backward()
        rec_opt.step()

    # Stage 2: 7-day forecasting from recovered demand signal proxy
    y = x[:, :1]
    forecaster = ForecastHead(local_dim=16, global_dim=16, horizon=1)
    f_opt = torch.optim.Adam(forecaster.parameters(), lr=1e-3)

    for _ in range(5):
        pred = forecaster(local.detach(), global_latent.detach())
        f_loss = nn.functional.l1_loss(pred, y)
        f_opt.zero_grad()
        f_loss.backward()
        f_opt.step()

    print(f"stage1_recovery_mse={loss.item():.6f}")
    print(f"stage2_wape={wape(y.numpy().reshape(-1), pred.detach().numpy().reshape(-1)):.4f}")


if __name__ == "__main__":
    main()
