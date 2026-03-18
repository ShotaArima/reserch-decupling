"""Scenario 4: two-stage pipeline (recovery -> forecasting)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from src.data import FreshRetailConfig, coerce_numeric_columns, load_freshretail_dataframe, normalize_columns
from src.metrics import wape
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead
from src.plotting import save_learning_curve

FEATURES = ["sale_amount", "hours_stock_status", "discount", "holiday_flag", "activity_flag"]


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = coerce_numeric_columns(df, FEATURES)
    df = df.dropna(subset=FEATURES)
    x = torch.tensor(normalize_columns(df, FEATURES)[FEATURES].values, dtype=torch.float32)

    # Stage 1: latent demand recovery
    recovery = DecouplingAutoEncoder(DecouplingConfig(input_dim=len(FEATURES)))
    rec_opt = torch.optim.Adam(recovery.parameters(), lr=1e-3)
    stage1_losses: list[float] = []

    for _ in range(100):
        print(f'stage1-step: {_}')
        rec, local, global_latent = recovery(x)
        loss = nn.functional.mse_loss(rec, x)
        rec_opt.zero_grad()
        loss.backward()
        rec_opt.step()
        stage1_losses.append(loss.item())

    # Stage 2: 7-day forecasting from recovered demand signal proxy
    y = x[:, :1]
    forecaster = ForecastHead(local_dim=16, global_dim=16, horizon=1)
    f_opt = torch.optim.Adam(forecaster.parameters(), lr=1e-3)
    stage2_losses: list[float] = []

    for _ in range(100):
        print(f'stage2-step: {_}')
        pred = forecaster(local.detach(), global_latent.detach())
        f_loss = nn.functional.l1_loss(pred, y)
        f_opt.zero_grad()
        f_loss.backward()
        f_opt.step()
        stage2_losses.append(f_loss.item())

    print(f"stage1_recovery_mse={loss.item():.6f}")
    print(f"stage2_wape={wape(y.numpy().reshape(-1), pred.detach().numpy().reshape(-1)):.4f}")
    stage1_curve = save_learning_curve(
        stage1_losses,
        SCENARIO_DIR / "stage1_train_loss_curve.png",
        title="Scenario 4 Stage 1 Train Loss Curve",
        ylabel="MSE Loss",
    )
    stage2_curve = save_learning_curve(
        stage2_losses,
        SCENARIO_DIR / "stage2_train_loss_curve.png",
        title="Scenario 4 Stage 2 Train Loss Curve",
        ylabel="L1 Loss",
    )
    print(f"Saved stage1 loss curve to: {stage1_curve}")
    print(f"Saved stage2 loss curve to: {stage2_curve}")


if __name__ == "__main__":
    main()
