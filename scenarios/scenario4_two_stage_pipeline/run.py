"""Scenario 4: two-stage pipeline (recovery -> forecasting)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    FreshRetailConfig,
    build_window_tensor,
    extract_last_timestep_feature,
    load_freshretail_dataframe,
    normalize_by_train_stats,
    split_train_valid_test,
)
from src.metrics import wape
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead
from src.plotting import save_learning_curve

FEATURES = ["sale_amount", "hours_stock_status", "discount", "holiday_flag", "activity_flag"]
WINDOW_SIZES = [7, 14]
TARGET_FEATURE_INDEX = 0


def _shift_x_y(x_split: torch.Tensor, y_split: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x_split[:-1], y_split[1:]


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())

    for window_size in WINDOW_SIZES:
        print(f"\n=== Scenario4 / window_size={window_size} ===")
        x_all_raw = build_window_tensor(df, FEATURES, window_size=window_size)
        y_all_raw = extract_last_timestep_feature(x_all_raw, TARGET_FEATURE_INDEX)

        train_raw, valid_raw, test_raw = split_train_valid_test(x_all_raw)
        y_train_raw, y_valid_raw, y_test_raw = split_train_valid_test(y_all_raw)
        train_x, valid_x, test_x = normalize_by_train_stats(train_raw, valid_raw, test_raw)

        x_train = torch.tensor(train_x, dtype=torch.float32)
        x_valid = torch.tensor(valid_x, dtype=torch.float32)
        x_test = torch.tensor(test_x, dtype=torch.float32)

        y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(-1)
        y_valid = torch.tensor(y_valid_raw, dtype=torch.float32).unsqueeze(-1)
        y_test = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(-1)

        recovery = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(FEATURES), window_size=window_size))
        rec_opt = torch.optim.Adam(recovery.parameters(), lr=1e-3)
        stage1_losses: list[float] = []

        for step in range(100):
            rec, _, _ = recovery(x_train)
            loss = nn.functional.mse_loss(rec, x_train)
            rec_opt.zero_grad()
            loss.backward()
            rec_opt.step()
            stage1_losses.append(loss.item())
            if step % 20 == 0:
                print(f"stage1-step={step} train_mse={loss.item():.6f}")

        with torch.no_grad():
            _, local_train, global_train = recovery(x_train)
            _, local_valid, global_valid = recovery(x_valid)
            _, local_test, global_test = recovery(x_test)

        local_train, y_train_s = _shift_x_y(local_train, y_train)
        local_valid, y_valid_s = _shift_x_y(local_valid, y_valid)
        local_test, y_test_s = _shift_x_y(local_test, y_test)
        global_train, _ = _shift_x_y(global_train, y_train)
        global_valid, _ = _shift_x_y(global_valid, y_valid)
        global_test, _ = _shift_x_y(global_test, y_test)

        forecaster = ForecastHead(local_dim=16, global_dim=16, horizon=1)
        f_opt = torch.optim.Adam(forecaster.parameters(), lr=1e-3)
        stage2_losses: list[float] = []

        for step in range(100):
            pred = forecaster(local_train.detach(), global_train.detach())
            f_loss = nn.functional.l1_loss(pred, y_train_s)
            f_opt.zero_grad()
            f_loss.backward()
            f_opt.step()
            stage2_losses.append(f_loss.item())
            if step % 20 == 0:
                print(f"stage2-step={step} train_l1={f_loss.item():.6f}")

        with torch.no_grad():
            pred_valid = forecaster(local_valid, global_valid)
            pred_test = forecaster(local_test, global_test)

        print(f"window={window_size} stage1_train_mse={stage1_losses[-1]:.6f}")
        print(f"window={window_size} stage2_valid_wape={wape(y_valid_s.numpy().reshape(-1), pred_valid.numpy().reshape(-1)):.4f}")
        print(f"window={window_size} stage2_test_wape={wape(y_test_s.numpy().reshape(-1), pred_test.numpy().reshape(-1)):.4f}")

        stage1_curve = save_learning_curve(
            stage1_losses,
            SCENARIO_DIR / f"stage1_train_loss_curve_w{window_size}.png",
            title=f"Scenario 4 Stage 1 Train Loss Curve (window={window_size})",
            ylabel="MSE Loss",
        )
        stage2_curve = save_learning_curve(
            stage2_losses,
            SCENARIO_DIR / f"stage2_train_loss_curve_w{window_size}.png",
            title=f"Scenario 4 Stage 2 Train Loss Curve (window={window_size})",
            ylabel="L1 Loss",
        )
        print(f"Saved stage1 loss curve to: {stage1_curve}")
        print(f"Saved stage2 loss curve to: {stage2_curve}")


if __name__ == "__main__":
    main()
