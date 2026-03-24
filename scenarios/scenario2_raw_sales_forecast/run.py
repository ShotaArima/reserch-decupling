"""Scenario 2: raw daily sales forecasting with windowed local inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent

from src.data import (
    FreshRetailConfig,
    build_window_tensor,
    extract_last_timestep_feature,
    load_freshretail_dataframe,
    normalize_by_train_stats,
    split_train_valid_test,
)
from src.metrics import wape, wpe
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead
from src.plotting import save_learning_curve

INPUT_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
TARGET_FEATURE_INDEX = 0
WINDOW_SIZES = [7, 14]


def _make_forecast_pairs(x_split: torch.Tensor, y_split_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x_t -> y_{t+1}
    return x_split[:-1], y_split_raw[1:]


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())

    for window_size in WINDOW_SIZES:
        print(f"\n=== Scenario2 / window_size={window_size} ===")
        x_all_raw = build_window_tensor(df, INPUT_FEATURES, window_size=window_size)
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

        x_train, y_train = _make_forecast_pairs(x_train, y_train)
        x_valid, y_valid = _make_forecast_pairs(x_valid, y_valid)
        x_test, y_test = _make_forecast_pairs(x_test, y_test)

        body = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(INPUT_FEATURES), window_size=window_size))
        head = ForecastHead(local_dim=16, global_dim=16, horizon=1)
        opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
        loss_fn = nn.L1Loss()
        losses: list[float] = []

        for step in range(100):
            _, local, global_latent = body(x_train)
            pred = head(local, global_latent)
            loss = loss_fn(pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if step % 20 == 0:
                print(f"step={step} train_l1={loss.item():.6f}")

        with torch.no_grad():
            pred_valid = head(*body(x_valid)[1:])
            pred_test = head(*body(x_test)[1:])

        print(f"valid_WAPE={wape(y_valid.numpy().reshape(-1), pred_valid.numpy().reshape(-1)):.4f}")
        print(f"valid_WPE={wpe(y_valid.numpy().reshape(-1), pred_valid.numpy().reshape(-1)):.4f}")
        print(f"test_WAPE={wape(y_test.numpy().reshape(-1), pred_test.numpy().reshape(-1)):.4f}")
        print(f"test_WPE={wpe(y_test.numpy().reshape(-1), pred_test.numpy().reshape(-1)):.4f}")

        curve_path = save_learning_curve(
            losses,
            SCENARIO_DIR / f"train_loss_curve_w{window_size}.png",
            title=f"Scenario 2 Train Loss Curve (window={window_size})",
        )
        print(f"Saved train loss curve to: {curve_path}")


if __name__ == "__main__":
    main()
