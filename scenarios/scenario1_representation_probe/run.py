"""Scenario 1: representation split sanity test on FreshRetailNet-50K."""

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
    load_freshretail_dataframe,
    normalize_by_train_stats,
    split_train_valid_test,
)
from src.models import DecouplingAutoEncoder, DecouplingConfig
from src.plotting import save_learning_curve

FEATURES = [
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]
WINDOW_SIZES = [7, 14]


def _to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())

    for window_size in WINDOW_SIZES:
        print(f"\n=== Scenario1 / window_size={window_size} ===")
        x_all = build_window_tensor(df, FEATURES, window_size=window_size)
        train_x, valid_x, test_x = split_train_valid_test(x_all)
        train_x, valid_x, test_x = normalize_by_train_stats(train_x, valid_x, test_x)

        x_train = _to_tensor(train_x)
        x_valid = _to_tensor(valid_x)
        x_test = _to_tensor(test_x)

        model = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(FEATURES), window_size=window_size))
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        losses: list[float] = []

        for step in range(100):
            rec, _, _ = model(x_train)
            loss = loss_fn(rec, x_train)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if step % 20 == 0:
                print(f"step={step} train_mse={loss.item():.6f}")

        with torch.no_grad():
            valid_mse = loss_fn(model(x_valid)[0], x_valid).item()
            test_mse = loss_fn(model(x_test)[0], x_test).item()

        print(f"window={window_size} train_mse={losses[-1]:.6f} valid_mse={valid_mse:.6f} test_mse={test_mse:.6f}")

        curve_path = save_learning_curve(
            losses,
            SCENARIO_DIR / f"train_loss_curve_w{window_size}.png",
            title=f"Scenario 1 Train Loss Curve (window={window_size})",
        )
        print(f"Saved train loss curve to: {curve_path}")


if __name__ == "__main__":
    main()
