"""Scenario 3: stockout-mask-based latent demand recovery."""

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
from src.metrics import wape, wpe
from src.models import DecouplingAutoEncoder, DecouplingConfig
from src.plotting import save_learning_curve

FEATURES = [
    "hours_sale",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]
WINDOW_SIZES = [7, 14]
STOCK_STATUS_IDX = 1
SALES_IDX = 0


def _stockout_mask(x_raw: torch.Tensor) -> torch.Tensor:
    stock = x_raw[:, :, STOCK_STATUS_IDX]
    # stock_status <= 0 を欠品として強く学習
    return (stock <= 0).float().unsqueeze(-1)


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())

    for window_size in WINDOW_SIZES:
        print(f"\n=== Scenario3 / window_size={window_size} ===")
        x_all_raw = build_window_tensor(df, FEATURES, window_size=window_size)
        train_raw, valid_raw, test_raw = split_train_valid_test(x_all_raw)
        train_x, valid_x, test_x = normalize_by_train_stats(train_raw, valid_raw, test_raw)

        x_train = torch.tensor(train_x, dtype=torch.float32)
        x_valid = torch.tensor(valid_x, dtype=torch.float32)
        x_test = torch.tensor(test_x, dtype=torch.float32)
        mask_train = _stockout_mask(torch.tensor(train_raw, dtype=torch.float32))

        model = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(FEATURES), window_size=window_size))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss(reduction="none")
        losses: list[float] = []

        for step in range(100):
            rec, _, _ = model(x_train)
            raw_loss = loss_fn(rec, x_train)
            weighted = mask_train * raw_loss + 0.1 * (1 - mask_train) * raw_loss
            loss = weighted.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if step % 20 == 0:
                print(f"step={step} train_weighted_mse={loss.item():.6f}")

        with torch.no_grad():
            rec_valid = model(x_valid)[0]
            rec_test = model(x_test)[0]

        valid_true = x_valid[:, :, SALES_IDX].numpy().reshape(-1)
        valid_pred = rec_valid[:, :, SALES_IDX].numpy().reshape(-1)
        test_true = x_test[:, :, SALES_IDX].numpy().reshape(-1)
        test_pred = rec_test[:, :, SALES_IDX].numpy().reshape(-1)

        print(f"valid_WAPE={wape(valid_true, valid_pred):.4f}")
        print(f"valid_WPE={wpe(valid_true, valid_pred):.4f}")
        print(f"test_WAPE={wape(test_true, test_pred):.4f}")
        print(f"test_WPE={wpe(test_true, test_pred):.4f}")

        curve_path = save_learning_curve(
            losses,
            SCENARIO_DIR / f"train_loss_curve_w{window_size}.png",
            title=f"Scenario 3 Train Loss Curve (window={window_size})",
        )
        print(f"Saved train loss curve to: {curve_path}")


if __name__ == "__main__":
    main()
