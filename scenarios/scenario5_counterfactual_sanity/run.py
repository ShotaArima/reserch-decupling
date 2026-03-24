"""Scenario 5: counterfactual generation sanity check."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
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

FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag", "avg_temperature"]
WINDOW_SIZES = [7, 14]


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())

    for window_size in WINDOW_SIZES:
        print(f"\n=== Scenario5 / window_size={window_size} ===")
        x_all = build_window_tensor(df, FEATURES, window_size=window_size)
        train_x, valid_x, test_x = split_train_valid_test(x_all)
        train_x, valid_x, test_x = normalize_by_train_stats(train_x, valid_x, test_x)

        x_train = torch.tensor(train_x, dtype=torch.float32)
        x_test = torch.tensor(test_x, dtype=torch.float32)

        model = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(FEATURES), window_size=window_size))
        rec, local, global_latent = model(x_train)

        # Evaluate counterfactual behavior on test split
        _, local_t, global_t = model(x_test)

        shuffled_global = global_t[torch.randperm(global_t.size(0))]
        cf_a = model.decoder(torch.cat([local_t, shuffled_global], dim=-1)).reshape_as(x_test)

        shuffled_local = local_t[torch.randperm(local_t.size(0))]
        cf_b = model.decoder(torch.cat([shuffled_local, global_t], dim=-1)).reshape_as(x_test)

        print(f"train_rec_mean_sale={rec[:, :, 0].mean().item():.4f}")
        print(f"test_cf_global_swap_mean_sale={cf_a[:, :, 0].mean().item():.4f}")
        print(f"test_cf_local_swap_mean_sale={cf_b[:, :, 0].mean().item():.4f}")


if __name__ == "__main__":
    main()
