"""Scenario 1: representation split sanity test on FreshRetailNet-50K."""

from __future__ import annotations

import torch
from torch import nn

from src.data import FreshRetailConfig, load_freshretail_dataframe, normalize_columns
from src.models import DecouplingAutoEncoder, DecouplingConfig

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


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = df.dropna(subset=FEATURES)
    x = torch.tensor(normalize_columns(df, FEATURES)[FEATURES].values, dtype=torch.float32)

    model = DecouplingAutoEncoder(DecouplingConfig(input_dim=len(FEATURES)))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(5):
        rec, _, _ = model(x)
        loss = loss_fn(rec, x)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(f"reconstruction_mse={loss.item():.6f}")


if __name__ == "__main__":
    main()
