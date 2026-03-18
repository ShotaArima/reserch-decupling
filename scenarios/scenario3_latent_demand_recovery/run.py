"""Scenario 3: stockout-mask-based latent demand recovery."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from src.data import (
    FreshRetailConfig,
    coerce_numeric_columns,
    load_freshretail_dataframe,
    normalize_columns,
)
from src.metrics import wape, wpe
from src.models import DecouplingAutoEncoder, DecouplingConfig

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


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = coerce_numeric_columns(df, FEATURES)
    df = df.dropna(subset=FEATURES)
    norm = normalize_columns(df, FEATURES)

    x = torch.tensor(norm[FEATURES].values, dtype=torch.float32)
    mask = torch.tensor((df["hours_stock_status"].values > 0).astype(np.float32)).unsqueeze(-1)

    model = DecouplingAutoEncoder(DecouplingConfig(input_dim=len(FEATURES)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="none")

    for _ in range(5):
        rec, _, _ = model(x)
        raw_loss = loss_fn(rec, x)
        weighted = (1 - mask) * raw_loss + 0.1 * mask * raw_loss
        loss = weighted.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    rec_np = rec.detach().cpu().numpy()[:, 0]
    true_np = x.detach().cpu().numpy()[:, 0]
    print(f"WAPE={wape(true_np, rec_np):.4f}")
    print(f"WPE={wpe(true_np, rec_np):.4f}")


if __name__ == "__main__":
    main()
