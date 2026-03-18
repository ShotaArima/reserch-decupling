"""Scenario 5: counterfactual generation sanity check."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data import FreshRetailConfig, load_freshretail_dataframe, normalize_columns
from src.models import DecouplingAutoEncoder, DecouplingConfig

FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag", "avg_temperature"]


def main() -> None:
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = df.dropna(subset=FEATURES)
    x = torch.tensor(normalize_columns(df, FEATURES)[FEATURES].values, dtype=torch.float32)

    model = DecouplingAutoEncoder(DecouplingConfig(input_dim=len(FEATURES)))
    rec, local, global_latent = model(x)

    # Counterfactual A: keep local, shuffle global
    shuffled_global = global_latent[torch.randperm(global_latent.size(0))]
    cf_a = model.decoder(torch.cat([local, shuffled_global], dim=-1))

    # Counterfactual B: keep global, shuffle local
    shuffled_local = local[torch.randperm(local.size(0))]
    cf_b = model.decoder(torch.cat([shuffled_local, global_latent], dim=-1))

    print(f"orig_mean_sale={rec[:, 0].mean().item():.4f}")
    print(f"cf_global_swap_mean_sale={cf_a[:, 0].mean().item():.4f}")
    print(f"cf_local_swap_mean_sale={cf_b[:, 0].mean().item():.4f}")


if __name__ == "__main__":
    main()
