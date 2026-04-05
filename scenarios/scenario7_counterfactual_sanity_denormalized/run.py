"""Scenario 7: denormalized counterfactual sanity check."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "outputs"

from src.data import (
    FreshRetailConfig,
    apply_normalization,
    build_window_tensor,
    denormalize,
    fit_train_normalization_stats,
    load_freshretail_dataframe,
    split_train_valid_test,
)
from src.models import DecouplingAutoEncoder, DecouplingConfig
from src.plotting import save_difference_histogram, save_learning_curve, save_sample_series_plot

FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag", "avg_temperature"]
WINDOW_SIZE = 14
SALE_FEATURE_INDEX = 0
TRAIN_STEPS = 100
SEED = 42


def _write_summary_csv(out_path: Path, rows: list[tuple[str, float]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in rows:
            writer.writerow([key, f"{value:.6f}"])


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_freshretail_dataframe(FreshRetailConfig())

    x_all = build_window_tensor(df, FEATURES, window_size=WINDOW_SIZE)
    train_raw, valid_raw, test_raw = split_train_valid_test(x_all)

    mu, sigma = fit_train_normalization_stats(train_raw)
    train_x = apply_normalization(train_raw, mu, sigma)
    valid_x = apply_normalization(valid_raw, mu, sigma)
    test_x = apply_normalization(test_raw, mu, sigma)

    x_train = torch.tensor(train_x, dtype=torch.float32)
    x_valid = torch.tensor(valid_x, dtype=torch.float32)
    x_test = torch.tensor(test_x, dtype=torch.float32)

    model = DecouplingAutoEncoder(
        DecouplingConfig(
            feature_dim=len(FEATURES),
            window_size=WINDOW_SIZE,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses: list[float] = []
    for step in range(TRAIN_STEPS):
        rec_train, _, _ = model(x_train)
        train_loss = loss_fn(rec_train, x_train)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        losses.append(float(train_loss.item()))

        if step % 20 == 0:
            with torch.no_grad():
                rec_valid, _, _ = model(x_valid)
                valid_loss = loss_fn(rec_valid, x_valid).item()
            print(f"step={step:03d} train_mse={train_loss.item():.6f} valid_mse={valid_loss:.6f}")

    with torch.no_grad():
        rec_norm, local_t, global_t = model(x_test)

        shuffled_global = global_t[torch.randperm(global_t.size(0))]
        cf_global_norm = model.decoder(torch.cat([local_t, shuffled_global], dim=-1)).reshape_as(x_test)

        shuffled_local = local_t[torch.randperm(local_t.size(0))]
        cf_local_norm = model.decoder(torch.cat([shuffled_local, global_t], dim=-1)).reshape_as(x_test)

    rec_np = rec_norm.cpu().numpy()
    cf_global_np = cf_global_norm.cpu().numpy()
    cf_local_np = cf_local_norm.cpu().numpy()

    rec_denorm = denormalize(rec_np, mu, sigma)
    cf_global_denorm = denormalize(cf_global_np, mu, sigma)
    cf_local_denorm = denormalize(cf_local_np, mu, sigma)

    rec_sale = rec_denorm[:, :, SALE_FEATURE_INDEX]
    cf_global_sale = cf_global_denorm[:, :, SALE_FEATURE_INDEX]
    cf_local_sale = cf_local_denorm[:, :, SALE_FEATURE_INDEX]

    global_diff = (cf_global_sale - rec_sale).reshape(-1)
    local_diff = (cf_local_sale - rec_sale).reshape(-1)

    summary_rows = [
        ("orig_mean_sale_denorm", float(rec_sale.mean())),
        ("cf_global_swap_mean_sale_denorm", float(cf_global_sale.mean())),
        ("cf_local_swap_mean_sale_denorm", float(cf_local_sale.mean())),
        ("diff_global_minus_orig_mean", float(global_diff.mean())),
        ("diff_local_minus_orig_mean", float(local_diff.mean())),
        ("diff_global_minus_local_mean", float((cf_global_sale - cf_local_sale).mean())),
        ("diff_global_minus_orig_abs_mean", float(np.abs(global_diff).mean())),
        ("diff_local_minus_orig_abs_mean", float(np.abs(local_diff).mean())),
    ]

    summary_csv = OUTPUT_DIR / "scenario7_summary_denorm.csv"
    _write_summary_csv(summary_csv, summary_rows)

    curve_path = save_learning_curve(
        losses,
        OUTPUT_DIR / "scenario7_train_loss.png",
        title="Scenario 7 Train Loss",
        ylabel="MSE Loss",
    )
    hist_path = save_difference_histogram(
        global_diff,
        local_diff,
        OUTPUT_DIR / "scenario7_diff_histogram.png",
        title="Scenario 7 Difference Histogram (Denormalized)",
    )
    sample_plot_path = save_sample_series_plot(
        rec_sale,
        cf_global_sale,
        cf_local_sale,
        OUTPUT_DIR / "scenario7_sample_series.png",
        title="Scenario 7 Sample Series (Denormalized Sale)",
        max_samples=12,
    )

    print("\n=== Scenario 7 (denormalized) summary ===")
    for key, value in summary_rows:
        print(f"{key}={value:.6f}")

    print(f"Saved: {summary_csv}")
    print(f"Saved: {curve_path}")
    print(f"Saved: {hist_path}")
    print(f"Saved: {sample_plot_path}")


if __name__ == "__main__":
    main()
