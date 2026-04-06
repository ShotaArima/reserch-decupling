"""Scenario 3: stockout-mask-based latent demand recovery.

Extended with Experiment 3 diagnostics:
- subset evaluation (non-stockout / near-stockout / all)
- mask-weight sweep
- train/valid gap reporting
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Allow running as `python scenarios/.../run.py` by adding repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "outputs"
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
DEFAULT_WINDOW_SIZES = [7, 14]
STOCK_STATUS_IDX = 1
SALES_IDX = 0


@dataclass
class SubsetEval:
    split: str
    subset: str
    wape_value: float
    wpe_value: float
    n_points: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario 3 latent demand recovery with diagnostics.")
    parser.add_argument(
        "--window-sizes",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WINDOW_SIZES),
        help="Comma-separated window sizes (e.g. 7,14).",
    )
    parser.add_argument("--steps", type=int, default=100, help="Training steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--mask-weight",
        type=float,
        default=0.1,
        help="Stockout weight alpha. Ignored when --mask-weight-sweep is used.",
    )
    parser.add_argument(
        "--mask-weight-sweep",
        type=str,
        default="",
        help="Comma-separated stockout weights (e.g. 0.1,0.3,0.5,1.0).",
    )
    parser.add_argument(
        "--near-window",
        type=int,
        default=3,
        help="Near-stockout width delta for subset evaluation.",
    )
    return parser.parse_args()


def _stockout_indicator(x_raw: np.ndarray) -> np.ndarray:
    stock = x_raw[:, :, STOCK_STATUS_IDX]
    return stock <= 0


def _near_stockout_mask(stockout: np.ndarray, delta: int) -> np.ndarray:
    if delta <= 0:
        return stockout.copy()
    kernel = np.ones(2 * delta + 1, dtype=np.int32)
    return np.convolve(stockout.astype(np.int32), kernel, mode="same") > 0


def _subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, subset_mask: np.ndarray) -> tuple[float, float, int]:
    n_points = int(np.sum(subset_mask))
    if n_points == 0:
        return float("nan"), float("nan"), 0
    yt = y_true[subset_mask]
    yp = y_pred[subset_mask]
    return wape(yt, yp), wpe(yt, yp), n_points


def _evaluate_subsets(
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stockout_indicator: np.ndarray,
    delta: int,
) -> list[SubsetEval]:
    stockout_flat = stockout_indicator.reshape(-1)
    near_flat = _near_stockout_mask(stockout_flat, delta=delta)
    non_flat = ~stockout_flat
    all_flat = np.ones_like(stockout_flat, dtype=bool)

    rows: list[SubsetEval] = []
    for subset_name, subset_mask in (
        ("non_stockout", non_flat),
        ("near_stockout", near_flat),
        ("all", all_flat),
    ):
        subset_wape, subset_wpe, n_points = _subset_metrics(y_true, y_pred, subset_mask)
        rows.append(
            SubsetEval(
                split=split_name,
                subset=subset_name,
                wape_value=subset_wape,
                wpe_value=subset_wpe,
                n_points=n_points,
            )
        )
    return rows


def _write_subset_csv(rows: list[SubsetEval], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "subset", "wape", "wpe", "n_points"])
        for row in rows:
            writer.writerow([row.split, row.subset, f"{row.wape_value:.6f}", f"{row.wpe_value:.6f}", row.n_points])


def _parse_list(value: str, cast_type):
    return [cast_type(v.strip()) for v in value.split(",") if v.strip()]


def _format_weight(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def _run_single_setting(
    window_size: int,
    mask_weight: float,
    near_window: int,
    steps: int,
    seed: int,
    df,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n=== Scenario3 / window_size={window_size} / mask_weight={mask_weight} ===")
    x_all_raw = build_window_tensor(df, FEATURES, window_size=window_size)
    train_raw, valid_raw, test_raw = split_train_valid_test(x_all_raw)
    train_x, valid_x, test_x = normalize_by_train_stats(train_raw, valid_raw, test_raw)

    x_train = torch.tensor(train_x, dtype=torch.float32)
    x_valid = torch.tensor(valid_x, dtype=torch.float32)
    x_test = torch.tensor(test_x, dtype=torch.float32)

    stockout_train = _stockout_indicator(train_raw)
    stockout_valid = _stockout_indicator(valid_raw)
    stockout_test = _stockout_indicator(test_raw)
    stockout_train_t = torch.tensor(stockout_train, dtype=torch.float32).unsqueeze(-1)

    model = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(FEATURES), window_size=window_size))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="none")
    losses: list[float] = []

    for step in range(steps):
        rec, _, _ = model(x_train)
        raw_loss = loss_fn(rec, x_train)
        # L = mean(((1 - s) + alpha * s) * mse), s=1 indicates stockout.
        sample_weight = (1.0 - stockout_train_t) + mask_weight * stockout_train_t
        loss = (sample_weight * raw_loss).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"step={step} train_weighted_mse={loss.item():.6f}")

    with torch.no_grad():
        rec_train = model(x_train)[0]
        rec_valid = model(x_valid)[0]
        rec_test = model(x_test)[0]

    train_true = x_train[:, :, SALES_IDX].numpy().reshape(-1)
    train_pred = rec_train[:, :, SALES_IDX].numpy().reshape(-1)
    valid_true = x_valid[:, :, SALES_IDX].numpy().reshape(-1)
    valid_pred = rec_valid[:, :, SALES_IDX].numpy().reshape(-1)
    test_true = x_test[:, :, SALES_IDX].numpy().reshape(-1)
    test_pred = rec_test[:, :, SALES_IDX].numpy().reshape(-1)

    eval_rows: list[SubsetEval] = []
    eval_rows.extend(_evaluate_subsets("train", train_true, train_pred, stockout_train, delta=near_window))
    eval_rows.extend(_evaluate_subsets("valid", valid_true, valid_pred, stockout_valid, delta=near_window))
    eval_rows.extend(_evaluate_subsets("test", test_true, test_pred, stockout_test, delta=near_window))

    # train/valid gap (WAPE only)
    print("subset metrics (train/valid/test):")
    by_subset = {r.subset for r in eval_rows}
    for subset in sorted(by_subset):
        train_row = next((r for r in eval_rows if r.split == "train" and r.subset == subset), None)
        valid_row = next((r for r in eval_rows if r.split == "valid" and r.subset == subset), None)
        test_row = next((r for r in eval_rows if r.split == "test" and r.subset == subset), None)
        if train_row is None or valid_row is None or test_row is None:
            continue
        gap = valid_row.wape_value - train_row.wape_value
        print(
            f"  {subset}: "
            f"train_wape={train_row.wape_value:.4f} valid_wape={valid_row.wape_value:.4f} "
            f"test_wape={test_row.wape_value:.4f} gap={gap:.4f} "
            f"valid_wpe={valid_row.wpe_value:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alpha_tag = _format_weight(mask_weight)

    curve_path = save_learning_curve(
        losses,
        SCENARIO_DIR / f"train_loss_curve_w{window_size}_a{alpha_tag}.png",
        title=f"Scenario 3 Train Loss Curve (window={window_size}, alpha={mask_weight})",
    )
    print(f"Saved train loss curve to: {curve_path}")

    csv_path = OUTPUT_DIR / f"scenario3_subset_metrics_w{window_size}_a{alpha_tag}.csv"
    _write_subset_csv(eval_rows, csv_path)
    print(f"Saved subset metrics to: {csv_path}")


def main() -> None:
    args = _parse_args()
    df = load_freshretail_dataframe(FreshRetailConfig())

    window_sizes = _parse_list(args.window_sizes, int)
    mask_weights = (
        _parse_list(args.mask_weight_sweep, float)
        if args.mask_weight_sweep
        else [args.mask_weight]
    )

    for window_size in window_sizes:
        for mask_weight in mask_weights:
            _run_single_setting(
                window_size=window_size,
                mask_weight=mask_weight,
                near_window=args.near_window,
                steps=args.steps,
                seed=args.seed,
                df=df,
            )


if __name__ == "__main__":
    main()
