"""Scenario 8: recovery subset diagnosis (A3).

Compare raw baseline / Scenario2 / Scenario4 on test subset metrics:
- all
- stockout
- non_stockout
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
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
from src.forecast_baselines import make_one_step_pairs, train_flatten_mlp, train_scenario2_model, train_scenario4_pipeline
from src.plotting import save_learning_curve
from src.subset_evaluation import SubsetDiffRow, SubsetMetricRow, compute_diff_rows, compute_subset_metrics

BASE_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
SCENARIO4_FEATURES = ["sale_amount", "hours_stock_status", "discount", "holiday_flag", "activity_flag"]
TARGET_FEATURE_INDEX = 0
STOCK_STATUS_FEATURE_INDEX = 1
DEFAULT_WINDOW_SIZE = 14
DEFAULT_STEPS = 100
DEFAULT_SEED = 42
DEFAULT_HIDDEN_DIMS = (128, 64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario 8 recovery subset diagnosis.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=",".join(str(v) for v in DEFAULT_HIDDEN_DIMS),
        help="FlattenMLP hidden dims, comma-separated (e.g., 128,64).",
    )
    return parser.parse_args()


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _prepare_tensors(df, features: list[str], window_size: int):
    x_all_raw = build_window_tensor(df, features, window_size=window_size)
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

    return (x_train, x_valid, x_test, y_train, y_valid, y_test, train_raw, valid_raw, test_raw)


def _extract_label_stockout_mask(test_raw_s4: np.ndarray) -> np.ndarray:
    stock_status = test_raw_s4[:, -1, STOCK_STATUS_FEATURE_INDEX]
    mask = stock_status <= 0
    # one-step evaluation uses y[t+1], so align with shifted labels.
    return mask[1:]


def _write_subset_metrics(rows: list[SubsetMetricRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "model", "n", "target_sum", "stockout_ratio", "WAPE", "WPE", "MAE"])
        for r in rows:
            writer.writerow(
                [
                    r.subset,
                    r.model,
                    r.n,
                    f"{r.target_sum:.6f}",
                    f"{r.stockout_ratio:.6f}",
                    f"{r.wape_value:.6f}",
                    f"{r.wpe_value:.6f}",
                    f"{r.mae_value:.6f}",
                ]
            )


def _write_diff_table(rows: list[SubsetDiffRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "metric", "S4-S2", "S4-raw", "S2-raw"])
        for r in rows:
            writer.writerow(
                [
                    r.subset,
                    r.metric,
                    f"{r.s4_minus_s2:.6f}",
                    f"{r.s4_minus_raw:.6f}",
                    f"{r.s2_minus_raw:.6f}",
                ]
            )


def _write_prediction_rows(
    out_path: Path,
    *,
    y_true: np.ndarray,
    pred_raw: np.ndarray,
    pred_s2: np.ndarray,
    pred_s4: np.ndarray,
    stockout_mask: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "y_true", "y_pred_raw", "y_pred_s2", "y_pred_s4", "is_stockout", "is_non_stockout"])
        for idx in range(len(y_true)):
            is_stockout = bool(stockout_mask[idx])
            writer.writerow(
                [
                    idx,
                    f"{y_true[idx]:.6f}",
                    f"{pred_raw[idx]:.6f}",
                    f"{pred_s2[idx]:.6f}",
                    f"{pred_s4[idx]:.6f}",
                    int(is_stockout),
                    int(not is_stockout),
                ]
            )


def main() -> None:
    args = _parse_args()
    hidden_dims = _parse_int_list(args.hidden_dims)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_freshretail_dataframe(FreshRetailConfig())

    # raw baseline and scenario2 share the same feature set.
    x_train, x_valid, x_test, y_train, y_valid, y_test, _, _, _ = _prepare_tensors(
        df,
        BASE_FEATURES,
        args.window_size,
    )
    x_train_s, y_train_s = make_one_step_pairs(x_train, y_train)
    x_test_s, y_test_s = make_one_step_pairs(x_test, y_test)

    y_test_np = y_test_s.numpy().reshape(-1)

    # Scenario4 tensors include stock status and use the same split/window.
    x4_train, _, x4_test, y4_train, _, y4_test, _, _, test_raw_s4 = _prepare_tensors(
        df,
        SCENARIO4_FEATURES,
        args.window_size,
    )
    _, y4_test_s = make_one_step_pairs(x4_test, y4_test)

    # Ensure label alignment across models.
    if not np.allclose(y_test_np, y4_test_s.numpy().reshape(-1), atol=1e-6):
        raise RuntimeError("Label mismatch between BASE_FEATURES and SCENARIO4_FEATURES pipelines.")

    # label-side stockout mask for test y[t+1]
    stockout_mask = _extract_label_stockout_mask(test_raw_s4)
    if len(stockout_mask) != len(y_test_np):
        raise RuntimeError("Stockout mask length does not match one-step test labels.")

    # 1) raw baseline: FlattenMLP
    raw_model, raw_losses = train_flatten_mlp(
        x_train_s,
        y_train_s,
        hidden_dims=hidden_dims,
        steps=args.steps,
    )
    with torch.no_grad():
        pred_raw = raw_model(x_test_s.reshape(x_test_s.shape[0], -1)).numpy().reshape(-1)

    raw_curve = save_learning_curve(
        raw_losses,
        OUTPUT_DIR / "flatten_mlp_train_loss.png",
        title="Scenario8 FlattenMLP Train Loss",
        ylabel="L1 Loss",
    )

    # 2) Scenario2
    s2_body, s2_head, s2_losses = train_scenario2_model(
        x_train_s,
        y_train_s,
        feature_dim=len(BASE_FEATURES),
        window_size=args.window_size,
        steps=args.steps,
    )
    with torch.no_grad():
        pred_s2 = s2_head(*s2_body(x_test_s)[1:]).numpy().reshape(-1)

    s2_curve = save_learning_curve(
        s2_losses,
        OUTPUT_DIR / "scenario2_train_loss.png",
        title="Scenario8 Scenario2 Train Loss",
        ylabel="L1 Loss",
    )

    # 3) Scenario4 (two-stage)
    recovery, forecaster, stage1_losses, stage2_losses = train_scenario4_pipeline(
        x4_train,
        y4_train,
        feature_dim=len(SCENARIO4_FEATURES),
        window_size=args.window_size,
        steps=args.steps,
    )
    with torch.no_grad():
        _, local_test, global_test = recovery(x4_test)

    local_test_s, _ = make_one_step_pairs(local_test, y4_test)
    global_test_s, _ = make_one_step_pairs(global_test, y4_test)
    with torch.no_grad():
        pred_s4 = forecaster(local_test_s, global_test_s).numpy().reshape(-1)

    stage1_curve = save_learning_curve(
        stage1_losses,
        OUTPUT_DIR / "scenario4_stage1_train_loss.png",
        title="Scenario8 Scenario4 Stage1 Train Loss",
        ylabel="MSE Loss",
    )
    stage2_curve = save_learning_curve(
        stage2_losses,
        OUTPUT_DIR / "scenario4_stage2_train_loss.png",
        title="Scenario8 Scenario4 Stage2 Train Loss",
        ylabel="L1 Loss",
    )

    metric_rows: list[SubsetMetricRow] = []
    metric_rows.extend(compute_subset_metrics(y_true=y_test_np, y_pred=pred_raw, stockout_mask=stockout_mask, model_name="raw_baseline"))
    metric_rows.extend(compute_subset_metrics(y_true=y_test_np, y_pred=pred_s2, stockout_mask=stockout_mask, model_name="Scenario2"))
    metric_rows.extend(compute_subset_metrics(y_true=y_test_np, y_pred=pred_s4, stockout_mask=stockout_mask, model_name="Scenario4"))

    diff_rows = compute_diff_rows(metric_rows)

    subset_csv = OUTPUT_DIR / "scenario8_subset_metrics.csv"
    diff_csv = OUTPUT_DIR / "scenario8_diff_metrics.csv"
    pred_csv = OUTPUT_DIR / "scenario8_test_predictions.csv"

    _write_subset_metrics(metric_rows, subset_csv)
    _write_diff_table(diff_rows, diff_csv)
    _write_prediction_rows(
        pred_csv,
        y_true=y_test_np,
        pred_raw=pred_raw,
        pred_s2=pred_s2,
        pred_s4=pred_s4,
        stockout_mask=stockout_mask,
    )

    print("=== Scenario8 completed ===")
    print(f"Saved: {subset_csv}")
    print(f"Saved: {diff_csv}")
    print(f"Saved: {pred_csv}")
    print(f"Saved: {raw_curve}")
    print(f"Saved: {s2_curve}")
    print(f"Saved: {stage1_curve}")
    print(f"Saved: {stage2_curve}")


if __name__ == "__main__":
    main()
