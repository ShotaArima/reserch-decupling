"""Baseline: Forecast baseline block (Scenario2/4 + naive + Prophet)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASELINE_DIR / "outputs"

from src.data import (
    FreshRetailConfig,
    build_window_tensor,
    extract_last_timestep_feature,
    load_freshretail_dataframe,
    normalize_by_train_stats,
    split_train_valid_test,
)
from src.forecast_baselines import (
    ForecastResult,
    make_one_step_pairs,
    predict_last_value,
    predict_moving_average,
    train_flatten_linear,
    train_flatten_mlp,
    train_scenario2_model,
    train_scenario4_pipeline,
    predict_prophet_next_step_per_sample,
)
from src.metrics import wape, wpe
from src.plotting import save_learning_curve, save_metric_bar_chart

BASE_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
SCENARIO4_FEATURES = ["sale_amount", "hours_stock_status", "discount", "holiday_flag", "activity_flag"]
WINDOW_SIZE = 14
TARGET_FEATURE_INDEX = 0
MOVING_AVERAGE_KS = [3, 7, 14]
MLP_HIDDEN_CANDIDATES = ([128, 64], [128, 64, 32])
TRAIN_STEPS = 100


def _score(name: str, y_valid: np.ndarray, p_valid: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> ForecastResult:
    return ForecastResult(
        name=name,
        valid_wape=wape(y_valid, p_valid),
        valid_wpe=wpe(y_valid, p_valid),
        test_wape=wape(y_test, p_test),
        test_wpe=wpe(y_test, p_test),
    )


def _write_results_csv(results: list[ForecastResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "valid_wape", "valid_wpe", "test_wape", "test_wpe"])
        for r in results:
            writer.writerow([r.name, f"{r.valid_wape:.6f}", f"{r.valid_wpe:.6f}", f"{r.test_wape:.6f}", f"{r.test_wpe:.6f}"])


def _prepare_split_tensors(df, features: list[str], window_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
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

    return x_train, x_valid, x_test, y_train, y_valid, y_test, train_raw, valid_raw, test_raw


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_freshretail_dataframe(FreshRetailConfig())

    x_train, x_valid, x_test, y_train, y_valid, y_test, train_raw, valid_raw, test_raw = _prepare_split_tensors(
        df,
        BASE_FEATURES,
        WINDOW_SIZE,
    )

    x_train_s, y_train_s = make_one_step_pairs(x_train, y_train)
    x_valid_s, y_valid_s = make_one_step_pairs(x_valid, y_valid)
    x_test_s, y_test_s = make_one_step_pairs(x_test, y_test)

    y_valid_np = y_valid_s.numpy().reshape(-1)
    y_test_np = y_test_s.numpy().reshape(-1)

    results: list[ForecastResult] = []

    # 1) Last Value
    valid_last = predict_last_value(valid_raw[:-1], target_feature_index=TARGET_FEATURE_INDEX)
    test_last = predict_last_value(test_raw[:-1], target_feature_index=TARGET_FEATURE_INDEX)
    results.append(_score("LastValue", y_valid_np, valid_last, y_test_np, test_last))

    # 2) Moving Average (select k by valid WAPE)
    best_k = MOVING_AVERAGE_KS[0]
    best_valid_pred = predict_moving_average(valid_raw[:-1], k=best_k, target_feature_index=TARGET_FEATURE_INDEX)
    best_test_pred = predict_moving_average(test_raw[:-1], k=best_k, target_feature_index=TARGET_FEATURE_INDEX)
    best_valid_wape = wape(y_valid_np, best_valid_pred)

    for k in MOVING_AVERAGE_KS[1:]:
        valid_pred = predict_moving_average(valid_raw[:-1], k=k, target_feature_index=TARGET_FEATURE_INDEX)
        candidate_wape = wape(y_valid_np, valid_pred)
        if candidate_wape < best_valid_wape:
            best_k = k
            best_valid_wape = candidate_wape
            best_valid_pred = valid_pred
            best_test_pred = predict_moving_average(test_raw[:-1], k=k, target_feature_index=TARGET_FEATURE_INDEX)

    print(f"Selected MovingAverage k={best_k}")
    results.append(_score("MovingAverage", y_valid_np, best_valid_pred, y_test_np, best_test_pred))

    # 3) Flatten + Linear
    linear_model, linear_losses = train_flatten_linear(x_train_s, y_train_s, steps=TRAIN_STEPS)
    with torch.no_grad():
        linear_valid = linear_model(x_valid_s.reshape(x_valid_s.shape[0], -1)).numpy().reshape(-1)
        linear_test = linear_model(x_test_s.reshape(x_test_s.shape[0], -1)).numpy().reshape(-1)
    results.append(_score("FlattenLinear", y_valid_np, linear_valid, y_test_np, linear_test))

    linear_curve = save_learning_curve(
        linear_losses,
        OUTPUT_DIR / "flatten_linear_train_loss.png",
        title="Flatten Linear Train Loss",
        ylabel="L1 Loss",
    )
    print(f"Saved: {linear_curve}")

    # 4) Flatten + MLP (select depth by valid WAPE)
    mlp_runs = []
    for hidden_dims in MLP_HIDDEN_CANDIDATES:
        mlp_model, mlp_losses = train_flatten_mlp(
            x_train_s,
            y_train_s,
            hidden_dims=hidden_dims,
            steps=TRAIN_STEPS,
        )
        with torch.no_grad():
            pred_valid = mlp_model(x_valid_s.reshape(x_valid_s.shape[0], -1)).numpy().reshape(-1)
            pred_test = mlp_model(x_test_s.reshape(x_test_s.shape[0], -1)).numpy().reshape(-1)
        mlp_runs.append((hidden_dims, mlp_losses, pred_valid, pred_test))

    best_hidden_dims, best_mlp_losses, best_mlp_valid, best_mlp_test = min(
        mlp_runs,
        key=lambda x: wape(y_valid_np, x[2]),
    )
    print(f"Selected FlattenMLP hidden_dims={list(best_hidden_dims)}")
    results.append(_score("FlattenMLP", y_valid_np, best_mlp_valid, y_test_np, best_mlp_test))

    mlp_curve = save_learning_curve(
        best_mlp_losses,
        OUTPUT_DIR / "flatten_mlp_train_loss.png",
        title="Flatten MLP Train Loss",
        ylabel="L1 Loss",
    )
    print(f"Saved: {mlp_curve}")

    # 5) Prophet (per-sample univariate forecast)
    print("Running Prophet baseline...")
    try:
        prophet_valid = predict_prophet_next_step_per_sample(valid_raw[:-1], target_feature_index=TARGET_FEATURE_INDEX)
        prophet_test = predict_prophet_next_step_per_sample(test_raw[:-1], target_feature_index=TARGET_FEATURE_INDEX)
        results.append(_score("Prophet", y_valid_np, prophet_valid, y_test_np, prophet_test))
    except Exception as exc:
        print(f"Skipped Prophet baseline: {exc}")

    # 6) Scenario 2
    s2_body, s2_head, s2_losses = train_scenario2_model(
        x_train_s,
        y_train_s,
        feature_dim=len(BASE_FEATURES),
        window_size=WINDOW_SIZE,
        steps=TRAIN_STEPS,
    )
    with torch.no_grad():
        s2_valid = s2_head(*s2_body(x_valid_s)[1:]).numpy().reshape(-1)
        s2_test = s2_head(*s2_body(x_test_s)[1:]).numpy().reshape(-1)
    results.append(_score("Scenario2", y_valid_np, s2_valid, y_test_np, s2_test))

    s2_curve = save_learning_curve(
        s2_losses,
        OUTPUT_DIR / "scenario2_train_loss.png",
        title="Scenario2 Train Loss",
        ylabel="L1 Loss",
    )
    print(f"Saved: {s2_curve}")

    # 7) Scenario 4
    x4_train, x4_valid, x4_test, y4_train, y4_valid, y4_test, _, _, _ = _prepare_split_tensors(
        df,
        SCENARIO4_FEATURES,
        WINDOW_SIZE,
    )

    recovery, forecaster, stage1_losses, stage2_losses = train_scenario4_pipeline(
        x4_train,
        y4_train,
        feature_dim=len(SCENARIO4_FEATURES),
        window_size=WINDOW_SIZE,
        steps=TRAIN_STEPS,
    )

    with torch.no_grad():
        _, local_valid, global_valid = recovery(x4_valid)
        _, local_test, global_test = recovery(x4_test)

    local_valid_s, y4_valid_s = make_one_step_pairs(local_valid, y4_valid)
    global_valid_s, _ = make_one_step_pairs(global_valid, y4_valid)
    local_test_s, y4_test_s = make_one_step_pairs(local_test, y4_test)
    global_test_s, _ = make_one_step_pairs(global_test, y4_test)

    with torch.no_grad():
        s4_valid = forecaster(local_valid_s, global_valid_s).numpy().reshape(-1)
        s4_test = forecaster(local_test_s, global_test_s).numpy().reshape(-1)

    results.append(
        _score(
            "Scenario4",
            y4_valid_s.numpy().reshape(-1),
            s4_valid,
            y4_test_s.numpy().reshape(-1),
            s4_test,
        )
    )

    stage1_curve = save_learning_curve(
        stage1_losses,
        OUTPUT_DIR / "scenario4_stage1_train_loss.png",
        title="Scenario4 Stage1 Train Loss",
        ylabel="MSE Loss",
    )
    stage2_curve = save_learning_curve(
        stage2_losses,
        OUTPUT_DIR / "scenario4_stage2_train_loss.png",
        title="Scenario4 Stage2 Train Loss",
        ylabel="L1 Loss",
    )
    print(f"Saved: {stage1_curve}")
    print(f"Saved: {stage2_curve}")

    # Save summary artifacts
    results_csv = OUTPUT_DIR / "forecast_baseline_results.csv"
    _write_results_csv(results, results_csv)
    print(f"Saved: {results_csv}")

    model_names = [r.name for r in results]
    valid_wape_values = [r.valid_wape for r in results]
    test_wape_values = [r.test_wape for r in results]

    valid_bar = save_metric_bar_chart(
        model_names,
        valid_wape_values,
        OUTPUT_DIR / "valid_wape_comparison.png",
        title="Valid WAPE by Model",
        ylabel="WAPE",
    )
    test_bar = save_metric_bar_chart(
        model_names,
        test_wape_values,
        OUTPUT_DIR / "test_wape_comparison.png",
        title="Test WAPE by Model",
        ylabel="WAPE",
    )
    print(f"Saved: {valid_bar}")
    print(f"Saved: {test_bar}")

    print("\n=== Result Summary ===")
    for r in results:
        print(
            f"{r.name:14s} "
            f"valid_wape={r.valid_wape:.4f} valid_wpe={r.valid_wpe:.4f} "
            f"test_wape={r.test_wape:.4f} test_wpe={r.test_wpe:.4f}"
        )


if __name__ == "__main__":
    main()
