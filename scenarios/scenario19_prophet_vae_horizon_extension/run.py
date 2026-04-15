"""Scenario 19: Prophet vs VAE-decoupling horizon extension experiment."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "output"

from src.data import (
    FreshRetailConfig,
    build_window_tensor,
    extract_last_timestep_feature,
    load_freshretail_dataframe,
    normalize_by_train_stats,
    split_train_valid_test,
)
from src.forecast_baselines import (
    make_one_step_pairs,
    predict_prophet_next_step_per_sample,
    train_scenario2_model,
)
from src.metrics import wape, wpe
from src.plotting import save_horizon_error_plot, save_learning_curve

BASE_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
TARGET_FEATURE_INDEX = 0
HORIZONS = [14, 21, 28, 35, 42]
TRAIN_STEPS = 100
SEED = 42


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def _prepare_split_tensors(
    df,
    features: list[str],
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
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

    return x_train, x_valid, x_test, y_train, y_valid, y_test, valid_raw, test_raw


def _score_row(
    horizon: int,
    model: str,
    y_valid: np.ndarray,
    p_valid: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
) -> dict[str, float | str | int]:
    return {
        "horizon": horizon,
        "model": model,
        "valid_wape": wape(y_valid, p_valid),
        "valid_wpe": wpe(y_valid, p_valid),
        "test_wape": wape(y_test, p_test),
        "test_wpe": wpe(y_test, p_test),
    }


def _write_results(rows: list[dict[str, float | str | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["horizon", "model", "valid_wape", "valid_wpe", "test_wape", "test_wpe"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _log(f"Seed fixed at {SEED}")
    _log("Loading FreshRetailNet-50K dataframe...")
    df = load_freshretail_dataframe(FreshRetailConfig())
    _log(f"Loaded dataframe rows={len(df)} cols={len(df.columns)}")

    rows: list[dict[str, float | str | int]] = []

    for horizon in HORIZONS:
        _log("-" * 80)
        _log(f"Start horizon={horizon}")
        x_train, x_valid, x_test, y_train, y_valid, y_test, valid_raw, test_raw = _prepare_split_tensors(
            df,
            BASE_FEATURES,
            horizon,
        )
        _log(
            "Prepared tensors: "
            f"x_train={tuple(x_train.shape)} x_valid={tuple(x_valid.shape)} x_test={tuple(x_test.shape)}"
        )

        x_train_s, y_train_s = make_one_step_pairs(x_train, y_train)
        x_valid_s, y_valid_s = make_one_step_pairs(x_valid, y_valid)
        x_test_s, y_test_s = make_one_step_pairs(x_test, y_test)

        y_valid_np = y_valid_s.numpy().reshape(-1)
        y_test_np = y_test_s.numpy().reshape(-1)
        _log(f"One-step pairs: train={len(x_train_s)} valid={len(x_valid_s)} test={len(x_test_s)}")

        _log("Running Prophet baseline...")
        try:
            prophet_valid = predict_prophet_next_step_per_sample(
                valid_raw[:-1],
                target_feature_index=TARGET_FEATURE_INDEX,
            )
            prophet_test = predict_prophet_next_step_per_sample(
                test_raw[:-1],
                target_feature_index=TARGET_FEATURE_INDEX,
            )
            prophet_row = _score_row(horizon, "Prophet", y_valid_np, prophet_valid, y_test_np, prophet_test)
            rows.append(prophet_row)
            _log(
                "Prophet finished: "
                f"valid_wape={prophet_row['valid_wape']:.4f} test_wape={prophet_row['test_wape']:.4f}"
            )
        except Exception as exc:
            _log(f"Prophet skipped on horizon={horizon}: {exc}")

        _log(f"Training VAE decoupling model (Scenario2) steps={TRAIN_STEPS}...")
        s2_body, s2_head, s2_losses = train_scenario2_model(
            x_train_s,
            y_train_s,
            feature_dim=len(BASE_FEATURES),
            window_size=horizon,
            steps=TRAIN_STEPS,
        )
        _log(
            f"Scenario2 loss snapshot: start={s2_losses[0]:.4f} "
            f"mid={s2_losses[len(s2_losses)//2]:.4f} end={s2_losses[-1]:.4f}"
        )

        with torch.no_grad():
            s2_valid = s2_head(*s2_body(x_valid_s)[1:]).numpy().reshape(-1)
            s2_test = s2_head(*s2_body(x_test_s)[1:]).numpy().reshape(-1)

        s2_row = _score_row(horizon, "Scenario2", y_valid_np, s2_valid, y_test_np, s2_test)
        rows.append(s2_row)
        _log(
            "Scenario2 finished: "
            f"valid_wape={s2_row['valid_wape']:.4f} test_wape={s2_row['test_wape']:.4f}"
        )

        loss_path = save_learning_curve(
            s2_losses,
            OUTPUT_DIR / f"h{horizon}_scenario2_train_loss.png",
            title=f"Scenario2 Train Loss (window={horizon})",
            ylabel="L1 Loss",
        )
        _log(f"Saved learning curve: {loss_path}")

    if not rows:
        raise RuntimeError("No result rows were produced.")

    csv_path = OUTPUT_DIR / "horizon_extension_results.csv"
    _write_results(rows, csv_path)
    _log(f"Saved result csv: {csv_path}")

    model_order = ["Prophet", "Scenario2"]
    for split in ("valid_wape", "test_wape"):
        plot_path = OUTPUT_DIR / f"horizon_extension_{split}.png"
        rendered = save_horizon_error_plot(
            rows=rows,
            model_order=model_order,
            metric_key=split,
            out_path=plot_path,
            title=f"Scenario19 Horizon Extension ({split})",
        )
        _log(f"Saved horizon plot: {rendered}")

    _log("=" * 80)
    _log("Result summary")
    for horizon in HORIZONS:
        horizon_rows = [r for r in rows if r["horizon"] == horizon]
        for row in sorted(horizon_rows, key=lambda x: str(x["model"])):
            _log(
                f"h={horizon:2d} {row['model']:10s} "
                f"valid_wape={row['valid_wape']:.4f} test_wape={row['test_wape']:.4f}"
            )


if __name__ == "__main__":
    main()
