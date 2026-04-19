"""Scenario 19: Prophet vs VAE-decoupling horizon extension experiment."""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario19: Prophet vs VAE horizon extension.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--horizons", type=str, default="14,21,28,35,42")
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--skip-prophet", action="store_true")
    parser.add_argument(
        "--prophet-max-samples",
        type=int,
        default=256,
        help="Limit samples per split for Prophet to avoid very long runtime.",
    )
    return parser.parse_args()


def _parse_horizons(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cap_prophet_indices(length: int, max_samples: int | None, *, seed: int) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or length <= max_samples:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_samples, replace=False))


def _predict_scenario2_batched(body, head, x: torch.Tensor, batch_size: int) -> np.ndarray:
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            batch = x[start:end]
            pred = head(*body(batch)[1:]).numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


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
    args = _parse_args()
    seed = args.seed
    horizons = _parse_horizons(args.horizons)
    if not horizons:
        raise ValueError("No horizons provided.")

    device = _resolve_device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _log(f"Seed fixed at {seed}")
    _log(
        f"Runtime config: device={device} horizons={horizons} train_steps={args.train_steps} "
        f"batch_size={args.batch_size} skip_prophet={args.skip_prophet} "
        f"prophet_max_samples={args.prophet_max_samples}"
    )
    _log("Loading FreshRetailNet-50K dataframe...")
    df = load_freshretail_dataframe(FreshRetailConfig())
    _log(f"Loaded dataframe rows={len(df)} cols={len(df.columns)}")

    rows: list[dict[str, float | str | int]] = []

    for horizon in horizons:
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

        if not args.skip_prophet:
            _log("Running Prophet baseline...")
            try:
                valid_indices = _cap_prophet_indices(len(y_valid_np), args.prophet_max_samples, seed=seed + horizon)
                test_indices = _cap_prophet_indices(len(y_test_np), args.prophet_max_samples, seed=seed + horizon + 10_000)
                valid_raw_prophet = valid_raw[:-1][valid_indices]
                test_raw_prophet = test_raw[:-1][test_indices]

                prophet_valid = predict_prophet_next_step_per_sample(
                    valid_raw_prophet,
                    target_feature_index=TARGET_FEATURE_INDEX,
                )
                prophet_test = predict_prophet_next_step_per_sample(
                    test_raw_prophet,
                    target_feature_index=TARGET_FEATURE_INDEX,
                )
                y_valid_prophet = y_valid_np[valid_indices]
                y_test_prophet = y_test_np[test_indices]
                prophet_row = _score_row(
                    horizon,
                    f"Prophet(n={len(valid_raw_prophet)}/{len(test_raw_prophet)})",
                    y_valid_prophet,
                    prophet_valid,
                    y_test_prophet,
                    prophet_test,
                )
                rows.append(prophet_row)
                _log(
                    "Prophet finished: "
                    f"valid_wape={prophet_row['valid_wape']:.4f} test_wape={prophet_row['test_wape']:.4f}"
                )
            except Exception as exc:
                _log(f"Prophet skipped on horizon={horizon}: {exc}")
        else:
            _log("Skipping Prophet baseline by --skip-prophet.")

        _log(f"Training VAE decoupling model (Scenario2) steps={args.train_steps}...")
        s2_body, s2_head, s2_losses = train_scenario2_model(
            x_train_s,
            y_train_s,
            feature_dim=len(BASE_FEATURES),
            window_size=horizon,
            steps=args.train_steps,
            batch_size=args.batch_size,
            device=device,
            log_interval=args.log_interval,
        )
        _log(
            f"Scenario2 loss snapshot: start={s2_losses[0]:.4f} "
            f"mid={s2_losses[len(s2_losses)//2]:.4f} end={s2_losses[-1]:.4f}"
        )

        s2_valid = _predict_scenario2_batched(s2_body, s2_head, x_valid_s, batch_size=max(1, args.batch_size))
        s2_test = _predict_scenario2_batched(s2_body, s2_head, x_test_s, batch_size=max(1, args.batch_size))

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

    model_order = list(dict.fromkeys(str(r["model"]) for r in rows))
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
    for horizon in horizons:
        horizon_rows = [r for r in rows if r["horizon"] == horizon]
        for row in sorted(horizon_rows, key=lambda x: str(x["model"])):
            _log(
                f"h={horizon:2d} {row['model']:10s} "
                f"valid_wape={row['valid_wape']:.4f} test_wape={row['test_wape']:.4f}"
            )


if __name__ == "__main__":
    main()
