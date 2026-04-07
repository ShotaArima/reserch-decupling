"""Scenario 17: horizon-dependent common/specific role-gap experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "output"

from src.data import FreshRetailConfig, load_freshretail_dataframe
from src.horizon_role_gap import evaluate_model_for_horizon, predict_for_split_for_horizon, train_model_for_horizon
from src.plotting import save_learning_curve
from src.scenario9_pipeline import (
    COMMON_FEATURE_CANDIDATES,
    SPECIFIC_FEATURE_CANDIDATES,
    AblationMode,
    TrainConfig,
    add_dt_features,
    build_splits,
    resolve_features,
)


@dataclass
class ResultRow:
    horizon: int
    mode: AblationMode
    seed: int
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    test_wape: float
    test_wpe: float
    test_mae: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario17 horizon role-gap experiments.")
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Single-seed run when --seeds is not provided")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds (e.g. 42,43,44)")
    parser.add_argument("--forecast-horizons", type=str, default="1,3,7")
    parser.add_argument("--ablation-modes", type=str, default="both,common_only,specific_only")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_mode_list(raw: str) -> list[AblationMode]:
    valid = {"both", "common_only", "specific_only"}
    modes: list[AblationMode] = []
    for token in raw.split(","):
        mode = token.strip()
        if not mode:
            continue
        if mode not in valid:
            raise ValueError(f"Unsupported ablation mode: {mode}")
        modes.append(mode)  # type: ignore[arg-type]
    return modes


def _write_single_metrics(row: ResultRow, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon", "mode", "seed", "valid_wape", "valid_wpe", "valid_mae", "test_wape", "test_wpe", "test_mae"])
        writer.writerow(
            [
                row.horizon,
                row.mode,
                row.seed,
                f"{row.valid_wape:.6f}",
                f"{row.valid_wpe:.6f}",
                f"{row.valid_mae:.6f}",
                f"{row.test_wape:.6f}",
                f"{row.test_wpe:.6f}",
                f"{row.test_mae:.6f}",
            ]
        )


def _write_predictions_csv(pred_rows: list[tuple[int, str, int, str, int, float, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon", "mode", "seed", "split", "sample_index", "y_true", "y_pred"])
        writer.writerows(pred_rows)


def _write_summary(rows: list[ResultRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon", "mode", "seed", "valid_wape", "valid_wpe", "valid_mae", "test_wape", "test_wpe", "test_mae"])
        for row in rows:
            writer.writerow(
                [
                    row.horizon,
                    row.mode,
                    row.seed,
                    f"{row.valid_wape:.6f}",
                    f"{row.valid_wpe:.6f}",
                    f"{row.valid_mae:.6f}",
                    f"{row.test_wape:.6f}",
                    f"{row.test_wpe:.6f}",
                    f"{row.test_mae:.6f}",
                ]
            )


def _write_relative_contribution(rows: list[ResultRow], out_path: Path) -> None:
    grouped: dict[tuple[int, int], dict[str, ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.horizon, row.seed), {})[row.mode] = row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "horizon",
                "seed",
                "valid_drop_common_wape",
                "valid_drop_specific_wape",
                "test_drop_common_wape",
                "test_drop_specific_wape",
                "valid_drop_common_wpe",
                "valid_drop_specific_wpe",
                "test_drop_common_wpe",
                "test_drop_specific_wpe",
                "valid_drop_common_mae",
                "valid_drop_specific_mae",
                "test_drop_common_mae",
                "test_drop_specific_mae",
            ]
        )

        for (horizon, seed), by_mode in sorted(grouped.items()):
            if not {"both", "common_only", "specific_only"}.issubset(set(by_mode.keys())):
                continue
            both = by_mode["both"]
            common_only = by_mode["common_only"]
            specific_only = by_mode["specific_only"]
            writer.writerow(
                [
                    horizon,
                    seed,
                    f"{common_only.valid_wape - both.valid_wape:.6f}",
                    f"{specific_only.valid_wape - both.valid_wape:.6f}",
                    f"{common_only.test_wape - both.test_wape:.6f}",
                    f"{specific_only.test_wape - both.test_wape:.6f}",
                    f"{common_only.valid_wpe - both.valid_wpe:.6f}",
                    f"{specific_only.valid_wpe - both.valid_wpe:.6f}",
                    f"{common_only.test_wpe - both.test_wpe:.6f}",
                    f"{specific_only.test_wpe - both.test_wpe:.6f}",
                    f"{common_only.valid_mae - both.valid_mae:.6f}",
                    f"{specific_only.valid_mae - both.valid_mae:.6f}",
                    f"{common_only.test_mae - both.test_mae:.6f}",
                    f"{specific_only.test_mae - both.test_mae:.6f}",
                ]
            )


def main() -> None:
    args = _parse_args()
    horizons = _parse_int_list(args.forecast_horizons)
    if not horizons:
        raise ValueError("No forecast horizons were provided")

    modes = _parse_mode_list(args.ablation_modes)
    if not modes:
        raise ValueError("No ablation modes were provided")

    seeds = _parse_int_list(args.seeds) if args.seeds.strip() else [args.seed]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[progress] Loading dataset...")
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = add_dt_features(df)

    common_features, missing_common = resolve_features(df, COMMON_FEATURE_CANDIDATES)
    specific_features, missing_specific = resolve_features(df, SPECIFIC_FEATURE_CANDIDATES)
    print(f"[features] common={common_features}")
    print(f"[features] specific={specific_features}")
    if missing_common:
        print(f"[features] missing_common={missing_common}")
    if missing_specific:
        print(f"[features] missing_specific={missing_specific}")

    if not common_features or not specific_features:
        raise RuntimeError("Scenario17 requires both common and specific feature sets.")

    print(
        f"[config] window_size={args.window_size} steps={args.steps} lr={args.lr} hidden_dim={args.hidden_dim} "
        f"latent_dim={args.latent_dim} horizons={horizons} modes={modes} seeds={seeds}"
    )

    all_rows: list[ResultRow] = []
    all_pred_rows: list[tuple[int, str, int, str, int, float, float]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_cfg = TrainConfig(
            steps=args.steps,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            seed=seed,
            log_interval=args.log_interval,
        )

        for horizon in horizons:
            print(f"\n[progress] Build splits for horizon={horizon}, seed={seed}")
            splits = build_splits(
                df,
                common_features=common_features,
                specific_features=specific_features,
                window_size=args.window_size,
            )

            train_count = max(0, splits.y_train.shape[0] - horizon)
            valid_count = max(0, splits.y_valid.shape[0] - horizon)
            test_count = max(0, splits.y_test.shape[0] - horizon)
            print(
                f"[progress] horizon={horizon} seed={seed} sample_count train={train_count} valid={valid_count} test={test_count}"
            )

            if min(train_count, valid_count, test_count) <= 0:
                print(f"[warn] Skip horizon={horizon} seed={seed} due to insufficient samples.")
                continue

            for mode in modes:
                exp_name = f"scenario17_h{horizon}_{mode}_seed{seed}"
                print(f"[progress] Start training: {exp_name}")
                model, losses = train_model_for_horizon(
                    splits,
                    config=train_cfg,
                    forecast_horizon=horizon,
                    mode=mode,
                    experiment_name=exp_name,
                )

                loss_path = save_learning_curve(
                    losses,
                    OUTPUT_DIR / f"train_loss_h{horizon}_{mode}_seed{seed}.png",
                    title=f"Scenario17 h={horizon} mode={mode} seed={seed}",
                    ylabel="L1 Loss",
                )
                print(f"[saved] {loss_path}")

                metrics = evaluate_model_for_horizon(model, splits, forecast_horizon=horizon, mode=mode)
                row = ResultRow(
                    horizon=horizon,
                    mode=mode,
                    seed=seed,
                    valid_wape=metrics.valid_wape,
                    valid_wpe=metrics.valid_wpe,
                    valid_mae=metrics.valid_mae,
                    test_wape=metrics.test_wape,
                    test_wpe=metrics.test_wpe,
                    test_mae=metrics.test_mae,
                )
                all_rows.append(row)

                metrics_path = OUTPUT_DIR / f"metrics_h{horizon}_{mode}_seed{seed}.csv"
                _write_single_metrics(row, metrics_path)
                print(
                    f"[result] h={horizon} mode={mode} seed={seed} "
                    f"valid_wape={metrics.valid_wape:.4f} test_wape={metrics.test_wape:.4f}"
                )
                print(f"[saved] {metrics_path}")

                pred_path = OUTPUT_DIR / f"predictions_h{horizon}_{mode}_seed{seed}.csv"
                combo_rows: list[tuple[int, str, int, str, int, float, float]] = []
                for split in ("valid", "test"):
                    y_true, y_pred = predict_for_split_for_horizon(
                        model,
                        splits,
                        split=split,
                        forecast_horizon=horizon,
                        mode=mode,
                    )
                    for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
                        row_tuple = (horizon, mode, seed, split, idx, float(yt), float(yp))
                        combo_rows.append(row_tuple)
                        all_pred_rows.append(row_tuple)
                _write_predictions_csv(combo_rows, pred_path)
                print(f"[saved] {pred_path} (rows={len(combo_rows)})")

    summary_path = OUTPUT_DIR / "summary_by_horizon.csv"
    _write_summary(all_rows, summary_path)
    print(f"[saved] {summary_path}")

    all_preds_path = OUTPUT_DIR / "scenario17_predictions_all.csv"
    _write_predictions_csv(all_pred_rows, all_preds_path)
    print(f"[saved] {all_preds_path}")

    contrib_path = OUTPUT_DIR / "relative_contribution.csv"
    _write_relative_contribution(all_rows, contrib_path)
    print(f"[saved] {contrib_path}")

    print("\n[summary] Scenario17 completed.")


if __name__ == "__main__":
    main()
