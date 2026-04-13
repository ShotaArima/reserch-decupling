"""Scenario 18: Prophet-oriented input schema vs sequential VAE(common/specific)."""

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
from src.plotting import save_learning_curve
from src.scenario18_pipeline import (
    ModelName,
    Scenario18TrainConfig,
    build_splits_for_scenario18,
    evaluate_vae_model,
    run_prophet_baseline,
    train_vae_model,
)
from src.scenario9_pipeline import STOCK_FEATURE_CANDIDATES, add_dt_features, resolve_features


@dataclass
class ResultRow:
    model: str
    ablation_mode: str
    lookback: int
    horizon: int
    seed: int
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    valid_rmse: float
    valid_smape: float
    test_wape: float
    test_wpe: float
    test_mae: float
    test_rmse: float
    test_smape: float


COMMON_FEATURE_CANDIDATES: tuple[str, ...] = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "holiday_flag",
    "dt_weekday",
    "dt_month",
    "dt_day",
    "dt_is_weekend",
    "discount",
    "activity_flag",
)

SPECIFIC_FEATURE_CANDIDATES: tuple[str, ...] = (
    "sale_amount",
    "hours_sale",
    "discount",
    "activity_flag",
    *STOCK_FEATURE_CANDIDATES,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario18 experiments.")
    parser.add_argument("--lookback", type=int, default=28)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument(
        "--model",
        type=str,
        default="v2_seq_vae_transition",
        choices=["p0_prophet", "p1_prophet_reg", "v0_flatten_vae", "v1_seq_vae", "v2_seq_vae_transition"],
    )
    parser.add_argument("--ablation-mode", type=str, default="both", choices=["both", "common_only", "specific_only"])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--transition-weight", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def _write_metrics(row: ResultRow, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "ablation_mode",
                "lookback",
                "horizon",
                "seed",
                "valid_wape",
                "valid_wpe",
                "valid_mae",
                "valid_rmse",
                "valid_smape",
                "test_wape",
                "test_wpe",
                "test_mae",
                "test_rmse",
                "test_smape",
            ]
        )
        writer.writerow(
            [
                row.model,
                row.ablation_mode,
                row.lookback,
                row.horizon,
                row.seed,
                f"{row.valid_wape:.6f}",
                f"{row.valid_wpe:.6f}",
                f"{row.valid_mae:.6f}",
                f"{row.valid_rmse:.6f}",
                f"{row.valid_smape:.6f}",
                f"{row.test_wape:.6f}",
                f"{row.test_wpe:.6f}",
                f"{row.test_mae:.6f}",
                f"{row.test_rmse:.6f}",
                f"{row.test_smape:.6f}",
            ]
        )


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        raise RuntimeError("Scenario18 requires both common and specific features")

    splits = build_splits_for_scenario18(
        df,
        lookback=args.lookback,
        common_features=common_features,
        specific_features=specific_features,
    )

    train_count = max(0, splits.y_train.shape[0] - args.horizon)
    valid_count = max(0, splits.y_valid.shape[0] - args.horizon)
    test_count = max(0, splits.y_test.shape[0] - args.horizon)
    print(f"[split] sample_count train={train_count} valid={valid_count} test={test_count}")

    if min(train_count, valid_count, test_count) <= 0:
        raise RuntimeError("Insufficient samples for the given horizon")

    model_name = args.model

    if model_name in {"p0_prophet", "p1_prophet_reg"}:
        use_reg = model_name == "p1_prophet_reg"
        metrics = run_prophet_baseline(splits, horizon=args.horizon, use_regressor=use_reg)
        losses: list[float] = []
    else:
        train_cfg = Scenario18TrainConfig(
            steps=args.steps,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            beta_kl=args.beta_kl,
            transition_weight=args.transition_weight,
            seed=args.seed,
            log_interval=args.log_interval,
        )

        model, losses = train_vae_model(
            splits,
            model_name=model_name,  # type: ignore[arg-type]
            horizon=args.horizon,
            mode=args.ablation_mode,  # type: ignore[arg-type]
            config=train_cfg,
        )
        metrics = evaluate_vae_model(
            model,
            splits,
            model_name=model_name,  # type: ignore[arg-type]
            horizon=args.horizon,
            mode=args.ablation_mode,  # type: ignore[arg-type]
        )

    row = ResultRow(
        model=model_name,
        ablation_mode=args.ablation_mode,
        lookback=args.lookback,
        horizon=args.horizon,
        seed=args.seed,
        valid_wape=metrics.valid_wape,
        valid_wpe=metrics.valid_wpe,
        valid_mae=metrics.valid_mae,
        valid_rmse=metrics.valid_rmse,
        valid_smape=metrics.valid_smape,
        test_wape=metrics.test_wape,
        test_wpe=metrics.test_wpe,
        test_mae=metrics.test_mae,
        test_rmse=metrics.test_rmse,
        test_smape=metrics.test_smape,
    )

    metrics_file = OUTPUT_DIR / f"metrics_{model_name}_{args.ablation_mode}_seed{args.seed}.csv"
    _write_metrics(row, metrics_file)

    if losses:
        fig_path = OUTPUT_DIR / f"train_loss_{model_name}_{args.ablation_mode}_seed{args.seed}.png"
        save_learning_curve(losses, title=f"scenario18 {model_name} ({args.ablation_mode})", output_path=fig_path)

    print("[result] ")
    print(
        f"model={model_name} mode={args.ablation_mode} "
        f"valid_wape={metrics.valid_wape:.4f} test_wape={metrics.test_wape:.4f} "
        f"valid_mae={metrics.valid_mae:.4f} test_mae={metrics.test_mae:.4f}"
    )
    print(f"[saved] {metrics_file}")


if __name__ == "__main__":
    main()
