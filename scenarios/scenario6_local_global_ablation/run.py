"""Scenario 6: local / global / both ablation for forecast head."""

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

from src.data import (
    FreshRetailConfig,
    build_window_tensor,
    extract_last_timestep_feature,
    load_freshretail_dataframe,
    normalize_by_train_stats,
    split_train_valid_test,
)
from src.metrics import wape, wpe
from src.plotting import save_learning_curve, save_metric_bar_chart
from src.scenario6_ablation import (
    AblationMode,
    LatentSplits,
    select_eval_inputs,
    train_ablation_head,
    train_scenario2_latents,
    train_stage1_recovery_latents,
)

BASE_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
STAGE1_FEATURES = ["sale_amount", "hours_stock_status", "discount", "holiday_flag", "activity_flag"]
WINDOW_SIZE = 14
TARGET_FEATURE_INDEX = 0
ABLATION_MODES: tuple[AblationMode, ...] = ("local_only", "global_only", "both")


@dataclass
class EvalResult:
    condition: str
    valid_wape: float
    valid_wpe: float
    test_wape: float
    test_wpe: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario 6 local/global/both ablation.")
    parser.add_argument(
        "--latent-source",
        choices=["scenario2", "stage1"],
        default="scenario2",
        help="Use scenario2 end-to-end latents or stage1 recovery latents.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Training steps for latent and head training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _prepare_tensors(df, features: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_all_raw = build_window_tensor(df, features, window_size=WINDOW_SIZE)
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
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def _write_results_csv(results: list[EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "valid_wape", "valid_wpe", "test_wape", "test_wpe"])
        for r in results:
            writer.writerow([r.condition, f"{r.valid_wape:.6f}", f"{r.valid_wpe:.6f}", f"{r.test_wape:.6f}", f"{r.test_wpe:.6f}"])


def _evaluate_mode(mode: AblationMode, latents: LatentSplits, y_train: torch.Tensor, y_valid: torch.Tensor, y_test: torch.Tensor, steps: int) -> tuple[EvalResult, list[float]]:
    head, losses = train_ablation_head(
        latents.local_train,
        latents.global_train,
        y_train,
        mode=mode,
        steps=steps,
    )

    with torch.no_grad():
        local_valid, global_valid, y_valid_s = select_eval_inputs(latents.local_valid, latents.global_valid, y_valid, mode=mode)
        local_test, global_test, y_test_s = select_eval_inputs(latents.local_test, latents.global_test, y_test, mode=mode)
        pred_valid = head(local_valid, global_valid).numpy().reshape(-1)
        pred_test = head(local_test, global_test).numpy().reshape(-1)

    y_valid_np = y_valid_s.numpy().reshape(-1)
    y_test_np = y_test_s.numpy().reshape(-1)

    result = EvalResult(
        condition=mode,
        valid_wape=wape(y_valid_np, pred_valid),
        valid_wpe=wpe(y_valid_np, pred_valid),
        test_wape=wape(y_test_np, pred_test),
        test_wpe=wpe(y_test_np, pred_test),
    )
    return result, losses


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_freshretail_dataframe(FreshRetailConfig())

    if args.latent_source == "scenario2":
        features = BASE_FEATURES
    else:
        features = STAGE1_FEATURES

    x_train, x_valid, x_test, y_train, y_valid, y_test = _prepare_tensors(df, features)

    if args.latent_source == "scenario2":
        latent_artifacts = train_scenario2_latents(
            x_train,
            x_valid,
            x_test,
            y_train,
            feature_dim=len(BASE_FEATURES),
            window_size=WINDOW_SIZE,
            steps=args.steps,
        )
    else:
        latent_artifacts = train_stage1_recovery_latents(
            x_train,
            x_valid,
            x_test,
            feature_dim=len(STAGE1_FEATURES),
            window_size=WINDOW_SIZE,
            steps=args.steps,
        )

    source_curve = save_learning_curve(
        latent_artifacts.train_losses,
        OUTPUT_DIR / f"latent_source_{args.latent_source}_train_loss.png",
        title=f"Scenario6 Latent Source Train Loss ({args.latent_source})",
        ylabel="Loss",
    )
    print(f"Saved: {source_curve}")

    results: list[EvalResult] = []

    for mode in ABLATION_MODES:
        result, losses = _evaluate_mode(
            mode,
            latent_artifacts.latents,
            y_train,
            y_valid,
            y_test,
            steps=args.steps,
        )
        results.append(result)
        curve = save_learning_curve(
            losses,
            OUTPUT_DIR / f"{mode}_train_loss.png",
            title=f"Scenario6 {mode} Train Loss",
            ylabel="L1 Loss",
        )
        print(f"Saved: {curve}")
        print(
            f"{mode}: valid_wape={result.valid_wape:.4f} valid_wpe={result.valid_wpe:.4f} "
            f"test_wape={result.test_wape:.4f} test_wpe={result.test_wpe:.4f}"
        )

    out_csv = OUTPUT_DIR / f"scenario6_results_{args.latent_source}.csv"
    _write_results_csv(results, out_csv)
    print(f"Saved: {out_csv}")

    labels = [r.condition for r in results]
    valid_wape_values = [r.valid_wape for r in results]
    test_wape_values = [r.test_wape for r in results]

    valid_plot = save_metric_bar_chart(
        labels,
        valid_wape_values,
        OUTPUT_DIR / f"scenario6_valid_wape_{args.latent_source}.png",
        title=f"Scenario6 valid WAPE ({args.latent_source})",
        ylabel="WAPE",
    )
    test_plot = save_metric_bar_chart(
        labels,
        test_wape_values,
        OUTPUT_DIR / f"scenario6_test_wape_{args.latent_source}.png",
        title=f"Scenario6 test WAPE ({args.latent_source})",
        ylabel="WAPE",
    )
    print(f"Saved: {valid_plot}")
    print(f"Saved: {test_plot}")


if __name__ == "__main__":
    main()
