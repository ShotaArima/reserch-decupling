"""Scenario 9: common/specific feature-assignment experiments.

Exp-0: same features for both branches (Scenario2-like baseline)
Exp-1: role-aware feature split (main)
Exp-2: swapped split (control)
"""

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
from src.plotting import save_learning_curve, save_metric_bar_chart
from src.scenario9_pipeline import (
    COMMON_FEATURE_CANDIDATES,
    SPECIFIC_FEATURE_CANDIDATES,
    AblationMode,
    TrainConfig,
    add_dt_features,
    build_splits,
    evaluate_model,
    predict_for_split,
    resolve_features,
    train_model,
)


@dataclass
class ExperimentDef:
    name: str
    common_features: list[str]
    specific_features: list[str]


@dataclass
class ResultRow:
    experiment: str
    mode: AblationMode
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    test_wape: float
    test_wpe: float
    test_mae: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario9 common/specific assignment experiments.")
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def _write_metrics_csv(rows: list[ResultRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "mode", "valid_wape", "valid_wpe", "valid_mae", "test_wape", "test_wpe", "test_mae"])
        for row in rows:
            writer.writerow(
                [
                    row.experiment,
                    row.mode,
                    f"{row.valid_wape:.6f}",
                    f"{row.valid_wpe:.6f}",
                    f"{row.valid_mae:.6f}",
                    f"{row.test_wape:.6f}",
                    f"{row.test_wpe:.6f}",
                    f"{row.test_mae:.6f}",
                ]
            )


def _write_predictions_csv(pred_rows: list[tuple[str, str, str, int, float, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "mode", "split", "sample_index", "y_true", "y_pred"])
        writer.writerows(pred_rows)


def _log_feature_resolution(tag: str, present: list[str], missing: list[str]) -> None:
    print(f"[features] {tag}: present={present}")
    if missing:
        print(f"[features] {tag}: missing={missing}")


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[progress] Loading dataset...")
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = add_dt_features(df)

    common_features, missing_common = resolve_features(df, COMMON_FEATURE_CANDIDATES)
    specific_features, missing_specific = resolve_features(df, SPECIFIC_FEATURE_CANDIDATES)

    _log_feature_resolution("common_candidates", common_features, missing_common)
    _log_feature_resolution("specific_candidates", specific_features, missing_specific)

    if not common_features:
        raise RuntimeError("No common features available for Scenario9.")
    if not specific_features:
        raise RuntimeError("No specific features available for Scenario9.")

    combined_features = sorted(set(common_features + specific_features))

    experiments = [
        ExperimentDef(name="exp0_same_input", common_features=combined_features, specific_features=combined_features),
        ExperimentDef(name="exp1_role_split", common_features=common_features, specific_features=specific_features),
        ExperimentDef(name="exp2_swapped_split", common_features=specific_features, specific_features=common_features),
    ]

    train_cfg = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    print(
        f"[config] window_size={args.window_size}, steps={args.steps}, lr={args.lr}, "
        f"hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}, seed={args.seed}"
    )

    all_rows: list[ResultRow] = []
    pred_rows: list[tuple[str, str, str, int, float, float]] = []

    for exp in experiments:
        print(f"\n[progress] Running {exp.name}")
        print(f"[params] common_features={exp.common_features}")
        print(f"[params] specific_features={exp.specific_features}")

        splits = build_splits(
            df,
            common_features=exp.common_features,
            specific_features=exp.specific_features,
            window_size=args.window_size,
        )

        model, losses = train_model(splits, config=train_cfg, experiment_name=exp.name)
        curve_path = save_learning_curve(
            losses,
            OUTPUT_DIR / f"{exp.name}_train_loss.png",
            title=f"Scenario9 {exp.name} Train Loss",
            ylabel="L1 Loss",
        )
        print(f"[saved] {curve_path}")

        for mode in ("both", "common_only", "specific_only"):
            metrics = evaluate_model(model, splits, mode=mode)
            all_rows.append(
                ResultRow(
                    experiment=exp.name,
                    mode=mode,
                    valid_wape=metrics.valid_wape,
                    valid_wpe=metrics.valid_wpe,
                    valid_mae=metrics.valid_mae,
                    test_wape=metrics.test_wape,
                    test_wpe=metrics.test_wpe,
                    test_mae=metrics.test_mae,
                )
            )
            print(
                f"[result] exp={exp.name} mode={mode} "
                f"valid_wape={metrics.valid_wape:.4f} valid_wpe={metrics.valid_wpe:.4f} valid_mae={metrics.valid_mae:.4f} "
                f"test_wape={metrics.test_wape:.4f} test_wpe={metrics.test_wpe:.4f} test_mae={metrics.test_mae:.4f}"
            )

            for split in ("valid", "test"):
                y_true, y_pred = predict_for_split(model, splits, split=split, mode=mode)
                for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
                    pred_rows.append((exp.name, mode, split, idx, float(yt), float(yp)))

    metrics_csv = OUTPUT_DIR / "scenario9_metrics.csv"
    _write_metrics_csv(all_rows, metrics_csv)
    print(f"[saved] {metrics_csv}")

    pred_csv = OUTPUT_DIR / "scenario9_predictions.csv"
    _write_predictions_csv(pred_rows, pred_csv)
    print(f"[saved] {pred_csv}")

    labels = [f"{r.experiment}:{r.mode}" for r in all_rows]
    valid_wape_values = [r.valid_wape for r in all_rows]
    test_wape_values = [r.test_wape for r in all_rows]

    valid_plot = save_metric_bar_chart(
        labels,
        valid_wape_values,
        OUTPUT_DIR / "scenario9_valid_wape.png",
        title="Scenario9 valid WAPE (exp x mode)",
        ylabel="WAPE",
    )
    test_plot = save_metric_bar_chart(
        labels,
        test_wape_values,
        OUTPUT_DIR / "scenario9_test_wape.png",
        title="Scenario9 test WAPE (exp x mode)",
        ylabel="WAPE",
    )
    print(f"[saved] {valid_plot}")
    print(f"[saved] {test_plot}")

    print("\n[summary] Scenario9 completed.")


if __name__ == "__main__":
    main()
