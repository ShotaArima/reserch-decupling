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
from src.scenario15_pipeline import (
    AblationMode,
    TrainConfig,
    add_scenario15_features,
    build_splits,
    collect_common_latent,
    evaluate_model,
    resolve_arm_features,
    train_model,
    train_probe_classifier,
)


@dataclass
class Scenario15Row:
    arm: str
    seed: int
    hierarchy_dim: int
    mode: AblationMode
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    test_wape: float
    test_wpe: float
    test_mae: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario15: strengthen common branch with calendar/weather/hierarchy features.")
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--probe-steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-lr", type=float, default=5e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="42,52,62")
    return parser.parse_args()


def _write_metrics(rows: list[Scenario15Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arm",
            "seed",
            "hierarchy_dim",
            "mode",
            "valid_wape",
            "valid_wpe",
            "valid_mae",
            "test_wape",
            "test_wpe",
            "test_mae",
        ])
        for row in rows:
            writer.writerow([
                row.arm,
                row.seed,
                row.hierarchy_dim,
                row.mode,
                f"{row.valid_wape:.6f}",
                f"{row.valid_wpe:.6f}",
                f"{row.valid_mae:.6f}",
                f"{row.test_wape:.6f}",
                f"{row.test_wpe:.6f}",
                f"{row.test_mae:.6f}",
            ])


def _write_probe_metrics(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "seed", "hierarchy_dim", "split", "target", "accuracy", "macro_f1"])
        for row in rows:
            writer.writerow([
                row["arm"],
                row["seed"],
                row["hierarchy_dim"],
                row["split"],
                row["target"],
                f"{float(row['accuracy']):.6f}",
                f"{float(row['macro_f1']):.6f}",
            ])


def _write_config_summary(rows: list[Scenario15Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        key = (row.arm, row.hierarchy_dim, row.mode)
        grouped.setdefault(key, []).append(row.test_wape)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "hierarchy_dim", "mode", "test_wape_mean", "test_wape_std", "num_runs"])
        for (arm, hier_dim, mode), values in sorted(grouped.items()):
            arr = np.asarray(values, dtype=np.float64)
            writer.writerow([arm, hier_dim, mode, f"{arr.mean():.6f}", f"{arr.std(ddof=0):.6f}", len(values)])


def _probe_targets_for_split(splits, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split == "valid":
        return (
            splits.probe_weekday_valid[:-1],
            splits.probe_month_valid[:-1],
            splits.probe_hierarchy_valid[:-1],
        )
    return (
        splits.probe_weekday_test[:-1],
        splits.probe_month_test[:-1],
        splits.probe_hierarchy_test[:-1],
    )


def _run_probes(
    arm: str,
    seed: int,
    hierarchy_dim: int,
    z_train: np.ndarray,
    z_valid: np.ndarray,
    z_test: np.ndarray,
    splits,
    *,
    probe_steps: int,
    probe_lr: float,
    log_interval: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    y_weekday_train = splits.probe_weekday_train[:-1]
    y_month_train = splits.probe_month_train[:-1]
    y_hier_train = splits.probe_hierarchy_train[:-1]

    for split, z_eval in (("valid", z_valid), ("test", z_test)):
        y_weekday_eval, y_month_eval, y_hier_eval = _probe_targets_for_split(splits, split)

        for target_name, y_train, y_eval in (
            ("weekday", y_weekday_train, y_weekday_eval),
            ("month", y_month_train, y_month_eval),
            ("hierarchy", y_hier_train, y_hier_eval),
        ):
            acc, macro_f1 = train_probe_classifier(
                z_train,
                y_train,
                z_eval,
                y_eval,
                steps=probe_steps,
                lr=probe_lr,
                log_interval=log_interval,
                tag=f"{arm}-seed{seed}-h{hierarchy_dim}-{split}-{target_name}",
            )
            record = {
                "arm": arm,
                "seed": seed,
                "hierarchy_dim": hierarchy_dim,
                "split": split,
                "target": target_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
            }
            records.append(record)
            print(
                f"[probe-result] arm={arm} seed={seed} hdim={hierarchy_dim} split={split} "
                f"target={target_name} accuracy={acc:.4f} macro_f1={macro_f1:.4f}"
            )

    return records


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("[progress] Loading dataset...")
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = add_scenario15_features(df)

    arm_hierarchy_dims = {
        "A0": [8],
        "A1": [8],
        "A2": [8],
        "A3": [4, 8, 16],
    }

    metric_rows: list[Scenario15Row] = []
    probe_rows: list[dict[str, object]] = []

    for arm in ("A0", "A1", "A2", "A3"):
        common_features, specific_features, cat_features = resolve_arm_features(df, arm)
        if not common_features:
            raise RuntimeError(f"No common features resolved for arm={arm}")
        if not specific_features:
            raise RuntimeError(f"No specific features resolved for arm={arm}")

        print(f"\n[progress] arm={arm} common_features={len(common_features)} specific_features={len(specific_features)}")
        print(f"[progress] arm={arm} categorical_features={cat_features}")

        for hierarchy_dim in arm_hierarchy_dims[arm]:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                run_name = f"{arm}_seed{seed}_h{hierarchy_dim}"
                print(f"\n[progress] Training {run_name}")

                splits = build_splits(
                    df,
                    common_features=common_features,
                    specific_features=specific_features,
                    cat_features=cat_features,
                    window_size=args.window_size,
                )

                train_cfg = TrainConfig(
                    steps=args.steps,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    latent_dim=args.latent_dim,
                    seed=seed,
                    log_interval=args.log_interval,
                    hierarchy_dim=hierarchy_dim,
                )

                model, losses = train_model(splits, config=train_cfg, experiment_name=run_name)
                curve_path = save_learning_curve(
                    losses,
                    OUTPUT_DIR / f"{run_name}_train_loss.png",
                    title=f"Scenario15 {run_name} Train Loss",
                    ylabel="L1 Loss",
                )
                print(f"[saved] {curve_path}")

                for mode in ("local_only", "common_only", "both"):
                    metrics = evaluate_model(model, splits, mode=mode)
                    metric_rows.append(
                        Scenario15Row(
                            arm=arm,
                            seed=seed,
                            hierarchy_dim=hierarchy_dim,
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
                        f"[result] run={run_name} mode={mode} "
                        f"valid_wape={metrics.valid_wape:.4f} valid_wpe={metrics.valid_wpe:.4f} valid_mae={metrics.valid_mae:.4f} "
                        f"test_wape={metrics.test_wape:.4f} test_wpe={metrics.test_wpe:.4f} test_mae={metrics.test_mae:.4f}"
                    )

                z_train = collect_common_latent(model, splits, split="train")
                z_valid = collect_common_latent(model, splits, split="valid")
                z_test = collect_common_latent(model, splits, split="test")
                probe_rows.extend(
                    _run_probes(
                        arm,
                        seed,
                        hierarchy_dim,
                        z_train,
                        z_valid,
                        z_test,
                        splits,
                        probe_steps=args.probe_steps,
                        probe_lr=args.probe_lr,
                        log_interval=args.log_interval,
                    )
                )

    metrics_csv = OUTPUT_DIR / "scenario15_metrics.csv"
    _write_metrics(metric_rows, metrics_csv)
    print(f"[saved] {metrics_csv}")

    probe_csv = OUTPUT_DIR / "scenario15_probe_metrics.csv"
    _write_probe_metrics(probe_rows, probe_csv)
    print(f"[saved] {probe_csv}")

    config_csv = OUTPUT_DIR / "scenario15_config_results.csv"
    _write_config_summary(metric_rows, config_csv)
    print(f"[saved] {config_csv}")

    labels = [f"{row.arm}:{row.mode}:h{row.hierarchy_dim}:s{row.seed}" for row in metric_rows]
    valid_wape = [row.valid_wape for row in metric_rows]
    test_wape = [row.test_wape for row in metric_rows]

    valid_plot = save_metric_bar_chart(
        labels,
        valid_wape,
        OUTPUT_DIR / "scenario15_valid_wape.png",
        title="Scenario15 Valid WAPE",
        ylabel="WAPE",
    )
    test_plot = save_metric_bar_chart(
        labels,
        test_wape,
        OUTPUT_DIR / "scenario15_test_wape.png",
        title="Scenario15 Test WAPE",
        ylabel="WAPE",
    )
    print(f"[saved] {valid_plot}")
    print(f"[saved] {test_plot}")

    print("\n[summary] Scenario15 completed.")


if __name__ == "__main__":
    main()
