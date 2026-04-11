"""Scenario16: stock feature placement comparison (specific vs common)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "output"

from src.data import FreshRetailConfig, build_window_tensor, load_freshretail_dataframe, split_train_valid_test
from src.metrics import mae, wape, wpe
from src.plotting import save_learning_curve
from src.scenario16_pipeline import (
    build_scenario16_experiments,
    extract_latents,
    fit_linear_probe_accuracy,
    write_csv,
)
from src.scenario9_pipeline import (
    COMMON_FEATURE_CANDIDATES,
    SPECIFIC_FEATURE_CANDIDATES,
    STOCK_FEATURE_CANDIDATES,
    TrainConfig,
    add_dt_features,
    build_splits,
    evaluate_model,
    predict_for_split,
    resolve_features,
    train_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario16 stock placement experiments.")
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def _parse_seed_list(seed_text: str) -> list[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def _build_stock_status_targets(df, window_size: int) -> dict[str, np.ndarray]:
    if "hours_stock_status" not in df.columns:
        return {}
    stock_status = build_window_tensor(df, ["hours_stock_status"], window_size=window_size)
    target = stock_status[:, -1, 0]
    train, valid, test = split_train_valid_test(target)
    return {
        "train": (train[1:] > 0).astype(np.float32),
        "valid": (valid[1:] > 0).astype(np.float32),
        "test": (test[1:] > 0).astype(np.float32),
    }


def _compute_subset_rows(experiment: str, seed: int, mode: str, split: str, y_true: np.ndarray, y_pred: np.ndarray, stock_mask: np.ndarray) -> list[list[object]]:
    rows: list[list[object]] = []
    subset_masks = {
        "all": np.ones_like(stock_mask, dtype=bool),
        "stockout": stock_mask,
        "non_stockout": ~stock_mask,
    }
    for subset_name, mask in subset_masks.items():
        if int(mask.sum()) == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append(
            [
                experiment,
                seed,
                mode,
                split,
                subset_name,
                f"{wape(yt, yp):.6f}",
                f"{wpe(yt, yp):.6f}",
                f"{mae(yt, yp):.6f}",
                int(mask.sum()),
            ]
        )
    return rows


def main() -> None:
    args = _parse_args()
    seeds = _parse_seed_list(args.seeds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[progress] loading dataset")
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = add_dt_features(df)

    common_features, _ = resolve_features(df, COMMON_FEATURE_CANDIDATES)
    specific_base, _ = resolve_features(df, SPECIFIC_FEATURE_CANDIDATES)
    stock_features, missing_stock = resolve_features(df, STOCK_FEATURE_CANDIDATES)

    print(f"[features] common={common_features}")
    print(f"[features] specific_base={specific_base}")
    print(f"[features] stock_present={stock_features}")
    if missing_stock:
        print(f"[features] stock_missing={missing_stock}")

    experiments = build_scenario16_experiments(
        common_features=common_features,
        specific_base_features=specific_base,
        stock_features=stock_features,
    )

    stock_targets = _build_stock_status_targets(df, window_size=args.window_size)

    metric_rows: list[list[object]] = []
    subset_rows: list[list[object]] = []
    probe_rows: list[list[object]] = []

    for exp in experiments:
        print(f"\n[progress] scenario16 experiment start: {exp.name}")
        print(f"[progress] common_features={exp.common_features}")
        print(f"[progress] specific_features={exp.specific_features}")

        for seed in seeds:
            print(f"[progress] training exp={exp.name} seed={seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            splits = build_splits(
                df,
                common_features=exp.common_features,
                specific_features=exp.specific_features,
                window_size=args.window_size,
            )

            train_cfg = TrainConfig(
                steps=args.steps,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                seed=seed,
                log_interval=args.log_interval,
            )
            model, losses = train_model(splits, config=train_cfg, experiment_name=f"{exp.name}_seed{seed}")
            curve_path = save_learning_curve(
                losses,
                OUTPUT_DIR / f"{exp.name}_seed{seed}_train_loss.png",
                title=f"Scenario16 {exp.name} seed={seed} train loss",
                ylabel="L1 loss",
            )
            print(f"[saved] {curve_path}")

            for mode in ("both", "common_only", "specific_only"):
                metrics = evaluate_model(model, splits, mode=mode)
                metric_rows.append(
                    [
                        exp.name,
                        seed,
                        mode,
                        f"{metrics.valid_wape:.6f}",
                        f"{metrics.valid_wpe:.6f}",
                        f"{metrics.valid_mae:.6f}",
                        f"{metrics.test_wape:.6f}",
                        f"{metrics.test_wpe:.6f}",
                        f"{metrics.test_mae:.6f}",
                    ]
                )
                print(
                    f"[result] exp={exp.name} seed={seed} mode={mode} "
                    f"valid_wape={metrics.valid_wape:.4f} test_wape={metrics.test_wape:.4f}"
                )

                for split in ("valid", "test"):
                    y_true, y_pred = predict_for_split(model, splits, split=split, mode=mode)
                    if split in stock_targets:
                        stock_mask = stock_targets[split].astype(bool)
                        subset_rows.extend(_compute_subset_rows(exp.name, seed, mode, split, y_true, y_pred, stock_mask))

            if stock_targets:
                zc_train, zs_train = extract_latents(model, splits, split="train")
                zc_test, zs_test = extract_latents(model, splits, split="test")
                y_probe_train = stock_targets["train"]
                y_probe_test = stock_targets["test"]

                common_acc = fit_linear_probe_accuracy(zc_train, y_probe_train, zc_test, y_probe_test)
                specific_acc = fit_linear_probe_accuracy(zs_train, y_probe_train, zs_test, y_probe_test)
                probe_rows.append([exp.name, seed, "stock_status_binary", "z_common", f"{common_acc:.6f}"])
                probe_rows.append([exp.name, seed, "stock_status_binary", "z_specific", f"{specific_acc:.6f}"])
                print(
                    f"[probe] exp={exp.name} seed={seed} "
                    f"common_acc={common_acc:.4f} specific_acc={specific_acc:.4f}"
                )

    write_csv(
        OUTPUT_DIR / "scenario16_metrics_overall.csv",
        [
            "experiment",
            "seed",
            "mode",
            "valid_wape",
            "valid_wpe",
            "valid_mae",
            "test_wape",
            "test_wpe",
            "test_mae",
        ],
        metric_rows,
    )
    write_csv(
        OUTPUT_DIR / "scenario16_metrics_by_subset.csv",
        ["experiment", "seed", "mode", "split", "subset", "wape", "wpe", "mae", "n"],
        subset_rows,
    )
    write_csv(
        OUTPUT_DIR / "scenario16_probe_scores.csv",
        ["experiment", "seed", "probe_task", "latent", "accuracy"],
        probe_rows,
    )

    summary_lines = [
        "# Scenario16 summary",
        "",
        f"- experiments: {[exp.name for exp in experiments]}",
        f"- seeds: {seeds}",
        f"- steps: {args.steps}",
        f"- window_size: {args.window_size}",
        "",
        "## output files",
        "- scenario16_metrics_overall.csv",
        "- scenario16_metrics_by_subset.csv",
        "- scenario16_probe_scores.csv",
    ]
    (OUTPUT_DIR / "scenario16_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[saved] {OUTPUT_DIR / 'scenario16_metrics_overall.csv'}")
    print(f"[saved] {OUTPUT_DIR / 'scenario16_metrics_by_subset.csv'}")
    print(f"[saved] {OUTPUT_DIR / 'scenario16_probe_scores.csv'}")
    print(f"[saved] {OUTPUT_DIR / 'scenario16_summary.md'}")
    print("[summary] scenario16 completed")


if __name__ == "__main__":
    main()
