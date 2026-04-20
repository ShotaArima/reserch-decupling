"""Scenario 20: reproducible role clarification with runtime-aware acceleration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "output"

from src.plotting import (
    save_condition_series_examples,
    save_learning_curve,
    save_residual_boxplot,
    save_residual_histogram,
    save_residual_scatter,
)
from src.scenario20_role_clarification import (
    CONDITIONS,
    ConditionResult,
    compute_volatility_masks,
    evaluate_metrics,
    infer_condition,
    prepare_tensor_splits,
    resolve_runtime_config,
    train_condition_head,
    train_latents,
    write_metrics_csv,
    write_prediction_samples,
    write_subset_csv,
    write_summary_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario 20 role clarification experiment.")
    parser.add_argument("--steps", type=int, default=120, help="Training steps for latent and head training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62], help="Seeds for multi-seed evaluation.")
    parser.add_argument("--print-every", type=int, default=20, help="Print progress every N iterations.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Torch CPU threads (auto: available cores - 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Mini-batch size. Set 0 to auto-tune based on CPU/GPU memory budget.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_config(
        device_arg=args.device,
        cpu_threads=args.cpu_threads,
        batch_size=None if args.batch_size <= 0 else args.batch_size,
    )

    print(
        f"[Runtime] device={runtime.device} cpu_threads={runtime.cpu_threads} "
        f"batch_size={runtime.batch_size} pin_memory={runtime.pin_memory}"
    )
    if runtime.device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(runtime.device).total_memory / (1024**3)
        print(f"[Runtime] CUDA total memory={total_gb:.2f}GB (8GB budget aware settings enabled)")

    splits = prepare_tensor_splits()
    masks = compute_volatility_masks(splits.x_test)

    all_rows: list[ConditionResult] = []
    subset_rows: list[tuple[str, int, str, ConditionResult]] = []
    pred_accumulator: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    y_true_ref: np.ndarray | None = None

    for seed in args.seeds:
        print(f"\n===== Scenario20 seed={seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if runtime.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        latents, latent_losses = train_latents(
            splits,
            steps=args.steps,
            lr=args.lr,
            print_every=args.print_every,
            runtime=runtime,
        )
        latent_curve_path = save_learning_curve(
            latent_losses,
            OUTPUT_DIR / f"scenario20_latent_train_loss_seed{seed}.png",
            title=f"Scenario20 Latent Train Loss (seed={seed})",
            ylabel="L1 Loss",
        )
        print(f"Saved: {latent_curve_path}")

        for condition in CONDITIONS:
            head, head_losses = train_condition_head(
                latents,
                splits.y_train,
                mode=condition,
                steps=args.steps,
                lr=args.lr,
                print_every=args.print_every,
                runtime=runtime,
            )
            head_curve_path = save_learning_curve(
                head_losses,
                OUTPUT_DIR / f"scenario20_head_{condition}_train_loss_seed{seed}.png",
                title=f"Scenario20 Head Train Loss ({condition}, seed={seed})",
                ylabel="L1 Loss",
            )
            print(f"Saved: {head_curve_path}")

            y_true, y_pred = infer_condition(head, latents, splits.y_test, split="test", mode=condition, runtime=runtime)
            metric = evaluate_metrics(y_true, y_pred)
            metric.condition = condition
            metric.seed = seed
            all_rows.append(metric)
            pred_accumulator[condition].append(y_pred)
            y_true_ref = y_true

            print(
                f"[Eval:{condition}] wape={metric.wape:.4f} wpe={metric.wpe:.4f} "
                f"mae={metric.mae:.4f} mean_error={metric.mean_error:.4f} "
                f"residual_std={metric.residual_std:.4f} diff_corr={metric.diff_corr:.4f}"
            )

            for subset_name, mask in masks.items():
                subset_metric = evaluate_metrics(y_true[mask], y_pred[mask])
                subset_metric.condition = condition
                subset_metric.seed = seed
                subset_rows.append((subset_name, seed, condition, subset_metric))

        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()

    write_metrics_csv(all_rows, OUTPUT_DIR / "scenario20_metrics_overall.csv")
    write_summary_csv(all_rows, OUTPUT_DIR / "scenario20_metrics_summary.csv")
    write_subset_csv(subset_rows, OUTPUT_DIR / "scenario20_subset_metrics.csv")
    print(f"Saved: {OUTPUT_DIR / 'scenario20_metrics_overall.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'scenario20_metrics_summary.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'scenario20_subset_metrics.csv'}")

    assert y_true_ref is not None
    avg_pred_map = {condition: np.mean(np.stack(preds, axis=0), axis=0) for condition, preds in pred_accumulator.items()}
    residual_map = {condition: pred - y_true_ref for condition, pred in avg_pred_map.items()}

    n = len(y_true_ref)
    segment = max(60, n // 10)
    segment_ranges = [
        (0, min(segment, n)),
        (max(0, n // 2 - segment // 2), min(n, n // 2 + segment // 2)),
        (max(0, n - segment), n),
    ]

    plot_series = save_condition_series_examples(
        y_true_ref,
        avg_pred_map,
        segment_ranges,
        OUTPUT_DIR / "scenario20_plot_series_examples.png",
        title="Scenario20 Prediction Examples (seed-average)",
    )
    plot_hist = save_residual_histogram(
        residual_map,
        OUTPUT_DIR / "scenario20_plot_residual_hist.png",
        title="Scenario20 Residual Histogram (seed-average)",
    )
    plot_box = save_residual_boxplot(
        residual_map,
        OUTPUT_DIR / "scenario20_plot_residual_box.png",
        title="Scenario20 Residual Boxplot (seed-average)",
    )
    plot_scatter = save_residual_scatter(
        y_true_ref,
        avg_pred_map,
        OUTPUT_DIR / "scenario20_plot_residual_scatter.png",
        title="Scenario20 Residual Scatter (seed-average)",
    )

    print(f"Saved: {plot_series}")
    print(f"Saved: {plot_hist}")
    print(f"Saved: {plot_box}")
    print(f"Saved: {plot_scatter}")

    write_prediction_samples(
        y_true_ref,
        avg_pred_map,
        OUTPUT_DIR / "scenario20_predictions_sample.csv",
    )
    print(f"Saved: {OUTPUT_DIR / 'scenario20_predictions_sample.csv'}")

    notes_path = OUTPUT_DIR / "scenario20_interpretation_notes.md"
    notes_path.write_text(
        "# Scenario20 Interpretation Notes\n\n"
        "- Fill this file after reviewing metrics and plots.\n"
        "- Confirm whether common_only underpredicts and specific_only has higher residual variance.\n"
        "- Record whether both improves bias and absolute error simultaneously.\n",
        encoding="utf-8",
    )
    print(f"Saved: {notes_path}")


if __name__ == "__main__":
    main()
