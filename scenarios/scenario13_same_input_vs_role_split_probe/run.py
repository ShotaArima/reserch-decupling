"""Scenario13: same-input vs role-split probe comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "output"

from src.data import FreshRetailConfig, load_freshretail_dataframe
from src.scenario9_pipeline import (
    COMMON_FEATURE_CANDIDATES,
    SPECIFIC_FEATURE_CANDIDATES,
    TrainConfig,
    add_dt_features,
    build_splits,
    resolve_features,
    train_model,
)
from src.scenario13_probe import (
    ProbeConfig,
    ProbeTask,
    build_probe_labels,
    compute_latent_similarity,
    extract_latents,
    run_probes,
    save_probe_heatmap,
    save_similarity_bar,
    summarize_role_separation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario13 probe comparison.")
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--probe-steps", type=int, default=120)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-l2", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[progress] Scenario13 start")
    print("[progress] Loading FreshRetailNet-50K dataframe...")
    df = load_freshretail_dataframe(FreshRetailConfig())
    df = add_dt_features(df)

    common_features, missing_common = resolve_features(df, COMMON_FEATURE_CANDIDATES)
    specific_features, missing_specific = resolve_features(df, SPECIFIC_FEATURE_CANDIDATES)

    print(f"[features] common={common_features}")
    if missing_common:
        print(f"[features] missing_common={missing_common}")
    print(f"[features] specific={specific_features}")
    if missing_specific:
        print(f"[features] missing_specific={missing_specific}")

    combined_features = sorted(set(common_features + specific_features))

    experiments = [
        ("exp0_same_input", combined_features, combined_features),
        ("exp1_role_split", common_features, specific_features),
    ]

    tasks = [
        ProbeTask("city_id", "common", "multiclass"),
        ProbeTask("store_id", "common", "multiclass"),
        ProbeTask("first_category_id", "common", "multiclass"),
        ProbeTask("holiday_flag", "common", "binary"),
        ProbeTask("next_direction", "specific", "binary"),
        ProbeTask("recent_volatility_high", "specific", "binary"),
        ProbeTask("discount", "specific", "binary"),
        ProbeTask("activity_flag", "specific", "binary"),
    ]

    train_cfg = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    probe_cfg = ProbeConfig(
        steps=args.probe_steps,
        lr=args.probe_lr,
        l2=args.probe_l2,
        log_interval=20,
    )

    train_labels = build_probe_labels(df, split="train", window_size=args.window_size)
    test_labels = build_probe_labels(df, split="test", window_size=args.window_size)

    probe_frames: list[pd.DataFrame] = []
    similarity_rows: list[dict[str, float | str]] = []

    for name, exp_common, exp_specific in experiments:
        print(f"\n[progress] Running experiment={name}")
        print(f"[progress] Building split tensors for {name}...")
        splits = build_splits(
            df,
            common_features=exp_common,
            specific_features=exp_specific,
            window_size=args.window_size,
        )

        print(f"[progress] Training model for {name}...")
        model, _ = train_model(splits, config=train_cfg, experiment_name=name)

        print(f"[progress] Extracting latents for {name}...")
        train_latents = extract_latents(model, splits, split="train")
        test_latents = extract_latents(model, splits, split="test")

        print(f"[progress] Running probes for {name}...")
        probe_df = run_probes(
            train_latents=train_latents,
            eval_latents=test_latents,
            train_labels=train_labels,
            eval_labels=test_labels,
            tasks=tasks,
            probe_config=probe_cfg,
            experiment_name=name,
        )
        probe_frames.append(probe_df)

        sim = compute_latent_similarity(test_latents["z_common"], test_latents["z_specific"])
        sim["experiment"] = name
        similarity_rows.append(sim)

        print(
            f"[summary] {name}: cka={sim['cka']:.4f} cosine_mean={sim['cosine_mean']:.4f} "
            f"probe_rows={len(probe_df)}"
        )

    all_probe_df = pd.concat(probe_frames, ignore_index=True)
    sim_df = pd.DataFrame(similarity_rows)
    rsi_df = summarize_role_separation(all_probe_df)

    probe_csv = OUTPUT_DIR / "scenario13_probe_scores.csv"
    sim_csv = OUTPUT_DIR / "scenario13_latent_similarity.csv"
    rsi_csv = OUTPUT_DIR / "scenario13_probe_gap_summary.csv"

    all_probe_df.to_csv(probe_csv, index=False)
    sim_df.to_csv(sim_csv, index=False)
    rsi_df.to_csv(rsi_csv, index=False)

    print(f"[saved] {probe_csv}")
    print(f"[saved] {sim_csv}")
    print(f"[saved] {rsi_csv}")

    heatmap_path = save_probe_heatmap(all_probe_df, OUTPUT_DIR / "scenario13_probe_score_heatmap.png")
    summary_bar_path = save_similarity_bar(sim_df, rsi_df, OUTPUT_DIR / "scenario13_similarity_bar.png")
    print(f"[saved] {heatmap_path}")
    print(f"[saved] {summary_bar_path}")

    print("[progress] Scenario13 finished.")


if __name__ == "__main__":
    main()
