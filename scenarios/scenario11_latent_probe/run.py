from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENARIO_DIR / "output"

from src.data import FreshRetailConfig, load_freshretail_dataframe
from src.scenario11_probe import ProbeConfig, build_latents_and_tasks, run_probe_suite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario11 latent probe experiments")
    p.add_argument("--window-size", type=int, default=14)
    p.add_argument("--steps", type=int, default=120, help="Probe and latent extractor train steps")
    p.add_argument("--probe-lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="42,43,44", help="comma-separated probe seeds")
    p.add_argument("--log-interval", type=int, default=20)
    return p.parse_args()


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["task", "group", "input_type", "metric"], as_index=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["task", "group", "input_type", "metric", "mean", "std"]
    return agg


def _build_main_table(agg: pd.DataFrame) -> pd.DataFrame:
    test_macro = agg[agg["metric"] == "test_macro_f1"].copy()
    pivot = test_macro.pivot_table(index=["task", "group"], columns="input_type", values="mean").reset_index()
    for col in ["common", "specific", "concat", "random"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["gap_c_minus_s"] = pivot["common"] - pivot["specific"]
    return pivot[["task", "group", "common", "specific", "concat", "random", "gap_c_minus_s"]]


def _build_group_summary(main_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sign in [("common", 1.0), ("specific", -1.0)]:
        sub = main_table[main_table["group"] == group]
        if sub.empty:
            continue
        avg_gap = float(sub["gap_c_minus_s"].mean())
        if sign * avg_gap > 0:
            verdict = "expected_direction"
        else:
            verdict = "unexpected_direction"
        rows.append(
            {
                "group": group,
                "n_tasks": int(sub.shape[0]),
                "avg_gap_c_minus_s": avg_gap,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    print(
        f"[scenario11] start window_size={args.window_size} steps={args.steps} "
        f"probe_lr={args.probe_lr} latent_dim={args.latent_dim} hidden_dim={args.hidden_dim} seeds={seeds}"
    )

    print("[scenario11] loading dataset")
    df = load_freshretail_dataframe(FreshRetailConfig())

    cfg = ProbeConfig(
        steps=args.steps,
        lr=args.probe_lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    print("[scenario11] building latent features and probe tasks")
    latent_by_split, tasks, feature_df, label_df = build_latents_and_tasks(df, cfg, window_size=args.window_size)

    feature_path = OUTPUT_DIR / "latent_probe_features.csv"
    label_path = OUTPUT_DIR / "latent_probe_labels.csv"
    feature_df.to_csv(feature_path, index=False)
    label_df.to_csv(label_path, index=False)
    print(f"[scenario11] saved {feature_path}")
    print(f"[scenario11] saved {label_path}")

    print("[scenario11] running probe suite")
    result_df = run_probe_suite(latent_by_split, tasks, cfg, seeds)

    raw_result_path = OUTPUT_DIR / "scenario11_probe_results_raw.csv"
    result_df.to_csv(raw_result_path, index=False)
    print(f"[scenario11] saved {raw_result_path}")

    agg = _aggregate(result_df)
    agg_path = OUTPUT_DIR / "scenario11_probe_results_agg.csv"
    agg.to_csv(agg_path, index=False)
    print(f"[scenario11] saved {agg_path}")

    main_table = _build_main_table(agg)
    main_table_path = OUTPUT_DIR / "scenario11_main_table.csv"
    main_table.to_csv(main_table_path, index=False)
    print(f"[scenario11] saved {main_table_path}")

    group_summary = _build_group_summary(main_table)
    group_summary_path = OUTPUT_DIR / "scenario11_group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)
    print(f"[scenario11] saved {group_summary_path}")

    print("[scenario11] done")


if __name__ == "__main__":
    main()
