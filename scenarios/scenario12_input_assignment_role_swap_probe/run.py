"""Scenario 12: input-assignment role-swap probe experiments.

Runs probes on Scenario9 exp1/exp2 and tests whether informative latent side swaps
when common/specific feature assignment is swapped.
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

from src.data import FreshRetailConfig, build_window_tensor, extract_last_timestep_feature, load_freshretail_dataframe, split_train_valid_test
from src.plotting import save_probe_heatmap, save_swap_direction_plot
from src.scenario12_probe import (
    ProbeConfig,
    accuracy_score,
    fit_linear_classification_probe,
    macro_f1_score,
    predict_classification,
)
from src.scenario9_pipeline import (
    COMMON_FEATURE_CANDIDATES,
    SPECIFIC_FEATURE_CANDIDATES,
    TrainConfig,
    add_dt_features,
    build_splits,
    resolve_features,
    train_model,
)


@dataclass
class ExperimentDef:
    name: str
    common_features: list[str]
    specific_features: list[str]


@dataclass
class ProbeTask:
    name: str
    target_kind: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario12 role-swap probes")
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--steps", type=int, default=120, help="Scenario9 model training steps")
    parser.add_argument("--probe-steps", type=int, default=120, help="Probe training steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    return parser.parse_args()


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _make_one_step_pairs(common_x: np.ndarray, specific_x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return common_x[:-1], specific_x[:-1], y[1:]


def _build_label_last_step(df, feature: str, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = build_window_tensor(df, [feature], window_size=window_size)
    values = extract_last_timestep_feature(tensor, 0)
    return split_train_valid_test(values)


def _encode_by_train(train: np.ndarray, valid: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    unique = np.unique(train)
    class_to_idx = {v: i for i, v in enumerate(unique)}

    def _map(arr: np.ndarray) -> np.ndarray:
        out = np.array([class_to_idx.get(v, -1) for v in arr], dtype=np.int64)
        return out

    return _map(train), _map(valid), _map(test), len(unique)


def _mask_unknown(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = y >= 0
    return x[mask], y[mask]


def _quantile_bins(train_values: np.ndarray, values: np.ndarray, bins: int = 4) -> np.ndarray:
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(train_values, q)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(values, dtype=np.int64)
    binned = np.digitize(values, edges[1:-1], right=False)
    return binned.astype(np.int64)


def _build_probe_targets(df, window_size: int) -> dict[str, dict[str, np.ndarray]]:
    sale_train, sale_valid, sale_test = _build_label_last_step(df, "sale_amount", window_size)
    store_train, store_valid, store_test = _build_label_last_step(df, "store_id", window_size)
    cat_train, cat_valid, cat_test = _build_label_last_step(df, "first_category_id", window_size)
    hol_train, hol_valid, hol_test = _build_label_last_step(df, "holiday_flag", window_size)
    dow_train, dow_valid, dow_test = _build_label_last_step(df, "dt_weekday", window_size)
    disc_train, disc_valid, disc_test = _build_label_last_step(df, "discount", window_size)
    act_train, act_valid, act_test = _build_label_last_step(df, "activity_flag", window_size)

    sale_diff_train = sale_train[1:] - sale_train[:-1]
    sale_diff_valid = sale_valid[1:] - sale_valid[:-1]
    sale_diff_test = sale_test[1:] - sale_test[:-1]

    out = {
        "store_id": {"train": store_train[:-1], "valid": store_valid[:-1], "test": store_test[:-1]},
        "first_category_id": {"train": cat_train[:-1], "valid": cat_valid[:-1], "test": cat_test[:-1]},
        "holiday_flag": {"train": hol_train[:-1], "valid": hol_valid[:-1], "test": hol_test[:-1]},
        "dt_weekday": {"train": dow_train[:-1], "valid": dow_valid[:-1], "test": dow_test[:-1]},
        "discount_flag": {
            "train": (disc_train[:-1] > 0).astype(np.int64),
            "valid": (disc_valid[:-1] > 0).astype(np.int64),
            "test": (disc_test[:-1] > 0).astype(np.int64),
        },
        "activity_flag": {
            "train": (act_train[:-1] > 0).astype(np.int64),
            "valid": (act_valid[:-1] > 0).astype(np.int64),
            "test": (act_test[:-1] > 0).astype(np.int64),
        },
        "next_sale_up": {
            "train": (sale_diff_train > 0).astype(np.int64),
            "valid": (sale_diff_valid > 0).astype(np.int64),
            "test": (sale_diff_test > 0).astype(np.int64),
        },
        "sale_diff_bin": {
            "train": _quantile_bins(np.abs(sale_diff_train), np.abs(sale_diff_train), bins=4),
            "valid": _quantile_bins(np.abs(sale_diff_train), np.abs(sale_diff_valid), bins=4),
            "test": _quantile_bins(np.abs(sale_diff_train), np.abs(sale_diff_test), bins=4),
        },
    }
    return out


def _extract_latents(model, splits) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for split_name, common_raw, specific_raw, y_raw in (
        ("train", splits.common_train, splits.specific_train, splits.y_train),
        ("valid", splits.common_valid, splits.specific_valid, splits.y_valid),
        ("test", splits.common_test, splits.specific_test, splits.y_test),
    ):
        common_x, specific_x, _ = _make_one_step_pairs(common_raw, specific_raw, y_raw)
        common_t = torch.tensor(_flatten_windows(common_x), dtype=torch.float32)
        specific_t = torch.tensor(_flatten_windows(specific_x), dtype=torch.float32)
        with torch.no_grad():
            _, z_common, z_specific = model(common_t, specific_t)
        out[split_name] = {
            "common": z_common.numpy(),
            "specific": z_specific.numpy(),
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[progress] Loading FreshRetailNet-50K...")
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

    experiments = [
        ExperimentDef(name="exp1_role_split", common_features=common_features, specific_features=specific_features),
        ExperimentDef(name="exp2_swapped_split", common_features=specific_features, specific_features=common_features),
    ]

    probe_tasks = [
        ProbeTask("store_id", "structural"),
        ProbeTask("first_category_id", "structural"),
        ProbeTask("holiday_flag", "structural"),
        ProbeTask("dt_weekday", "structural"),
        ProbeTask("next_sale_up", "temporal"),
        ProbeTask("sale_diff_bin", "temporal"),
        ProbeTask("discount_flag", "temporal"),
        ProbeTask("activity_flag", "temporal"),
    ]

    targets = _build_probe_targets(df, args.window_size)

    long_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        print(f"\n[progress] Seed {seed} started")
        train_cfg = TrainConfig(
            steps=args.steps,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            seed=seed,
            log_interval=args.log_interval,
        )

        for exp in experiments:
            print(f"\n[progress] Train backbone {exp.name} (seed={seed})")
            splits = build_splits(
                df,
                common_features=exp.common_features,
                specific_features=exp.specific_features,
                window_size=args.window_size,
            )
            model, _ = train_model(splits, config=train_cfg, experiment_name=f"scenario12_{exp.name}_seed{seed}")
            latents = _extract_latents(model, splits)

            for probe in probe_tasks:
                print(f"[progress] Probe {probe.name} on {exp.name} (seed={seed})")
                y_train = targets[probe.name]["train"]
                y_valid = targets[probe.name]["valid"]
                y_test = targets[probe.name]["test"]
                y_train_enc, y_valid_enc, y_test_enc, num_classes = _encode_by_train(y_train, y_valid, y_test)

                probe_cfg = ProbeConfig(
                    steps=args.probe_steps,
                    lr=args.probe_lr,
                    batch_size=args.batch_size,
                    seed=seed,
                    log_interval=args.log_interval,
                )

                for latent_name in ("common", "specific"):
                    x_train = latents["train"][latent_name]
                    x_valid = latents["valid"][latent_name]
                    x_test = latents["test"][latent_name]

                    x_train_m, y_train_m = _mask_unknown(x_train, y_train_enc)
                    x_valid_m, y_valid_m = _mask_unknown(x_valid, y_valid_enc)
                    x_test_m, y_test_m = _mask_unknown(x_test, y_test_enc)

                    if x_train_m.shape[0] == 0 or num_classes <= 1:
                        print(f"[warn] Skip {probe.name} / {latent_name}: insufficient classes")
                        continue

                    clf = fit_linear_classification_probe(
                        x_train_m,
                        y_train_m,
                        num_classes=num_classes,
                        config=probe_cfg,
                        probe_name=probe.name,
                        latent_name=latent_name,
                    )

                    for split_name, x_split, y_split in (
                        ("valid", x_valid_m, y_valid_m),
                        ("test", x_test_m, y_test_m),
                    ):
                        pred = predict_classification(clf, x_split)
                        acc = accuracy_score(y_split, pred)
                        f1 = macro_f1_score(y_split, pred, num_classes=num_classes)
                        long_rows.append(
                            {
                                "experiment": exp.name,
                                "latent": latent_name,
                                "probe_name": probe.name,
                                "probe_kind": probe.target_kind,
                                "split": split_name,
                                "seed": seed,
                                "metric": "accuracy",
                                "value": acc,
                            }
                        )
                        long_rows.append(
                            {
                                "experiment": exp.name,
                                "latent": latent_name,
                                "probe_name": probe.name,
                                "probe_kind": probe.target_kind,
                                "split": split_name,
                                "seed": seed,
                                "metric": "macro_f1",
                                "value": f1,
                            }
                        )
                        print(
                            f"[result] exp={exp.name} probe={probe.name} latent={latent_name} "
                            f"split={split_name} acc={acc:.4f} macro_f1={f1:.4f}"
                        )

    long_path = OUTPUT_DIR / "probe_scores_long.csv"
    _write_csv(
        long_path,
        long_rows,
        ["experiment", "latent", "probe_name", "probe_kind", "split", "seed", "metric", "value"],
    )
    print(f"[saved] {long_path}")

    # Summary (test / macro_f1)
    summary_rows: list[dict[str, object]] = []
    swap_rows: list[dict[str, object]] = []

    probe_names = sorted({r["probe_name"] for r in long_rows})
    exp1_gaps: list[float] = []
    exp2_gaps: list[float] = []

    for probe_name in probe_names:
        def _mean(exp: str, latent: str) -> float:
            vals = [
                float(r["value"])
                for r in long_rows
                if r["probe_name"] == probe_name
                and r["split"] == "test"
                and r["metric"] == "macro_f1"
                and r["experiment"] == exp
                and r["latent"] == latent
            ]
            return float(np.mean(vals)) if vals else float("nan")

        exp1_common = _mean("exp1_role_split", "common")
        exp1_specific = _mean("exp1_role_split", "specific")
        exp2_common = _mean("exp2_swapped_split", "common")
        exp2_specific = _mean("exp2_swapped_split", "specific")

        gap_exp1 = exp1_common - exp1_specific
        gap_exp2 = exp2_common - exp2_specific
        swap_index = gap_exp1 * gap_exp2
        swapped = bool(swap_index < 0)

        summary_rows.append(
            {
                "probe_name": probe_name,
                "metric": "macro_f1",
                "exp1_common": exp1_common,
                "exp1_specific": exp1_specific,
                "exp2_common": exp2_common,
                "exp2_specific": exp2_specific,
                "swapped": swapped,
            }
        )
        swap_rows.append(
            {
                "probe_name": probe_name,
                "gap_exp1": gap_exp1,
                "gap_exp2": gap_exp2,
                "swap_index": swap_index,
                "swapped": swapped,
            }
        )
        exp1_gaps.append(gap_exp1)
        exp2_gaps.append(gap_exp2)

    summary_path = OUTPUT_DIR / "probe_summary.csv"
    _write_csv(
        summary_path,
        summary_rows,
        ["probe_name", "metric", "exp1_common", "exp1_specific", "exp2_common", "exp2_specific", "swapped"],
    )
    print(f"[saved] {summary_path}")

    swap_path = OUTPUT_DIR / "swap_index.csv"
    _write_csv(swap_path, swap_rows, ["probe_name", "gap_exp1", "gap_exp2", "swap_index", "swapped"])
    print(f"[saved] {swap_path}")

    # Heatmap matrix: rows=(exp,latent) cols=probe ; values=macro_f1(test mean over seeds)
    row_labels = [
        "exp1_common",
        "exp1_specific",
        "exp2_common",
        "exp2_specific",
    ]
    matrix = np.zeros((len(row_labels), len(probe_names)), dtype=np.float32)

    for j, probe_name in enumerate(probe_names):
        for i, (exp, latent) in enumerate(
            [
                ("exp1_role_split", "common"),
                ("exp1_role_split", "specific"),
                ("exp2_swapped_split", "common"),
                ("exp2_swapped_split", "specific"),
            ]
        ):
            vals = [
                float(r["value"])
                for r in long_rows
                if r["probe_name"] == probe_name
                and r["split"] == "test"
                and r["metric"] == "macro_f1"
                and r["experiment"] == exp
                and r["latent"] == latent
            ]
            matrix[i, j] = float(np.mean(vals)) if vals else np.nan

    heatmap_path = save_probe_heatmap(
        matrix,
        row_labels,
        probe_names,
        OUTPUT_DIR / "fig_probe_heatmap.png",
        title="Scenario12 Probe Macro-F1 Heatmap (test mean over seeds)",
    )
    print(f"[saved] {heatmap_path}")

    swap_fig_path = save_swap_direction_plot(
        probe_names,
        exp1_gaps,
        exp2_gaps,
        OUTPUT_DIR / "fig_swap_direction.png",
        title="Scenario12 Gap Direction (common - specific)",
    )
    print(f"[saved] {swap_fig_path}")

    report_path = OUTPUT_DIR / "scenario12_report.md"
    swapped_count = sum(1 for r in swap_rows if r["swapped"])
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Scenario12 Report\n\n")
        f.write(f"- total_probes: {len(probe_names)}\n")
        f.write(f"- swapped_probes: {swapped_count}\n")
        f.write("\n## Swap Index\n")
        for row in swap_rows:
            f.write(
                f"- {row['probe_name']}: gap_exp1={row['gap_exp1']:.4f}, "
                f"gap_exp2={row['gap_exp2']:.4f}, swap_index={row['swap_index']:.6f}, swapped={row['swapped']}\n"
            )
    print(f"[saved] {report_path}")

    print("\n[summary] Scenario12 completed.")


if __name__ == "__main__":
    main()
