from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from src.data import (
    FreshRetailConfig,
    build_window_tensor,
    extract_last_timestep_feature,
    load_freshretail_dataframe,
    normalize_by_train_stats,
    split_train_valid_test,
)
from src.forecast_baselines import make_one_step_pairs
from src.metrics import diff_correlation, mae, mean_error, residual_std, wape, wpe
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead

Condition = Literal["common_only", "specific_only", "both"]
CONDITIONS: tuple[Condition, ...] = ("common_only", "specific_only", "both")

BASE_FEATURES = ["sale_amount", "discount", "holiday_flag", "activity_flag"]
WINDOW_SIZE = 14
TARGET_FEATURE_INDEX = 0


@dataclass
class TensorSplits:
    x_train: torch.Tensor
    x_valid: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor
    y_valid: torch.Tensor
    y_test: torch.Tensor


@dataclass
class LatentSplits:
    common_train: torch.Tensor
    specific_train: torch.Tensor
    common_valid: torch.Tensor
    specific_valid: torch.Tensor
    common_test: torch.Tensor
    specific_test: torch.Tensor


@dataclass
class ConditionResult:
    condition: str
    seed: int
    wape: float
    wpe: float
    mae: float
    mean_error: float
    residual_std: float
    diff_corr: float


def prepare_tensor_splits() -> TensorSplits:
    df = load_freshretail_dataframe(FreshRetailConfig())
    x_all_raw = build_window_tensor(df, BASE_FEATURES, window_size=WINDOW_SIZE)
    y_all_raw = extract_last_timestep_feature(x_all_raw, TARGET_FEATURE_INDEX)

    train_raw, valid_raw, test_raw = split_train_valid_test(x_all_raw)
    y_train_raw, y_valid_raw, y_test_raw = split_train_valid_test(y_all_raw)

    train_x, valid_x, test_x = normalize_by_train_stats(train_raw, valid_raw, test_raw)

    return TensorSplits(
        x_train=torch.tensor(train_x, dtype=torch.float32),
        x_valid=torch.tensor(valid_x, dtype=torch.float32),
        x_test=torch.tensor(test_x, dtype=torch.float32),
        y_train=torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(-1),
        y_valid=torch.tensor(y_valid_raw, dtype=torch.float32).unsqueeze(-1),
        y_test=torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(-1),
    )


def train_latents(
    splits: TensorSplits,
    *,
    steps: int,
    lr: float,
    print_every: int,
) -> tuple[LatentSplits, list[float]]:
    model = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(BASE_FEATURES), window_size=WINDOW_SIZE))
    head = ForecastHead(local_dim=16, global_dim=16, horizon=1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)

    losses: list[float] = []
    for step in range(1, steps + 1):
        _, specific_latent, common_latent = model(splits.x_train)
        specific_s, y_train_s = make_one_step_pairs(specific_latent, splits.y_train)
        common_s, _ = make_one_step_pairs(common_latent, splits.y_train)
        pred = head(specific_s, common_s)
        loss = nn.functional.l1_loss(pred, y_train_s)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step % print_every == 0 or step == 1 or step == steps:
            print(f"[LatentTrain] step={step:04d}/{steps} loss={loss.item():.6f}")

    with torch.no_grad():
        _, specific_train, common_train = model(splits.x_train)
        _, specific_valid, common_valid = model(splits.x_valid)
        _, specific_test, common_test = model(splits.x_test)

    return (
        LatentSplits(
            common_train=common_train,
            specific_train=specific_train,
            common_valid=common_valid,
            specific_valid=specific_valid,
            common_test=common_test,
            specific_test=specific_test,
        ),
        losses,
    )


def train_condition_head(
    latents: LatentSplits,
    y_train: torch.Tensor,
    *,
    mode: Condition,
    steps: int,
    lr: float,
    print_every: int,
) -> tuple[ForecastHead, list[float]]:
    specific_train_s, y_train_s = make_one_step_pairs(latents.specific_train, y_train)
    common_train_s, _ = make_one_step_pairs(latents.common_train, y_train)

    if mode == "specific_only":
        local_dim, global_dim = specific_train_s.shape[-1], 0
    elif mode == "common_only":
        local_dim, global_dim = 0, common_train_s.shape[-1]
    else:
        local_dim, global_dim = specific_train_s.shape[-1], common_train_s.shape[-1]

    head = ForecastHead(local_dim=local_dim, global_dim=global_dim, horizon=1)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    losses: list[float] = []
    for step in range(1, steps + 1):
        local_in, global_in = select_inputs(specific_train_s, common_train_s, mode=mode)
        pred = head(local_in, global_in)
        loss = nn.functional.l1_loss(pred, y_train_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step % print_every == 0 or step == 1 or step == steps:
            print(f"[HeadTrain:{mode}] step={step:04d}/{steps} loss={loss.item():.6f}")

    return head, losses


def select_inputs(specific_s: torch.Tensor, common_s: torch.Tensor, *, mode: Condition) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "specific_only":
        return specific_s, torch.zeros(common_s.shape[0], 0, dtype=common_s.dtype, device=common_s.device)
    if mode == "common_only":
        return torch.zeros(specific_s.shape[0], 0, dtype=specific_s.dtype, device=specific_s.device), common_s
    return specific_s, common_s


def infer_condition(head: ForecastHead, latents: LatentSplits, y_split: torch.Tensor, *, split: str, mode: Condition) -> tuple[np.ndarray, np.ndarray]:
    if split == "valid":
        specific, common = latents.specific_valid, latents.common_valid
    elif split == "test":
        specific, common = latents.specific_test, latents.common_test
    else:
        raise ValueError(f"Unsupported split: {split}")

    with torch.no_grad():
        specific_s, y_s = make_one_step_pairs(specific, y_split)
        common_s, _ = make_one_step_pairs(common, y_split)
        local_in, global_in = select_inputs(specific_s, common_s, mode=mode)
        pred = head(local_in, global_in).numpy().reshape(-1)

    return y_s.numpy().reshape(-1), pred


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ConditionResult:
    return ConditionResult(
        condition="",
        seed=-1,
        wape=wape(y_true, y_pred),
        wpe=wpe(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mean_error=mean_error(y_true, y_pred),
        residual_std=residual_std(y_true, y_pred),
        diff_corr=diff_correlation(y_true, y_pred),
    )


def compute_volatility_masks(x_test: torch.Tensor) -> dict[str, np.ndarray]:
    sale_hist = x_test[:-1, :, TARGET_FEATURE_INDEX].numpy()
    volatility = sale_hist.std(axis=1)
    q_low = float(np.quantile(volatility, 0.3))
    q_high = float(np.quantile(volatility, 0.7))

    return {
        "high_volatility": volatility >= q_high,
        "low_volatility": volatility <= q_low,
        "all": np.ones_like(volatility, dtype=bool),
    }


def write_metrics_csv(rows: list[ConditionResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "wape", "wpe", "mae", "mean_error", "residual_std", "diff_corr"])
        for r in rows:
            writer.writerow([
                r.condition,
                r.seed,
                f"{r.wape:.6f}",
                f"{r.wpe:.6f}",
                f"{r.mae:.6f}",
                f"{r.mean_error:.6f}",
                f"{r.residual_std:.6f}",
                f"{r.diff_corr:.6f}",
            ])


def write_summary_csv(rows: list[ConditionResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[ConditionResult]] = {}
    for row in rows:
        grouped.setdefault(row.condition, []).append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "wape_mean",
                "wape_std",
                "wpe_mean",
                "wpe_std",
                "mae_mean",
                "mae_std",
                "mean_error_mean",
                "mean_error_std",
                "residual_std_mean",
                "residual_std_std",
                "diff_corr_mean",
                "diff_corr_std",
            ]
        )
        for condition, vals in grouped.items():
            writer.writerow(
                [
                    condition,
                    f"{np.mean([v.wape for v in vals]):.6f}",
                    f"{np.std([v.wape for v in vals]):.6f}",
                    f"{np.mean([v.wpe for v in vals]):.6f}",
                    f"{np.std([v.wpe for v in vals]):.6f}",
                    f"{np.mean([v.mae for v in vals]):.6f}",
                    f"{np.std([v.mae for v in vals]):.6f}",
                    f"{np.mean([v.mean_error for v in vals]):.6f}",
                    f"{np.std([v.mean_error for v in vals]):.6f}",
                    f"{np.mean([v.residual_std for v in vals]):.6f}",
                    f"{np.std([v.residual_std for v in vals]):.6f}",
                    f"{np.mean([v.diff_corr for v in vals]):.6f}",
                    f"{np.std([v.diff_corr for v in vals]):.6f}",
                ]
            )


def write_subset_csv(
    subset_rows: list[tuple[str, int, str, ConditionResult]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "condition", "seed", "wape", "wpe", "mae", "mean_error", "residual_std", "diff_corr"])
        for subset, seed, condition, row in subset_rows:
            writer.writerow(
                [
                    subset,
                    condition,
                    seed,
                    f"{row.wape:.6f}",
                    f"{row.wpe:.6f}",
                    f"{row.mae:.6f}",
                    f"{row.mean_error:.6f}",
                    f"{row.residual_std:.6f}",
                    f"{row.diff_corr:.6f}",
                ]
            )


def write_prediction_samples(
    y_true: np.ndarray,
    pred_map: dict[str, np.ndarray],
    out_path: Path,
    *,
    num_points: int = 240,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = y_true.shape[0]
    indices = np.linspace(0, n - 1, num=min(num_points, n), dtype=int)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "condition", "value"])
        for idx in indices:
            writer.writerow([int(idx), "y_true", f"{y_true[idx]:.6f}"])
            for condition, pred in pred_map.items():
                writer.writerow([int(idx), condition, f"{pred[idx]:.6f}"])
