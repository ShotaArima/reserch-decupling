from __future__ import annotations

import csv
import os
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
class RuntimeConfig:
    device: torch.device
    cpu_threads: int
    batch_size: int
    pin_memory: bool


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


def resolve_runtime_config(
    *,
    device_arg: str = "auto",
    cpu_threads: int | None = None,
    batch_size: int | None = None,
) -> RuntimeConfig:
    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available_cores = os.cpu_count() or 1
    chosen_threads = max(1, min(cpu_threads or available_cores, available_cores))
    torch.set_num_threads(chosen_threads)
    if chosen_threads > 1:
        torch.set_num_interop_threads(max(1, chosen_threads // 2))

    auto_batch = 512
    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        auto_batch = 1024 if total_gb <= 8.5 else 2048

    final_batch = max(64, batch_size or auto_batch)
    return RuntimeConfig(
        device=device,
        cpu_threads=chosen_threads,
        batch_size=final_batch,
        pin_memory=(device.type == "cuda"),
    )


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


def _sample_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] <= batch_size:
        return x.to(device), y.to(device)
    idx = torch.randint(low=0, high=x.shape[0], size=(batch_size,), device=device)
    return x[idx].to(device), y[idx].to(device)


def train_latents(
    splits: TensorSplits,
    *,
    steps: int,
    lr: float,
    print_every: int,
    runtime: RuntimeConfig,
) -> tuple[LatentSplits, list[float]]:
    model = DecouplingAutoEncoder(DecouplingConfig(feature_dim=len(BASE_FEATURES), window_size=WINDOW_SIZE)).to(runtime.device)
    head = ForecastHead(local_dim=16, global_dim=16, horizon=1).to(runtime.device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)

    losses: list[float] = []
    model.train()
    head.train()
    for step in range(1, steps + 1):
        x_batch, y_batch = _sample_batch(splits.x_train, splits.y_train, runtime.batch_size, runtime.device)
        _, specific_latent, common_latent = model(x_batch)
        specific_s, y_train_s = make_one_step_pairs(specific_latent, y_batch)
        common_s, _ = make_one_step_pairs(common_latent, y_batch)
        pred = head(specific_s, common_s)
        loss = nn.functional.l1_loss(pred, y_train_s)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step % print_every == 0 or step == 1 or step == steps:
            print(f"[LatentTrain] step={step:04d}/{steps} loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        _, specific_train, common_train = model(splits.x_train.to(runtime.device))
        _, specific_valid, common_valid = model(splits.x_valid.to(runtime.device))
        _, specific_test, common_test = model(splits.x_test.to(runtime.device))

    return (
        LatentSplits(
            common_train=common_train.detach().cpu(),
            specific_train=specific_train.detach().cpu(),
            common_valid=common_valid.detach().cpu(),
            specific_valid=specific_valid.detach().cpu(),
            common_test=common_test.detach().cpu(),
            specific_test=specific_test.detach().cpu(),
        ),
        losses,
    )


def select_inputs(specific_s: torch.Tensor, common_s: torch.Tensor, *, mode: Condition) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "specific_only":
        return specific_s, torch.zeros(common_s.shape[0], 0, dtype=common_s.dtype, device=common_s.device)
    if mode == "common_only":
        return torch.zeros(specific_s.shape[0], 0, dtype=specific_s.dtype, device=specific_s.device), common_s
    return specific_s, common_s


def train_condition_head(
    latents: LatentSplits,
    y_train: torch.Tensor,
    *,
    mode: Condition,
    steps: int,
    lr: float,
    print_every: int,
    runtime: RuntimeConfig,
) -> tuple[ForecastHead, list[float]]:
    specific_train_s, y_train_s = make_one_step_pairs(latents.specific_train, y_train)
    common_train_s, _ = make_one_step_pairs(latents.common_train, y_train)

    if mode == "specific_only":
        local_dim, global_dim = specific_train_s.shape[-1], 0
    elif mode == "common_only":
        local_dim, global_dim = 0, common_train_s.shape[-1]
    else:
        local_dim, global_dim = specific_train_s.shape[-1], common_train_s.shape[-1]

    head = ForecastHead(local_dim=local_dim, global_dim=global_dim, horizon=1).to(runtime.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    losses: list[float] = []
    train_count = specific_train_s.shape[0]
    for step in range(1, steps + 1):
        if train_count > runtime.batch_size:
            idx = torch.randint(low=0, high=train_count, size=(runtime.batch_size,))
            specific_batch = specific_train_s[idx].to(runtime.device)
            common_batch = common_train_s[idx].to(runtime.device)
            y_batch = y_train_s[idx].to(runtime.device)
        else:
            specific_batch = specific_train_s.to(runtime.device)
            common_batch = common_train_s.to(runtime.device)
            y_batch = y_train_s.to(runtime.device)

        local_in, global_in = select_inputs(specific_batch, common_batch, mode=mode)
        pred = head(local_in, global_in)
        loss = nn.functional.l1_loss(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step % print_every == 0 or step == 1 or step == steps:
            print(f"[HeadTrain:{mode}] step={step:04d}/{steps} loss={loss.item():.6f}")

    return head.cpu(), losses


def _predict_in_batches(head: ForecastHead, local_in: torch.Tensor, global_in: torch.Tensor, batch_size: int) -> np.ndarray:
    preds: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, local_in.shape[0], batch_size):
            end = min(start + batch_size, local_in.shape[0])
            pred = head(local_in[start:end], global_in[start:end]).cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def infer_condition(
    head: ForecastHead,
    latents: LatentSplits,
    y_split: torch.Tensor,
    *,
    split: str,
    mode: Condition,
    runtime: RuntimeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "valid":
        specific, common = latents.specific_valid, latents.common_valid
    elif split == "test":
        specific, common = latents.specific_test, latents.common_test
    else:
        raise ValueError(f"Unsupported split: {split}")

    specific_s, y_s = make_one_step_pairs(specific, y_split)
    common_s, _ = make_one_step_pairs(common, y_split)

    local_in, global_in = select_inputs(
        specific_s.to(runtime.device),
        common_s.to(runtime.device),
        mode=mode,
    )
    head = head.to(runtime.device)
    pred = _predict_in_batches(head, local_in, global_in, batch_size=runtime.batch_size)
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
