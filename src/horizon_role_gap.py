from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.metrics import mae, wape, wpe
from src.scenario9_pipeline import AblationMode, DatasetSplits, TrainConfig, TwoBranchForecaster


@dataclass
class EvalMetrics:
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    test_wape: float
    test_wpe: float
    test_mae: float


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def make_horizon_pairs(
    common_x: np.ndarray,
    specific_x: np.ndarray,
    y: np.ndarray,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1")
    if common_x.shape[0] <= forecast_horizon:
        raise ValueError("Not enough samples for the requested forecast_horizon")

    return (
        common_x[:-forecast_horizon],
        specific_x[:-forecast_horizon],
        y[forecast_horizon:],
    )


def _apply_mode(z_common: torch.Tensor, z_specific: torch.Tensor, mode: AblationMode) -> torch.Tensor:
    if mode == "both":
        return torch.cat([z_common, z_specific], dim=-1)
    if mode == "common_only":
        return torch.cat([z_common, torch.zeros_like(z_specific)], dim=-1)
    if mode == "specific_only":
        return torch.cat([torch.zeros_like(z_common), z_specific], dim=-1)
    raise ValueError(f"Unsupported mode: {mode}")


def train_model_for_horizon(
    splits: DatasetSplits,
    *,
    config: TrainConfig,
    forecast_horizon: int,
    mode: AblationMode,
    experiment_name: str,
) -> tuple[TwoBranchForecaster, list[float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    common_train, specific_train, y_train = make_horizon_pairs(
        splits.common_train,
        splits.specific_train,
        splits.y_train,
        forecast_horizon,
    )

    common_train_t = torch.tensor(_flatten_windows(common_train), dtype=torch.float32)
    specific_train_t = torch.tensor(_flatten_windows(specific_train), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    model = TwoBranchForecaster(
        common_dim=common_train_t.shape[1],
        specific_dim=specific_train_t.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.L1Loss()

    losses: list[float] = []
    print(
        f"[train] {experiment_name}: mode={mode} h={forecast_horizon} steps={config.steps} "
        f"common_input_dim={common_train_t.shape[1]} specific_input_dim={specific_train_t.shape[1]}"
    )

    for step in range(1, config.steps + 1):
        _, z_common, z_specific = model(common_train_t, specific_train_t)
        pred = model.head(_apply_mode(z_common, z_specific, mode=mode))
        loss = loss_fn(pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            print(
                f"[train] {experiment_name}: mode={mode} h={forecast_horizon} "
                f"step={step}/{config.steps} loss={loss.item():.6f}"
            )

    return model, losses


def evaluate_model_for_horizon(
    model: TwoBranchForecaster,
    splits: DatasetSplits,
    *,
    forecast_horizon: int,
    mode: AblationMode,
) -> EvalMetrics:
    common_valid, specific_valid, y_valid = make_horizon_pairs(
        splits.common_valid,
        splits.specific_valid,
        splits.y_valid,
        forecast_horizon,
    )
    common_test, specific_test, y_test = make_horizon_pairs(
        splits.common_test,
        splits.specific_test,
        splits.y_test,
        forecast_horizon,
    )

    common_valid_t = torch.tensor(_flatten_windows(common_valid), dtype=torch.float32)
    specific_valid_t = torch.tensor(_flatten_windows(specific_valid), dtype=torch.float32)
    common_test_t = torch.tensor(_flatten_windows(common_test), dtype=torch.float32)
    specific_test_t = torch.tensor(_flatten_windows(specific_test), dtype=torch.float32)

    with torch.no_grad():
        valid_pred = model.predict_with_mode(common_valid_t, specific_valid_t, mode=mode).numpy().reshape(-1)
        test_pred = model.predict_with_mode(common_test_t, specific_test_t, mode=mode).numpy().reshape(-1)

    y_valid_np = y_valid.reshape(-1)
    y_test_np = y_test.reshape(-1)

    return EvalMetrics(
        valid_wape=wape(y_valid_np, valid_pred),
        valid_wpe=wpe(y_valid_np, valid_pred),
        valid_mae=mae(y_valid_np, valid_pred),
        test_wape=wape(y_test_np, test_pred),
        test_wpe=wpe(y_test_np, test_pred),
        test_mae=mae(y_test_np, test_pred),
    )


def predict_for_split_for_horizon(
    model: TwoBranchForecaster,
    splits: DatasetSplits,
    *,
    split: str,
    forecast_horizon: int,
    mode: AblationMode,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "valid":
        common_x, specific_x, y = make_horizon_pairs(splits.common_valid, splits.specific_valid, splits.y_valid, forecast_horizon)
    elif split == "test":
        common_x, specific_x, y = make_horizon_pairs(splits.common_test, splits.specific_test, splits.y_test, forecast_horizon)
    else:
        raise ValueError("split must be one of: valid, test")

    common_t = torch.tensor(_flatten_windows(common_x), dtype=torch.float32)
    specific_t = torch.tensor(_flatten_windows(specific_x), dtype=torch.float32)

    with torch.no_grad():
        pred = model.predict_with_mode(common_t, specific_t, mode=mode).numpy().reshape(-1)

    return y.reshape(-1), pred
