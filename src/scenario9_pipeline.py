from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import build_window_tensor, extract_last_timestep_feature, normalize_by_train_stats, split_train_valid_test
from src.metrics import mae, wape, wpe

AblationMode = Literal["both", "common_only", "specific_only"]

COMMON_FEATURE_CANDIDATES: tuple[str, ...] = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "holiday_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "dt_weekday",
    "dt_month",
    "dt_day",
    "dt_weekofyear",
    "dt_is_weekend",
)

SPECIFIC_FEATURE_CANDIDATES: tuple[str, ...] = (
    "sale_amount",
    "hours_sale",
    "discount",
    "activity_flag",
)

STOCK_FEATURE_CANDIDATES: tuple[str, ...] = (
    "stock_hour6_22_cnt",
    "hours_stock_status",
)


@dataclass
class DatasetSplits:
    common_train: np.ndarray
    common_valid: np.ndarray
    common_test: np.ndarray
    specific_train: np.ndarray
    specific_valid: np.ndarray
    specific_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


@dataclass
class TrainConfig:
    steps: int = 100
    lr: float = 1e-3
    hidden_dim: int = 64
    latent_dim: int = 16
    seed: int = 42
    log_interval: int = 20


@dataclass
class EvalMetrics:
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    test_wape: float
    test_wpe: float
    test_mae: float


class TwoBranchForecaster(nn.Module):
    def __init__(self, common_dim: int, specific_dim: int, hidden_dim: int = 64, latent_dim: int = 16) -> None:
        super().__init__()
        self.common_encoder = nn.Sequential(
            nn.Linear(common_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.specific_encoder = nn.Sequential(
            nn.Linear(specific_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, common_x: torch.Tensor, specific_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_common = self.common_encoder(common_x)
        z_specific = self.specific_encoder(specific_x)
        pred = self.head(torch.cat([z_common, z_specific], dim=-1))
        return pred, z_common, z_specific

    def predict_with_mode(self, common_x: torch.Tensor, specific_x: torch.Tensor, mode: AblationMode) -> torch.Tensor:
        pred, z_common, z_specific = self(common_x, specific_x)
        del pred
        if mode == "both":
            z = torch.cat([z_common, z_specific], dim=-1)
        elif mode == "common_only":
            z = torch.cat([z_common, torch.zeros_like(z_specific)], dim=-1)
        else:
            z = torch.cat([torch.zeros_like(z_common), z_specific], dim=-1)
        return self.head(z)


def _parse_dt_seq(value: object) -> list[pd.Timestamp]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        seq = list(value)
    else:
        seq = [value]

    out: list[pd.Timestamp] = []
    for item in seq:
        ts = pd.to_datetime(item, errors="coerce")
        if pd.isna(ts):
            continue
        out.append(ts)
    return out


def add_dt_features(df: pd.DataFrame) -> pd.DataFrame:
    if "dt" not in df.columns:
        return df

    df = df.copy()
    dt_seq = df["dt"].map(_parse_dt_seq)

    def _map_seq(values: list[pd.Timestamp], fn) -> list[float]:
        if not values:
            return [np.nan]
        return [float(fn(v)) for v in values]

    df["dt_weekday"] = dt_seq.map(lambda arr: _map_seq(arr, lambda v: v.weekday()))
    df["dt_month"] = dt_seq.map(lambda arr: _map_seq(arr, lambda v: v.month))
    df["dt_day"] = dt_seq.map(lambda arr: _map_seq(arr, lambda v: v.day))
    df["dt_weekofyear"] = dt_seq.map(lambda arr: _map_seq(arr, lambda v: v.isocalendar().week))
    df["dt_is_weekend"] = dt_seq.map(lambda arr: _map_seq(arr, lambda v: 1 if v.weekday() >= 5 else 0))
    return df


def resolve_features(df: pd.DataFrame, candidates: Sequence[str]) -> tuple[list[str], list[str]]:
    present = [f for f in candidates if f in df.columns]
    missing = [f for f in candidates if f not in df.columns]
    return present, missing


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def build_splits(
    df: pd.DataFrame,
    *,
    common_features: Sequence[str],
    specific_features: Sequence[str],
    target_feature: str = "sale_amount",
    window_size: int = 14,
) -> DatasetSplits:
    common_all = build_window_tensor(df, common_features, window_size=window_size)
    specific_all = build_window_tensor(df, specific_features, window_size=window_size)
    target_all = build_window_tensor(df, [target_feature], window_size=window_size)
    y_all = extract_last_timestep_feature(target_all, 0)

    common_train_raw, common_valid_raw, common_test_raw = split_train_valid_test(common_all)
    specific_train_raw, specific_valid_raw, specific_test_raw = split_train_valid_test(specific_all)
    y_train, y_valid, y_test = split_train_valid_test(y_all)

    common_train, common_valid, common_test = normalize_by_train_stats(common_train_raw, common_valid_raw, common_test_raw)
    specific_train, specific_valid, specific_test = normalize_by_train_stats(
        specific_train_raw,
        specific_valid_raw,
        specific_test_raw,
    )

    return DatasetSplits(
        common_train=common_train,
        common_valid=common_valid,
        common_test=common_test,
        specific_train=specific_train,
        specific_valid=specific_valid,
        specific_test=specific_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )


def _make_one_step_pairs(common_x: np.ndarray, specific_x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return common_x[:-1], specific_x[:-1], y[1:]


def train_model(
    splits: DatasetSplits,
    *,
    config: TrainConfig,
    experiment_name: str,
) -> tuple[TwoBranchForecaster, list[float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    common_train, specific_train, y_train = _make_one_step_pairs(splits.common_train, splits.specific_train, splits.y_train)

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
        f"[train] {experiment_name}: steps={config.steps}, lr={config.lr}, "
        f"hidden_dim={config.hidden_dim}, latent_dim={config.latent_dim}, "
        f"common_input_dim={common_train_t.shape[1]}, specific_input_dim={specific_train_t.shape[1]}"
    )

    for step in range(1, config.steps + 1):
        pred, _, _ = model(common_train_t, specific_train_t)
        loss = loss_fn(pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            print(f"[train] {experiment_name}: step={step}/{config.steps} loss={loss.item():.6f}")

    return model, losses


def evaluate_model(model: TwoBranchForecaster, splits: DatasetSplits, mode: AblationMode = "both") -> EvalMetrics:
    common_valid, specific_valid, y_valid = _make_one_step_pairs(splits.common_valid, splits.specific_valid, splits.y_valid)
    common_test, specific_test, y_test = _make_one_step_pairs(splits.common_test, splits.specific_test, splits.y_test)

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


def predict_for_split(model: TwoBranchForecaster, splits: DatasetSplits, split: Literal["valid", "test"], mode: AblationMode = "both") -> tuple[np.ndarray, np.ndarray]:
    if split == "valid":
        common_x, specific_x, y = _make_one_step_pairs(splits.common_valid, splits.specific_valid, splits.y_valid)
    else:
        common_x, specific_x, y = _make_one_step_pairs(splits.common_test, splits.specific_test, splits.y_test)

    common_t = torch.tensor(_flatten_windows(common_x), dtype=torch.float32)
    specific_t = torch.tensor(_flatten_windows(specific_x), dtype=torch.float32)

    with torch.no_grad():
        pred = model.predict_with_mode(common_t, specific_t, mode=mode).numpy().reshape(-1)

    return y.reshape(-1), pred
