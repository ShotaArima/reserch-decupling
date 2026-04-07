from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import build_window_tensor, extract_last_timestep_feature, normalize_by_train_stats, split_train_valid_test
from src.metrics import mae, wape, wpe
from src.scenario9_pipeline import SPECIFIC_FEATURE_CANDIDATES, add_dt_features

AblationMode = Literal["both", "common_only", "local_only"]

BASE_COMMON_FEATURES: tuple[str, ...] = (
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
)

CALENDAR_FEATURES: tuple[str, ...] = (
    "dt_weekday",
    "dt_month",
    "dt_day",
    "dt_weekday_sin",
    "dt_weekday_cos",
    "dt_month_sin",
    "dt_month_cos",
    "dt_day_sin",
    "dt_day_cos",
)

WEATHER_LAG_FEATURES: tuple[str, ...] = (
    "avg_temperature_lag1",
    "avg_temperature_lag7",
    "precpt_lag1",
    "precpt_lag7",
    "avg_temperature_ma7",
    "avg_temperature_ma14",
    "precpt_ma7",
    "precpt_ma14",
    "weather_history_available",
)

HIERARCHY_FEATURES: tuple[str, ...] = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
)


@dataclass
class DatasetSplits:
    common_train: np.ndarray
    common_valid: np.ndarray
    common_test: np.ndarray
    specific_train: np.ndarray
    specific_valid: np.ndarray
    specific_test: np.ndarray
    cat_train: np.ndarray
    cat_valid: np.ndarray
    cat_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    probe_weekday_train: np.ndarray
    probe_weekday_valid: np.ndarray
    probe_weekday_test: np.ndarray
    probe_month_train: np.ndarray
    probe_month_valid: np.ndarray
    probe_month_test: np.ndarray
    probe_hierarchy_train: np.ndarray
    probe_hierarchy_valid: np.ndarray
    probe_hierarchy_test: np.ndarray
    cat_cardinalities: list[int]


@dataclass
class TrainConfig:
    steps: int = 120
    lr: float = 1e-3
    hidden_dim: int = 64
    latent_dim: int = 16
    seed: int = 42
    log_interval: int = 20
    hierarchy_dim: int = 8


@dataclass
class EvalMetrics:
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    test_wape: float
    test_wpe: float
    test_mae: float


@dataclass
class ProbeMetrics:
    split: str
    target: str
    accuracy: float
    macro_f1: float


def _to_array(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr
    if pd.isna(value):
        return np.array([], dtype=np.float32)
    return np.array([float(value)], dtype=np.float32)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = np.zeros_like(arr, dtype=np.float32)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        out[idx] = float(np.mean(arr[start : idx + 1]))
    return out


def _lag(arr: np.ndarray, lag: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = np.zeros_like(arr, dtype=np.float32)
    if lag < arr.size:
        out[lag:] = arr[:-lag]
    return out


def _resolve_present(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [name for name in candidates if name in df.columns]


def add_scenario15_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_dt_features(df).copy()

    if "dt_weekday" in out.columns:
        out["dt_weekday_sin"] = out["dt_weekday"].map(
            lambda v: np.sin(2 * np.pi * (_to_array(v) % 7) / 7).astype(np.float32).tolist()
        )
        out["dt_weekday_cos"] = out["dt_weekday"].map(
            lambda v: np.cos(2 * np.pi * (_to_array(v) % 7) / 7).astype(np.float32).tolist()
        )

    if "dt_month" in out.columns:
        out["dt_month_sin"] = out["dt_month"].map(
            lambda v: np.sin(2 * np.pi * ((_to_array(v) - 1) % 12) / 12).astype(np.float32).tolist()
        )
        out["dt_month_cos"] = out["dt_month"].map(
            lambda v: np.cos(2 * np.pi * ((_to_array(v) - 1) % 12) / 12).astype(np.float32).tolist()
        )

    if "dt_day" in out.columns:
        out["dt_day_sin"] = out["dt_day"].map(
            lambda v: np.sin(2 * np.pi * ((_to_array(v) - 1) % 31) / 31).astype(np.float32).tolist()
        )
        out["dt_day_cos"] = out["dt_day"].map(
            lambda v: np.cos(2 * np.pi * ((_to_array(v) - 1) % 31) / 31).astype(np.float32).tolist()
        )

    for base_col in ("avg_temperature", "precpt"):
        if base_col not in out.columns:
            continue
        out[f"{base_col}_lag1"] = out[base_col].map(lambda v: _lag(_to_array(v), lag=1).tolist())
        out[f"{base_col}_lag7"] = out[base_col].map(lambda v: _lag(_to_array(v), lag=7).tolist())
        out[f"{base_col}_ma7"] = out[base_col].map(lambda v: _rolling_mean(_to_array(v), window=7).tolist())
        out[f"{base_col}_ma14"] = out[base_col].map(lambda v: _rolling_mean(_to_array(v), window=14).tolist())

    out["weather_history_available"] = out["avg_temperature"].map(
        lambda v: np.where(np.arange(_to_array(v).size) >= 7, 1.0, 0.0).astype(np.float32).tolist()
        if _to_array(v).size
        else [0.0]
    )
    return out


def resolve_arm_features(df: pd.DataFrame, arm: str) -> tuple[list[str], list[str], list[str]]:
    common_features = list(_resolve_present(df, BASE_COMMON_FEATURES))
    specific_features = list(_resolve_present(df, SPECIFIC_FEATURE_CANDIDATES))

    if arm in {"A1", "A2", "A3"}:
        common_features.extend(_resolve_present(df, CALENDAR_FEATURES))
    if arm in {"A2", "A3"}:
        common_features.extend(_resolve_present(df, WEATHER_LAG_FEATURES))

    common_features = sorted(set(common_features))
    cat_features = [name for name in HIERARCHY_FEATURES if name in common_features]
    return common_features, specific_features, cat_features


def _build_categorical_last_step(df: pd.DataFrame, cat_features: Sequence[str], window_size: int) -> np.ndarray:
    if not cat_features:
        return np.zeros((len(df), 0), dtype=np.int64)

    cat_windows = build_window_tensor(df, cat_features, window_size=window_size)
    last = cat_windows[:, -1, :]
    last = np.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(last.astype(np.int64), a_min=0, a_max=None)


def _make_probe_target(df: pd.DataFrame, column: str, window_size: int, default: int = 0) -> np.ndarray:
    if column not in df.columns:
        return np.full((len(df),), default, dtype=np.int64)
    arr = build_window_tensor(df, [column], window_size=window_size)
    out = extract_last_timestep_feature(arr, 0)
    out = np.nan_to_num(out, nan=float(default), posinf=float(default), neginf=float(default))
    return np.clip(out.astype(np.int64), a_min=0, a_max=None)


def build_splits(
    df: pd.DataFrame,
    *,
    common_features: Sequence[str],
    specific_features: Sequence[str],
    cat_features: Sequence[str],
    target_feature: str = "sale_amount",
    window_size: int = 14,
) -> DatasetSplits:
    common_all = build_window_tensor(df, common_features, window_size=window_size)
    specific_all = build_window_tensor(df, specific_features, window_size=window_size)
    target_all = build_window_tensor(df, [target_feature], window_size=window_size)
    y_all = extract_last_timestep_feature(target_all, 0)

    cat_all = _build_categorical_last_step(df, cat_features, window_size=window_size)
    probe_weekday = _make_probe_target(df, "dt_weekday", window_size=window_size)
    probe_month = _make_probe_target(df, "dt_month", window_size=window_size)
    probe_hierarchy = _make_probe_target(df, "first_category_id", window_size=window_size)

    common_train_raw, common_valid_raw, common_test_raw = split_train_valid_test(common_all)
    specific_train_raw, specific_valid_raw, specific_test_raw = split_train_valid_test(specific_all)
    cat_train, cat_valid, cat_test = split_train_valid_test(cat_all)

    y_train, y_valid, y_test = split_train_valid_test(y_all)
    probe_weekday_train, probe_weekday_valid, probe_weekday_test = split_train_valid_test(probe_weekday)
    probe_month_train, probe_month_valid, probe_month_test = split_train_valid_test(probe_month)
    probe_hierarchy_train, probe_hierarchy_valid, probe_hierarchy_test = split_train_valid_test(probe_hierarchy)

    common_train, common_valid, common_test = normalize_by_train_stats(common_train_raw, common_valid_raw, common_test_raw)
    specific_train, specific_valid, specific_test = normalize_by_train_stats(
        specific_train_raw,
        specific_valid_raw,
        specific_test_raw,
    )

    cardinalities: list[int] = []
    for col_idx in range(cat_train.shape[1] if cat_train.ndim == 2 else 0):
        max_idx = int(np.max(cat_train[:, col_idx])) if cat_train.shape[0] else 0
        cardinalities.append(max(2, max_idx + 1))

    return DatasetSplits(
        common_train=common_train,
        common_valid=common_valid,
        common_test=common_test,
        specific_train=specific_train,
        specific_valid=specific_valid,
        specific_test=specific_test,
        cat_train=cat_train,
        cat_valid=cat_valid,
        cat_test=cat_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        probe_weekday_train=probe_weekday_train,
        probe_weekday_valid=probe_weekday_valid,
        probe_weekday_test=probe_weekday_test,
        probe_month_train=probe_month_train,
        probe_month_valid=probe_month_valid,
        probe_month_test=probe_month_test,
        probe_hierarchy_train=probe_hierarchy_train,
        probe_hierarchy_valid=probe_hierarchy_valid,
        probe_hierarchy_test=probe_hierarchy_test,
        cat_cardinalities=cardinalities,
    )


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _make_one_step_pairs(common_x: np.ndarray, specific_x: np.ndarray, cat_x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return common_x[:-1], specific_x[:-1], cat_x[:-1], y[1:]


class CommonBranchForecaster(nn.Module):
    def __init__(
        self,
        common_dim: int,
        specific_dim: int,
        cat_cardinalities: Sequence[int],
        hierarchy_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, hierarchy_dim) for card in cat_cardinalities])
        cat_total_dim = hierarchy_dim * len(cat_cardinalities)

        self.common_encoder = nn.Sequential(
            nn.Linear(common_dim + cat_total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(specific_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _concat_common(self, common_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) == 0:
            return common_x
        embs = []
        for idx, emb in enumerate(self.embeddings):
            embs.append(emb(cat_x[:, idx]))
        return torch.cat([common_x, torch.cat(embs, dim=-1)], dim=-1)

    def forward(self, common_x: torch.Tensor, specific_x: torch.Tensor, cat_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        common_in = self._concat_common(common_x, cat_x)
        z_common = self.common_encoder(common_in)
        z_local = self.local_encoder(specific_x)
        pred = self.head(torch.cat([z_common, z_local], dim=-1))
        return pred, z_common, z_local

    def predict_with_mode(self, common_x: torch.Tensor, specific_x: torch.Tensor, cat_x: torch.Tensor, mode: AblationMode) -> torch.Tensor:
        _, z_common, z_local = self(common_x, specific_x, cat_x)
        if mode == "both":
            z = torch.cat([z_common, z_local], dim=-1)
        elif mode == "common_only":
            z = torch.cat([z_common, torch.zeros_like(z_local)], dim=-1)
        else:
            z = torch.cat([torch.zeros_like(z_common), z_local], dim=-1)
        return self.head(z)

    def encode_common(self, common_x: torch.Tensor, specific_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        _, z_common, _ = self(common_x, specific_x, cat_x)
        return z_common


def train_model(splits: DatasetSplits, *, config: TrainConfig, experiment_name: str) -> tuple[CommonBranchForecaster, list[float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    common_train, specific_train, cat_train, y_train = _make_one_step_pairs(
        splits.common_train,
        splits.specific_train,
        splits.cat_train,
        splits.y_train,
    )

    common_train_t = torch.tensor(_flatten_windows(common_train), dtype=torch.float32)
    specific_train_t = torch.tensor(_flatten_windows(specific_train), dtype=torch.float32)
    cat_train_t = torch.tensor(cat_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    model = CommonBranchForecaster(
        common_dim=common_train_t.shape[1],
        specific_dim=specific_train_t.shape[1],
        cat_cardinalities=splits.cat_cardinalities,
        hierarchy_dim=config.hierarchy_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses: list[float] = []

    print(
        f"[train] {experiment_name}: steps={config.steps} lr={config.lr} hidden_dim={config.hidden_dim} "
        f"latent_dim={config.latent_dim} hierarchy_dim={config.hierarchy_dim} log_interval={config.log_interval}"
    )

    for step in range(1, config.steps + 1):
        pred, _, _ = model(common_train_t, specific_train_t, cat_train_t)
        loss = nn.functional.l1_loss(pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            print(f"[train] {experiment_name}: step={step}/{config.steps} loss={loss.item():.6f}")

    return model, losses


def evaluate_model(model: CommonBranchForecaster, splits: DatasetSplits, mode: AblationMode) -> EvalMetrics:
    common_valid, specific_valid, cat_valid, y_valid = _make_one_step_pairs(
        splits.common_valid,
        splits.specific_valid,
        splits.cat_valid,
        splits.y_valid,
    )
    common_test, specific_test, cat_test, y_test = _make_one_step_pairs(
        splits.common_test,
        splits.specific_test,
        splits.cat_test,
        splits.y_test,
    )

    common_valid_t = torch.tensor(_flatten_windows(common_valid), dtype=torch.float32)
    specific_valid_t = torch.tensor(_flatten_windows(specific_valid), dtype=torch.float32)
    cat_valid_t = torch.tensor(cat_valid, dtype=torch.long)
    common_test_t = torch.tensor(_flatten_windows(common_test), dtype=torch.float32)
    specific_test_t = torch.tensor(_flatten_windows(specific_test), dtype=torch.float32)
    cat_test_t = torch.tensor(cat_test, dtype=torch.long)

    with torch.no_grad():
        valid_pred = model.predict_with_mode(common_valid_t, specific_valid_t, cat_valid_t, mode=mode).numpy().reshape(-1)
        test_pred = model.predict_with_mode(common_test_t, specific_test_t, cat_test_t, mode=mode).numpy().reshape(-1)

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


def collect_common_latent(model: CommonBranchForecaster, splits: DatasetSplits, split: Literal["train", "valid", "test"]) -> np.ndarray:
    if split == "train":
        common_x, specific_x, cat_x, _ = _make_one_step_pairs(splits.common_train, splits.specific_train, splits.cat_train, splits.y_train)
    elif split == "valid":
        common_x, specific_x, cat_x, _ = _make_one_step_pairs(splits.common_valid, splits.specific_valid, splits.cat_valid, splits.y_valid)
    else:
        common_x, specific_x, cat_x, _ = _make_one_step_pairs(splits.common_test, splits.specific_test, splits.cat_test, splits.y_test)

    common_t = torch.tensor(_flatten_windows(common_x), dtype=torch.float32)
    specific_t = torch.tensor(_flatten_windows(specific_x), dtype=torch.float32)
    cat_t = torch.tensor(cat_x, dtype=torch.long)

    with torch.no_grad():
        z = model.encode_common(common_t, specific_t, cat_t).numpy()
    return z


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores: list[float] = []
    for cls in labels:
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        scores.append(2 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(scores)) if scores else 0.0


def train_probe_classifier(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    steps: int = 120,
    lr: float = 5e-3,
    log_interval: int = 20,
    tag: str,
) -> tuple[float, float]:
    y_train = y_train.astype(np.int64)
    y_eval = y_eval.astype(np.int64)

    num_classes = int(max(np.max(y_train, initial=0), np.max(y_eval, initial=0)) + 1)
    if num_classes <= 1:
        return 1.0, 1.0

    model = nn.Linear(z_train.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    z_train_t = torch.tensor(z_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for step in range(1, steps + 1):
        logits = model(z_train_t)
        loss = nn.functional.cross_entropy(logits, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_interval == 0 or step == steps:
            print(f"[probe] {tag}: step={step}/{steps} loss={loss.item():.6f}")

    with torch.no_grad():
        pred = model(torch.tensor(z_eval, dtype=torch.float32)).argmax(dim=-1).numpy().astype(np.int64)

    accuracy = float(np.mean(pred == y_eval)) if y_eval.size else 0.0
    macro_f1 = _macro_f1(y_eval, pred)
    return accuracy, macro_f1
