from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


@dataclass
class FreshRetailConfig:
    dataset_name: str = "Dingdong-Inc/FreshRetailNet-50K"
    split: str = "train"
    local_data_dir: Path = Path("data")


def _dataset_disk_path(config: FreshRetailConfig) -> Path:
    dataset_slug = config.dataset_name.split("/")[-1]
    return config.local_data_dir / dataset_slug


def _ensure_dataset_on_disk(config: FreshRetailConfig) -> Path:
    dataset_path = _dataset_disk_path(config)
    if dataset_path.exists():
        return dataset_path

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(config.dataset_name)
    ds.save_to_disk(str(dataset_path))
    return dataset_path


def _extract_split_from_saved_dataset(saved: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(saved, Dataset):
        if split != "train":
            raise ValueError(
                f"Saved dataset contains a single split only. Requested split='{split}'."
            )
        return saved

    if split not in saved:
        available_splits = ", ".join(saved.keys())
        raise ValueError(
            f"Split '{split}' not found in saved dataset. Available splits: {available_splits}."
        )
    return saved[split]


def load_freshretail_dataframe(config: FreshRetailConfig) -> pd.DataFrame:
    dataset_path = _ensure_dataset_on_disk(config)
    saved_ds = load_from_disk(str(dataset_path))
    ds = _extract_split_from_saved_dataset(saved_ds, config.split)
    return ds.to_pandas()


def _cell_to_window(value: object, window_size: int) -> np.ndarray:
    """Convert a cell value to a fixed-length time window.

    Sequence-like values are truncated to the most recent `window_size` values.
    Scalar values are repeated over the window.
    """
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            arr = np.full(window_size, np.nan, dtype=np.float32)
        elif arr.size >= window_size:
            arr = arr[-window_size:]
        else:
            pad = np.full(window_size - arr.size, np.nan, dtype=np.float32)
            arr = np.concatenate([pad, arr])
        return arr

    if pd.isna(value):
        return np.full(window_size, np.nan, dtype=np.float32)

    scalar = float(value)
    return np.full(window_size, scalar, dtype=np.float32)


def build_window_tensor(df: pd.DataFrame, features: Iterable[str], window_size: int) -> np.ndarray:
    """Build a [N, W, F] tensor from dataframe cells while preserving time windows."""
    feature_windows: list[np.ndarray] = []
    for col in features:
        col_windows = np.stack(df[col].map(lambda v: _cell_to_window(v, window_size)).to_numpy())
        feature_windows.append(col_windows)

    # [F, N, W] -> [N, W, F]
    return np.stack(feature_windows, axis=0).transpose(1, 2, 0)


def split_train_valid_test(
    x: np.ndarray,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be in (0, 1)")

    n = x.shape[0]
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return x[:train_end], x[train_end:valid_end], x[valid_end:]


def normalize_by_train_stats(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each feature using train split statistics only."""
    mu = np.nanmean(train_x, axis=(0, 1), keepdims=True)
    sigma = np.nanstd(train_x, axis=(0, 1), keepdims=True) + 1e-6

    def _norm(arr: np.ndarray) -> np.ndarray:
        out = (arr - mu) / sigma
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return _norm(train_x), _norm(valid_x), _norm(test_x)


def fit_train_normalization_stats(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only normalization stats with shape [1, 1, F]."""
    mu = np.nanmean(train_x, axis=(0, 1), keepdims=True)
    sigma = np.nanstd(train_x, axis=(0, 1), keepdims=True) + 1e-6
    return mu, sigma


def apply_normalization(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    out = (x - mu) / sigma
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def denormalize(x_norm: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x_norm * sigma) + mu


def extract_last_timestep_feature(x: np.ndarray, feature_idx: int) -> np.ndarray:
    return x[:, -1, feature_idx]
