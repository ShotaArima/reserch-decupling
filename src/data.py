from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import load_dataset


@dataclass
class FreshRetailConfig:
    dataset_name: str = "Dingdong-Inc/FreshRetailNet-50K"
    split: str = "train"


def load_freshretail_dataframe(config: FreshRetailConfig) -> pd.DataFrame:
    ds = load_dataset(config.dataset_name, split=config.split)
    return ds.to_pandas()


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convert object-like columns (e.g., list/array cells) into numeric scalars."""

    def _to_scalar(value: object) -> float:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(value).reshape(-1)
            if arr.size == 0:
                return np.nan
            # Use mean for sequence-like cells so hourly/vector features become scalars.
            return float(np.nanmean(arr.astype(np.float64)))
        if pd.isna(value):
            return np.nan
        return float(value)

    return series.map(_to_scalar)


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = _coerce_numeric_series(out[col])
    return out


def normalize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = coerce_numeric_columns(df, columns)
    for col in columns:
        mu = out[col].mean()
        sigma = out[col].std() + 1e-6
        out[col] = (out[col] - mu) / sigma
    return out
