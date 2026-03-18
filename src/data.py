from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from datasets import load_dataset


@dataclass
class FreshRetailConfig:
    dataset_name: str = "Dingdong-Inc/FreshRetailNet-50K"
    split: str = "train"


def load_freshretail_dataframe(config: FreshRetailConfig) -> pd.DataFrame:
    ds = load_dataset(config.dataset_name, split=config.split)
    return ds.to_pandas()


def normalize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        mu = out[col].mean()
        sigma = out[col].std() + 1e-6
        out[col] = (out[col] - mu) / sigma
    return out
