from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.metrics import mae, wape, wpe


@dataclass
class SubsetMetricRow:
    subset: str
    model: str
    n: int
    target_sum: float
    stockout_ratio: float
    wape_value: float
    wpe_value: float
    mae_value: float


@dataclass
class SubsetDiffRow:
    subset: str
    metric: str
    s4_minus_s2: float
    s4_minus_raw: float
    s2_minus_raw: float


def build_subset_masks(stockout_mask: np.ndarray) -> dict[str, np.ndarray]:
    stockout = stockout_mask.astype(bool)
    return {
        "all": np.ones_like(stockout, dtype=bool),
        "stockout": stockout,
        "non_stockout": ~stockout,
    }


def compute_subset_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stockout_mask: np.ndarray,
    model_name: str,
) -> list[SubsetMetricRow]:
    masks = build_subset_masks(stockout_mask)
    rows: list[SubsetMetricRow] = []

    for subset_name, mask in masks.items():
        n = int(np.sum(mask))
        if n == 0:
            rows.append(
                SubsetMetricRow(
                    subset=subset_name,
                    model=model_name,
                    n=0,
                    target_sum=0.0,
                    stockout_ratio=0.0,
                    wape_value=float("nan"),
                    wpe_value=float("nan"),
                    mae_value=float("nan"),
                )
            )
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        stockout_ratio = float(np.mean(stockout_mask[mask]))
        rows.append(
            SubsetMetricRow(
                subset=subset_name,
                model=model_name,
                n=n,
                target_sum=float(np.sum(np.abs(yt))),
                stockout_ratio=stockout_ratio,
                wape_value=wape(yt, yp),
                wpe_value=wpe(yt, yp),
                mae_value=mae(yt, yp),
            )
        )

    return rows


def compute_diff_rows(rows: Iterable[SubsetMetricRow]) -> list[SubsetDiffRow]:
    by_key: dict[tuple[str, str], float] = {}
    for r in rows:
        by_key[(r.subset, f"{r.model}:WAPE")] = r.wape_value
        by_key[(r.subset, f"{r.model}:WPE")] = r.wpe_value

    diff_rows: list[SubsetDiffRow] = []
    for subset in ("all", "stockout", "non_stockout"):
        for metric in ("WAPE", "WPE"):
            raw = by_key[(subset, f"raw_baseline:{metric}")]
            s2 = by_key[(subset, f"Scenario2:{metric}")]
            s4 = by_key[(subset, f"Scenario4:{metric}")]
            diff_rows.append(
                SubsetDiffRow(
                    subset=subset,
                    metric=metric,
                    s4_minus_s2=s4 - s2,
                    s4_minus_raw=s4 - raw,
                    s2_minus_raw=s2 - raw,
                )
            )
    return diff_rows
