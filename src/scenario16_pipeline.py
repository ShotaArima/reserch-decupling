from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.scenario9_pipeline import DatasetSplits, TwoBranchForecaster


@dataclass
class Scenario16Experiment:
    name: str
    common_features: list[str]
    specific_features: list[str]


def build_scenario16_experiments(
    *,
    common_features: list[str],
    specific_base_features: list[str],
    stock_features: list[str],
) -> list[Scenario16Experiment]:
    exp16a_stock = [f for f in ["stock_hour6_22_cnt"] if f in stock_features]
    exp16b_stock = [f for f in ["stock_hour6_22_cnt", "hours_stock_status"] if f in stock_features]

    experiments = [
        Scenario16Experiment(
            name="baseline_s9_exp1",
            common_features=common_features,
            specific_features=specific_base_features,
        ),
        Scenario16Experiment(
            name="exp16a_specific_stock_cnt_only",
            common_features=common_features,
            specific_features=sorted(set(specific_base_features + exp16a_stock)),
        ),
        Scenario16Experiment(
            name="exp16b_specific_stock_both",
            common_features=common_features,
            specific_features=sorted(set(specific_base_features + exp16b_stock)),
        ),
        Scenario16Experiment(
            name="exp16c_common_stock_both",
            common_features=sorted(set(common_features + exp16b_stock)),
            specific_features=specific_base_features,
        ),
    ]
    return experiments


def _to_one_step_pairs(common_x: np.ndarray, specific_x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return common_x[:-1], specific_x[:-1], y[1:]


def extract_latents(
    model: TwoBranchForecaster,
    splits: DatasetSplits,
    split: Literal["train", "valid", "test"],
) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        common_x, specific_x, y = _to_one_step_pairs(splits.common_train, splits.specific_train, splits.y_train)
    elif split == "valid":
        common_x, specific_x, y = _to_one_step_pairs(splits.common_valid, splits.specific_valid, splits.y_valid)
    else:
        common_x, specific_x, y = _to_one_step_pairs(splits.common_test, splits.specific_test, splits.y_test)

    del y
    common_t = torch.tensor(common_x.reshape(common_x.shape[0], -1), dtype=torch.float32)
    specific_t = torch.tensor(specific_x.reshape(specific_x.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        _, z_common, z_specific = model(common_t, specific_t)
    return z_common.numpy(), z_specific.numpy()


def fit_linear_probe_accuracy(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        return float("nan")

    y_train = train_y.astype(np.float32).reshape(-1)
    y_test = test_y.astype(np.float32).reshape(-1)

    x_train_bias = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=np.float32)], axis=1)
    x_test_bias = np.concatenate([test_x, np.ones((test_x.shape[0], 1), dtype=np.float32)], axis=1)

    w, *_ = np.linalg.lstsq(x_train_bias, y_train, rcond=None)
    pred = x_test_bias @ w
    pred_label = (pred >= 0.5).astype(np.float32)
    return float(np.mean(pred_label == y_test))


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
