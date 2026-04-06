from __future__ import annotations

import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8))


def wpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(y_pred - y_true) / (np.sum(np.abs(y_true)) + 1e-8))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
