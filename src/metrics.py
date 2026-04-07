from __future__ import annotations

import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8))


def wpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(y_pred - y_true) / (np.sum(np.abs(y_true)) + 1e-8))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def residual_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.std(y_pred - y_true))


def diff_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] < 3:
        return 0.0
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    true_std = np.std(true_diff)
    pred_std = np.std(pred_diff)
    if true_std < 1e-12 or pred_std < 1e-12:
        return 0.0
    return float(np.corrcoef(true_diff, pred_diff)[0, 1])
