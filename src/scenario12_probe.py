from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

TaskType = Literal["classification", "regression"]


@dataclass
class ProbeConfig:
    steps: int = 120
    lr: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 512
    seed: int = 42
    log_interval: int = 20


@dataclass
class ProbeResult:
    metric_name: str
    value: float


class _LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _LinearRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    if y_true.size == 0 or num_classes <= 0:
        return 0.0

    f1s: list[float] = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(float(f1))
    return float(np.mean(f1s))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _iter_batches(n: int, batch_size: int) -> list[tuple[int, int]]:
    if batch_size <= 0:
        return [(0, n)]
    bounds: list[tuple[int, int]] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bounds.append((start, end))
    return bounds


def fit_linear_classification_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_classes: int,
    config: ProbeConfig,
    probe_name: str,
    latent_name: str,
) -> nn.Module:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    model = _LinearClassifier(input_dim=x_t.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    print(
        f"[probe-train] {probe_name} / {latent_name}: steps={config.steps}, lr={config.lr}, "
        f"batch_size={config.batch_size}, classes={num_classes}, n_train={x_t.shape[0]}"
    )

    for step in range(1, config.steps + 1):
        perm = torch.randperm(x_t.shape[0])
        total_loss = 0.0
        for b_start, b_end in _iter_batches(x_t.shape[0], config.batch_size):
            idx = perm[b_start:b_end]
            logits = model(x_t[idx])
            loss = loss_fn(logits, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * (b_end - b_start)

        avg_loss = total_loss / max(1, x_t.shape[0])
        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            print(f"[probe-train] {probe_name} / {latent_name}: step={step}/{config.steps} loss={avg_loss:.6f}")

    return model


def predict_classification(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).numpy()
    return logits.argmax(axis=1)


def fit_linear_regression_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: ProbeConfig,
    probe_name: str,
    latent_name: str,
) -> nn.Module:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model = _LinearRegressor(input_dim=x_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    print(
        f"[probe-train] {probe_name} / {latent_name}: steps={config.steps}, lr={config.lr}, "
        f"batch_size={config.batch_size}, n_train={x_t.shape[0]}"
    )

    for step in range(1, config.steps + 1):
        perm = torch.randperm(x_t.shape[0])
        total_loss = 0.0
        for b_start, b_end in _iter_batches(x_t.shape[0], config.batch_size):
            idx = perm[b_start:b_end]
            pred = model(x_t[idx])
            loss = loss_fn(pred, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * (b_end - b_start)

        avg_loss = total_loss / max(1, x_t.shape[0])
        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            print(f"[probe-train] {probe_name} / {latent_name}: step={step}/{config.steps} loss={avg_loss:.6f}")

    return model


def predict_regression(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32)).numpy()
    return pred.reshape(-1)
