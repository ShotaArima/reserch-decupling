from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead


@dataclass
class ForecastResult:
    name: str
    valid_wape: float
    valid_wpe: float
    test_wape: float
    test_wpe: float


def make_one_step_pairs(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[:-1], y[1:]


def predict_last_value(x_raw: np.ndarray, target_feature_index: int = 0) -> np.ndarray:
    return x_raw[:, -1, target_feature_index]


def predict_moving_average(x_raw: np.ndarray, k: int, target_feature_index: int = 0) -> np.ndarray:
    return np.nanmean(x_raw[:, -k:, target_feature_index], axis=1)


def train_flatten_linear(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    lr: float = 1e-3,
    steps: int = 100,
) -> tuple[nn.Module, list[float]]:
    input_dim = x_train.shape[1] * x_train.shape[2]
    model = nn.Linear(input_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    losses: list[float] = []

    x_flat = x_train.reshape(x_train.shape[0], -1)
    for _ in range(steps):
        pred = model(x_flat)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return model, losses


class FlattenMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_flatten_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    hidden_dims: Sequence[int],
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    steps: int = 100,
) -> tuple[nn.Module, list[float]]:
    input_dim = x_train.shape[1] * x_train.shape[2]
    model = FlattenMLP(input_dim=input_dim, hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()
    losses: list[float] = []

    x_flat = x_train.reshape(x_train.shape[0], -1)
    for _ in range(steps):
        pred = model(x_flat)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return model, losses


def train_scenario2_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    feature_dim: int,
    window_size: int,
    lr: float = 1e-3,
    steps: int = 100,
) -> tuple[DecouplingAutoEncoder, ForecastHead, list[float]]:
    body = DecouplingAutoEncoder(DecouplingConfig(feature_dim=feature_dim, window_size=window_size))
    head = ForecastHead(local_dim=16, global_dim=16, horizon=1)
    optimizer = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.L1Loss()
    losses: list[float] = []

    for _ in range(steps):
        _, local, global_latent = body(x_train)
        pred = head(local, global_latent)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return body, head, losses


def train_scenario4_pipeline(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    feature_dim: int,
    window_size: int,
    lr: float = 1e-3,
    steps: int = 100,
) -> tuple[DecouplingAutoEncoder, ForecastHead, list[float], list[float]]:
    recovery = DecouplingAutoEncoder(DecouplingConfig(feature_dim=feature_dim, window_size=window_size))
    stage1_optimizer = torch.optim.Adam(recovery.parameters(), lr=lr)
    stage1_losses: list[float] = []

    for _ in range(steps):
        rec, _, _ = recovery(x_train)
        stage1_loss = nn.functional.mse_loss(rec, x_train)
        stage1_optimizer.zero_grad()
        stage1_loss.backward()
        stage1_optimizer.step()
        stage1_losses.append(float(stage1_loss.item()))

    with torch.no_grad():
        _, local_train, global_train = recovery(x_train)

    local_train_s, y_train_s = make_one_step_pairs(local_train, y_train)
    global_train_s, _ = make_one_step_pairs(global_train, y_train)

    forecaster = ForecastHead(local_dim=16, global_dim=16, horizon=1)
    stage2_optimizer = torch.optim.Adam(forecaster.parameters(), lr=lr)
    stage2_losses: list[float] = []

    for _ in range(steps):
        pred = forecaster(local_train_s.detach(), global_train_s.detach())
        stage2_loss = nn.functional.l1_loss(pred, y_train_s)
        stage2_optimizer.zero_grad()
        stage2_loss.backward()
        stage2_optimizer.step()
        stage2_losses.append(float(stage2_loss.item()))

    return recovery, forecaster, stage1_losses, stage2_losses
