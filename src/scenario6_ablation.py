from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.forecast_baselines import make_one_step_pairs
from src.models import DecouplingAutoEncoder, DecouplingConfig, ForecastHead

AblationMode = Literal["local_only", "global_only", "both"]


@dataclass
class LatentSplits:
    local_train: torch.Tensor
    global_train: torch.Tensor
    local_valid: torch.Tensor
    global_valid: torch.Tensor
    local_test: torch.Tensor
    global_test: torch.Tensor


@dataclass
class LatentSourceArtifacts:
    latents: LatentSplits
    train_losses: list[float]


def train_scenario2_latents(
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    x_test: torch.Tensor,
    y_train: torch.Tensor,
    *,
    feature_dim: int,
    window_size: int,
    steps: int = 100,
    lr: float = 1e-3,
) -> LatentSourceArtifacts:
    """Train end-to-end Scenario2 body (with auxiliary head loss) and extract latents."""
    body = DecouplingAutoEncoder(DecouplingConfig(feature_dim=feature_dim, window_size=window_size))
    head = ForecastHead(local_dim=16, global_dim=16, horizon=1)

    optimizer = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=lr)
    losses: list[float] = []

    for _ in range(steps):
        _, local, global_latent = body(x_train)
        local_s, target_train = make_one_step_pairs(local, y_train)
        global_s, _ = make_one_step_pairs(global_latent, y_train)
        pred = head(local_s, global_s)
        loss = nn.functional.l1_loss(pred, target_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        _, local_train, global_train = body(x_train)
        _, local_valid, global_valid = body(x_valid)
        _, local_test, global_test = body(x_test)

    return LatentSourceArtifacts(
        latents=LatentSplits(
            local_train=local_train,
            global_train=global_train,
            local_valid=local_valid,
            global_valid=global_valid,
            local_test=local_test,
            global_test=global_test,
        ),
        train_losses=losses,
    )


def train_stage1_recovery_latents(
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    x_test: torch.Tensor,
    *,
    feature_dim: int,
    window_size: int,
    steps: int = 100,
    lr: float = 1e-3,
) -> LatentSourceArtifacts:
    """Train Scenario4 Stage1-like recovery model and extract frozen latents."""
    recovery = DecouplingAutoEncoder(DecouplingConfig(feature_dim=feature_dim, window_size=window_size))
    optimizer = torch.optim.Adam(recovery.parameters(), lr=lr)
    losses: list[float] = []

    for _ in range(steps):
        rec, _, _ = recovery(x_train)
        loss = nn.functional.mse_loss(rec, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        _, local_train, global_train = recovery(x_train)
        _, local_valid, global_valid = recovery(x_valid)
        _, local_test, global_test = recovery(x_test)

    return LatentSourceArtifacts(
        latents=LatentSplits(
            local_train=local_train,
            global_train=global_train,
            local_valid=local_valid,
            global_valid=global_valid,
            local_test=local_test,
            global_test=global_test,
        ),
        train_losses=losses,
    )


def train_ablation_head(
    local_train: torch.Tensor,
    global_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    mode: AblationMode,
    steps: int = 100,
    lr: float = 1e-3,
) -> tuple[ForecastHead, list[float]]:
    local_train_s, y_train_s = make_one_step_pairs(local_train, y_train)
    global_train_s, _ = make_one_step_pairs(global_train, y_train)

    if mode == "local_only":
        local_dim, global_dim = local_train_s.shape[-1], 0
    elif mode == "global_only":
        local_dim, global_dim = 0, global_train_s.shape[-1]
    elif mode == "both":
        local_dim, global_dim = local_train_s.shape[-1], global_train_s.shape[-1]
    else:
        raise ValueError(f"Unsupported ablation mode: {mode}")

    head = ForecastHead(local_dim=local_dim, global_dim=global_dim, horizon=1)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        local_in, global_in = _select_inputs(local_train_s, global_train_s, mode=mode)
        pred = head(local_in, global_in)
        loss = nn.functional.l1_loss(pred, y_train_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return head, losses


def select_eval_inputs(
    local_split: torch.Tensor,
    global_split: torch.Tensor,
    y_split: torch.Tensor,
    *,
    mode: AblationMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_s, y_s = make_one_step_pairs(local_split, y_split)
    global_s, _ = make_one_step_pairs(global_split, y_split)
    local_in, global_in = _select_inputs(local_s, global_s, mode=mode)
    return local_in, global_in, y_s


def _select_inputs(local_s: torch.Tensor, global_s: torch.Tensor, *, mode: AblationMode) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "local_only":
        return local_s, torch.zeros(local_s.shape[0], 0, device=local_s.device, dtype=local_s.dtype)
    if mode == "global_only":
        return torch.zeros(global_s.shape[0], 0, device=global_s.device, dtype=global_s.dtype), global_s
    if mode == "both":
        return local_s, global_s
    raise ValueError(f"Unsupported ablation mode: {mode}")
