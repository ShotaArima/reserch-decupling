from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DecouplingConfig:
    input_dim: int
    global_dim: int = 16
    local_dim: int = 16
    hidden_dim: int = 64


class DecouplingAutoEncoder(nn.Module):
    """Minimal Decoupling-inspired model with global/local branches."""

    def __init__(self, config: DecouplingConfig) -> None:
        super().__init__()
        self.config = config
        self.local_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.local_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.global_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.local_dim + config.global_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.local_encoder(x)
        global_latent = self.global_encoder(x)
        rec = self.decoder(torch.cat([local, global_latent], dim=-1))
        return rec, local, global_latent


class ForecastHead(nn.Module):
    def __init__(self, local_dim: int, global_dim: int, horizon: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(local_dim + global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, local: torch.Tensor, global_latent: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([local, global_latent], dim=-1))
