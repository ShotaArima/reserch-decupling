from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from src.data import build_window_tensor, normalize_by_train_stats, split_train_valid_test
from src.metrics import mae, wape, wpe
from src.scenario9_pipeline import AblationMode

ModelName = Literal[
    "p0_prophet",
    "p1_prophet_reg",
    "v0_flatten_vae",
    "v1_seq_vae",
    "v2_seq_vae_transition",
]


@dataclass
class Scenario18Splits:
    common_train: np.ndarray
    common_valid: np.ndarray
    common_test: np.ndarray
    specific_train: np.ndarray
    specific_valid: np.ndarray
    specific_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


@dataclass
class Scenario18TrainConfig:
    steps: int = 120
    lr: float = 1e-3
    hidden_dim: int = 64
    latent_dim: int = 16
    beta_kl: float = 1e-3
    transition_weight: float = 1e-2
    seed: int = 42
    log_interval: int = 20


@dataclass
class Scenario18Metrics:
    valid_wape: float
    valid_wpe: float
    valid_mae: float
    valid_rmse: float
    valid_smape: float
    test_wape: float
    test_wpe: float
    test_mae: float
    test_rmse: float
    test_smape: float


class FlattenVAEForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, horizon: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = _reparameterize(mu, logvar)
        pred = self.decoder(z)
        return pred, mu, logvar, z


class SequentialVAEForecaster(nn.Module):
    def __init__(
        self,
        common_dim: int,
        specific_dim: int,
        hidden_dim: int,
        latent_dim: int,
        horizon: int,
        with_transition: bool,
    ) -> None:
        super().__init__()
        self.with_transition = with_transition
        self.common_encoder = nn.GRU(common_dim, hidden_dim, batch_first=True)
        self.specific_encoder = nn.GRU(specific_dim, hidden_dim, batch_first=True)

        self.common_mu = nn.Linear(hidden_dim, latent_dim)
        self.common_logvar = nn.Linear(hidden_dim, latent_dim)
        self.specific_mu = nn.Linear(hidden_dim, latent_dim)
        self.specific_logvar = nn.Linear(hidden_dim, latent_dim)

        if with_transition:
            self.specific_transition = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh(),
                nn.Linear(latent_dim, latent_dim),
            )
        else:
            self.specific_transition = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def encode(self, common_x: torch.Tensor, specific_x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, h_common = self.common_encoder(common_x)
        _, h_specific = self.specific_encoder(specific_x)

        h_common_last = h_common[-1]
        h_specific_last = h_specific[-1]

        mu_common = self.common_mu(h_common_last)
        logvar_common = self.common_logvar(h_common_last)
        mu_specific = self.specific_mu(h_specific_last)
        logvar_specific = self.specific_logvar(h_specific_last)

        z_common = _reparameterize(mu_common, logvar_common)
        z_specific_raw = _reparameterize(mu_specific, logvar_specific)
        z_specific = self.specific_transition(z_specific_raw)

        return mu_common, logvar_common, mu_specific, logvar_specific, z_common, z_specific

    def forward(self, common_x: torch.Tensor, specific_x: torch.Tensor, mode: AblationMode) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        mu_common, logvar_common, mu_specific, logvar_specific, z_common, z_specific = self.encode(common_x, specific_x)
        z = _apply_mode(z_common, z_specific, mode=mode)
        pred = self.head(z)
        return pred, (mu_common, logvar_common, mu_specific, logvar_specific, z_common, z_specific)


class ProphetWindowForecaster:
    """Per-window Prophet baseline with optional pseudo-regressor.

    Notes:
    - Fits Prophet independently per sample window (slow, but simple and leakage-safe).
    - If prophet package is unavailable, falls back to a deterministic naive strategy.
    """

    def __init__(self, use_regressor: bool) -> None:
        self.use_regressor = use_regressor

    def predict(self, y_hist: np.ndarray, reg_hist: np.ndarray, reg_future: np.ndarray, horizon: int) -> np.ndarray:
        try:
            import importlib
            import pandas as pd

            Prophet = importlib.import_module("prophet").Prophet

            ds = pd.date_range("2021-01-01", periods=len(y_hist), freq="D")
            train_df = pd.DataFrame({"ds": ds, "y": y_hist.astype(float)})

            model = Prophet(
                weekly_seasonality=True,
                daily_seasonality=False,
                yearly_seasonality=False,
            )

            if self.use_regressor:
                model.add_regressor("reg")
                train_df["reg"] = reg_hist.astype(float)

            model.fit(train_df)
            fut = pd.DataFrame({"ds": pd.date_range(ds[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")})
            if self.use_regressor:
                fut["reg"] = reg_future.astype(float)

            fcst = model.predict(fut)
            return fcst["yhat"].to_numpy(dtype=np.float64)
        except Exception:
            if self.use_regressor:
                base = float(y_hist[-1])
                drift = float(reg_future.mean() - reg_hist.mean()) if reg_hist.size else 0.0
                return np.full(horizon, base + 0.1 * drift, dtype=np.float64)
            trend = float(y_hist[-1] - y_hist[-7]) / 7.0 if y_hist.shape[0] >= 7 else 0.0
            return np.asarray([float(y_hist[-1] + trend * (i + 1)) for i in range(horizon)], dtype=np.float64)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _apply_mode(z_common: torch.Tensor, z_specific: torch.Tensor, mode: AblationMode) -> torch.Tensor:
    if mode == "both":
        return torch.cat([z_common, z_specific], dim=-1)
    if mode == "common_only":
        return torch.cat([z_common, torch.zeros_like(z_specific)], dim=-1)
    if mode == "specific_only":
        return torch.cat([torch.zeros_like(z_common), z_specific], dim=-1)
    raise ValueError(f"Unsupported ablation mode: {mode}")


def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def make_multi_horizon_pairs(common_x: np.ndarray, specific_x: np.ndarray, y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if common_x.shape[0] <= horizon:
        raise ValueError("Not enough samples for multi-horizon pairs")

    n = common_x.shape[0] - horizon
    ys = np.stack([y[i + 1 : i + horizon + 1] for i in range(n)], axis=0)
    return common_x[:n], specific_x[:n], ys


def build_splits_for_scenario18(
    df,
    *,
    lookback: int,
    common_features: list[str],
    specific_features: list[str],
    target_feature: str = "sale_amount",
) -> Scenario18Splits:
    common_all = build_window_tensor(df, common_features, window_size=lookback)
    specific_all = build_window_tensor(df, specific_features, window_size=lookback)
    target_all = build_window_tensor(df, [target_feature], window_size=lookback)
    y_all = target_all[:, -1, 0]

    common_train_raw, common_valid_raw, common_test_raw = split_train_valid_test(common_all)
    specific_train_raw, specific_valid_raw, specific_test_raw = split_train_valid_test(specific_all)
    y_train, y_valid, y_test = split_train_valid_test(y_all)

    common_train, common_valid, common_test = normalize_by_train_stats(common_train_raw, common_valid_raw, common_test_raw)
    specific_train, specific_valid, specific_test = normalize_by_train_stats(
        specific_train_raw,
        specific_valid_raw,
        specific_test_raw,
    )

    return Scenario18Splits(
        common_train=common_train,
        common_valid=common_valid,
        common_test=common_test,
        specific_train=specific_train,
        specific_valid=specific_valid,
        specific_test=specific_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )


def train_vae_model(
    splits: Scenario18Splits,
    *,
    model_name: ModelName,
    horizon: int,
    mode: AblationMode,
    config: Scenario18TrainConfig,
) -> tuple[nn.Module, list[float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    common_train, specific_train, y_train = make_multi_horizon_pairs(
        splits.common_train,
        splits.specific_train,
        splits.y_train,
        horizon,
    )

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    losses: list[float] = []

    if model_name == "v0_flatten_vae":
        x_train = np.concatenate([common_train, specific_train], axis=-1).reshape(common_train.shape[0], -1)
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        model = FlattenVAEForecaster(
            input_dim=x_train_t.shape[1],
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            horizon=horizon,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = nn.L1Loss()

        for step in range(1, config.steps + 1):
            pred, mu, logvar, _ = model(x_train_t)
            rec_loss = loss_fn(pred, y_train_t)
            kl = _kl_standard_normal(mu, logvar)
            loss = rec_loss + config.beta_kl * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            if step == 1 or step % config.log_interval == 0 or step == config.steps:
                print(f"[train] {model_name}: step={step}/{config.steps} loss={loss.item():.6f} rec={rec_loss.item():.6f} kl={kl.item():.6f}")
        return model, losses

    with_transition = model_name == "v2_seq_vae_transition"
    model = SequentialVAEForecaster(
        common_dim=common_train.shape[2],
        specific_dim=specific_train.shape[2],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        horizon=horizon,
        with_transition=with_transition,
    )

    common_train_t = torch.tensor(common_train, dtype=torch.float32)
    specific_train_t = torch.tensor(specific_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.L1Loss()

    for step in range(1, config.steps + 1):
        pred, info = model(common_train_t, specific_train_t, mode=mode)
        mu_common, logvar_common, mu_specific, logvar_specific, z_common, z_specific = info

        rec_loss = loss_fn(pred, y_train_t)
        kl = _kl_standard_normal(mu_common, logvar_common) + _kl_standard_normal(mu_specific, logvar_specific)

        if with_transition and z_specific.shape[0] > 1:
            smooth = torch.mean(torch.abs(z_specific[1:] - z_specific[:-1]))
        else:
            smooth = torch.tensor(0.0)

        loss = rec_loss + config.beta_kl * kl + config.transition_weight * smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            print(
                f"[train] {model_name}: mode={mode} step={step}/{config.steps} "
                f"loss={loss.item():.6f} rec={rec_loss.item():.6f} kl={kl.item():.6f} smooth={smooth.item():.6f}"
            )

    return model, losses


def predict_vae(
    model: nn.Module,
    common_x: np.ndarray,
    specific_x: np.ndarray,
    *,
    model_name: ModelName,
    mode: AblationMode,
) -> np.ndarray:
    if model_name == "v0_flatten_vae":
        x = np.concatenate([common_x, specific_x], axis=-1).reshape(common_x.shape[0], -1)
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            pred, _, _, _ = model(x_t)  # type: ignore[misc]
        return pred.numpy()

    common_t = torch.tensor(common_x, dtype=torch.float32)
    specific_t = torch.tensor(specific_x, dtype=torch.float32)
    with torch.no_grad():
        pred, _ = model(common_t, specific_t, mode=mode)  # type: ignore[misc]
    return pred.numpy()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float, float]:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    return (
        wape(y_true_flat, y_pred_flat),
        wpe(y_true_flat, y_pred_flat),
        mae(y_true_flat, y_pred_flat),
        _rmse(y_true_flat, y_pred_flat),
        _smape(y_true_flat, y_pred_flat),
    )


def evaluate_vae_model(
    model: nn.Module,
    splits: Scenario18Splits,
    *,
    model_name: ModelName,
    horizon: int,
    mode: AblationMode,
) -> Scenario18Metrics:
    c_valid, s_valid, y_valid = make_multi_horizon_pairs(splits.common_valid, splits.specific_valid, splits.y_valid, horizon)
    c_test, s_test, y_test = make_multi_horizon_pairs(splits.common_test, splits.specific_test, splits.y_test, horizon)

    valid_pred = predict_vae(model, c_valid, s_valid, model_name=model_name, mode=mode)
    test_pred = predict_vae(model, c_test, s_test, model_name=model_name, mode=mode)

    valid_scores = evaluate_predictions(y_valid, valid_pred)
    test_scores = evaluate_predictions(y_test, test_pred)

    return Scenario18Metrics(
        valid_wape=valid_scores[0],
        valid_wpe=valid_scores[1],
        valid_mae=valid_scores[2],
        valid_rmse=valid_scores[3],
        valid_smape=valid_scores[4],
        test_wape=test_scores[0],
        test_wpe=test_scores[1],
        test_mae=test_scores[2],
        test_rmse=test_scores[3],
        test_smape=test_scores[4],
    )


def run_prophet_baseline(
    splits: Scenario18Splits,
    *,
    horizon: int,
    use_regressor: bool,
) -> Scenario18Metrics:
    forecaster = ProphetWindowForecaster(use_regressor=use_regressor)

    def _predict_for_split(common_x: np.ndarray, specific_x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s, yy = make_multi_horizon_pairs(common_x, specific_x, y, horizon)
        preds: list[np.ndarray] = []
        for idx in range(c.shape[0]):
            y_hist = s[idx, :, 0]
            reg_hist = c[idx, :, 0] if c.shape[2] > 0 else np.zeros_like(y_hist)
            reg_future = np.repeat(reg_hist[-1], horizon)
            pred = forecaster.predict(y_hist=y_hist, reg_hist=reg_hist, reg_future=reg_future, horizon=horizon)
            preds.append(pred)
        return yy, np.stack(preds, axis=0)

    y_valid, pred_valid = _predict_for_split(splits.common_valid, splits.specific_valid, splits.y_valid)
    y_test, pred_test = _predict_for_split(splits.common_test, splits.specific_test, splits.y_test)

    valid_scores = evaluate_predictions(y_valid, pred_valid)
    test_scores = evaluate_predictions(y_test, pred_test)

    return Scenario18Metrics(
        valid_wape=valid_scores[0],
        valid_wpe=valid_scores[1],
        valid_mae=valid_scores[2],
        valid_rmse=valid_scores[3],
        valid_smape=valid_scores[4],
        test_wape=test_scores[0],
        test_wpe=test_scores[1],
        test_mae=test_scores[2],
        test_rmse=test_scores[3],
        test_smape=test_scores[4],
    )
