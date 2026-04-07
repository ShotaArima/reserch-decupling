from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import build_window_tensor, split_train_valid_test
from src.scenario9_pipeline import (
    COMMON_FEATURE_CANDIDATES,
    SPECIFIC_FEATURE_CANDIDATES,
    STOCK_FEATURE_CANDIDATES,
    TrainConfig,
    TwoBranchForecaster,
    add_dt_features,
    resolve_features,
    train_model,
)

SplitName = Literal["train", "valid", "test"]


@dataclass
class ProbeConfig:
    steps: int = 120
    lr: float = 5e-3
    batch_size: int = 2048
    log_interval: int = 20
    hidden_dim: int = 64
    latent_dim: int = 16
    seed: int = 42
    volatility_window: int = 5


@dataclass
class ProbeTask:
    name: str
    group: Literal["common", "specific"]
    labels: dict[SplitName, np.ndarray]
    n_classes: int


@dataclass
class SplitLatents:
    z_common: np.ndarray
    z_specific: np.ndarray


@dataclass
class ProbeResult:
    seed: int
    task: str
    group: str
    input_type: str
    metric: str
    value: float


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _make_one_step_pairs(x: np.ndarray) -> np.ndarray:
    return x[:-1]


def _shift_target(y: np.ndarray) -> np.ndarray:
    return y[1:]


def _last_scalar(df: pd.DataFrame, col: str, window_size: int) -> np.ndarray:
    t = build_window_tensor(df, [col], window_size=window_size)
    return t[:, -1, 0]


def _split(arr: np.ndarray) -> dict[SplitName, np.ndarray]:
    train, valid, test = split_train_valid_test(arr)
    return {"train": train, "valid": valid, "test": test}


def _normalize_train(train_x: np.ndarray, valid_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mu) / sigma, (valid_x - mu) / sigma, (test_x - mu) / sigma


def _to_int_labels(train: np.ndarray, valid: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    all_values = np.concatenate([train, valid, test])
    uniques = np.unique(all_values)
    mapper = {v: i for i, v in enumerate(uniques.tolist())}

    def _encode(a: np.ndarray) -> np.ndarray:
        return np.array([mapper[v] for v in a], dtype=np.int64)

    return _encode(train), _encode(valid), _encode(test), len(uniques)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s: list[float] = []
    for c in range(n_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1s.append((2.0 * precision * recall) / (precision + recall + 1e-12))
    return float(np.mean(f1s))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _average_precision_binary(y_true: np.ndarray, score_pos: np.ndarray) -> float:
    order = np.argsort(-score_pos)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.0
    ap = np.sum(precision[y_sorted == 1]) / positives
    return float(ap)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_and_eval_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
    cfg: ProbeConfig,
) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = LinearProbe(in_dim=x_train.shape[1], n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    print(
        f"[probe-train] n_train={x_train.shape[0]} n_valid={x_valid.shape[0]} n_test={x_test.shape[0]} "
        f"in_dim={x_train.shape[1]} n_classes={n_classes} steps={cfg.steps}"
    )

    for step in range(1, cfg.steps + 1):
        batch_idx = rng.integers(0, x_train.shape[0], size=min(cfg.batch_size, x_train.shape[0]))
        xb = x_train_t[batch_idx]
        yb = y_train_t[batch_idx]

        logits = model(xb)
        loss = loss_fn(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % cfg.log_interval == 0 or step == cfg.steps:
            print(f"[probe-train] step={step}/{cfg.steps} loss={loss.item():.6f}")

    with torch.no_grad():
        valid_logits = model(torch.tensor(x_valid, dtype=torch.float32)).numpy()
        test_logits = model(torch.tensor(x_test, dtype=torch.float32)).numpy()

    valid_pred = valid_logits.argmax(axis=1)
    test_pred = test_logits.argmax(axis=1)

    metrics = {
        "valid_macro_f1": _macro_f1(y_valid, valid_pred, n_classes),
        "valid_accuracy": _accuracy(y_valid, valid_pred),
        "test_macro_f1": _macro_f1(y_test, test_pred, n_classes),
        "test_accuracy": _accuracy(y_test, test_pred),
    }

    if n_classes == 2:
        valid_score_pos = torch.softmax(torch.tensor(valid_logits), dim=1).numpy()[:, 1]
        test_score_pos = torch.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
        metrics["valid_pr_auc"] = _average_precision_binary(y_valid, valid_score_pos)
        metrics["test_pr_auc"] = _average_precision_binary(y_test, test_score_pos)

    return metrics


def build_latents_and_tasks(df: pd.DataFrame, cfg: ProbeConfig, window_size: int) -> tuple[dict[SplitName, SplitLatents], list[ProbeTask], pd.DataFrame, pd.DataFrame]:
    df = add_dt_features(df)

    common_features, _ = resolve_features(df, COMMON_FEATURE_CANDIDATES)
    specific_candidates = tuple(dict.fromkeys(SPECIFIC_FEATURE_CANDIDATES + STOCK_FEATURE_CANDIDATES))
    specific_features, _ = resolve_features(df, specific_candidates)

    if not common_features:
        raise RuntimeError("No common features were found.")
    if not specific_features:
        raise RuntimeError("No specific features were found.")

    print(f"[scenario11] common_features={common_features}")
    print(f"[scenario11] specific_features={specific_features}")

    common_all = build_window_tensor(df, common_features, window_size=window_size)
    specific_all = build_window_tensor(df, specific_features, window_size=window_size)
    y_all = _last_scalar(df, "sale_amount", window_size)

    common_train, common_valid, common_test = split_train_valid_test(common_all)
    specific_train, specific_valid, specific_test = split_train_valid_test(specific_all)
    y_train, y_valid, y_test = split_train_valid_test(y_all)

    # Train a decoupled forecaster to obtain z_common / z_specific.
    from src.scenario9_pipeline import DatasetSplits

    splits = DatasetSplits(
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

    model, _ = train_model(
        splits,
        config=TrainConfig(
            steps=cfg.steps,
            lr=1e-3,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            seed=cfg.seed,
            log_interval=cfg.log_interval,
        ),
        experiment_name="scenario11_latent_extractor",
    )

    def _extract(common_split: np.ndarray, specific_split: np.ndarray) -> SplitLatents:
        c = torch.tensor(_flatten_windows(_make_one_step_pairs(common_split)), dtype=torch.float32)
        s = torch.tensor(_flatten_windows(_make_one_step_pairs(specific_split)), dtype=torch.float32)
        with torch.no_grad():
            _, zc, zs = model(c, s)
        return SplitLatents(z_common=zc.numpy(), z_specific=zs.numpy())

    latent_by_split = {
        "train": _extract(common_train, specific_train),
        "valid": _extract(common_valid, specific_valid),
        "test": _extract(common_test, specific_test),
    }

    # Build task labels on full split, then shift by +1 to align with one-step pair index.
    sale_splits = {"train": y_train, "valid": y_valid, "test": y_test}

    label_map_raw: dict[str, dict[SplitName, np.ndarray]] = {}

    for col in ["city_id", "store_id", "first_category_id", "product_id", "dt_weekday", "holiday_flag", "dt_month"]:
        if col in df.columns:
            col_values = _split(_last_scalar(df, col, window_size))
            label_map_raw[col] = col_values

    # long-term sales band (train quantiles only)
    q = np.quantile(sale_splits["train"], [0.25, 0.5, 0.75])
    for split_name, y in sale_splits.items():
        band = np.digitize(y, q, right=False)
        label_map_raw.setdefault("long_term_sales_band", {})[split_name] = band.astype(np.int64)

    # next delta sign (down/flat/up)
    eps = 1e-6
    for split_name, y in sale_splits.items():
        delta = np.diff(y, prepend=y[0])
        sign = np.where(delta > eps, 2, np.where(delta < -eps, 0, 1)).astype(np.int64)
        label_map_raw.setdefault("next_delta_sign", {})[split_name] = sign

    # recent volatility band
    for split_name, y in sale_splits.items():
        dif = np.abs(np.diff(y, prepend=y[0]))
        k = max(2, cfg.volatility_window)
        vol = np.convolve(dif, np.ones(k) / k, mode="same")
        if split_name == "train":
            vq = np.quantile(vol, [1 / 3, 2 / 3])
        band = np.digitize(vol, vq, right=False).astype(np.int64)
        label_map_raw.setdefault("recent_volatility_band", {})[split_name] = band

    if "discount" in df.columns:
        for split_name, x in _split(_last_scalar(df, "discount", window_size)).items():
            label_map_raw.setdefault("is_discount", {})[split_name] = (x > 0).astype(np.int64)

    if "activity_flag" in df.columns:
        for split_name, x in _split(_last_scalar(df, "activity_flag", window_size)).items():
            label_map_raw.setdefault("is_activity", {})[split_name] = (x > 0).astype(np.int64)

    stock_signal = None
    if "hours_stock_status" in df.columns:
        stock_signal = _last_scalar(df, "hours_stock_status", window_size)
    elif "stock_hour6_22_cnt" in df.columns:
        stock_signal = _last_scalar(df, "stock_hour6_22_cnt", window_size)
    if stock_signal is not None:
        for split_name, x in _split(stock_signal).items():
            label_map_raw.setdefault("is_stockout", {})[split_name] = (x <= 0).astype(np.int64)

    tasks: list[ProbeTask] = []
    task_group = {
        "city_id": "common",
        "store_id": "common",
        "first_category_id": "common",
        "product_id": "common",
        "dt_weekday": "common",
        "holiday_flag": "common",
        "dt_month": "common",
        "long_term_sales_band": "common",
        "next_delta_sign": "specific",
        "recent_volatility_band": "specific",
        "is_discount": "specific",
        "is_activity": "specific",
        "is_stockout": "specific",
    }

    label_records = []
    for task_name, split_labels in label_map_raw.items():
        train_raw = _shift_target(split_labels["train"])
        valid_raw = _shift_target(split_labels["valid"])
        test_raw = _shift_target(split_labels["test"])
        train_y, valid_y, test_y, n_classes = _to_int_labels(train_raw, valid_raw, test_raw)

        tasks.append(
            ProbeTask(
                name=task_name,
                group=task_group[task_name],
                labels={"train": train_y, "valid": valid_y, "test": test_y},
                n_classes=n_classes,
            )
        )

        for split_name, vals in [("train", train_y), ("valid", valid_y), ("test", test_y)]:
            for idx, label in enumerate(vals.tolist()):
                label_records.append((split_name, idx, task_name, int(label)))

    feat_records = []
    for split_name, latent in latent_by_split.items():
        for idx in range(latent.z_common.shape[0]):
            feat_records.append(
                {
                    "split": split_name,
                    "sample_index": idx,
                    "z_common": latent.z_common[idx].tolist(),
                    "z_specific": latent.z_specific[idx].tolist(),
                }
            )

    feature_df = pd.DataFrame(feat_records)
    labels_df = pd.DataFrame(label_records, columns=["split", "sample_index", "task", "label"])
    return latent_by_split, tasks, feature_df, labels_df


def run_probe_suite(latent_by_split: dict[SplitName, SplitLatents], tasks: list[ProbeTask], cfg: ProbeConfig, seeds: list[int]) -> pd.DataFrame:
    rows: list[ProbeResult] = []

    for seed in seeds:
        print(f"\n[scenario11] probe_seed={seed}")
        for task in tasks:
            print(f"[scenario11] task={task.name} group={task.group} n_classes={task.n_classes}")
            x_sources = {
                "common": (
                    latent_by_split["train"].z_common,
                    latent_by_split["valid"].z_common,
                    latent_by_split["test"].z_common,
                ),
                "specific": (
                    latent_by_split["train"].z_specific,
                    latent_by_split["valid"].z_specific,
                    latent_by_split["test"].z_specific,
                ),
                "concat": (
                    np.concatenate([latent_by_split["train"].z_common, latent_by_split["train"].z_specific], axis=1),
                    np.concatenate([latent_by_split["valid"].z_common, latent_by_split["valid"].z_specific], axis=1),
                    np.concatenate([latent_by_split["test"].z_common, latent_by_split["test"].z_specific], axis=1),
                ),
                "random": (
                    np.random.default_rng(seed).normal(size=latent_by_split["train"].z_common.shape).astype(np.float32),
                    np.random.default_rng(seed + 1).normal(size=latent_by_split["valid"].z_common.shape).astype(np.float32),
                    np.random.default_rng(seed + 2).normal(size=latent_by_split["test"].z_common.shape).astype(np.float32),
                ),
            }

            for input_type, (xtr, xva, xte) in x_sources.items():
                xtr, xva, xte = _normalize_train(xtr, xva, xte)
                result = train_and_eval_probe(
                    xtr,
                    task.labels["train"],
                    xva,
                    task.labels["valid"],
                    xte,
                    task.labels["test"],
                    n_classes=task.n_classes,
                    cfg=ProbeConfig(
                        steps=cfg.steps,
                        lr=cfg.lr,
                        batch_size=cfg.batch_size,
                        log_interval=cfg.log_interval,
                        seed=seed,
                    ),
                )
                for metric_name, value in result.items():
                    rows.append(
                        ProbeResult(
                            seed=seed,
                            task=task.name,
                            group=task.group,
                            input_type=input_type,
                            metric=metric_name,
                            value=float(value),
                        )
                    )

    return pd.DataFrame([r.__dict__ for r in rows])
