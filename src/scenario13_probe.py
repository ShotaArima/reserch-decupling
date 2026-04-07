from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import build_window_tensor, split_train_valid_test
from src.scenario9_pipeline import DatasetSplits, TwoBranchForecaster

SplitName = Literal["train", "valid", "test"]
TaskType = Literal["binary", "multiclass", "regression"]


@dataclass
class ProbeTask:
    name: str
    group: Literal["common", "specific"]
    task_type: TaskType


@dataclass
class ProbeConfig:
    steps: int = 120
    lr: float = 1e-2
    l2: float = 1e-4
    log_interval: int = 20


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _make_one_step_pairs(common_x: np.ndarray, specific_x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return common_x[:-1], specific_x[:-1], y[1:]


def extract_latents(model: TwoBranchForecaster, splits: DatasetSplits, split: SplitName) -> dict[str, np.ndarray]:
    if split == "train":
        common_x, specific_x, _ = _make_one_step_pairs(splits.common_train, splits.specific_train, splits.y_train)
    elif split == "valid":
        common_x, specific_x, _ = _make_one_step_pairs(splits.common_valid, splits.specific_valid, splits.y_valid)
    else:
        common_x, specific_x, _ = _make_one_step_pairs(splits.common_test, splits.specific_test, splits.y_test)

    common_t = torch.tensor(_flatten_windows(common_x), dtype=torch.float32)
    specific_t = torch.tensor(_flatten_windows(specific_x), dtype=torch.float32)

    with torch.no_grad():
        _, z_common, z_specific = model(common_t, specific_t)

    return {
        "z_common": z_common.numpy(),
        "z_specific": z_specific.numpy(),
    }


def _last_step_feature(df: pd.DataFrame, col: str, window_size: int) -> np.ndarray:
    tensor = build_window_tensor(df, [col], window_size=window_size)
    return tensor[:, -1, 0]


def _split_by_name(arr: np.ndarray, split: SplitName) -> np.ndarray:
    train_x, valid_x, test_x = split_train_valid_test(arr)
    if split == "train":
        return train_x
    if split == "valid":
        return valid_x
    return test_x


def build_probe_labels(df: pd.DataFrame, split: SplitName, window_size: int = 14) -> dict[str, np.ndarray]:
    sale = _last_step_feature(df, "sale_amount", window_size)
    sale_train = _split_by_name(sale, "train")
    sale_split = _split_by_name(sale, split)

    train_diff = np.abs(sale_train[1:] - sale_train[:-1])
    volatility_thr = np.quantile(train_diff, 0.75)

    out: dict[str, np.ndarray] = {}

    for col in ("city_id", "store_id", "first_category_id", "holiday_flag", "discount", "activity_flag"):
        values = _last_step_feature(df, col, window_size)
        split_values = _split_by_name(values, split)
        out[col] = split_values[:-1]

    diff = sale_split[1:] - sale_split[:-1]
    out["next_direction"] = (diff > 0).astype(np.int64)
    out["recent_volatility_high"] = (np.abs(diff) >= volatility_thr).astype(np.int64)

    return out


def _encode_classes(y: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    y_int = y.astype(np.int64)
    classes = np.unique(y_int)
    mapping = {int(c): i for i, c in enumerate(classes.tolist())}
    encoded = np.array([mapping[int(v)] for v in y_int], dtype=np.int64)
    return encoded, mapping


def _encode_with_mapping(y: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    y_int = y.astype(np.int64)
    out = np.full(y_int.shape[0], -1, dtype=np.int64)
    for idx, value in enumerate(y_int.tolist()):
        if int(value) in mapping:
            out[idx] = mapping[int(value)]
    return out


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score)) + 1
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s: list[float] = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(float(f1))
    return float(np.mean(f1s))


def _train_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    task_type: TaskType,
    cfg: ProbeConfig,
    task_name: str,
    latent_name: str,
) -> dict[str, float]:
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_eval_t = torch.tensor(x_eval, dtype=torch.float32)

    if task_type == "regression":
        model = nn.Linear(x_train.shape[1], 1)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        y_eval_np = y_eval.astype(np.float32)
        loss_fn = nn.MSELoss()
    else:
        y_train_encoded, mapping = _encode_classes(y_train)
        y_eval_encoded = _encode_with_mapping(y_eval, mapping)
        num_classes = len(mapping)
        y_train_t = torch.tensor(y_train_encoded, dtype=torch.long)
        y_eval_np = y_eval_encoded.astype(np.int64)

        if task_type == "binary":
            model = nn.Linear(x_train.shape[1], 2)
        else:
            model = nn.Linear(x_train.shape[1], num_classes)
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)

    for step in range(1, cfg.steps + 1):
        logits = model(x_train_t)
        loss = loss_fn(logits, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % cfg.log_interval == 0 or step == cfg.steps:
            print(
                f"[probe] task={task_name} latent={latent_name} "
                f"step={step}/{cfg.steps} loss={loss.item():.6f}"
            )

    with torch.no_grad():
        pred_eval = model(x_eval_t).numpy()

    if task_type == "regression":
        y_pred = pred_eval.reshape(-1)
        mae = float(np.mean(np.abs(y_eval_np - y_pred)))
        var = float(np.var(y_eval_np))
        r2 = 1.0 - float(np.mean((y_eval_np - y_pred) ** 2)) / (var + 1e-12)
        return {"mae": mae, "r2": r2}

    valid_mask = y_eval_np >= 0
    if not np.any(valid_mask):
        out_nan = {"accuracy": float("nan"), "macro_f1": float("nan")}
        if task_type == "binary":
            out_nan["roc_auc"] = float("nan")
        return out_nan

    y_pred = pred_eval.argmax(axis=1)[valid_mask]
    y_eval_valid = y_eval_np[valid_mask]
    acc = float(np.mean(y_pred == y_eval_valid))
    macro_f1 = _macro_f1(y_eval_valid, y_pred)

    out = {"accuracy": acc, "macro_f1": macro_f1}
    if task_type == "binary":
        probs = np.exp(pred_eval - pred_eval.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        out["roc_auc"] = _binary_auc(y_eval_valid, probs[valid_mask, 1])
    return out


def run_probes(
    train_latents: dict[str, np.ndarray],
    eval_latents: dict[str, np.ndarray],
    train_labels: dict[str, np.ndarray],
    eval_labels: dict[str, np.ndarray],
    tasks: list[ProbeTask],
    probe_config: ProbeConfig,
    experiment_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for task in tasks:
        y_train = train_labels[task.name]
        y_eval = eval_labels[task.name]

        for latent_name in ("z_common", "z_specific"):
            scores = _train_linear_probe(
                train_latents[latent_name],
                y_train,
                eval_latents[latent_name],
                y_eval,
                task.task_type,
                probe_config,
                task_name=task.name,
                latent_name=latent_name,
            )
            for metric, score in scores.items():
                rows.append(
                    {
                        "experiment": experiment_name,
                        "task": task.name,
                        "task_group": task.group,
                        "latent": latent_name,
                        "metric": metric,
                        "score": float(score),
                    }
                )

    return pd.DataFrame(rows)


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_c = _center(x)
    y_c = _center(y)
    hsic_xy = np.linalg.norm(x_c.T @ y_c, ord="fro") ** 2
    hsic_xx = np.linalg.norm(x_c.T @ x_c, ord="fro") ** 2
    hsic_yy = np.linalg.norm(y_c.T @ y_c, ord="fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def compute_latent_similarity(z_common: np.ndarray, z_specific: np.ndarray) -> dict[str, float]:
    cosine = np.sum(z_common * z_specific, axis=1) / (
        np.linalg.norm(z_common, axis=1) * np.linalg.norm(z_specific, axis=1) + 1e-12
    )
    return {
        "cka": _linear_cka(z_common, z_specific),
        "cosine_mean": float(np.mean(cosine)),
    }


def summarize_role_separation(probe_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        probe_df[probe_df["metric"] == "macro_f1"]
        .pivot_table(index=["experiment", "task", "task_group"], columns="latent", values="score")
        .reset_index()
    )
    pivot["gap_common_minus_specific"] = pivot["z_common"] - pivot["z_specific"]

    rows: list[dict[str, float | str]] = []
    for exp, group_df in pivot.groupby("experiment"):
        common_df = group_df[group_df["task_group"] == "common"]
        specific_df = group_df[group_df["task_group"] == "specific"]

        rsi_common = float(common_df["gap_common_minus_specific"].mean()) if not common_df.empty else float("nan")
        rsi_specific = float((-specific_df["gap_common_minus_specific"]).mean()) if not specific_df.empty else float("nan")
        rsi_total = float(np.nanmean([rsi_common, rsi_specific]))

        rows.append(
            {
                "experiment": exp,
                "rsi_common": rsi_common,
                "rsi_specific": rsi_specific,
                "rsi_total": rsi_total,
            }
        )

    return pd.DataFrame(rows)


def save_probe_heatmap(probe_df: pd.DataFrame, out_path: str | Path) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_df = probe_df[probe_df["metric"] == "macro_f1"].copy()
    metric_df["key"] = metric_df["experiment"] + "|" + metric_df["latent"]
    table = metric_df.pivot_table(index="task", columns="key", values="score")

    fig, ax = plt.subplots(figsize=(9, 4 + 0.35 * len(table.index)))
    im = ax.imshow(table.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title("Scenario13 Probe Macro-F1 Heatmap")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, label="Macro-F1")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_similarity_bar(sim_df: pd.DataFrame, rsi_df: pd.DataFrame, out_path: str | Path) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    merged = sim_df.merge(rsi_df, on="experiment", how="inner")
    labels = merged["experiment"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width, merged["cka"], width=width, label="CKA")
    ax.bar(x, merged["cosine_mean"], width=width, label="Cosine mean")
    ax.bar(x + width, merged["rsi_total"], width=width, label="RSI total")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Scenario13 Similarity / RSI Summary")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
