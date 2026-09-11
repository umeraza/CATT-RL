"""Plot compact attention diagnostics and learning curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_attention(
    input_path: str | Path,
    output_path: str | Path,
    key: str = "head_feature_projection",
) -> Path:
    input_path = Path(input_path)
    with np.load(input_path, allow_pickle=False) as archive:
        if key not in archive:
            available = ", ".join(sorted(archive.files))
            raise KeyError(f"Attention key {key!r} not found. Available: {available}")
        values = np.asarray(archive[key], dtype=float)
        if values.ndim > 2:
            values = values.mean(axis=tuple(range(values.ndim - 2)))
        if values.ndim == 1:
            values = values[None, :]
        feature_names = (
            archive["feature_names"].astype(str).tolist()
            if key == "head_feature_projection" and "feature_names" in archive
            else [str(index) for index in range(values.shape[1])]
        )
    width = max(7.0, min(18.0, 0.35 * values.shape[1]))
    height = max(3.0, 0.7 * values.shape[0] + 1.5)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xlabel("Feature / key position")
    axis.set_ylabel("Attention head")
    axis.set_xticks(np.arange(len(feature_names)))
    axis.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(np.arange(values.shape[0]))
    axis.set_yticklabels([f"H{index + 1}" for index in range(values.shape[0])])
    axis.set_title(key.replace("_", " ").title())
    figure.colorbar(image, ax=axis, shrink=0.8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_training(input_path: str | Path, output_path: str | Path) -> Path:
    records = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Training log is empty")
    updates = [record["update"] for record in records]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plots = [
        ("mean_rollout_reward", "Rollout reward"),
        ("validation_sharpe", "Validation Sharpe"),
        ("approx_kl", "Approximate KL"),
        ("mean_rollout_turnover", "Turnover"),
    ]
    for axis, (key, title) in zip(axes.flat, plots, strict=True):
        x = [update for update, record in zip(updates, records, strict=True) if key in record]
        y = [record[key] for record in records if key in record]
        axis.plot(x, y, linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("PPO update")
        axis.grid(alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path
