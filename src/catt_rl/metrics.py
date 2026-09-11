"""Financial performance metrics and moving-block bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _returns(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def annual_to_periodic(rate: float, periods: int = 252) -> float:
    return float(np.expm1(np.log1p(rate) / periods))


def cumulative_return(returns: np.ndarray | list[float]) -> float:
    values = _returns(returns)
    if values.size == 0:
        return 0.0
    return float(np.prod(1.0 + values) - 1.0)


def sharpe_ratio(
    returns: np.ndarray | list[float], risk_free_rate: float = 0.03, annualization: int = 252
) -> float:
    values = _returns(returns)
    if values.size < 2:
        return 0.0
    excess = values - annual_to_periodic(risk_free_rate, annualization)
    scale = np.std(excess, ddof=1)
    return float(np.sqrt(annualization) * np.mean(excess) / scale) if scale > 0 else 0.0


def sortino_ratio(
    returns: np.ndarray | list[float], threshold: float = 0.03, annualization: int = 252
) -> float:
    values = _returns(returns)
    if values.size == 0:
        return 0.0
    target = annual_to_periodic(threshold, annualization)
    excess = values - target
    downside = np.minimum(excess, 0.0)
    deviation = np.sqrt(np.mean(np.square(downside)))
    return float(np.sqrt(annualization) * np.mean(excess) / deviation) if deviation > 0 else 0.0


def omega_ratio(
    returns: np.ndarray | list[float], threshold: float = 0.03, annualization: int = 252
) -> float:
    values = _returns(returns)
    if values.size == 0:
        return 0.0
    target = annual_to_periodic(threshold, annualization)
    gains = np.maximum(values - target, 0.0).sum()
    losses = np.maximum(target - values, 0.0).sum()
    return float(gains / losses) if losses > 0 else float("inf")


def maximum_drawdown(returns: np.ndarray | list[float]) -> float:
    values = _returns(returns)
    if values.size == 0:
        return 0.0
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / np.maximum(peaks, np.finfo(np.float64).tiny)
    return float(np.max(drawdowns))


def performance_metrics(
    returns: np.ndarray | list[float],
    turnovers: np.ndarray | list[float] | None = None,
    risk_free_rate: float = 0.03,
    threshold: float = 0.03,
    annualization: int = 252,
) -> dict[str, float]:
    values = _returns(returns)
    result = {
        "cumulative_return": cumulative_return(values),
        "sharpe": sharpe_ratio(values, risk_free_rate, annualization),
        "sortino": sortino_ratio(values, threshold, annualization),
        "omega": omega_ratio(values, threshold, annualization),
        "maximum_drawdown": maximum_drawdown(values),
        "mean_daily_return": float(values.mean()) if values.size else 0.0,
        "daily_volatility": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "observations": int(values.size),
    }
    if turnovers is not None:
        turnover_values = _returns(turnovers)
        result["average_turnover"] = float(turnover_values.mean()) if turnover_values.size else 0.0
    return result


def moving_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    block_length = max(1, min(int(block_length), length))
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length - block_length + 1, size=blocks)
    indices = np.concatenate([np.arange(start, start + block_length) for start in starts])
    return indices[:length]


def block_bootstrap_confidence_intervals(
    returns: np.ndarray | list[float],
    samples: int = 2000,
    block_length: int = 21,
    confidence: float = 0.95,
    seed: int = 7,
    risk_free_rate: float = 0.03,
    threshold: float = 0.03,
    annualization: int = 252,
) -> dict[str, dict[str, float]]:
    values = _returns(returns)
    if values.size < 2 or samples <= 0:
        return {}
    functions: dict[str, Callable[[np.ndarray], float]] = {
        "cumulative_return": cumulative_return,
        "sharpe": lambda x: sharpe_ratio(x, risk_free_rate, annualization),
        "sortino": lambda x: sortino_ratio(x, threshold, annualization),
        "omega": lambda x: omega_ratio(x, threshold, annualization),
        "maximum_drawdown": maximum_drawdown,
    }
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in functions}
    for _ in range(int(samples)):
        sample = values[moving_block_indices(values.size, block_length, rng)]
        for name, function in functions.items():
            value = function(sample)
            if np.isfinite(value):
                draws[name].append(value)
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float]] = {}
    for name, values_for_metric in draws.items():
        if not values_for_metric:
            continue
        result[name] = {
            "lower": float(np.quantile(values_for_metric, alpha)),
            "upper": float(np.quantile(values_for_metric, 1.0 - alpha)),
            "confidence": float(confidence),
        }
    return result


def sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Replace non-finite JSON values with null-equivalent `None`."""

    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            result[key] = sanitize_metrics(value)
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            result[key] = None
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result
