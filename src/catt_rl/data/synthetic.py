"""Deterministic synthetic regime data for tests and smoke runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from catt_rl.data.panel import MarketPanel


def make_synthetic_panel(
    output: str | Path | None = None,
    assets: int = 4,
    days: int = 220,
    seed: int = 7,
    start: str = "2018-01-01",
) -> MarketPanel:
    if assets < 2 or days < 80:
        raise ValueError("Synthetic smoke data needs at least 2 assets and 80 days")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days).values.astype("datetime64[D]")
    market = rng.normal(0.00025, 0.007, size=days)
    regimes = np.where((np.arange(days) // 45) % 2 == 0, 1.0, -0.35)
    market = market * regimes
    beta = np.linspace(0.65, 1.25, assets)
    idiosyncratic = rng.normal(0.0, 0.009, size=(days, assets))
    returns = np.clip(market[:, None] * beta[None, :] + idiosyncratic, -0.18, 0.18)
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)

    frame = pd.DataFrame(returns)
    momentum_20 = frame.rolling(20, min_periods=1).sum().to_numpy()
    momentum_63 = frame.rolling(63, min_periods=1).sum().to_numpy()
    momentum_126 = frame.rolling(126, min_periods=1).sum().to_numpy()
    volatility_20 = frame.rolling(20, min_periods=2).std().fillna(0).to_numpy()
    volatility_63 = frame.rolling(63, min_periods=2).std().fillna(0).to_numpy()
    up = pd.DataFrame(np.maximum(returns, 0)).rolling(14, min_periods=1).mean()
    down = pd.DataFrame(np.maximum(-returns, 0)).rolling(14, min_periods=1).mean()
    rsi = (100.0 - 100.0 / (1.0 + up / (down + 1e-8))).to_numpy() / 100.0
    ema12 = frame.ewm(span=12, adjust=False).mean().to_numpy()
    ema26 = frame.ewm(span=26, adjust=False).mean().to_numpy()
    macd = ema12 - ema26
    ranges = np.abs(rng.normal(0.012, 0.004, size=(days, assets)))
    liquidity = np.log1p(rng.lognormal(15.0, 0.8, size=(days, assets)))
    close_level = np.log(prices)
    feature_names = (
        "adj_close",
        "return_1",
        "log_return_1",
        "close_to_ma20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "momentum_20",
        "momentum_63",
        "momentum_126",
        "true_range",
        "atr_14",
        "high_low_range",
        "volatility_20",
        "volatility_63",
        "dollar_volume_log",
    )
    features = np.stack(
        [
            close_level,
            returns,
            np.log1p(returns),
            momentum_20 / 20.0,
            rsi,
            macd,
            pd.DataFrame(macd).ewm(span=9, adjust=False).mean().to_numpy(),
            macd - pd.DataFrame(macd).ewm(span=9, adjust=False).mean().to_numpy(),
            momentum_20,
            momentum_63,
            momentum_126,
            ranges,
            pd.DataFrame(ranges).rolling(14, min_periods=1).mean().to_numpy(),
            ranges,
            volatility_20,
            volatility_63,
            liquidity,
        ],
        axis=-1,
    )
    # Daily cross-sectional normalization mirrors the real preprocessing path.
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    features = (features - mean) / np.where(std > 1e-6, std, 1.0)
    panel = MarketPanel(
        dates=dates,
        tickers=np.asarray([f"SYN{i:02d}" for i in range(assets)]),
        prices=prices.astype(np.float32),
        features=features.astype(np.float32),
        tradable=np.ones((days, assets), dtype=bool),
        feature_names=feature_names,
        metadata={"source": "synthetic", "seed": seed},
    )
    if output is not None:
        panel.save(output)
    return panel
