"""Causal technical-factor construction and cross-sectional normalization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from catt_rl.data.panel import MarketPanel
from catt_rl.utils import sha256_file

RAW_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}


def read_raw_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame]
    missing = RAW_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Raw OHLCV file is missing columns: {sorted(missing)}")
    if "adj_close" not in frame:
        frame["adj_close"] = frame["close"]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    numeric = ["open", "high", "low", "close", "adj_close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    return frame


def _pivot(
    frame: pd.DataFrame, field: str, dates: pd.DatetimeIndex, tickers: list[str]
) -> pd.DataFrame:
    result = frame.pivot(index="date", columns="ticker", values=field)
    return result.reindex(index=dates, columns=tickers).astype(float)


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return numerator.divide(denominator.replace(0.0, np.nan))


def _true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    previous = close.shift(1)
    values = np.maximum.reduce(
        [
            (high - low).to_numpy(),
            (high - previous).abs().to_numpy(),
            (low - previous).abs().to_numpy(),
        ]
    )
    return pd.DataFrame(values, index=close.index, columns=close.columns)


def _technical_features(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    epsilon = 1e-12
    returns = close.pct_change(fill_method=None)
    log_returns = np.log(close).diff()
    ma20 = close.rolling(20, min_periods=20).mean()
    std20_price = close.rolling(20, min_periods=20).std(ddof=0)
    upper = ma20 + 2.0 * std20_price
    lower = ma20 - 2.0 * std20_price

    delta = close.diff()
    average_gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rsi = 100.0 - 100.0 / (1.0 + average_gain / (average_loss + epsilon))

    typical = (high + low + close) / 3.0
    typical_mean = typical.rolling(20, min_periods=20).mean()
    mean_deviation = typical.rolling(20, min_periods=20).apply(
        lambda values: np.mean(np.abs(values - values.mean())), raw=True
    )
    cci = (typical - typical_mean) / (0.015 * mean_deviation + epsilon)

    tr = _true_range(high, low, close)
    atr = tr.rolling(14, min_periods=14).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    plus_di = (
        100.0
        * plus_dm.rolling(14, min_periods=14).sum()
        / (tr.rolling(14, min_periods=14).sum() + epsilon)
    )
    minus_di = (
        100.0
        * minus_dm.rolling(14, min_periods=14).sum()
        / (tr.rolling(14, min_periods=14).sum() + epsilon)
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + epsilon)
    adx = dx.rolling(14, min_periods=14).mean()

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_raw = ema12 - ema26
    macd_signal_raw = macd_raw.ewm(span=9, adjust=False, min_periods=9).mean()

    raw_money = typical * volume.clip(lower=0.0)
    direction = typical.diff()
    positive_money = raw_money.where(direction > 0.0, 0.0).rolling(14, min_periods=14).sum()
    negative_money = raw_money.where(direction < 0.0, 0.0).rolling(14, min_periods=14).sum()
    money_ratio = positive_money / (negative_money + epsilon)
    mfi = 100.0 - 100.0 / (1.0 + money_ratio)

    dollar_volume = close * volume.clip(lower=0.0)
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)

    return {
        "adj_close": np.log(close),
        "return_1": returns,
        "log_return_1": log_returns,
        "close_to_ma20": _safe_divide(close, ma20) - 1.0,
        "bollinger_mid": _safe_divide(ma20, close) - 1.0,
        "bollinger_upper": _safe_divide(upper, close) - 1.0,
        "bollinger_lower": _safe_divide(lower, close) - 1.0,
        "bollinger_width": _safe_divide(upper - lower, ma20),
        "cci_20": cci,
        "rsi_14": rsi / 100.0,
        "true_range": _safe_divide(tr, close),
        "atr_14": _safe_divide(atr, close),
        "plus_di_14": plus_di / 100.0,
        "minus_di_14": minus_di / 100.0,
        "adx_14": adx / 100.0,
        "macd": _safe_divide(macd_raw, close),
        "macd_signal": _safe_divide(macd_signal_raw, close),
        "macd_hist": _safe_divide(macd_raw - macd_signal_raw, close),
        "mfi_14": mfi / 100.0,
        "momentum_20": close.pct_change(20, fill_method=None),
        "momentum_63": close.pct_change(63, fill_method=None),
        "momentum_126": close.pct_change(126, fill_method=None),
        "high_low_range": _safe_divide(high - low, close),
        "volatility_20": returns.rolling(20, min_periods=20).std(ddof=0),
        "volatility_63": returns.rolling(63, min_periods=63).std(ddof=0),
        "volume_z_20": (volume - volume_mean) / (volume_std + epsilon),
        "dollar_volume_log": np.log1p(dollar_volume),
    }


def _load_external_factors(
    path: str | Path,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    selected_columns: Iterable[str] | None,
) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame]
    if not {"date", "ticker"}.issubset(frame.columns):
        raise ValueError("External factors require date and ticker columns")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    available = [column for column in frame if column not in {"date", "ticker"}]
    columns = list(selected_columns) if selected_columns is not None else available
    unknown = set(columns) - set(available)
    if unknown:
        raise ValueError(f"Unknown external factor columns: {sorted(unknown)}")
    factors: dict[str, pd.DataFrame] = {}
    for column in columns:
        values = frame.pivot(index="date", columns="ticker", values=column)
        factors[f"alpha158_{column}"] = values.reindex(index=dates, columns=tickers)
    return factors


def cross_sectional_normalize(
    raw: np.ndarray,
    tradable: np.ndarray,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> np.ndarray:
    """Winsorize and z-score using only assets available on the same date."""

    output = np.zeros_like(raw, dtype=np.float32)
    for date_index in range(raw.shape[0]):
        available = tradable[date_index]
        for feature_index in range(raw.shape[2]):
            values = raw[date_index, :, feature_index]
            valid = available & np.isfinite(values)
            if not np.any(valid):
                continue
            selected = values[valid]
            lower, upper = np.quantile(selected, [lower_quantile, upper_quantile])
            selected = np.clip(selected, lower, upper)
            mean = selected.mean()
            std = selected.std(ddof=0)
            output[date_index, valid, feature_index] = (
                (selected - mean) / std if std > 1e-8 else selected - mean
            )
    return output


def build_panel(
    raw: pd.DataFrame | str | Path,
    availability: float = 0.80,
    feature_set: str = "full",
    warmup: int = 126,
    availability_start: str | None = None,
    availability_end: str | None = None,
    external_factors: str | Path | None = None,
    selected_external_columns: Iterable[str] | None = None,
) -> MarketPanel:
    frame = read_raw_ohlcv(raw) if isinstance(raw, (str, Path)) else raw.copy()
    if not 0.0 <= availability <= 1.0:
        raise ValueError("availability must be in [0, 1]")
    ticker_order = list(dict.fromkeys(frame["ticker"].astype(str).tolist()))
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    close_raw = _pivot(frame, "close", dates, ticker_order)
    adjusted_raw = _pivot(frame, "adj_close", dates, ticker_order).combine_first(close_raw)

    availability_view = adjusted_raw
    if availability_start is not None:
        availability_view = availability_view.loc[pd.Timestamp(availability_start) :]
    if availability_end is not None:
        availability_view = availability_view.loc[: pd.Timestamp(availability_end)]
    ratios = availability_view.notna().mean(axis=0)
    tickers = [ticker for ticker in ticker_order if ratios.get(ticker, 0.0) >= availability]
    if not tickers:
        raise ValueError("No assets satisfy the requested availability threshold")

    close_raw = close_raw[tickers]
    adjusted_raw = adjusted_raw[tickers]
    original_tradable = adjusted_raw.notna().to_numpy()
    scale = adjusted_raw / close_raw.replace(0.0, np.nan)
    open_adjusted = _pivot(frame, "open", dates, tickers) * scale
    high_adjusted = _pivot(frame, "high", dates, tickers) * scale
    low_adjusted = _pivot(frame, "low", dates, tickers) * scale
    volume = _pivot(frame, "volume", dates, tickers)

    close = adjusted_raw.ffill()
    open_adjusted = open_adjusted.ffill().combine_first(close)
    high_adjusted = high_adjusted.ffill().combine_first(close)
    low_adjusted = low_adjusted.ffill().combine_first(close)
    volume = volume.fillna(0.0)
    # Pre-listing prices are neutral placeholders and remain non-tradable.
    close = close.fillna(1.0).clip(lower=np.finfo(float).tiny)
    open_adjusted = open_adjusted.fillna(close)
    high_adjusted = high_adjusted.fillna(close)
    low_adjusted = low_adjusted.fillna(close)

    factor_frames = _technical_features(close, high_adjusted, low_adjusted, volume)
    if external_factors is not None:
        factor_frames.update(
            _load_external_factors(external_factors, dates, tickers, selected_external_columns)
        )
    feature_names = tuple(factor_frames)
    raw_features = np.stack(
        [factor_frames[name].to_numpy(dtype=np.float64) for name in feature_names], axis=-1
    )
    normalized = cross_sectional_normalize(raw_features, original_tradable)

    start_index = max(int(warmup), 0)
    if len(dates) - start_index < 3:
        raise ValueError(
            f"Only {len(dates)} dates are available; warmup={start_index} leaves fewer than 3"
        )
    panel = MarketPanel(
        dates=dates.values.astype("datetime64[D]")[start_index:],
        tickers=np.asarray(tickers),
        prices=close.to_numpy(dtype=np.float32)[start_index:],
        features=normalized[start_index:],
        tradable=original_tradable[start_index:],
        feature_names=feature_names,
        metadata={
            "source": "raw_ohlcv",
            "availability_threshold": availability,
            "availability_start": availability_start,
            "availability_end": availability_end,
            "warmup_rows_removed": start_index,
            "winsorization": [0.01, 0.99],
            "normalization": "daily_cross_sectional_zscore",
            "external_factors": str(external_factors) if external_factors else None,
        },
    )
    return panel.select_features(feature_set)


def prepare_panel_file(
    raw_path: str | Path,
    output_path: str | Path,
    **kwargs: object,
) -> Path:
    panel = build_panel(raw_path, **kwargs)
    return panel.save(
        output_path,
        metadata={"raw_file": str(raw_path), "raw_sha256": sha256_file(raw_path)},
    )
