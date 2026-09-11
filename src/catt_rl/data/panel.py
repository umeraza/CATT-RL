"""Typed in-memory panel and compressed, pickle-free persistence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from catt_rl.utils import sha256_file, write_json

PRICE_FEATURES = {
    "adj_close",
    "return_1",
    "log_return_1",
    "close_to_ma20",
}
MOMENTUM_FEATURES = {
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "momentum_20",
    "momentum_63",
    "momentum_126",
}
VOLATILITY_FEATURES = {
    "bollinger_mid",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_width",
    "true_range",
    "atr_14",
    "plus_di_14",
    "minus_di_14",
    "adx_14",
    "high_low_range",
    "volatility_20",
    "volatility_63",
}


@dataclass(frozen=True)
class MarketPanel:
    dates: np.ndarray
    tickers: np.ndarray
    prices: np.ndarray
    features: np.ndarray
    tradable: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        dates = np.asarray(self.dates).astype("datetime64[D]")
        tickers = np.asarray(self.tickers).astype(str)
        prices = np.asarray(self.prices, dtype=np.float32)
        features = np.asarray(self.features, dtype=np.float32)
        tradable = np.asarray(self.tradable, dtype=bool)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "tickers", tickers)
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "tradable", tradable)
        object.__setattr__(self, "feature_names", tuple(map(str, self.feature_names)))
        self.validate()

    @property
    def num_dates(self) -> int:
        return int(self.dates.shape[0])

    @property
    def num_assets(self) -> int:
        return int(self.tickers.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[2])

    def validate(self) -> None:
        if self.prices.shape != (self.num_dates, self.num_assets):
            raise ValueError("prices must have shape [date, asset]")
        if self.tradable.shape != self.prices.shape:
            raise ValueError("tradable must match prices")
        if self.features.shape[:2] != self.prices.shape:
            raise ValueError("features must have shape [date, asset, feature]")
        if self.features.shape[2] != len(self.feature_names):
            raise ValueError("feature_names length does not match the feature axis")
        if self.num_dates < 3 or self.num_assets < 1 or self.num_features < 1:
            raise ValueError("panel is too small")
        if np.any(np.diff(self.dates.astype(np.int64)) <= 0):
            raise ValueError("dates must be strictly increasing")
        if len(set(self.tickers.tolist())) != self.num_assets:
            raise ValueError("tickers must be unique")
        if not np.isfinite(self.features).all():
            raise ValueError("features contain non-finite values")
        if not np.isfinite(self.prices).all() or np.any(self.prices <= 0):
            raise ValueError("prices must be finite and positive")

    def feature_indices(self, feature_set: str) -> np.ndarray:
        names = np.asarray(self.feature_names)
        if feature_set == "full":
            return np.arange(self.num_features)
        if feature_set == "prices":
            selected = PRICE_FEATURES
        elif feature_set == "momentum":
            selected = PRICE_FEATURES | MOMENTUM_FEATURES
        elif feature_set == "volatility_range":
            selected = PRICE_FEATURES | MOMENTUM_FEATURES | VOLATILITY_FEATURES
        else:
            raise ValueError(f"Unknown feature set {feature_set!r}")
        indices = np.flatnonzero(np.isin(names, list(selected)))
        if indices.size == 0:
            raise ValueError(
                f"Feature set {feature_set!r} selected no columns from {self.feature_names}"
            )
        return indices

    def select_features(self, feature_set: str) -> MarketPanel:
        indices = self.feature_indices(feature_set)
        return replace(
            self,
            features=self.features[:, :, indices],
            feature_names=tuple(self.feature_names[index] for index in indices),
        )

    def date_index(self, date: str, side: str = "left") -> int:
        return int(np.searchsorted(self.dates, np.datetime64(str(date), "D"), side=side))

    def date_bounds(self, start: str, end: str) -> tuple[int, int]:
        first = self.date_index(start, "left")
        last_exclusive = self.date_index(end, "right")
        if first >= last_exclusive or first >= self.num_dates:
            raise ValueError(f"No panel dates fall in [{start}, {end}]")
        return first, min(last_exclusive, self.num_dates)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dates=self.dates.astype("datetime64[D]").astype(str),
            tickers=self.tickers.astype(str),
            prices=self.prices.astype(np.float32),
            features=self.features.astype(np.float32),
            tradable=self.tradable.astype(np.uint8),
            feature_names=np.asarray(self.feature_names, dtype=str),
        )
        manifest = dict(self.metadata or {})
        manifest.update(metadata or {})
        manifest.update(
            {
                "panel_file": path.name,
                "sha256": sha256_file(path),
                "num_dates": self.num_dates,
                "num_assets": self.num_assets,
                "num_features": self.num_features,
                "first_date": str(self.dates[0]),
                "last_date": str(self.dates[-1]),
                "tickers": self.tickers.tolist(),
                "feature_names": list(self.feature_names),
            }
        )
        write_json(manifest, path.with_suffix(".json"))
        return path

    @classmethod
    def load(cls, path: str | Path) -> MarketPanel:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            panel = cls(
                dates=data["dates"].astype("datetime64[D]"),
                tickers=data["tickers"].astype(str),
                prices=data["prices"].astype(np.float32),
                features=data["features"].astype(np.float32),
                tradable=data["tradable"].astype(bool),
                feature_names=tuple(data["feature_names"].astype(str).tolist()),
            )
        return panel
