import numpy as np
import pandas as pd

from catt_rl.data.features import build_panel


def _raw_frame(days: int = 210, assets: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=days)
    rows = []
    for asset in range(assets):
        price = 80.0 + 10.0 * asset
        for index, date in enumerate(dates):
            price *= 1.0 + 0.0004 * (asset + 1) + 0.002 * np.sin(index / 9.0 + asset)
            rows.append(
                {
                    "date": date,
                    "ticker": f"A{asset}",
                    "open": price * 0.998,
                    "high": price * 1.008,
                    "low": price * 0.992,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000 + 1_000 * index + 10_000 * asset,
                }
            )
    return pd.DataFrame(rows)


def test_future_perturbation_does_not_change_past_features() -> None:
    raw = _raw_frame()
    cutoff = pd.Timestamp("2019-07-31")
    original = build_panel(raw, availability=1.0, warmup=20)
    changed = raw.copy()
    future = changed["date"] > cutoff
    changed.loc[future, ["open", "high", "low", "close", "adj_close"]] *= 3.0
    perturbed = build_panel(changed, availability=1.0, warmup=20)
    indices = original.dates <= np.datetime64(cutoff.date())
    np.testing.assert_allclose(
        original.features[indices], perturbed.features[indices], atol=1e-6, rtol=1e-6
    )


def test_cross_sectional_features_are_finite() -> None:
    panel = build_panel(_raw_frame(), availability=1.0, warmup=126)
    assert np.isfinite(panel.features).all()
    assert panel.features.shape[2] >= 20
