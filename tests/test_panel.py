import numpy as np

from catt_rl.data.panel import MarketPanel


def test_panel_round_trip(tmp_path, synthetic_panel) -> None:
    path = tmp_path / "panel.npz"
    synthetic_panel.save(path)
    restored = MarketPanel.load(path)
    np.testing.assert_array_equal(restored.dates, synthetic_panel.dates)
    np.testing.assert_array_equal(restored.tickers, synthetic_panel.tickers)
    np.testing.assert_allclose(restored.prices, synthetic_panel.prices)
    np.testing.assert_allclose(restored.features, synthetic_panel.features)
    assert path.with_suffix(".json").exists()


def test_feature_group_is_a_strict_subset(synthetic_panel) -> None:
    prices = synthetic_panel.select_features("prices")
    momentum = synthetic_panel.select_features("momentum")
    assert 0 < prices.num_features < momentum.num_features < synthetic_panel.num_features
