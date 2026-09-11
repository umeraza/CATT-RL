import numpy as np

from catt_rl.data.panel import MarketPanel
from catt_rl.envs.portfolio import PortfolioEnv


def _accounting_panel() -> MarketPanel:
    dates = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2020-01-08"), dtype="datetime64[D]"
    )
    prices = np.ones((len(dates), 2), dtype=np.float32) * 100.0
    prices[2:, 0] = 110.0
    return MarketPanel(
        dates=dates,
        tickers=np.asarray(["UP", "FLAT"]),
        prices=prices,
        features=np.zeros((len(dates), 2, 1), dtype=np.float32),
        tradable=np.ones((len(dates), 2), dtype=bool),
        feature_names=("return_1",),
    )


def test_cost_and_next_return_accounting() -> None:
    env = PortfolioEnv(
        _accounting_panel(),
        "2020-01-01",
        "2020-01-07",
        lookback=2,
        initial_value=100.0,
        transaction_cost=0.001,
        blend=1.0,
        turnover_cap=2.0,
        allocation_floor=0.0,
    )
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.asarray([1.0, 0.0, 0.0]))
    assert np.isclose(info["gross_return"], 0.10)
    assert np.isclose(info["risky_turnover"], 1.0)
    assert np.isclose(info["net_return"], 0.999 * 1.10 - 1.0)
    assert np.isclose(reward, info["portfolio_value"] / 100.0 - 1.0)


def test_projection_respects_simplex_and_l1_cap() -> None:
    env = PortfolioEnv(
        _accounting_panel(),
        "2020-01-01",
        "2020-01-07",
        lookback=2,
        blend=1.0,
        turnover_cap=0.25,
        allocation_floor=1e-4,
    )
    env.reset(seed=1)
    target = env.project_action(np.asarray([100.0, -3.0, np.nan]))
    assert np.all(target >= 0.0)
    assert np.isclose(target.sum(), 1.0)
    assert np.abs(target - env.weights).sum() <= 0.25 + 1e-10


def test_gymnasium_signatures(synthetic_panel) -> None:
    env = PortfolioEnv(
        synthetic_panel,
        str(synthetic_panel.dates[0]),
        str(synthetic_panel.dates[-1]),
        lookback=8,
        episode_steps=2,
    )
    observation, info = env.reset(seed=5)
    assert set(observation) == {"market", "weights", "tradable", "index"}
    assert "portfolio_value" in info
    transition = env.step(np.ones(synthetic_panel.num_assets + 1))
    assert len(transition) == 5
