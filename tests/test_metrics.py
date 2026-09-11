import numpy as np

from catt_rl.metrics import (
    block_bootstrap_confidence_intervals,
    cumulative_return,
    maximum_drawdown,
    omega_ratio,
    performance_metrics,
)


def test_known_cumulative_return_and_drawdown() -> None:
    returns = np.asarray([0.10, -0.10, 0.05])
    assert np.isclose(cumulative_return(returns), 1.10 * 0.90 * 1.05 - 1.0)
    assert np.isclose(maximum_drawdown(returns), 0.10)


def test_metrics_and_bootstrap_are_deterministic() -> None:
    returns = np.asarray([0.01, -0.005, 0.007, -0.002] * 20)
    metrics = performance_metrics(returns, np.ones_like(returns) * 0.2)
    assert metrics["sharpe"] > 0
    assert omega_ratio(returns) > 1
    first = block_bootstrap_confidence_intervals(returns, samples=50, block_length=5, seed=3)
    second = block_bootstrap_confidence_intervals(returns, samples=50, block_length=5, seed=3)
    assert first == second
