"""Transaction-aware, long-only asset-plus-cash portfolio MDP."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from catt_rl.data.panel import MarketPanel


class PortfolioEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Daily close-to-close portfolio allocation environment.

    The observation at index `t` contains data no later than close `t`. The
    selected target is rebalanced at that close, transaction costs are charged,
    and the target earns the `t -> t+1` adjusted-price relative.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: MarketPanel,
        start_date: str,
        end_date: str,
        lookback: int = 64,
        initial_value: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        blend: float = 0.35,
        turnover_cap: float = 0.25,
        allocation_floor: float = 1e-4,
        reward_type: str = "value_delta",
        reward_scale: str | float = "initial_value",
        mean_variance_alpha: float = 0.005,
        variance_window: int = 20,
        random_start: bool = False,
        episode_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.panel = panel
        self.lookback = int(lookback)
        self.initial_value = float(initial_value)
        self.transaction_cost = float(transaction_cost)
        self.blend = float(blend)
        self.turnover_cap = float(turnover_cap)
        self.allocation_floor = float(allocation_floor)
        self.reward_type = str(reward_type)
        self.reward_scale = reward_scale
        self.mean_variance_alpha = float(mean_variance_alpha)
        self.variance_window = int(variance_window)
        self.random_start = bool(random_start)
        self.episode_steps = int(episode_steps) if episode_steps else None

        first, last_exclusive = panel.date_bounds(start_date, end_date)
        self.first_step = max(first, self.lookback - 1)
        self.last_step = last_exclusive - 2  # final action requires a next price
        if self.first_step > self.last_step:
            raise ValueError("Date range does not contain one complete lookback and transition")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if self.reward_type not in {"value_delta", "log_return", "mean_variance"}:
            raise ValueError(f"Unsupported reward type {self.reward_type!r}")

        assets, features = panel.num_assets, panel.num_features
        self.observation_space = spaces.Dict(
            {
                "market": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.lookback, assets, features),
                    dtype=np.float32,
                ),
                "weights": spaces.Box(0.0, 1.0, shape=(assets + 1,), dtype=np.float32),
                "tradable": spaces.Box(0.0, 1.0, shape=(assets,), dtype=np.float32),
                "index": spaces.Box(0, panel.num_dates - 1, shape=(1,), dtype=np.int64),
            }
        )
        self.action_space = spaces.Box(0.0, 1.0, shape=(assets + 1,), dtype=np.float32)
        self.current_step = self.first_step
        self.episode_last_step = self.last_step
        self.portfolio_value = self.initial_value
        self.weights = np.zeros(assets + 1, dtype=np.float64)
        self.weights[-1] = 1.0
        self.return_history: deque[float] = deque(maxlen=max(self.variance_window, 2))
        self.steps_taken = 0

    @property
    def num_assets(self) -> int:
        return self.panel.num_assets

    def _get_obs(self) -> dict[str, np.ndarray]:
        begin = self.current_step - self.lookback + 1
        return {
            "market": self.panel.features[begin : self.current_step + 1].astype(
                np.float32, copy=True
            ),
            "weights": self.weights.astype(np.float32, copy=True),
            "tradable": self.panel.tradable[self.current_step].astype(np.float32, copy=True),
            "index": np.asarray([self.current_step], dtype=np.int64),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "date": str(self.panel.dates[self.current_step]),
            "portfolio_value": float(self.portfolio_value),
            "weights": self.weights.copy(),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        requested_start = options.get("start_step")
        if requested_start is not None:
            start = int(requested_start)
            if not self.first_step <= start <= self.last_step:
                raise ValueError("Requested start_step is outside the configured date range")
        elif self.random_start:
            required = self.episode_steps or 1
            latest = max(self.first_step, self.last_step - required + 1)
            start = int(self.np_random.integers(self.first_step, latest + 1))
        else:
            start = self.first_step

        self.current_step = start
        if self.episode_steps:
            self.episode_last_step = min(self.last_step, start + self.episode_steps - 1)
        else:
            self.episode_last_step = self.last_step
        self.portfolio_value = self.initial_value
        self.weights = np.zeros(self.num_assets + 1, dtype=np.float64)
        self.weights[-1] = 1.0
        self.return_history.clear()
        self.steps_taken = 0
        return self._get_obs(), self._get_info()

    def project_action(self, action: np.ndarray) -> np.ndarray:
        proposed = np.asarray(action, dtype=np.float64).reshape(-1)
        if proposed.shape != (self.num_assets + 1,):
            raise ValueError(
                f"Expected action shape {(self.num_assets + 1,)}, got {proposed.shape}"
            )
        proposed = np.where(np.isfinite(proposed), proposed, 0.0)
        proposed = np.maximum(proposed, 0.0)
        tradable = self.panel.tradable[self.current_step]
        floor = np.concatenate(
            [tradable.astype(np.float64) * self.allocation_floor, [self.allocation_floor]]
        )
        proposed = np.maximum(proposed, floor)
        proposed[:-1] *= tradable
        total = proposed.sum()
        if total <= 0.0:
            proposed[-1] = 1.0
            total = 1.0
        proposed /= total

        target = (1.0 - self.blend) * self.weights + self.blend * proposed
        # Assets unavailable at this close cannot receive a new target; their
        # modeled value is conservatively returned to cash at the last known close.
        unavailable_value = target[:-1][~tradable].sum()
        target[:-1][~tradable] = 0.0
        target[-1] += unavailable_value
        delta = target - self.weights
        l1 = float(np.abs(delta).sum())
        if self.turnover_cap > 0.0 and l1 > self.turnover_cap:
            target = self.weights + delta * (self.turnover_cap / l1)
        target = np.maximum(target, 0.0)
        target /= target.sum()
        return target

    def _scaled_value_delta(self, delta_value: float, old_value: float) -> float:
        if isinstance(self.reward_scale, (int, float)):
            divisor = float(self.reward_scale)
        elif self.reward_scale == "initial_value":
            divisor = self.initial_value
        elif self.reward_scale == "current_value":
            divisor = old_value
        elif self.reward_scale in {"none", None}:
            divisor = 1.0
        else:
            raise ValueError(f"Unknown reward_scale {self.reward_scale!r}")
        return float(delta_value / max(divisor, np.finfo(float).tiny))

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        target = self.project_action(action)
        pre_trade_weights = self.weights.copy()
        risky_turnover = float(np.abs(target[:-1] - pre_trade_weights[:-1]).sum())
        l1_turnover = float(np.abs(target - pre_trade_weights).sum())
        cost_fraction = self.transaction_cost * risky_turnover

        prices_now = self.panel.prices[self.current_step].astype(np.float64)
        prices_next = self.panel.prices[self.current_step + 1].astype(np.float64)
        relatives = np.divide(
            prices_next,
            prices_now,
            out=np.ones_like(prices_next),
            where=prices_now > 0.0,
        )
        gross_growth = float(target[-1] + np.dot(target[:-1], relatives))
        net_growth = max((1.0 - cost_fraction) * gross_growth, np.finfo(float).tiny)
        old_value = self.portfolio_value
        new_value = old_value * net_growth
        delta_value = new_value - old_value
        net_return = net_growth - 1.0
        gross_return = gross_growth - 1.0
        self.return_history.append(net_return)

        if self.reward_type == "value_delta":
            reward = self._scaled_value_delta(delta_value, old_value)
        elif self.reward_type == "log_return":
            # Costs still affect accounting/evaluation but are absent from this
            # ablation's training reward, as stated in the manuscript.
            reward = float(np.log(max(gross_growth, np.finfo(float).tiny)))
        else:
            variance = float(np.var(self.return_history, ddof=0))
            reward = (
                self._scaled_value_delta(delta_value, old_value)
                - self.mean_variance_alpha * variance
            )

        component_values = np.concatenate([target[:-1] * relatives, [target[-1]]])
        self.weights = component_values / max(component_values.sum(), np.finfo(float).tiny)
        self.portfolio_value = new_value
        execution_date = str(self.panel.dates[self.current_step])
        next_date = str(self.panel.dates[self.current_step + 1])
        self.steps_taken += 1

        at_data_end = self.current_step >= self.last_step
        at_episode_end = self.current_step >= self.episode_last_step
        terminated = bool(at_data_end)
        truncated = bool(at_episode_end and not at_data_end)
        self.current_step += 1
        info = {
            "date": execution_date,
            "next_date": next_date,
            "portfolio_value": float(new_value),
            "previous_portfolio_value": float(old_value),
            "gross_return": gross_return,
            "net_return": net_return,
            "transaction_cost_fraction": cost_fraction,
            "transaction_cost_amount": float(old_value * cost_fraction),
            "risky_turnover": risky_turnover,
            "turnover": l1_turnover,
            "pre_trade_weights": pre_trade_weights,
            "target_weights": target.copy(),
            "post_return_weights": self.weights.copy(),
            "price_relatives": relatives.copy(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
