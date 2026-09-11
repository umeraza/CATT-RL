"""Memory-aware PPO rollout storage."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        steps: int,
        environments: int,
        observation: dict[str, np.ndarray],
        action_dimension: int,
        memory_depth: int,
        model_dimension: int,
        panel_features: np.ndarray | None = None,
        lookback: int | None = None,
    ) -> None:
        self.steps = int(steps)
        self.environments = int(environments)
        self.position = 0
        self.panel_features = panel_features
        self.lookback = int(lookback) if lookback is not None else None
        if self.panel_features is not None and self.lookback is None:
            raise ValueError("lookback is required with panel_features")
        self.observations: dict[str, np.ndarray] = {}
        for key, value in observation.items():
            if key == "market" and self.panel_features is not None:
                continue
            shape = (steps, environments, *value.shape[1:])
            self.observations[key] = np.empty(shape, dtype=np.float32)
        self.raw_actions = np.empty((steps, environments, action_dimension), dtype=np.float32)
        self.log_probabilities = np.empty((steps, environments), dtype=np.float32)
        self.rewards = np.empty((steps, environments), dtype=np.float32)
        self.dones = np.empty((steps, environments), dtype=bool)
        self.values = np.empty((steps, environments), dtype=np.float32)
        self.memories = np.empty(
            (steps, environments, memory_depth, model_dimension), dtype=np.float32
        )
        self.memory_masks = np.empty((steps, environments, memory_depth), dtype=bool)
        self.advantages = np.empty((steps, environments), dtype=np.float32)
        self.returns = np.empty((steps, environments), dtype=np.float32)

    def add(
        self,
        observations: dict[str, np.ndarray],
        raw_actions: np.ndarray,
        log_probabilities: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        memories: np.ndarray,
        memory_masks: np.ndarray,
    ) -> None:
        if self.position >= self.steps:
            raise RuntimeError("Rollout buffer is full")
        for key, value in observations.items():
            if key in self.observations:
                self.observations[key][self.position] = value
        self.raw_actions[self.position] = raw_actions
        self.log_probabilities[self.position] = log_probabilities
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        self.values[self.position] = values
        self.memories[self.position] = memories
        self.memory_masks[self.position] = memory_masks
        self.position += 1

    def compute_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.steps:
            raise RuntimeError("Cannot finalize a partially filled rollout")
        advantage = np.zeros(self.environments, dtype=np.float32)
        for step in reversed(range(self.steps)):
            next_values = last_values if step == self.steps - 1 else self.values[step + 1]
            nonterminal = 1.0 - self.dones[step].astype(np.float32)
            delta = self.rewards[step] + gamma * next_values * nonterminal - self.values[step]
            advantage = delta + gamma * gae_lambda * nonterminal * advantage
            self.advantages[step] = advantage
        self.returns = self.advantages + self.values

    @property
    def size(self) -> int:
        return self.steps * self.environments

    def minibatches(
        self,
        minibatch_size: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> Iterator[dict[str, object]]:
        indices = rng.permutation(self.size)
        flat_observations = {
            key: value.reshape(self.size, *value.shape[2:])
            for key, value in self.observations.items()
        }
        for start in range(0, self.size, minibatch_size):
            selected = indices[start : start + minibatch_size]
            observation_batch = {
                key: torch.as_tensor(value[selected], dtype=torch.float32, device=device)
                for key, value in flat_observations.items()
            }
            if self.panel_features is not None:
                current = flat_observations["index"][selected].reshape(-1).astype(np.int64)
                offsets = np.arange(-self.lookback + 1, 1, dtype=np.int64)  # type: ignore[operator]
                market_indices = current[:, None] + offsets[None, :]
                market = self.panel_features[market_indices]
                observation_batch["market"] = torch.as_tensor(
                    market, dtype=torch.float32, device=device
                )
            yield {
                "observations": observation_batch,
                "raw_actions": torch.as_tensor(
                    self.raw_actions.reshape(self.size, -1)[selected],
                    dtype=torch.float32,
                    device=device,
                ),
                "old_log_probabilities": torch.as_tensor(
                    self.log_probabilities.reshape(-1)[selected],
                    dtype=torch.float32,
                    device=device,
                ),
                "advantages": torch.as_tensor(
                    self.advantages.reshape(-1)[selected], dtype=torch.float32, device=device
                ),
                "returns": torch.as_tensor(
                    self.returns.reshape(-1)[selected], dtype=torch.float32, device=device
                ),
                "old_values": torch.as_tensor(
                    self.values.reshape(-1)[selected], dtype=torch.float32, device=device
                ),
                "memory": torch.as_tensor(
                    self.memories.reshape(
                        self.size, self.memories.shape[2], self.memories.shape[3]
                    )[selected],
                    dtype=torch.float32,
                    device=device,
                ),
                "memory_mask": torch.as_tensor(
                    self.memory_masks.reshape(self.size, self.memory_masks.shape[2])[selected],
                    dtype=torch.bool,
                    device=device,
                ),
            }
