"""Logistic-normal simplex actor with a separate portfolio-value critic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Independent, Normal

from catt_rl.models.catt import CATTEncoder
from catt_rl.models.layers import initialize_linear_layers


@dataclass
class PolicyOutput:
    mean_logits: torch.Tensor
    value: torch.Tensor
    next_memory: torch.Tensor
    next_memory_mask: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class PortfolioActorCritic(nn.Module):
    def __init__(
        self,
        num_assets: int,
        market_features: int,
        model_config: dict[str, Any],
        policy_config: dict[str, Any],
        include_weights: bool = True,
        include_cash: bool = True,
    ) -> None:
        super().__init__()
        if policy_config.get("distribution") != "logistic_normal":
            raise ValueError("Only the documented logistic_normal policy is supported")
        self.num_assets = int(num_assets)
        self.temperature = float(policy_config["temperature"])
        dimension = int(model_config["d_model"])
        hidden = int(model_config["policy_hidden"])
        self.encoder = CATTEncoder(market_features, model_config, include_weights, include_cash)
        self.weight_condition = nn.Linear(1, dimension)
        self.cash_condition = nn.Linear(1, dimension)
        self.policy_hidden = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU())
        self.asset_score = nn.Linear(hidden, 1)
        self.cash_score = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.log_std = nn.Parameter(
            torch.full((num_assets + 1,), float(model_config["initial_log_std"]))
        )
        self.fan_in_scale = 1.0 / math.sqrt(dimension)
        self.apply(initialize_linear_layers)

    @property
    def memory_depth(self) -> int:
        return self.encoder.memory_depth

    @property
    def dimension(self) -> int:
        return self.encoder.dimension

    def initial_memory(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.initial_memory(batch_size, device)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> PolicyOutput:
        assets, context, next_memory, next_mask, diagnostics = self.encoder(
            observations, memory, memory_mask, need_weights
        )
        previous = observations["weights"]
        conditioned_assets = assets + self.weight_condition(previous[:, :-1].unsqueeze(-1))
        conditioned_cash = context + self.cash_condition(previous[:, -1:].unsqueeze(-1)).squeeze(1)
        asset_logits = self.asset_score(self.policy_hidden(conditioned_assets)).squeeze(-1)
        asset_logits = asset_logits.masked_fill(observations["tradable"] <= 0.5, -20.0)
        cash_logits = self.cash_score(self.policy_hidden(conditioned_cash)).squeeze(-1)
        mean_logits = torch.cat([asset_logits, cash_logits.unsqueeze(-1)], dim=-1)
        mean_logits = mean_logits * self.fan_in_scale
        value = self.value_head(context).squeeze(-1)
        return PolicyOutput(mean_logits, value, next_memory, next_mask, diagnostics)

    def distribution(self, mean_logits: torch.Tensor) -> Independent:
        standard_deviation = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean_logits)
        return Independent(Normal(mean_logits, standard_deviation), 1)

    def act(
        self,
        observations: dict[str, torch.Tensor],
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        deterministic: bool = False,
        need_weights: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        output = self(observations, memory, memory_mask, need_weights)
        distribution = self.distribution(output.mean_logits)
        raw_action = output.mean_logits if deterministic else distribution.sample()
        simplex_action = torch.softmax(raw_action / self.temperature, dim=-1)
        log_probability = distribution.log_prob(raw_action)
        return (
            simplex_action,
            raw_action,
            log_probability,
            output.value,
            output.next_memory,
            output.next_memory_mask,
            output.diagnostics,
        )

    def evaluate_raw_actions(
        self,
        observations: dict[str, torch.Tensor],
        raw_actions: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self(observations, memory, memory_mask)
        distribution = self.distribution(output.mean_logits)
        return distribution.log_prob(raw_actions), distribution.entropy(), output.value
