"""PPO-Clip update with GAE rollouts and a logistic-normal actor."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn

from catt_rl.models.policy import PortfolioActorCritic
from catt_rl.rl.buffer import RolloutBuffer
from catt_rl.utils import optimizer_groups


class PPOOptimizer:
    def __init__(
        self,
        model: PortfolioActorCritic,
        config: dict[str, Any],
        total_updates: int,
        device: torch.device,
        amp: bool = True,
        seed: int = 7,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.clip_epsilon = float(config["clip_epsilon"])
        self.value_coefficient = float(config["value_coefficient"])
        self.entropy_coefficient = float(config["entropy_coefficient"])
        self.gradient_clip = float(config["gradient_clip"])
        self.epochs = int(config["epochs"])
        self.minibatch_size = int(config["minibatch_size"])
        self.target_kl = float(config["target_kl"]) if config.get("target_kl") is not None else None
        learning_rate = float(config["learning_rate"])
        minimum = float(config["minimum_learning_rate"])
        self.optimizer = torch.optim.AdamW(
            optimizer_groups(model, float(config["weight_decay"])), lr=learning_rate
        )
        minimum_ratio = minimum / learning_rate

        def schedule(step: int) -> float:
            progress = min(max(step / max(total_updates, 1), 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.amp = bool(amp and device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):  # PyTorch 2.1 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.rng = np.random.default_rng(seed)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        flattened_advantages = buffer.advantages.reshape(-1)
        mean = float(flattened_advantages.mean())
        std = float(flattened_advantages.std(ddof=0))
        buffer.advantages = (buffer.advantages - mean) / max(std, 1e-8)

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }
        updates = 0
        stop_early = False
        self.model.train()
        for _ in range(self.epochs):
            for batch in buffer.minibatches(self.minibatch_size, self.device, self.rng):
                observations = batch["observations"]
                if self.amp:
                    try:
                        autocast = torch.amp.autocast("cuda", enabled=True)
                    except (AttributeError, TypeError):  # PyTorch 2.1 compatibility
                        autocast = torch.cuda.amp.autocast(enabled=True)
                else:
                    autocast = nullcontext()
                with autocast:
                    new_log_probability, entropy, value = self.model.evaluate_raw_actions(
                        observations,  # type: ignore[arg-type]
                        batch["raw_actions"],  # type: ignore[arg-type]
                        batch["memory"],  # type: ignore[arg-type]
                        batch["memory_mask"],  # type: ignore[arg-type]
                    )
                    log_ratio = new_log_probability - batch["old_log_probabilities"]  # type: ignore[operator]
                    ratio = log_ratio.exp()
                    advantages = batch["advantages"]  # type: ignore[assignment]
                    unclipped = ratio * advantages
                    clipped = (
                        torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                        * advantages
                    )
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * torch.mean((value - batch["returns"]) ** 2)  # type: ignore[operator]
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        + self.value_coefficient * value_loss
                        - self.entropy_coefficient * entropy_mean
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy_mean,
                    "approx_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for key, tensor in values.items():
                    totals[key] += float(tensor.detach().cpu())
                updates += 1
                if self.target_kl is not None and float(approximate_kl) > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        self.scheduler.step()
        result = {key: value / max(updates, 1) for key, value in totals.items()}
        result["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        result["minibatches"] = float(updates)
        result["early_kl_stop"] = float(stop_early)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
