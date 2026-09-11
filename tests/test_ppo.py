import numpy as np
import torch

from catt_rl.models.policy import PortfolioActorCritic
from catt_rl.rl.buffer import RolloutBuffer
from catt_rl.rl.ppo import PPOOptimizer


def test_single_ppo_update_is_finite(synthetic_panel, small_model_config) -> None:
    environments, steps, lookback = 2, 4, 8
    model = PortfolioActorCritic(
        synthetic_panel.num_assets,
        synthetic_panel.num_features,
        small_model_config,
        {"distribution": "logistic_normal", "temperature": 1.0},
    )
    observation = {
        "market": np.stack([synthetic_panel.features[:lookback]] * environments),
        "weights": np.tile(
            np.asarray([0.0] * synthetic_panel.num_assets + [1.0], dtype=np.float32),
            (environments, 1),
        ),
        "tradable": np.ones((environments, synthetic_panel.num_assets), dtype=np.float32),
    }
    buffer = RolloutBuffer(
        steps,
        environments,
        observation,
        synthetic_panel.num_assets + 1,
        model.memory_depth,
        model.dimension,
    )
    memory, mask = model.initial_memory(environments, torch.device("cpu"))
    tensors = {key: torch.tensor(value) for key, value in observation.items()}
    for _ in range(steps):
        with torch.no_grad():
            _, raw, log_probability, value, next_memory, next_mask, _ = model.act(
                tensors, memory, mask
            )
        buffer.add(
            observation,
            raw.numpy(),
            log_probability.numpy(),
            np.asarray([0.01, -0.002], dtype=np.float32),
            np.asarray([False, False]),
            value.numpy(),
            memory.numpy(),
            mask.numpy(),
        )
        memory, mask = next_memory, next_mask
    buffer.compute_advantages(np.zeros(environments), gamma=0.995, gae_lambda=0.95)
    optimizer = PPOOptimizer(
        model,
        {
            "clip_epsilon": 0.2,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.001,
            "learning_rate": 3e-4,
            "minimum_learning_rate": 3e-5,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "epochs": 1,
            "minibatch_size": 4,
            "target_kl": None,
        },
        total_updates=2,
        device=torch.device("cpu"),
        amp=False,
    )
    metrics = optimizer.update(buffer)
    assert all(np.isfinite(value) for value in metrics.values())
