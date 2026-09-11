import numpy as np
import torch

from catt_rl.models.policy import PortfolioActorCritic


def _observations(panel, lookback: int = 8, batch: int = 2):
    market = np.stack([panel.features[:lookback]] * batch)
    weights = np.zeros((batch, panel.num_assets + 1), dtype=np.float32)
    weights[:, -1] = 1.0
    tradable = np.ones((batch, panel.num_assets), dtype=np.float32)
    return {
        "market": torch.tensor(market),
        "weights": torch.tensor(weights),
        "tradable": torch.tensor(tradable),
    }


def test_policy_shapes_simplex_and_memory(synthetic_panel, small_model_config) -> None:
    model = PortfolioActorCritic(
        synthetic_panel.num_assets,
        synthetic_panel.num_features,
        small_model_config,
        {"distribution": "logistic_normal", "temperature": 1.0},
    )
    observations = _observations(synthetic_panel)
    memory, mask = model.initial_memory(2, torch.device("cpu"))
    action, raw, log_probability, value, memory, mask, diagnostics = model.act(
        observations, memory, mask, deterministic=True, need_weights=True
    )
    assert action.shape == (2, synthetic_panel.num_assets + 1)
    assert raw.shape == action.shape
    assert log_probability.shape == (2,)
    assert value.shape == (2,)
    assert torch.allclose(action.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert mask[:, -1].all()
    assert diagnostics
    _, _, _, _, _, next_mask, _ = model.act(observations, memory, mask, deterministic=True)
    assert next_mask.all()


def test_latent_context_mode(synthetic_panel, small_model_config) -> None:
    config = dict(small_model_config)
    config["cross_asset_mode"] = "latent"
    model = PortfolioActorCritic(
        synthetic_panel.num_assets,
        synthetic_panel.num_features,
        config,
        {"distribution": "logistic_normal", "temperature": 1.0},
    )
    observations = _observations(synthetic_panel, batch=1)
    memory, mask = model.initial_memory(1, torch.device("cpu"))
    output = model(observations, memory, mask)
    assert output.mean_logits.shape == (1, synthetic_panel.num_assets + 1)


def test_vanilla_ablation_is_parameter_aligned(synthetic_panel, small_model_config) -> None:
    policy = {"distribution": "logistic_normal", "temperature": 1.0}
    catt = PortfolioActorCritic(
        synthetic_panel.num_assets,
        synthetic_panel.num_features,
        small_model_config,
        policy,
    )
    vanilla_config = dict(small_model_config)
    vanilla_config["backbone"] = "vanilla"
    vanilla = PortfolioActorCritic(
        synthetic_panel.num_assets,
        synthetic_panel.num_features,
        vanilla_config,
        policy,
    )
    catt_parameters = sum(parameter.numel() for parameter in catt.parameters())
    vanilla_parameters = sum(parameter.numel() for parameter in vanilla.parameters())
    assert abs(catt_parameters - vanilla_parameters) / catt_parameters < 0.002
