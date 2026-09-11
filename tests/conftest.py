from __future__ import annotations

from typing import Any

import pytest

from catt_rl.data.synthetic import make_synthetic_panel


@pytest.fixture
def synthetic_panel():
    return make_synthetic_panel(assets=4, days=120, seed=11)


@pytest.fixture
def small_model_config() -> dict[str, Any]:
    return {
        "backbone": "catt",
        "d_model": 16,
        "layers": 1,
        "heads": 4,
        "pw_hidden_1": 12,
        "pw_hidden_2": 8,
        "dropout": 0.0,
        "memory_depth": 2,
        "normalization": "instance",
        "normalization_placement": "pre",
        "use_lgu": True,
        "temporal_position_encoding": "sinusoidal",
        "cross_asset_mode": "full",
        "context_latents": 2,
        "policy_hidden": 16,
        "initial_log_std": -0.5,
        "min_concentration": 0.001,
    }
