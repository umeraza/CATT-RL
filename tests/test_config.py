from pathlib import Path

import pytest

from catt_rl.config import ConfigError, config_hash, load_config


def test_inheritance_and_override() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "smoke.yaml", ["model.memory_depth=4", "experiment.amp=false"]
    )
    assert config["model"]["memory_depth"] == 4
    assert config["model"]["d_model"] == 32
    assert config["environment"]["transaction_cost"] == 0.001
    assert len(config_hash(config)) == 64


def test_invalid_head_width_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ConfigError):
        load_config(root / "configs" / "smoke.yaml", ["model.d_model=31"])
