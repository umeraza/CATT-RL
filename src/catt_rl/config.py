"""YAML configuration loading, inheritance, overrides, and validation."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_recursive(path: Path, active: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    active = set() if active is None else active
    if path in active:
        raise ConfigError(f"Cyclic config inheritance detected at {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    active.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent = current.pop("defaults", None)
    if parent is None:
        result = current
    else:
        parents = [parent] if isinstance(parent, str) else list(parent)
        result: dict[str, Any] = {}
        for item in parents:
            parent_path = (path.parent / item).resolve()
            result = _deep_merge(result, _load_recursive(parent_path, active))
        result = _deep_merge(result, current)
    active.remove(path)
    return result


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    """Set a nested value using a dotted path, creating dictionaries as needed."""

    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot set {key}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"Override must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        if not key.strip():
            raise ConfigError(f"Override has an empty key: {item!r}")
        set_dotted(parsed, key.strip(), yaml.safe_load(raw))
    return parsed


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = _load_recursive(Path(path))
    config = _deep_merge(config, parse_overrides(overrides))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "experiment",
        "data",
        "environment",
        "model",
        "policy",
        "ppo",
        "evaluation",
    }
    missing = required_sections - set(config)
    if missing:
        raise ConfigError(f"Missing config sections: {sorted(missing)}")

    data = config["data"]
    for name in ("train_start", "train_end", "test_start", "test_end"):
        if name not in data:
            raise ConfigError(f"data.{name} is required")
    if int(data["lookback"]) < 2:
        raise ConfigError("data.lookback must be at least 2")

    model = config["model"]
    if int(model["d_model"]) % int(model["heads"]) != 0:
        raise ConfigError("model.d_model must be divisible by model.heads")
    if model["normalization"] not in {"instance", "layer"}:
        raise ConfigError("model.normalization must be 'instance' or 'layer'")
    if model["normalization_placement"] not in {"pre", "post"}:
        raise ConfigError("model.normalization_placement must be 'pre' or 'post'")
    if model["cross_asset_mode"] not in {"full", "latent"}:
        raise ConfigError("model.cross_asset_mode must be 'full' or 'latent'")
    if model["backbone"] not in {"catt", "vanilla"}:
        raise ConfigError("model.backbone must be 'catt' or 'vanilla'")

    env = config["environment"]
    for key in ("transaction_cost", "blend", "turnover_cap", "allocation_floor"):
        if float(env[key]) < 0:
            raise ConfigError(f"environment.{key} cannot be negative")
    if not 0 <= float(env["blend"]) <= 1:
        raise ConfigError("environment.blend must be in [0, 1]")
    if env["reward_type"] not in {"value_delta", "log_return", "mean_variance"}:
        raise ConfigError("Unsupported environment.reward_type")

    ppo = config["ppo"]
    rollout_size = int(ppo["rollout_steps"]) * int(ppo["num_envs"])
    if int(ppo["minibatch_size"]) > rollout_size:
        raise ConfigError("ppo.minibatch_size cannot exceed rollout_steps * num_envs")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime, Path)):
        return str(value)
    return value


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable(config), handle, sort_keys=False)


def apply_mapping_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply `{dotted.key: value}` entries used by ablation/protocol files."""

    result = copy.deepcopy(config)
    for key, value in overrides.items():
        set_dotted(result, key, value)
    validate_config(result)
    return result
