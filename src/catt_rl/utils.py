"""Shared reproducibility, serialization, and tensor helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older supported PyTorch
            torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable")
    return device


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True, default=str, allow_nan=False) + "\n")


def git_revision(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Exclude biases and normalization parameters from AdamW decay."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for _module_name, module in model.named_modules():
        is_norm = isinstance(
            module,
            (
                torch.nn.LayerNorm,
                torch.nn.InstanceNorm1d,
                torch.nn.BatchNorm1d,
                torch.nn.GroupNorm,
            ),
        )
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if is_norm or parameter_name.endswith("bias") or parameter.ndim < 2:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def stack_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = observations[0].keys()
    return {key: np.stack([obs[key] for obs in observations], axis=0) for key in keys}


def observations_to_torch(
    observations: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in observations.items()
    }
