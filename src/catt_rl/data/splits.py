"""Explicit rolling-protocol loading and temporal validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle) or {}
    windows = protocol.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Protocol must contain a non-empty 'windows' list")
    required = {"name", "train_start", "train_end", "test_start", "test_end"}
    names: set[str] = set()
    for window in windows:
        missing = required - set(window)
        if missing:
            raise ValueError(f"Protocol window missing fields: {sorted(missing)}")
        if window["name"] in names:
            raise ValueError(f"Duplicate protocol window name: {window['name']}")
        names.add(window["name"])
        if str(window["train_end"]) >= str(window["test_start"]):
            raise ValueError(f"Train/test overlap in protocol window {window['name']}")
    return protocol
