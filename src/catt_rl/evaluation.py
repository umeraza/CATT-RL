"""Deterministic backtesting, metric calculation, and artifact export."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from catt_rl.data.panel import MarketPanel
from catt_rl.envs.portfolio import PortfolioEnv
from catt_rl.metrics import (
    block_bootstrap_confidence_intervals,
    cumulative_return,
    performance_metrics,
    sanitize_metrics,
)
from catt_rl.models.policy import PortfolioActorCritic
from catt_rl.utils import observations_to_torch, write_json


def _environment_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    environment = config["environment"]
    return {
        "lookback": int(config["data"]["lookback"]),
        "initial_value": float(environment["initial_value"]),
        "transaction_cost": float(environment["transaction_cost"]),
        "blend": float(environment["blend"]),
        "turnover_cap": float(environment["turnover_cap"]),
        "allocation_floor": float(environment["allocation_floor"]),
        "reward_type": str(environment["reward_type"]),
        "reward_scale": environment["reward_scale"],
        "mean_variance_alpha": float(environment["mean_variance_alpha"]),
        "variance_window": int(environment["variance_window"]),
    }


def _compress_attention(diagnostics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    compressed: dict[str, np.ndarray] = {}
    for name, value in diagnostics.items():
        tensor = value.detach().float().cpu()
        if name.endswith("_temporal") and tensor.ndim == 5:  # B,N,H,Q,K
            tensor = tensor.mean(dim=(0, 1, 3))
        elif tensor.ndim == 4:  # B,H,Q,K
            tensor = tensor.mean(dim=(0, 2))
        elif tensor.ndim > 2:
            tensor = tensor.mean(dim=tuple(range(tensor.ndim - 2)))
        compressed[name] = tensor.numpy().astype(np.float32)
    return compressed


def _head_feature_projection(
    model: PortfolioActorCritic, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    weights = model.encoder.input_projection.weight.detach().abs().float().cpu()
    heads = model.encoder.blocks[0].temporal_attention.num_heads
    per_head = weights.reshape(heads, weights.shape[0] // heads, weights.shape[1]).mean(dim=1)
    names = list(feature_names)
    if model.encoder.include_weights:
        names.append("portfolio_weight")
    if model.encoder.include_cash:
        names.append("cash_fraction")
    return per_head.numpy().astype(np.float32), np.asarray(names, dtype=str)


def evaluate_policy(
    model: PortfolioActorCritic,
    panel: MarketPanel,
    config: dict[str, Any],
    start_date: str,
    end_date: str,
    device: torch.device,
    output_dir: str | Path | None = None,
    bootstrap: bool = True,
    save_attention: bool = False,
) -> dict[str, Any]:
    env = PortfolioEnv(
        panel,
        str(start_date),
        str(end_date),
        random_start=False,
        episode_steps=None,
        **_environment_kwargs(config),
    )
    observation, _ = env.reset(seed=int(config["experiment"]["seed"]))
    memory, memory_mask = model.initial_memory(1, device)
    model.eval()
    records: list[dict[str, Any]] = []
    attention_records: dict[str, list[np.ndarray]] = defaultdict(list)
    done = False
    while not done:
        batch = {key: value[None, ...] for key, value in observation.items()}
        tensors = observations_to_torch(batch, device)
        with torch.no_grad():
            (
                action,
                _,
                _,
                _,
                memory,
                memory_mask,
                diagnostics,
            ) = model.act(
                tensors,
                memory,
                memory_mask,
                deterministic=True,
                need_weights=save_attention,
            )
        observation, _, terminated, truncated, info = env.step(action[0].detach().cpu().numpy())
        record = {
            "date": info["next_date"],
            "execution_date": info["date"],
            "portfolio_value": info["portfolio_value"],
            "gross_return": info["gross_return"],
            "net_return": info["net_return"],
            "transaction_cost_fraction": info["transaction_cost_fraction"],
            "transaction_cost_amount": info["transaction_cost_amount"],
            "turnover": info["turnover"],
            "risky_turnover": info["risky_turnover"],
            "pre_trade_weights": info["pre_trade_weights"],
            "target_weights": info["target_weights"],
            "post_return_weights": info["post_return_weights"],
        }
        records.append(record)
        if save_attention:
            for key, value in _compress_attention(diagnostics).items():
                attention_records[key].append(value)
        done = terminated or truncated

    returns = np.asarray([record["net_return"] for record in records])
    turnovers = np.asarray([record["turnover"] for record in records])
    evaluation = config["evaluation"]
    metrics = performance_metrics(
        returns,
        turnovers,
        risk_free_rate=float(evaluation["risk_free_rate"]),
        threshold=float(evaluation["minimum_acceptable_return"]),
        annualization=int(evaluation["annualization"]),
    )
    gross_returns = np.asarray([record["gross_return"] for record in records])
    metrics.update(
        {
            "gross_cumulative_return": cumulative_return(gross_returns),
            "final_portfolio_value": float(records[-1]["portfolio_value"]),
            "total_transaction_cost": float(
                sum(record["transaction_cost_amount"] for record in records)
            ),
            "average_risky_turnover": float(
                np.mean([record["risky_turnover"] for record in records])
            ),
        }
    )
    if bootstrap:
        metrics["confidence_intervals"] = block_bootstrap_confidence_intervals(
            returns,
            samples=int(evaluation["bootstrap_samples"]),
            block_length=int(evaluation["bootstrap_block_length"]),
            confidence=float(evaluation["bootstrap_confidence"]),
            seed=int(config["experiment"]["seed"]),
            risk_free_rate=float(evaluation["risk_free_rate"]),
            threshold=float(evaluation["minimum_acceptable_return"]),
            annualization=int(evaluation["annualization"]),
        )
    metrics = sanitize_metrics(metrics)

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        scalar_columns = [
            "date",
            "execution_date",
            "portfolio_value",
            "gross_return",
            "net_return",
            "transaction_cost_fraction",
            "transaction_cost_amount",
            "turnover",
            "risky_turnover",
        ]
        pd.DataFrame([{key: record[key] for key in scalar_columns} for record in records]).to_csv(
            output / "daily_returns.csv", index=False
        )
        columns = [*panel.tickers.tolist(), "CASH"]
        pd.DataFrame(
            [record["post_return_weights"] for record in records],
            index=[record["date"] for record in records],
            columns=columns,
        ).rename_axis("date").to_csv(output / "weights.csv")
        pd.DataFrame(
            [record["target_weights"] - record["pre_trade_weights"] for record in records],
            index=[record["execution_date"] for record in records],
            columns=columns,
        ).rename_axis("date").to_csv(output / "trades.csv")
        pd.DataFrame(
            {
                "date": [record["date"] for record in records],
                "portfolio_value": [record["portfolio_value"] for record in records],
            }
        ).to_csv(output / "portfolio_values.csv", index=False)
        write_json(metrics, output / "metrics.json")
        if save_attention:
            arrays: dict[str, np.ndarray] = {
                key: np.stack(value, axis=0) for key, value in attention_records.items() if value
            }
            projection, names = _head_feature_projection(model, panel.feature_names)
            arrays["head_feature_projection"] = projection
            arrays["feature_names"] = names
            arrays["dates"] = np.asarray([record["date"] for record in records], dtype=str)
            np.savez_compressed(output / "attention.npz", **arrays)
    return {"metrics": metrics, "records": records}


def load_checkpoint_model(
    checkpoint_path: str | Path,
    panel: MarketPanel,
    config: dict[str, Any],
    device: torch.device,
) -> PortfolioActorCritic:
    model = PortfolioActorCritic(
        panel.num_assets,
        panel.num_features,
        config["model"],
        config["policy"],
        bool(config["data"]["include_weights"]),
        bool(config["data"]["include_cash"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model
