"""End-to-end rollout collection, PPO training, checkpointing, and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from catt_rl.config import config_hash, save_config
from catt_rl.data.panel import MarketPanel
from catt_rl.envs.portfolio import PortfolioEnv
from catt_rl.evaluation import _environment_kwargs, evaluate_policy
from catt_rl.models.policy import PortfolioActorCritic
from catt_rl.rl.buffer import RolloutBuffer
from catt_rl.rl.ppo import PPOOptimizer
from catt_rl.utils import (
    append_jsonl,
    git_revision,
    observations_to_torch,
    resolve_device,
    runtime_manifest,
    seed_everything,
    sha256_file,
    stack_observations,
    write_json,
)


def _training_end(config: dict[str, Any]) -> str:
    data = config["data"]
    if data.get("validation_start"):
        day_before = np.datetime64(str(data["validation_start"]), "D") - np.timedelta64(1, "D")
        return str(min(np.datetime64(str(data["train_end"]), "D"), day_before))
    return str(data["train_end"])


def _save_checkpoint(
    path: Path,
    model: PortfolioActorCritic,
    optimizer: PPOOptimizer,
    config: dict[str, Any],
    update: int,
    best_sharpe: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ppo": optimizer.state_dict(),
            "config": config,
            "config_hash": config_hash(config),
            "update": int(update),
            "best_validation_sharpe": float(best_sharpe),
        },
        temporary,
    )
    os.replace(temporary, path)


def build_model(
    panel: MarketPanel, config: dict[str, Any], device: torch.device
) -> PortfolioActorCritic:
    return PortfolioActorCritic(
        panel.num_assets,
        panel.num_features,
        config["model"],
        config["policy"],
        bool(config["data"]["include_weights"]),
        bool(config["data"]["include_cash"]),
    ).to(device)


def train_experiment(config: dict[str, Any]) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))
    device = resolve_device(str(config["experiment"]["device"]))
    panel_path = Path(config["data"]["panel_path"])
    panel = MarketPanel.load(panel_path).select_features(str(config["data"]["feature_set"]))
    digest = config_hash(config)
    run_dir = Path(config["experiment"]["output_dir"]) / f"seed_{seed}_{digest[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "resolved_config.yaml")
    write_json(
        {
            "config_sha256": digest,
            "panel_path": str(panel_path),
            "panel_sha256": sha256_file(panel_path),
            "git_revision": git_revision(Path(__file__).resolve().parents[2]),
            "effective_training_end": _training_end(config),
            "runtime": runtime_manifest(),
        },
        run_dir / "run_manifest.json",
    )
    metrics_path = run_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    model = build_model(panel, config, device)
    maximum_updates = int(config["experiment"]["max_updates"])
    ppo = PPOOptimizer(
        model,
        config["ppo"],
        maximum_updates,
        device,
        amp=bool(config["experiment"]["amp"]),
        seed=seed,
    )
    train_end = _training_end(config)
    envs = [
        PortfolioEnv(
            panel,
            str(config["data"]["train_start"]),
            train_end,
            random_start=True,
            episode_steps=int(config["data"]["episode_steps"]),
            **_environment_kwargs(config),
        )
        for _ in range(int(config["ppo"]["num_envs"]))
    ]
    observations = [env.reset(seed=seed + index)[0] for index, env in enumerate(envs)]
    observation_batch = stack_observations(observations)
    memory, memory_mask = model.initial_memory(len(envs), device)
    best_sharpe = -float("inf")
    stale_updates = 0

    progress = trange(maximum_updates, desc="PPO", leave=False)
    for update in progress:
        buffer = RolloutBuffer(
            int(config["ppo"]["rollout_steps"]),
            len(envs),
            observation_batch,
            panel.num_assets + 1,
            model.memory_depth,
            model.dimension,
            panel_features=panel.features,
            lookback=int(config["data"]["lookback"]),
        )
        rollout_rewards: list[float] = []
        rollout_turnover: list[float] = []
        for _ in range(int(config["ppo"]["rollout_steps"])):
            tensor_observations = observations_to_torch(observation_batch, device)
            memory_before = memory.detach().cpu().numpy()
            mask_before = memory_mask.detach().cpu().numpy()
            with torch.no_grad():
                (
                    actions,
                    raw_actions,
                    log_probabilities,
                    values,
                    next_memory,
                    next_mask,
                    _,
                ) = model.act(tensor_observations, memory, memory_mask)
            actions_numpy = actions.detach().cpu().numpy()
            next_observations: list[dict[str, np.ndarray]] = []
            rewards = np.empty(len(envs), dtype=np.float32)
            dones = np.empty(len(envs), dtype=bool)
            for index, env in enumerate(envs):
                next_observation, reward, terminated, truncated, info = env.step(
                    actions_numpy[index]
                )
                done = terminated or truncated
                rewards[index] = reward
                dones[index] = done
                rollout_rewards.append(float(reward))
                rollout_turnover.append(float(info["turnover"]))
                if done:
                    next_observation, _ = env.reset()
                    next_memory[index].zero_()
                    next_mask[index].zero_()
                next_observations.append(next_observation)
            buffer.add(
                observation_batch,
                raw_actions.detach().cpu().numpy(),
                log_probabilities.detach().cpu().numpy(),
                rewards,
                dones,
                values.detach().cpu().numpy(),
                memory_before,
                mask_before,
            )
            observations = next_observations
            observation_batch = stack_observations(observations)
            memory, memory_mask = next_memory.detach(), next_mask.detach()

        with torch.no_grad():
            last_output = model(
                observations_to_torch(observation_batch, device), memory, memory_mask
            )
        buffer.compute_advantages(
            last_output.value.detach().cpu().numpy(),
            float(config["ppo"]["gamma"]),
            float(config["ppo"]["gae_lambda"]),
        )
        optimization_metrics = ppo.update(buffer)
        training_metrics: dict[str, Any] = {
            "update": update + 1,
            "mean_rollout_reward": float(np.mean(rollout_rewards)),
            "mean_rollout_turnover": float(np.mean(rollout_turnover)),
            **optimization_metrics,
        }

        evaluate_every = int(config["experiment"]["evaluate_every"])
        if (update + 1) % evaluate_every == 0:
            validation = evaluate_policy(
                model,
                panel,
                config,
                str(config["data"]["validation_start"]),
                str(config["data"]["validation_end"]),
                device,
                bootstrap=False,
            )["metrics"]
            validation_sharpe = float(validation["sharpe"])
            training_metrics["validation_sharpe"] = validation_sharpe
            training_metrics["validation_turnover"] = validation.get("average_turnover", 0.0)
            if validation_sharpe > best_sharpe:
                best_sharpe = validation_sharpe
                stale_updates = 0
                _save_checkpoint(run_dir / "best.pt", model, ppo, config, update + 1, best_sharpe)
            else:
                stale_updates += evaluate_every
        append_jsonl(training_metrics, metrics_path)
        _save_checkpoint(run_dir / "last.pt", model, ppo, config, update + 1, best_sharpe)
        progress.set_postfix(sharpe=f"{best_sharpe:.3f}")
        if stale_updates >= int(config["experiment"]["early_stopping_patience"]):
            break

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        _save_checkpoint(best_path, model, ppo, config, update + 1, best_sharpe)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation_result = evaluate_policy(
        model,
        panel,
        config,
        str(config["data"]["validation_start"]),
        str(config["data"]["validation_end"]),
        device,
        output_dir=run_dir / "validation",
        bootstrap=True,
        save_attention=False,
    )
    test_result = evaluate_policy(
        model,
        panel,
        config,
        str(config["data"]["test_start"]),
        str(config["data"]["test_end"]),
        device,
        output_dir=run_dir / "test",
        bootstrap=True,
        save_attention=bool(config["experiment"]["save_attention"]),
    )
    write_json(validation_result["metrics"], run_dir / "validation_metrics.json")
    write_json(test_result["metrics"], run_dir / "test_metrics.json")
    return run_dir
