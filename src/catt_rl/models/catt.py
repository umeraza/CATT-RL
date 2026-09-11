"""Context-Aware Temporal Transformer with a detached FIFO trajectory."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from catt_rl.models.layers import (
    CATTBlock,
    SinusoidalPositionEncoding,
    initialize_linear_layers,
)


class CATTEncoder(nn.Module):
    def __init__(
        self,
        market_features: int,
        config: dict[str, Any],
        include_weights: bool = True,
        include_cash: bool = True,
    ) -> None:
        super().__init__()
        self.dimension = int(config["d_model"])
        self.memory_depth = int(config["memory_depth"])
        self.include_weights = bool(include_weights)
        self.include_cash = bool(include_cash)
        input_features = market_features + int(include_weights) + int(include_cash)
        self.input_projection = nn.Linear(input_features, self.dimension)
        self.position_mode = config.get("temporal_position_encoding", "sinusoidal")
        self.position = SinusoidalPositionEncoding(self.dimension)
        vanilla = config.get("backbone", "catt") == "vanilla"
        self.blocks = nn.ModuleList(
            [
                CATTBlock(
                    dimension=self.dimension,
                    heads=int(config["heads"]),
                    hidden_1=int(config["pw_hidden_1"]),
                    hidden_2=int(config["pw_hidden_2"]),
                    dropout=float(config["dropout"]),
                    normalization=str(config["normalization"]),
                    normalization_placement=str(config["normalization_placement"]),
                    use_lgu=bool(config["use_lgu"]),
                    cross_asset_mode=str(config["cross_asset_mode"]),
                    context_latents=int(config["context_latents"]),
                    vanilla=vanilla,
                )
                for _ in range(int(config["layers"]))
            ]
        )
        self.final_norm = nn.LayerNorm(self.dimension)
        self.apply(initialize_linear_layers)

    def initial_memory(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = torch.zeros(batch_size, self.memory_depth, self.dimension, device=device)
        mask = torch.zeros(batch_size, self.memory_depth, dtype=torch.bool, device=device)
        return memory, mask

    def _augment_inputs(self, market: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        additions: list[torch.Tensor] = [market]
        batch, time, assets, _ = market.shape
        if self.include_weights:
            risky = weights[:, :-1].view(batch, 1, assets, 1).expand(-1, time, -1, -1)
            additions.append(risky)
        if self.include_cash:
            cash = weights[:, -1:].view(batch, 1, 1, 1).expand(-1, time, assets, -1)
            additions.append(cash)
        return torch.cat(additions, dim=-1)

    def _update_memory(
        self,
        memory: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_depth == 0:
            return memory, mask
        new_memory = torch.cat([memory[:, 1:], context.detach().unsqueeze(1)], dim=1)
        valid = torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
        new_mask = torch.cat([mask[:, 1:], valid], dim=1)
        return new_memory, new_mask

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        market = observations["market"]
        weights = observations["weights"]
        asset_mask = observations["tradable"] > 0.5
        batch, time, _, _ = market.shape
        if memory is None or memory_mask is None:
            memory, memory_mask = self.initial_memory(batch, market.device)
        augmented = self._augment_inputs(market, weights)
        tokens = self.input_projection(augmented)
        if self.position_mode == "sinusoidal":
            positions = self.position(time, tokens.dtype).view(1, time, 1, self.dimension)
            tokens = tokens + positions
        elif self.position_mode not in {"none", None}:
            raise ValueError(f"Unknown temporal position encoding {self.position_mode!r}")

        diagnostics: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            tokens, block_diagnostics = block(
                tokens, memory, memory_mask, asset_mask, need_weights=need_weights
            )
            if need_weights:
                diagnostics.update(
                    {f"block_{index}_{name}": value for name, value in block_diagnostics.items()}
                )
        assets = self.final_norm(tokens[:, -1])
        mask_float = asset_mask.to(dtype=assets.dtype).unsqueeze(-1)
        context = (assets * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        new_memory, new_mask = self._update_memory(memory, memory_mask, context)
        return assets, context, new_memory, new_mask, diagnostics
