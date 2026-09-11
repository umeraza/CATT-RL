"""Reusable CATT normalization, gating, attention, and MLP layers."""

from __future__ import annotations

import math

import torch
from torch import nn


class SequenceNorm(nn.Module):
    """LayerNorm or per-sequence InstanceNorm for `[batch, tokens, channels]`."""

    def __init__(self, dimension: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "instance":
            self.norm: nn.Module = nn.InstanceNorm1d(
                dimension, affine=True, track_running_stats=False
            )
        elif kind == "layer":
            self.norm = nn.LayerNorm(dimension)
        else:
            raise ValueError(f"Unknown normalization {kind!r}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind == "instance":
            # InstanceNorm needs more than one spatial element in training. The
            # one-token fallback uses its exact affine-free normalization limit.
            if inputs.shape[1] == 1:
                weight = self.norm.weight.view(1, 1, -1)  # type: ignore[union-attr]
                bias = self.norm.bias.view(1, 1, -1)  # type: ignore[union-attr]
                return inputs * weight + bias
            return self.norm(inputs.transpose(1, 2)).transpose(1, 2)
        return self.norm(inputs)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10_000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div)
        encoding[:, 1::2] = torch.cos(positions * div[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, length: int, dtype: torch.dtype) -> torch.Tensor:
        if length > self.encoding.shape[0]:
            raise ValueError(f"Sequence length {length} exceeds positional limit")
        return self.encoding[:length].to(dtype=dtype)


class LightweightGatingUnit(nn.Module):
    """Residual LGU defined by the equations in the manuscript."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.w_z = nn.Linear(dimension, dimension)
        self.u_z = nn.Linear(dimension, dimension, bias=False)
        self.w_h = nn.Linear(dimension, dimension)
        self.u_h = nn.Linear(dimension, dimension, bias=False)
        self.bias_z = nn.Parameter(torch.zeros(dimension))

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.w_z(update) + self.u_z(residual) - self.bias_z)
        hidden = torch.sigmoid(self.w_h(update) + self.u_h(gate * residual))
        return (1.0 - gate) * residual + gate * hidden


class PositionWiseMLP(nn.Module):
    """Paper bottleneck 128→64→32 with projection back to the residual width."""

    def __init__(self, dimension: int, hidden_1: int, hidden_2: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, dimension),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CATTBlock(nn.Module):
    """Axial temporal/context attention with optional latent scaling and LGU."""

    def __init__(
        self,
        dimension: int,
        heads: int,
        hidden_1: int,
        hidden_2: int,
        dropout: float,
        normalization: str,
        normalization_placement: str,
        use_lgu: bool,
        cross_asset_mode: str,
        context_latents: int,
        vanilla: bool = False,
    ) -> None:
        super().__init__()
        norm_kind = "layer" if vanilla else normalization
        self.normalization_placement = "pre" if vanilla else normalization_placement
        self.use_lgu = bool(use_lgu and not vanilla)
        self.cross_asset_mode = cross_asset_mode
        self.temporal_norm = SequenceNorm(dimension, norm_kind)
        self.context_norm = SequenceNorm(dimension, norm_kind)
        self.ff_norm = nn.LayerNorm(dimension)
        self.temporal_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.context_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        if cross_asset_mode == "latent":
            if context_latents < 1:
                raise ValueError("context_latents must be positive in latent mode")
            self.latents = nn.Parameter(torch.empty(context_latents, dimension))
            nn.init.normal_(self.latents, std=0.02)
            self.context_output_attention = nn.MultiheadAttention(
                dimension, heads, dropout=dropout, batch_first=True
            )
            self.latent_norm = nn.LayerNorm(dimension)
        else:
            self.register_parameter("latents", None)
            self.context_output_attention = None
            self.latent_norm = None
        if self.use_lgu:
            self.temporal_lgu: nn.Module | None = LightweightGatingUnit(dimension)
            self.context_lgu: nn.Module | None = LightweightGatingUnit(dimension)
        else:
            self.temporal_lgu = None
            self.context_lgu = None
        if vanilla:
            # Choose the closest standard FFN width to the parameter count of
            # CATT's two LGUs plus its 3-layer PW-MLP.
            lgu_parameters = 4 * dimension * dimension + 3 * dimension
            pw_parameters = (
                dimension * hidden_1
                + hidden_1
                + hidden_1 * hidden_2
                + hidden_2
                + hidden_2 * dimension
                + dimension
            )
            target_parameters = 2 * lgu_parameters + pw_parameters
            vanilla_hidden = max(1, round((target_parameters - dimension) / (2 * dimension + 1)))
            self.feed_forward: nn.Module = nn.Sequential(
                nn.Linear(dimension, vanilla_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(vanilla_hidden, dimension),
            )
        else:
            self.feed_forward = PositionWiseMLP(dimension, hidden_1, hidden_2, dropout)
        self.dropout = nn.Dropout(dropout)

    def _residual_update(
        self, residual: torch.Tensor, update: torch.Tensor, gate: nn.Module | None
    ) -> torch.Tensor:
        update = self.dropout(update)
        return gate(residual, update) if gate is not None else residual + update

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        asset_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, assets, dimension = tokens.shape
        temporal = tokens.permute(0, 2, 1, 3).reshape(batch * assets, time, dimension)
        temporal_input = (
            self.temporal_norm(temporal) if self.normalization_placement == "pre" else temporal
        )
        temporal_update, temporal_weights = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        temporal = self._residual_update(temporal, temporal_update, self.temporal_lgu)
        if self.normalization_placement == "post":
            temporal = self.temporal_norm(temporal)
        tokens = temporal.reshape(batch, assets, time, dimension).permute(0, 2, 1, 3)
        current = tokens[:, -1]

        context = self.context_norm(current) if self.normalization_placement == "pre" else current
        key_values = torch.cat([memory, context], dim=1)
        padding_mask = torch.cat([~memory_mask, ~asset_mask], dim=1)
        if self.cross_asset_mode == "full":
            context_update, context_weights = self.context_attention(
                context,
                key_values,
                key_values,
                key_padding_mask=padding_mask,
                need_weights=need_weights,
                average_attn_weights=False,
            )
        else:
            latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
            latent_update, latent_weights = self.context_attention(
                latents,
                key_values,
                key_values,
                key_padding_mask=padding_mask,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            latents = self.latent_norm(latents + self.dropout(latent_update))
            context_update, context_weights = self.context_output_attention(
                context,
                latents,
                latents,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            if need_weights:
                context_weights = context_weights
            else:
                latent_weights = torch.empty(0, device=tokens.device)
        current = self._residual_update(current, context_update, self.context_lgu)
        if self.normalization_placement == "post":
            current = self.context_norm(current)
        current = current + self.dropout(self.feed_forward(self.ff_norm(current)))
        tokens = torch.cat([tokens[:, :-1], current.unsqueeze(1)], dim=1)

        diagnostics: dict[str, torch.Tensor] = {}
        if need_weights:
            diagnostics["temporal"] = temporal_weights.reshape(
                batch, assets, temporal_weights.shape[1], time, time
            )
            diagnostics["context"] = context_weights
            if self.cross_asset_mode == "latent":
                diagnostics["latent"] = latent_weights
        return tokens, diagnostics


def initialize_linear_layers(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
