from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .chunked_attention import patch_model_chunked


MEMORY_MAX_NORM = 10.0


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-Boolean integer")
    return value


def _validate_odd_kernel_size(value: object) -> int:
    kernel_size = _validate_positive_int("kernel_size", value)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd to preserve spatial dimensions")
    return kernel_size


def _validate_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability in [0, 1)")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1)")
    return probability


def _validate_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _validate_hidden_stack(
    hidden_dims: object,
    dropout_rates: object,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not isinstance(hidden_dims, tuple) or not hidden_dims:
        raise ValueError("hidden_dims must be a nonempty tuple")
    if not isinstance(dropout_rates, tuple):
        raise ValueError("dropout_rates must be a tuple")
    if len(hidden_dims) != len(dropout_rates):
        raise ValueError("hidden_dims and dropout_rates must have the same length")
    dimensions = tuple(
        _validate_positive_int("hidden_dims entry", dimension)
        for dimension in hidden_dims
    )
    dropouts = tuple(
        _validate_probability("dropout_rates entry", rate) for rate in dropout_rates
    )
    return dimensions, dropouts


def _validate_attention_channels(
    hidden_dims: tuple[int, ...], attention_reduction: int
) -> None:
    if any(channels // attention_reduction < 1 for channels in hidden_dims):
        raise ValueError("attention_reduction would reduce attention channels to zero")


@dataclass(frozen=True, slots=True)
class MSAMConvLSTMConfig:

    input_channels: int = 1
    hidden_dims: tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 3
    attention_reduction: int = 4
    spatial_reduction: int = 8
    attention_dropout: float = 0.1
    dropout_rates: tuple[float, ...] = (0.1, 0.1, 0.2)
    memory_max_norm: float | None = MEMORY_MAX_NORM

    def __post_init__(self) -> None:
        _validate_positive_int("input_channels", self.input_channels)
        hidden_dims, _ = _validate_hidden_stack(
            self.hidden_dims, self.dropout_rates
        )
        _validate_odd_kernel_size(self.kernel_size)
        attention_reduction = _validate_positive_int(
            "attention_reduction", self.attention_reduction
        )
        _validate_attention_channels(hidden_dims, attention_reduction)
        _validate_positive_int("spatial_reduction", self.spatial_reduction)
        _validate_probability("attention_dropout", self.attention_dropout)
        if self.memory_max_norm is not None:
            _validate_positive_float("memory_max_norm", self.memory_max_norm)


@dataclass(frozen=True, slots=True)
class SAConvLSTMConfig:

    input_channels: int = 1
    hidden_dims: tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 3
    attention_reduction: int = 4
    dropout_rates: tuple[float, ...] = (0.1, 0.1, 0.2)

    def __post_init__(self) -> None:
        _validate_positive_int("input_channels", self.input_channels)
        hidden_dims, _ = _validate_hidden_stack(
            self.hidden_dims, self.dropout_rates
        )
        _validate_odd_kernel_size(self.kernel_size)
        attention_reduction = _validate_positive_int(
            "attention_reduction", self.attention_reduction
        )
        _validate_attention_channels(hidden_dims, attention_reduction)


@dataclass(frozen=True, slots=True)
class SimVPConfig:

    input_channels: int = 1
    input_steps: int = 15
    hidden_dim: int = 32
    kernel_size: int = 3
    num_blocks: int = 4

    def __post_init__(self) -> None:
        _validate_positive_int("input_channels", self.input_channels)
        _validate_positive_int("input_steps", self.input_steps)
        _validate_positive_int("hidden_dim", self.hidden_dim)
        _validate_odd_kernel_size(self.kernel_size)
        _validate_positive_int("num_blocks", self.num_blocks)


def _validate_predictor_input(
    observed: torch.Tensor,
    input_channels: int,
    future_seq: object,
    *,
    expected_steps: int | None = None,
    spatial_reduction: int | None = None,
) -> int:
    if not isinstance(observed, torch.Tensor):
        raise TypeError("predictor input must be a torch.Tensor")
    steps = _validate_positive_int("future_seq", future_seq)
    if observed.ndim != 5:
        raise ValueError("predictor input must have five dimensions")
    batch, observed_steps, channels, height, width = observed.shape
    if batch <= 0 or observed_steps <= 0 or height <= 0 or width <= 0:
        raise ValueError("predictor input dimensions must be positive")
    if channels != input_channels:
        raise ValueError(
            f"expected {input_channels} input channel(s), got {channels}"
        )
    if expected_steps is not None and observed_steps != expected_steps:
        raise ValueError(
            f"expected input_steps={expected_steps}, got {observed_steps}"
        )
    if not torch.is_floating_point(observed):
        raise TypeError("predictor input must be floating point")
    if not bool(torch.isfinite(observed).all()):
        raise ValueError("predictor input must contain only finite values")
    if spatial_reduction is not None and (
        height < spatial_reduction or width < spatial_reduction
    ):
        raise ValueError(
            "spatial_reduction must not exceed either frame dimension"
        )
    return steps


def _zero_states(
    reference: torch.Tensor, hidden_dims: tuple[int, ...]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    batch, _, _, height, width = reference.shape
    hidden = [
        reference.new_zeros((batch, channels, height, width))
        for channels in hidden_dims
    ]
    cells = [
        reference.new_zeros((batch, channels, height, width))
        for channels in hidden_dims
    ]
    return hidden, cells


class ConvLSTMCell(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim * 4,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = hidden_state
        gates = self.conv(torch.cat((x, hidden), dim=1))
        input_gate, forget_gate, output_gate, candidate = torch.split(
            gates, self.hidden_dim, dim=1
        )
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_cell)
        return next_hidden, next_cell


class SelfAttentionMemory(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        spatial_reduction: int = 8,
        memory_max_norm: float | None = MEMORY_MAX_NORM,
        *,
        attention_dropout: float = 0.1,
        hidden_gate_input: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_gate_input, bool):
            raise TypeError("hidden_gate_input must be a Boolean")
        self.hidden_gate_input = hidden_gate_input
        self.channels = channels
        self.d_k = channels // reduction
        self.spatial_reduction = spatial_reduction
        self.memory_max_norm = memory_max_norm
        self.W_hq = nn.Conv2d(channels, self.d_k, kernel_size=1)
        self.W_hk = nn.Conv2d(channels, self.d_k, kernel_size=1)
        self.W_hv = nn.Conv2d(channels, channels, kernel_size=1)
        self.W_mk = nn.Conv2d(channels, self.d_k, kernel_size=1)
        self.W_mv = nn.Conv2d(channels, channels, kernel_size=1)
        self.W_z = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.attn_dropout = (
            nn.Dropout2d(attention_dropout)
            if attention_dropout > 0.0
            else nn.Identity()
        )
        gate_channels = channels * (2 if hidden_gate_input else 1)
        self.W_mg = nn.Conv2d(gate_channels, channels, kernel_size=1)
        self.W_mo = nn.Conv2d(gate_channels, channels, kernel_size=1)
        self.W_mi = nn.Conv2d(gate_channels, channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, hidden: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = hidden.shape
        reduced_size = (
            height // self.spatial_reduction,
            width // self.spatial_reduction,
        )
        reduced_hidden = F.adaptive_avg_pool2d(hidden, reduced_size)
        reduced_memory = F.adaptive_avg_pool2d(memory, reduced_size)
        reduced_height, reduced_width = reduced_hidden.shape[-2:]
        positions = reduced_height * reduced_width
        scale = np.sqrt(self.d_k)

        query_hidden = self.W_hq(reduced_hidden).view(
            batch, self.d_k, positions
        ).permute(0, 2, 1)
        key_hidden = self.W_hk(reduced_hidden).view(
            batch, self.d_k, positions
        )
        key_memory = self.W_mk(reduced_memory).view(
            batch, self.d_k, positions
        )
        value_hidden = self.W_hv(reduced_hidden).view(
            batch, channels, positions
        )
        value_memory = self.W_mv(reduced_memory).view(
            batch, channels, positions
        )

        attention_hidden = F.softmax(
            torch.bmm(query_hidden, key_hidden) / scale, dim=-1
        )
        attention_memory = F.softmax(
            torch.bmm(query_hidden, key_memory) / scale, dim=-1
        )

        attended_hidden = torch.bmm(
            value_hidden, attention_hidden.permute(0, 2, 1)
        ).view(batch, channels, reduced_height, reduced_width)
        attended_memory = torch.bmm(
            value_memory, attention_memory.permute(0, 2, 1)
        ).view(batch, channels, reduced_height, reduced_width)

        attended_hidden = F.interpolate(
            attended_hidden,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        attended_memory = F.interpolate(
            attended_memory,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        fused = self.W_z(torch.cat((attended_hidden, attended_memory), dim=1))
        fused = self.attn_dropout(fused)
        return self._update_memory(fused, hidden, memory)

    def _update_memory(
        self, fused: torch.Tensor, hidden: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Figure 3(c) supplies H_t directly to the MSAM gates alongside Z.
        gate_input = (
            torch.cat((fused, hidden), dim=1) if self.hidden_gate_input else fused
        )
        output_gate = torch.sigmoid(self.W_mo(gate_input))
        attended_hidden_state = output_gate * torch.tanh(self.W_mg(gate_input))
        input_gate = torch.sigmoid(self.W_mi(gate_input))
        next_memory = (
            (1 - input_gate) * memory
            + input_gate * attended_hidden_state
        )
        if self.memory_max_norm is not None:
            memory_norm = torch.norm(
                next_memory, dim=1, keepdim=True
            ).clamp(min=1e-8)
            next_memory = torch.where(
                memory_norm > self.memory_max_norm,
                next_memory * self.memory_max_norm / memory_norm,
                next_memory,
            )
        return attended_hidden_state, next_memory


class SelfAttentionOnly(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.d_k = channels // reduction
        self.W_q = nn.Conv2d(channels, self.d_k, kernel_size=1)
        self.W_k = nn.Conv2d(channels, self.d_k, kernel_size=1)
        self.W_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.W_o = nn.Conv2d(channels, channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = hidden.shape
        positions = height * width
        scale = np.sqrt(self.d_k)
        query = self.W_q(hidden).view(
            batch, self.d_k, positions
        ).permute(0, 2, 1)
        key = self.W_k(hidden).view(batch, self.d_k, positions)
        value = self.W_v(hidden).view(batch, channels, positions)
        attention = F.softmax(torch.bmm(query, key) / scale, dim=-1)
        attended = torch.bmm(value, attention.permute(0, 2, 1)).view(
            batch, channels, height, width
        )
        return self.W_o(attended)


SpatialSelfAttention = SelfAttentionOnly


class MSAMConvLSTM(nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: list[int] = [64, 128, 128],
        kernel_size: int = 3,
        num_layers: int = 3,
        attention_reduction: int = 4,
        dropout_rates: list[float] = [0.1, 0.1, 0.2],
        *,
        config: MSAMConvLSTMConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None and not isinstance(config, MSAMConvLSTMConfig):
            raise TypeError("config must be an MSAMConvLSTMConfig")
        if config is None:
            dimensions = tuple(hidden_dims)
            dropouts = tuple(dropout_rates)
            validated_layers = _validate_positive_int("num_layers", num_layers)
            if validated_layers != len(dimensions):
                raise ValueError("num_layers must match the number of hidden_dims")
            self.config = MSAMConvLSTMConfig(
                input_channels=input_channels,
                hidden_dims=dimensions,
                kernel_size=kernel_size,
                attention_reduction=attention_reduction,
                dropout_rates=dropouts,
            )
        else:
            if (
                input_channels != 1
                or tuple(hidden_dims) != (64, 128, 128)
                or kernel_size != 3
                or num_layers != 3
                or attention_reduction != 4
                or tuple(dropout_rates) != (0.1, 0.1, 0.2)
            ):
                raise ValueError(
                    "config cannot be combined with nondefault constructor arguments"
                )
            self.config = config

        self.num_layers = len(self.config.hidden_dims)
        self.hidden_dims = list(self.config.hidden_dims)
        self.encoder_cells = nn.ModuleList()
        self.encoder_sam = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        self.encoder_drops = nn.ModuleList()
        self.decoder_cells = nn.ModuleList()
        self.decoder_sam = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()
        self.decoder_drops = nn.ModuleList()

        for index in range(self.num_layers):
            layer_input_channels = (
                self.config.input_channels
                if index == 0
                else self.hidden_dims[index - 1]
            )
            hidden_channels = self.hidden_dims[index]
            self.encoder_cells.append(
                ConvLSTMCell(
                    layer_input_channels,
                    hidden_channels,
                    self.config.kernel_size,
                )
            )
            self.encoder_sam.append(
                SelfAttentionMemory(
                    hidden_channels,
                    self.config.attention_reduction,
                    self.config.spatial_reduction,
                    self.config.memory_max_norm,
                    attention_dropout=self.config.attention_dropout,
                    hidden_gate_input=True,
                )
            )
            self.encoder_bns.append(nn.BatchNorm2d(hidden_channels))
            self.encoder_drops.append(
                nn.Dropout2d(self.config.dropout_rates[index])
            )
            self.decoder_cells.append(
                ConvLSTMCell(
                    layer_input_channels,
                    hidden_channels,
                    self.config.kernel_size,
                )
            )
            self.decoder_sam.append(
                SelfAttentionMemory(
                    hidden_channels,
                    self.config.attention_reduction,
                    self.config.spatial_reduction,
                    self.config.memory_max_norm,
                    attention_dropout=self.config.attention_dropout,
                    hidden_gate_input=True,
                )
            )
            self.decoder_bns.append(nn.BatchNorm2d(hidden_channels))
            self.decoder_drops.append(
                nn.Dropout2d(self.config.dropout_rates[index])
            )
        self.output_conv = nn.Conv2d(
            self.config.hidden_dims[-1], self.config.input_channels, kernel_size=1
        )

    def _advance(
        self,
        inputs: torch.Tensor,
        hidden: list[torch.Tensor],
        cells: list[torch.Tensor],
        memories: list[torch.Tensor],
        recurrent_layers: nn.ModuleList,
        attention_layers: nn.ModuleList,
        batch_norms: nn.ModuleList,
        dropouts: nn.ModuleList,
    ) -> torch.Tensor:
        layer_input = inputs
        for index in range(len(recurrent_layers)):
            hidden[index], cells[index] = recurrent_layers[index](
                layer_input, (hidden[index], cells[index])
            )
            features, memories[index] = attention_layers[index](
                hidden[index], memories[index]
            )
            # Figure 3(b) carries raw ConvLSTM h/c across time, not layer output.
            layer_input = dropouts[index](batch_norms[index](features))
        return layer_input

    def forward(
        self, x: torch.Tensor, future_seq: int = 5
    ) -> torch.Tensor:
        steps = _validate_predictor_input(
            x,
            self.config.input_channels,
            future_seq,
            spatial_reduction=self.config.spatial_reduction,
        )
        hidden, cells = _zero_states(x, tuple(self.hidden_dims))
        batch, _, _, height, width = x.shape
        memories = [
            x.new_zeros((batch, channels, height, width))
            for channels in self.hidden_dims
        ]

        for time_index in range(x.shape[1]):
            self._advance(
                x[:, time_index],
                hidden,
                cells,
                memories,
                self.encoder_cells,
                self.encoder_sam,
                self.encoder_bns,
                self.encoder_drops,
            )

        decoder_hidden = [state.clone() for state in hidden]
        decoder_cells = [state.clone() for state in cells]
        decoder_memories = [state.clone() for state in memories]
        current = x[:, -1]
        predictions: list[torch.Tensor] = []
        for _ in range(steps):
            features = self._advance(
                current,
                decoder_hidden,
                decoder_cells,
                decoder_memories,
                self.decoder_cells,
                self.decoder_sam,
                self.decoder_bns,
                self.decoder_drops,
            )
            current = torch.sigmoid(self.output_conv(features))
            predictions.append(current)
        return torch.stack(predictions, dim=1)


class SAConvLSTM(nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: list[int] = [64, 128, 128],
        kernel_size: int = 3,
        num_layers: int = 3,
        attention_reduction: int = 4,
        dropout_rates: list[float] = [0.1, 0.1, 0.2],
        *,
        config: SAConvLSTMConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None and not isinstance(config, SAConvLSTMConfig):
            raise TypeError("config must be an SAConvLSTMConfig")
        if config is None:
            dimensions = tuple(hidden_dims)
            dropouts = tuple(dropout_rates)
            validated_layers = _validate_positive_int("num_layers", num_layers)
            if validated_layers != len(dimensions):
                raise ValueError("num_layers must match the number of hidden_dims")
            self.config = SAConvLSTMConfig(
                input_channels=input_channels,
                hidden_dims=dimensions,
                kernel_size=kernel_size,
                attention_reduction=attention_reduction,
                dropout_rates=dropouts,
            )
        else:
            if (
                input_channels != 1
                or tuple(hidden_dims) != (64, 128, 128)
                or kernel_size != 3
                or num_layers != 3
                or attention_reduction != 4
                or tuple(dropout_rates) != (0.1, 0.1, 0.2)
            ):
                raise ValueError(
                    "config cannot be combined with nondefault constructor arguments"
                )
            self.config = config

        self.num_layers = len(self.config.hidden_dims)
        self.hidden_dims = list(self.config.hidden_dims)
        self.encoder_cells = nn.ModuleList()
        self.encoder_sa = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        self.encoder_drops = nn.ModuleList()
        self.decoder_cells = nn.ModuleList()
        self.decoder_sa = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()
        self.decoder_drops = nn.ModuleList()

        for index in range(self.num_layers):
            layer_input_channels = (
                self.config.input_channels
                if index == 0
                else self.hidden_dims[index - 1]
            )
            hidden_channels = self.hidden_dims[index]
            self.encoder_cells.append(
                ConvLSTMCell(
                    layer_input_channels,
                    hidden_channels,
                    self.config.kernel_size,
                )
            )
            self.encoder_sa.append(
                SelfAttentionMemory(
                    hidden_channels,
                    self.config.attention_reduction,
                    1,
                    None,
                    attention_dropout=0.0,
                )
            )
            self.encoder_bns.append(nn.BatchNorm2d(hidden_channels))
            self.encoder_drops.append(
                nn.Dropout2d(self.config.dropout_rates[index])
            )
            self.decoder_cells.append(
                ConvLSTMCell(
                    layer_input_channels,
                    hidden_channels,
                    self.config.kernel_size,
                )
            )
            self.decoder_sa.append(
                SelfAttentionMemory(
                    hidden_channels,
                    self.config.attention_reduction,
                    1,
                    None,
                    attention_dropout=0.0,
                )
            )
            self.decoder_bns.append(nn.BatchNorm2d(hidden_channels))
            self.decoder_drops.append(
                nn.Dropout2d(self.config.dropout_rates[index])
            )
        self.output_conv = nn.Conv2d(
            self.config.hidden_dims[-1], self.config.input_channels, kernel_size=1
        )
        patch_model_chunked(self)

    def _advance(
        self,
        inputs: torch.Tensor,
        hidden: list[torch.Tensor],
        cells: list[torch.Tensor],
        memories: list[torch.Tensor],
        recurrent_layers: nn.ModuleList,
        attention_layers: nn.ModuleList,
        batch_norms: nn.ModuleList,
        dropouts: nn.ModuleList,
    ) -> torch.Tensor:
        layer_input = inputs
        for index in range(len(recurrent_layers)):
            hidden[index], cells[index] = recurrent_layers[index](
                layer_input, (hidden[index], cells[index])
            )
            hidden[index], memories[index] = attention_layers[index](
                hidden[index], memories[index]
            )
            hidden[index] = dropouts[index](batch_norms[index](hidden[index]))
            layer_input = hidden[index]
        return layer_input

    def forward(
        self, x: torch.Tensor, future_seq: int = 5
    ) -> torch.Tensor:
        steps = _validate_predictor_input(
            x,
            self.config.input_channels,
            future_seq,
        )
        hidden, cells = _zero_states(x, tuple(self.hidden_dims))
        batch, _, _, height, width = x.shape
        memories = [
            x.new_zeros((batch, channels, height, width))
            for channels in self.hidden_dims
        ]

        for time_index in range(x.shape[1]):
            self._advance(
                x[:, time_index],
                hidden,
                cells,
                memories,
                self.encoder_cells,
                self.encoder_sa,
                self.encoder_bns,
                self.encoder_drops,
            )

        decoder_hidden = [state.clone() for state in hidden]
        decoder_cells = [state.clone() for state in cells]
        decoder_memories = [state.clone() for state in memories]
        current = x[:, -1]
        predictions: list[torch.Tensor] = []
        for _ in range(steps):
            features = self._advance(
                current,
                decoder_hidden,
                decoder_cells,
                decoder_memories,
                self.decoder_cells,
                self.decoder_sa,
                self.decoder_bns,
                self.decoder_drops,
            )
            current = torch.sigmoid(self.output_conv(features))
            predictions.append(current)
        return torch.stack(predictions, dim=1)


class ConvBlock(nn.Module):

    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels, channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.act(features + self.net(features))


_ResidualConvBlock = ConvBlock


class SimVP(nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        input_len: int = 15,
        hidden_dim: int = 32,
        num_blocks: int = 4,
        *,
        config: SimVPConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None and not isinstance(config, SimVPConfig):
            raise TypeError("config must be a SimVPConfig")
        if config is None:
            self.config = SimVPConfig(
                input_channels=input_channels,
                input_steps=input_len,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
            )
        else:
            if (
                input_channels != 1
                or input_len != 15
                or hidden_dim != 32
                or num_blocks != 4
            ):
                raise ValueError(
                    "config cannot be combined with nondefault constructor arguments"
                )
            self.config = config

        self.input_channels = self.config.input_channels
        self.input_len = self.config.input_steps
        self.hidden_dim = self.config.hidden_dim
        self.kernel_size = self.config.kernel_size

        padding = self.kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
        )
        temporal_channels = self.input_len * self.hidden_dim
        self.translator = nn.Sequential(
            *(
                ConvBlock(temporal_channels, kernel_size=self.kernel_size)
                for _ in range(self.config.num_blocks)
            )
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_dim, self.input_channels, kernel_size=1
            ),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, future_seq: int = 5
    ) -> torch.Tensor:
        steps = _validate_predictor_input(
            x,
            self.input_channels,
            future_seq,
            expected_steps=self.input_len,
        )
        batch, observed_steps, channels, height, width = x.shape
        encoded = self.encoder(
            x.reshape(batch * observed_steps, channels, height, width)
        )
        _, latent_channels, latent_height, latent_width = encoded.shape
        temporal_features = encoded.reshape(
            batch,
            observed_steps * latent_channels,
            latent_height,
            latent_width,
        )
        translated = self.translator(temporal_features).reshape(
            batch,
            observed_steps,
            latent_channels,
            latent_height,
            latent_width,
        )
        next_latent = translated[:, -1]
        predictions: list[torch.Tensor] = []
        for _ in range(steps):
            next_frame = self.decoder(next_latent)
            predictions.append(next_frame)
            next_latent = self.encoder(next_frame)
        return torch.stack(predictions, dim=1)


__all__ = [
    "ConvBlock",
    "ConvLSTMCell",
    "MEMORY_MAX_NORM",
    "MSAMConvLSTM",
    "MSAMConvLSTMConfig",
    "SAConvLSTM",
    "SAConvLSTMConfig",
    "SelfAttentionMemory",
    "SelfAttentionOnly",
    "SimVP",
    "SimVPConfig",
    "SpatialSelfAttention",
]
