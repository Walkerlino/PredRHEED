from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-Boolean integer")
    return value


@dataclass(frozen=True, slots=True)
class DirectSeq2LabelConfig:

    input_channels: int = 1
    num_classes: int = 3
    horizons: int = 5
    embed_dim: int = 256
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)

    def __post_init__(self) -> None:
        _validate_positive_int("input_channels", self.input_channels)
        if (
            isinstance(self.num_classes, bool)
            or not isinstance(self.num_classes, int)
            or self.num_classes < 2
        ):
            raise ValueError("num_classes must be an integer of at least two")
        _validate_positive_int("horizons", self.horizons)
        _validate_positive_int("embed_dim", self.embed_dim)
        if not isinstance(self.encoder_channels, tuple) or not self.encoder_channels:
            raise ValueError("encoder_channels must be a nonempty tuple")
        for channels in self.encoder_channels:
            _validate_positive_int("encoder_channels entry", channels)


class DirectSeq2Label(nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 3,
        horizons: int = 5,
        embed_dim: int = 256,
        *,
        config: DirectSeq2LabelConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None and not isinstance(config, DirectSeq2LabelConfig):
            raise TypeError("config must be a DirectSeq2LabelConfig")
        if config is None:
            config = DirectSeq2LabelConfig(
                input_channels=input_channels,
                num_classes=num_classes,
                horizons=horizons,
                embed_dim=embed_dim,
            )
        self.config = config

        channels = list(config.encoder_channels)
        layers: list[nn.Module] = []
        previous_channels = config.input_channels
        for out_channels in channels:
            layers += [
                nn.Conv2d(previous_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            previous_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(channels[-1], config.embed_dim, batch_first=True)
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.embed_dim, config.num_classes)
                for _ in range(config.horizons)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "input must have five dimensions (batch, time, channels, height, width)"
            )
        batch_size, time_steps, channels, height, width = x.shape
        features = self.encoder(x.reshape(batch_size * time_steps, channels, height, width))
        features = self.gap(features).flatten(1).view(batch_size, time_steps, -1)
        _, (hidden_state, _) = self.lstm(features)
        last_hidden = hidden_state[-1]
        return torch.stack([head(last_hidden) for head in self.heads], dim=1)
