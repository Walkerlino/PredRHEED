from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


_CNN_CHANNELS = (64, 128, 256, 256)
_CNN_KERNEL_SIZES = ((1, 3), (2, 3), (2, 3), (1, 3))


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-Boolean integer")
    return value


def _validate_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a probability in [0, 1)")
    probability = float(value)
    if not 0.0 <= probability < 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1)")
    return probability


def _validate_input_shape(shape: object) -> tuple[int, int, int]:
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("input_shape must be a three-dimensional tuple")
    return tuple(_validate_positive_int("input_shape dimension", value) for value in shape)  # type: ignore[return-value]


def _cnn_feature_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:

    _, height, width = shape
    for kernel_height, kernel_width in _CNN_KERNEL_SIZES:
        height = height - kernel_height + 1
        width = (width + 4 - kernel_width + 1) // 2
        if height <= 0 or width <= 0:
            raise ValueError("input_shape is too small for the CNN feature extractor")
    return _CNN_CHANNELS[-1], height, width


def _validate_attention_dimensions(projection_dim: int, num_heads: int) -> None:
    if projection_dim % num_heads != 0:
        raise ValueError("projection_dim must be divisible by num_heads")


@dataclass(frozen=True, slots=True)
class CNNTransformerConfig:

    input_shape: tuple[int, int, int] = (3, 32, 64)
    num_classes: int = 3
    feature_dropout: float = 0.3
    projection_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 8

    def __post_init__(self) -> None:
        shape = _validate_input_shape(self.input_shape)
        _cnn_feature_shape(shape)
        _validate_positive_int("num_classes", self.num_classes)
        _validate_probability("feature_dropout", self.feature_dropout)
        projection_dim = _validate_positive_int("projection_dim", self.projection_dim)
        num_heads = _validate_positive_int("num_heads", self.num_heads)
        _validate_attention_dimensions(projection_dim, num_heads)
        _validate_positive_int("transformer_layers", self.transformer_layers)


class CNNFeatureExtractor(nn.Module):

    def __init__(self, in_channels: int = 3, dr: float = 0.3) -> None:
        super().__init__()
        self.pad1 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.pad2 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(2, 3))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((1, 2))

        self.pad3 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2, 3))
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((1, 2))

        self.pad4 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(1, 3))
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((1, 2))

        self.dropout = nn.Dropout(dr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(self.pool1(x))

        x = self.pad2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(self.pool2(x))

        x = self.pad3(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(self.pool3(x))

        x = self.pad4(x)
        x = F.relu(self.bn4(self.conv4(x)))
        return self.dropout(self.pool4(x))


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        # Figure 2(a) applies normalization after each residual addition.
        self.post_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.post_norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attn(normalized, normalized, normalized)
        tokens = self.post_norm1(tokens + attended)
        return self.post_norm2(tokens + self.mlp(self.norm2(tokens)))


class CNNTransformer(nn.Module):

    def __init__(
        self,
        num_classes: int = 3,
        dr: float = 0.3,
        projection_dim: int = 64,
        num_heads: int = 4,
        transformer_layers: int = 8,
        input_shape: tuple[int, int, int] = (3, 32, 64),
        *,
        config: CNNTransformerConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None and not isinstance(config, CNNTransformerConfig):
            raise TypeError("config must be a CNNTransformerConfig")
        if config is None:
            config = CNNTransformerConfig(
                input_shape=input_shape,
                num_classes=num_classes,
                feature_dropout=dr,
                projection_dim=projection_dim,
                num_heads=num_heads,
                transformer_layers=transformer_layers,
            )
        num_classes = config.num_classes
        dr = config.feature_dropout
        projection_dim = config.projection_dim
        num_heads = config.num_heads
        transformer_layers = config.transformer_layers
        input_shape = config.input_shape
        self.config = config

        self.cnn = CNNFeatureExtractor(in_channels=input_shape[0], dr=dr)
        self.projection_dim = projection_dim
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            cnn_out = self.cnn(dummy)
            _, channels, height, width = cnn_out.shape
            num_patches = height * width
            flatten_dim = num_patches * projection_dim

        self.patch_proj = nn.Linear(channels, projection_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, projection_dim) * 0.02
        )
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(
                projection_dim,
                num_heads,
                mlp_ratio=2,
                dropout=0.1,
            )
            for _ in range(transformer_layers)
        )
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        batch, channels, height, width = features.shape
        features = features.view(batch, channels, height * width).transpose(1, 2)
        features = self.patch_proj(features) + self.pos_embed
        for block in self.transformer_blocks:
            features = block(features)
        features = self.norm(features)
        features = self.dropout(features)
        return self.classifier(features.flatten(1))


__all__ = [
    "CNNFeatureExtractor",
    "CNNTransformer",
    "CNNTransformerConfig",
    "TransformerBlock",
]
