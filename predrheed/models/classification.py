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


@dataclass(frozen=True, slots=True)
class CNNOnlyConfig:

    input_shape: tuple[int, int, int] = (3, 32, 64)
    num_classes: int = 3
    feature_dropout: float = 0.3

    def __post_init__(self) -> None:
        shape = _validate_input_shape(self.input_shape)
        # The CNNOnly trunk hard-codes a three-channel first convolution.
        if shape[0] != 3:
            raise ValueError("CNNOnly supports exactly 3 input channels")
        _cnn_feature_shape(shape)
        _validate_positive_int("num_classes", self.num_classes)
        _validate_probability("feature_dropout", self.feature_dropout)


@dataclass(frozen=True, slots=True)
class TransformerOnlyConfig:

    input_shape: tuple[int, int, int] = (3, 32, 64)
    num_classes: int = 3
    patch_size: tuple[int, int] = (8, 8)
    projection_dim: int = 256
    num_heads: int = 8
    depth: int = 8
    mlp_ratio: int = 4
    attention_dropout: float = 0.1

    def __post_init__(self) -> None:
        _validate_input_shape(self.input_shape)
        _validate_positive_int("num_classes", self.num_classes)
        if not isinstance(self.patch_size, tuple) or len(self.patch_size) != 2:
            raise ValueError("patch_size must be a two-dimensional tuple")
        for value in self.patch_size:
            _validate_positive_int("patch_size dimension", value)
        projection_dim = _validate_positive_int("projection_dim", self.projection_dim)
        num_heads = _validate_positive_int("num_heads", self.num_heads)
        _validate_attention_dimensions(projection_dim, num_heads)
        _validate_positive_int("depth", self.depth)
        _validate_positive_int("mlp_ratio", self.mlp_ratio)
        _validate_probability("attention_dropout", self.attention_dropout)


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


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        img_size: tuple[int, int] = (32, 64),
        patch_size: tuple[int, int] = (8, 8),
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.proj(frames).flatten(2).transpose(1, 2)


class TransformerBlockVit(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.norm1(x)
        attended, _ = self.attn(x1, x1, x1)
        x = x + attended
        return x + self.mlp(self.norm2(x))


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


class CNNOnly(nn.Module):

    def __init__(
        self,
        num_classes: int = 3,
        dr: float = 0.3,
        *,
        config: CNNOnlyConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None and not isinstance(config, CNNOnlyConfig):
            raise TypeError("config must be a CNNOnlyConfig")
        if config is None:
            config = CNNOnlyConfig(
                num_classes=num_classes,
                feature_dropout=dr,
            )
        num_classes = config.num_classes
        dr = config.feature_dropout
        self.config = config

        self.pad1 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(1, 3))
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
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(_CNN_CHANNELS[-1], 512),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for pad, conv, bn, pool in (
            (self.pad1, self.conv1, self.bn1, self.pool1),
            (self.pad2, self.conv2, self.bn2, self.pool2),
            (self.pad3, self.conv3, self.bn3, self.pool3),
            (self.pad4, self.conv4, self.bn4, self.pool4),
        ):
            x = pad(x)
            x = F.relu(bn(conv(x)))
            x = self.dropout(pool(x))
        return self.classifier(self.global_pool(x).flatten(1))


class TransformerOnly(nn.Module):

    def __init__(
        self,
        img_size: tuple[int, int] = (32, 64),
        patch_size: tuple[int, int] = (8, 8),
        in_channels: int = 3,
        num_classes: int = 3,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        *,
        config: TransformerOnlyConfig | None = None,
    ) -> None:
        super().__init__()
        if config is not None and not isinstance(config, TransformerOnlyConfig):
            raise TypeError("config must be a TransformerOnlyConfig")
        if config is None:
            config = TransformerOnlyConfig(
                input_shape=(in_channels, *img_size),
                num_classes=num_classes,
                patch_size=patch_size,
                projection_dim=embed_dim,
                num_heads=num_heads,
                depth=depth,
                mlp_ratio=mlp_ratio,
                attention_dropout=dropout,
            )
        in_channels, height, width = config.input_shape
        img_size = (height, width)
        patch_size = config.patch_size
        num_classes = config.num_classes
        embed_dim = config.projection_dim
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        dropout = config.attention_dropout
        self.config = config

        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlockVit(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])


__all__ = [
    "CNNFeatureExtractor",
    "CNNOnly",
    "CNNOnlyConfig",
    "CNNTransformer",
    "CNNTransformerConfig",
    "PatchEmbedding",
    "TransformerBlock",
    "TransformerBlockVit",
    "TransformerOnly",
    "TransformerOnlyConfig",
]
