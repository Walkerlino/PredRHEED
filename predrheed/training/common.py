from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


PAPER_SEED = 2025


class TrainingContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OptimizerContract:

    name: Literal["Adam", "AdamW"]
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise TrainingContractError("learning_rate must be positive")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise TrainingContractError("betas must contain two values in [0, 1)")
        if self.eps <= 0:
            raise TrainingContractError("eps must be positive")
        if self.weight_decay < 0:
            raise TrainingContractError("weight_decay must be nonnegative")


def require_nonempty_loader(loader: object, name: str) -> int:
    try:
        length = len(loader)  # type: ignore[arg-type]
    except TypeError as error:
        raise TrainingContractError(f"{name} loader must define its length") from error
    if length <= 0:
        raise TrainingContractError(f"{name} loader must not be empty")
    return length


def resolve_device(device: str | torch.device | None = None) -> torch.device:

    if device is not None:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_reproducible_seeds(seed: int = PAPER_SEED) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_dict(model: torch.nn.Module, path: str | Path) -> Path:

    resolved = Path(path).expanduser().resolve(strict=False)
    if resolved.exists() and resolved.is_dir():
        raise TrainingContractError(
            "checkpoint path must identify a file, not a directory"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), resolved)
    return resolved


def load_state_dict(
    model: torch.nn.Module,
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> object:

    resolved_device = resolve_device(device)
    resolved_path = Path(path).expanduser().resolve(strict=False)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved_path}")
    state = torch.load(
        resolved_path,
        map_location=resolved_device,
        weights_only=True,
    )
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TrainingContractError("checkpoint must contain only a state dictionary")
    return model.load_state_dict(state, strict=strict)
