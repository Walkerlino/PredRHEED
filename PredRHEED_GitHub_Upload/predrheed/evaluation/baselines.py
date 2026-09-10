from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np

from .decisions import PRIMARY_PROPORTION, predicted_pattern_classes, resolve_target_seconds


INPUT_STEPS: Final[int] = 15
FORECAST_STEPS: Final[int] = 5


class BaselineContractError(ValueError):
    pass


def _validate_history(history: np.ndarray) -> np.ndarray:
    frames = np.asarray(history, dtype=np.float32)
    if frames.ndim != 5 or frames.shape[1:] != (INPUT_STEPS, 1, 128, 128):
        raise BaselineContractError("history must have shape [samples, 15, 1, 128, 128]")
    if not np.all(np.isfinite(frames)):
        raise BaselineContractError("history must contain only finite values")
    if frames.size and (frames.min() < 0.0 or frames.max() > 1.0):
        raise BaselineContractError("history intensities must lie in [0, 1]")
    return frames


def last_frame_persistence(history: np.ndarray) -> np.ndarray:

    frames = _validate_history(history)
    return np.repeat(frames[:, -1:, :, :, :], FORECAST_STEPS, axis=1)


def label_persistence_oracle(
    last_observed_second_frame_labels: np.ndarray,
    *,
    proportion: float = PRIMARY_PROPORTION,
    number_of_classes: int = 3,
) -> np.ndarray:

    labels = resolve_target_seconds(
        last_observed_second_frame_labels,
        proportion=proportion,
        number_of_classes=number_of_classes,
    )
    return np.repeat(labels[..., None], FORECAST_STEPS, axis=-1)


def current_frame_classifier(
    final_observed_frames: np.ndarray,
    classifier: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:

    frames = np.asarray(final_observed_frames, dtype=np.float32)
    if frames.ndim < 4:
        raise BaselineContractError(
            "final observed frames must include sample, channel, and image dimensions"
        )
    logits = np.asarray(classifier(frames))
    labels = predicted_pattern_classes(logits)
    if labels.shape != frames.shape[:-3]:
        raise BaselineContractError("classifier output does not match the input sample dimensions")
    return np.repeat(labels[..., None], FORECAST_STEPS, axis=-1)
