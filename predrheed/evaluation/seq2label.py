from __future__ import annotations

from typing import Final

import numpy as np
import torch

from .baselines import _validate_history
from .decisions import (
    DecisionContractError,
    FRAMES_PER_TARGET_SECOND,
    PRIMARY_PROPORTION,
    resolve_target_seconds,
)


FORECAST_STEPS: Final[int] = 5


class Seq2LabelEvaluationError(ValueError):
    pass


def threshold_targets(
    frame_labels: np.ndarray,
    *,
    proportion: float = PRIMARY_PROPORTION,
    number_of_classes: int = 3,
) -> np.ndarray:

    try:
        return resolve_target_seconds(
            frame_labels,
            proportion=proportion,
            number_of_classes=number_of_classes,
        )
    except DecisionContractError as error:
        raise Seq2LabelEvaluationError(str(error)) from error


def direct_seq2label_predictions(
    model: torch.nn.Module,
    history: np.ndarray,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 4,
) -> np.ndarray:

    frames = _validate_history(history)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise Seq2LabelEvaluationError("batch_size must be a positive integer")
    if frames.shape[0] == 0:
        raise Seq2LabelEvaluationError("history must contain at least one sample")
    if device is None:
        parameter = next(model.parameters(), None)
        resolved_device = (
            parameter.device if parameter is not None else torch.device("cpu")
        )
    else:
        resolved_device = torch.device(device)
    model.eval()
    predicted_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            batch = torch.from_numpy(frames[start : start + batch_size]).to(
                resolved_device
            )
            logits = model(batch)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[0] != batch.shape[0]
                or logits.shape[1] != FORECAST_STEPS
                or logits.shape[2] < 2
            ):
                raise Seq2LabelEvaluationError(
                    "model must return logits of shape (batch, 5, classes)"
                )
            if not torch.isfinite(logits).all():
                raise Seq2LabelEvaluationError("model logits must contain only finite values")
            predicted_parts.append(logits.argmax(-1).cpu())
    return torch.cat(predicted_parts, dim=0).numpy().astype(np.int64, copy=False)


def threshold_accuracy(
    predicted_labels: np.ndarray,
    frame_labels: np.ndarray,
    *,
    proportion: float = PRIMARY_PROPORTION,
    number_of_classes: int = 3,
) -> tuple[float, tuple[float, ...]]:

    predicted = np.asarray(predicted_labels)
    if predicted.ndim != 2 or predicted.shape[1] != FORECAST_STEPS:
        raise Seq2LabelEvaluationError(
            "predicted labels must have shape (samples, 5)"
        )
    if predicted.shape[0] == 0:
        raise Seq2LabelEvaluationError("predicted labels must not be empty")
    if not np.issubdtype(predicted.dtype, np.integer):
        raise Seq2LabelEvaluationError("predicted labels must have an integer dtype")
    frames = np.asarray(frame_labels)
    if frames.shape != predicted.shape + (FRAMES_PER_TARGET_SECOND,):
        raise Seq2LabelEvaluationError(
            "frame labels must have shape (samples, 5, 30) matching the predictions"
        )
    targets = threshold_targets(
        frames,
        proportion=proportion,
        number_of_classes=number_of_classes,
    )
    if predicted.size and (
        int(predicted.min()) < 0 or int(predicted.max()) >= number_of_classes
    ):
        raise Seq2LabelEvaluationError(
            "predicted labels are outside the configured class range"
        )
    per_horizon = tuple(
        float(value) for value in (predicted == targets).mean(axis=0)
    )
    return float(np.mean(per_horizon)), per_horizon
