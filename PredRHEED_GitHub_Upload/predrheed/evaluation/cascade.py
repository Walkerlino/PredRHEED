from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from .decisions import predicted_pattern_classes


INPUT_STEPS: Final[int] = 15
FORECAST_STEPS: Final[int] = 5
PREDICTOR_FRAME_SHAPE: Final[tuple[int, int]] = (128, 128)
CLASSIFIER_FRAME_SHAPE: Final[tuple[int, int]] = (32, 64)


class CascadeContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CascadeOutput:

    predicted_frames: torch.Tensor
    classifier_logits: torch.Tensor
    predicted_labels: np.ndarray


def _validate_history(history: torch.Tensor) -> None:
    if history.ndim != 5 or tuple(history.shape[1:]) != (INPUT_STEPS, 1, 128, 128):
        raise CascadeContractError("history must have shape [batch, 15, 1, 128, 128]")
    if not history.is_floating_point():
        raise CascadeContractError("history must have a floating-point dtype")
    if history.shape[0] == 0:
        raise CascadeContractError("history must contain at least one sample")
    if not bool(torch.isfinite(history).all()):
        raise CascadeContractError("history must contain only finite values")
    if history.numel() and (float(history.min()) < 0.0 or float(history.max()) > 1.0):
        raise CascadeContractError("history intensities must lie in [0, 1]")


def prepare_predicted_frames_for_classifier(predicted_frames: torch.Tensor) -> torch.Tensor:

    if predicted_frames.ndim != 5 or tuple(predicted_frames.shape[1:]) != (5, 1, 128, 128):
        raise CascadeContractError("predicted frames must have shape [batch, 5, 1, 128, 128]")
    try:
        import cv2
    except ImportError as error:
        raise CascadeContractError("OpenCV is required for source-compatible linear resize") from error

    source_frames = (
        predicted_frames.detach().cpu().numpy().astype(np.float16).astype(np.float32)
    )
    flat = np.ascontiguousarray(source_frames).reshape(-1, 128, 128)
    prepared = np.empty((flat.shape[0], 3, 32, 64), dtype=np.float32)
    for index, frame in enumerate(flat):
        resized = cv2.resize(frame, (64, 32), interpolation=cv2.INTER_LINEAR)
        prepared[index] = np.repeat(resized[np.newaxis], 3, axis=0)
    return torch.from_numpy(prepared).to(device=predicted_frames.device)


def run_five_second_cascade(
    predictor: torch.nn.Module,
    classifier: torch.nn.Module,
    history: torch.Tensor,
    *,
    classifier_batch_size: int = 64,
) -> CascadeOutput:

    _validate_history(history)
    if isinstance(classifier_batch_size, bool) or classifier_batch_size <= 0:
        raise CascadeContractError("classifier_batch_size must be a positive integer")
    predictor.eval()
    classifier.eval()
    with torch.no_grad():
        predicted = predictor(history, future_seq=FORECAST_STEPS)
        if not isinstance(predicted, torch.Tensor):
            raise CascadeContractError("predictor must return a tensor")
        expected = (history.shape[0], FORECAST_STEPS, 1, 128, 128)
        if tuple(predicted.shape) != expected:
            raise CascadeContractError(f"predictor output must have shape {expected}")
        if not predicted.is_floating_point() or not bool(torch.isfinite(predicted).all()):
            raise CascadeContractError("predictor output must contain finite floating-point values")
        if float(predicted.min()) < 0.0 or float(predicted.max()) > 1.0:
            raise CascadeContractError("predictor output intensities must lie in [0, 1]")
        classifier_input = prepare_predicted_frames_for_classifier(predicted)
        logits_parts = [
            classifier(classifier_input[start : start + classifier_batch_size])
            for start in range(0, classifier_input.shape[0], classifier_batch_size)
        ]
        logits_flat = torch.cat(logits_parts, dim=0)
    if logits_flat.ndim != 2 or logits_flat.shape[0] != history.shape[0] * FORECAST_STEPS:
        raise CascadeContractError("classifier output must contain one logit vector per predicted second")
    logits = logits_flat.reshape(history.shape[0], FORECAST_STEPS, -1)
    labels = predicted_pattern_classes(logits.detach().cpu().numpy())
    return CascadeOutput(
        predicted_frames=predicted,
        classifier_logits=logits,
        predicted_labels=labels,
    )
