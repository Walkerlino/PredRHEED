from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


FRAMES_PER_TARGET_SECOND: Final[int] = 30
PRIMARY_PROPORTION: Final[float] = 0.60
SENSITIVITY_PROPORTIONS: Final[tuple[float, float]] = (0.65, 0.70)
UNRESOLVED_LABEL: Final[int] = -1


class DecisionContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecisionSummary:

    total: int
    resolved: int
    unresolved: int
    correct: int
    coverage: float
    overall_accuracy: float
    resolved_accuracy: float


def _validate_proportion(proportion: float) -> float:
    if isinstance(proportion, bool) or not isinstance(proportion, (int, float)):
        raise DecisionContractError("proportion must be a finite numeric value")
    value = float(proportion)
    if not np.isfinite(value) or not 0.5 < value <= 1.0:
        raise DecisionContractError("proportion must be greater than 0.5 and at most 1.0")
    return value


def resolve_target_seconds(
    frame_labels: np.ndarray,
    *,
    proportion: float = PRIMARY_PROPORTION,
    number_of_classes: int = 3,
) -> np.ndarray:

    labels = np.asarray(frame_labels)
    threshold = _validate_proportion(proportion)
    if labels.ndim < 1 or labels.shape[-1] != FRAMES_PER_TARGET_SECOND:
        raise DecisionContractError("each target second must contain exactly 30 labels")
    if (
        isinstance(number_of_classes, bool)
        or not isinstance(number_of_classes, int)
        or number_of_classes <= 1
    ):
        raise DecisionContractError("number_of_classes must be an integer greater than one")
    if not np.issubdtype(labels.dtype, np.integer):
        raise DecisionContractError("frame labels must have an integer dtype")
    if labels.size and (labels.min() < 0 or labels.max() >= number_of_classes):
        raise DecisionContractError("frame labels are outside the configured class range")

    counts = np.stack(
        [np.count_nonzero(labels == class_index, axis=-1) for class_index in range(number_of_classes)],
        axis=-1,
    )
    required = threshold * FRAMES_PER_TARGET_SECOND
    eligible = (counts >= required).reshape(-1, number_of_classes)
    resolved = np.full(eligible.shape[0], UNRESOLVED_LABEL, dtype=np.int64)
    has_eligible = np.any(eligible, axis=-1)
    if np.any(has_eligible):
        resolved[has_eligible] = np.argmax(eligible[has_eligible], axis=-1)
    return resolved.reshape(labels.shape[:-1])


def predicted_pattern_classes(logits: np.ndarray) -> np.ndarray:

    values = np.asarray(logits)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise DecisionContractError("logits must end in a class dimension")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise DecisionContractError("logits must contain only finite numeric values")
    return np.argmax(values, axis=-1).astype(np.int64, copy=False)


def summarize_decisions(
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> DecisionSummary:

    predicted = np.asarray(predicted_labels)
    reference = np.asarray(reference_labels)
    if predicted.shape != reference.shape:
        raise DecisionContractError("predicted and reference labels must have identical shapes")
    if not np.issubdtype(predicted.dtype, np.integer):
        raise DecisionContractError("predicted labels must have an integer dtype")
    if not np.issubdtype(reference.dtype, np.integer):
        raise DecisionContractError("reference labels must have an integer dtype")

    resolved_mask = reference != UNRESOLVED_LABEL
    total = int(reference.size)
    resolved = int(np.count_nonzero(resolved_mask))
    unresolved = total - resolved
    correct = int(np.count_nonzero(predicted[resolved_mask] == reference[resolved_mask]))
    coverage = float(resolved / total) if total else float("nan")
    overall_accuracy = float(correct / total) if total else float("nan")
    resolved_accuracy = float(correct / resolved) if resolved else float("nan")
    return DecisionSummary(
        total=total,
        resolved=resolved,
        unresolved=unresolved,
        correct=correct,
        coverage=coverage,
        overall_accuracy=overall_accuracy,
        resolved_accuracy=resolved_accuracy,
    )


def score_predicted_classes(
    predicted_labels: np.ndarray,
    frame_labels: np.ndarray,
    *,
    proportion: float = PRIMARY_PROPORTION,
    number_of_classes: int = 3,
) -> tuple[np.ndarray, DecisionSummary]:

    predicted = np.asarray(predicted_labels)
    frames = np.asarray(frame_labels)
    reference = resolve_target_seconds(
        frames,
        proportion=proportion,
        number_of_classes=number_of_classes,
    )
    if predicted.shape != frames.shape[:-1]:
        raise DecisionContractError(
            "predicted labels must identify every target-second frame-label sequence"
        )
    if not np.issubdtype(predicted.dtype, np.integer):
        raise DecisionContractError("predicted labels must have an integer dtype")
    if predicted.size and (predicted.min() < 0 or predicted.max() >= number_of_classes):
        raise DecisionContractError("predicted labels are outside the configured class range")

    flat_predicted = predicted.reshape(-1)
    flat_frames = frames.reshape(-1, FRAMES_PER_TARGET_SECOND)
    predicted_counts = np.count_nonzero(
        flat_frames == flat_predicted[:, None],
        axis=1,
    )
    correct_mask = predicted_counts >= _validate_proportion(proportion) * FRAMES_PER_TARGET_SECOND
    total = int(flat_predicted.size)
    resolved = int(np.count_nonzero(reference != UNRESOLVED_LABEL))
    correct = int(np.count_nonzero(correct_mask))
    unresolved = total - resolved
    coverage = float(resolved / total) if total else float("nan")
    overall_accuracy = float(correct / total) if total else float("nan")
    resolved_accuracy = float(correct / resolved) if resolved else float("nan")
    return reference, DecisionSummary(
        total=total,
        resolved=resolved,
        unresolved=unresolved,
        correct=correct,
        coverage=coverage,
        overall_accuracy=overall_accuracy,
        resolved_accuracy=resolved_accuracy,
    )


def evaluate_target_seconds(
    logits: np.ndarray,
    frame_labels: np.ndarray,
    *,
    proportion: float = PRIMARY_PROPORTION,
) -> tuple[np.ndarray, DecisionSummary]:

    logit_values = np.asarray(logits)
    predicted = predicted_pattern_classes(logit_values)
    return score_predicted_classes(
        predicted,
        frame_labels,
        proportion=proportion,
        number_of_classes=logit_values.shape[-1],
    )
