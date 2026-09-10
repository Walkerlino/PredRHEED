from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from skimage.metrics import structural_similarity


FORECAST_STEPS: Final[int] = 5
ERROR_SCALE: Final[float] = 1000.0


class FrameMetricContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FrameMetricValues:

    ssim: float
    mse_times_1000: float
    mae_times_1000: float


@dataclass(frozen=True, slots=True)
class FrameMetricSummary:

    overall: FrameMetricValues
    per_horizon: tuple[FrameMetricValues, ...]


def _validate_frames(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=np.float32)
    truth = np.asarray(reference, dtype=np.float32)
    if pred.shape != truth.shape:
        raise FrameMetricContractError("predicted and reference frames must have identical shapes")
    if pred.ndim != 5 or pred.shape[1:] != (FORECAST_STEPS, 1, 128, 128):
        raise FrameMetricContractError("frames must have shape [samples, 5, 1, 128, 128]")
    if pred.shape[0] == 0:
        raise FrameMetricContractError("at least one forecast sample is required")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise FrameMetricContractError("frames must contain only finite values")
    if pred.min() < 0.0 or pred.max() > 1.0 or truth.min() < 0.0 or truth.max() > 1.0:
        raise FrameMetricContractError("frame intensities must lie in [0, 1]")
    return pred, truth


def _aggregate(predicted: np.ndarray, reference: np.ndarray) -> FrameMetricValues:
    ssim_values = [
        structural_similarity(truth_frame, pred_frame, data_range=1.0)
        for pred_frame, truth_frame in zip(predicted, reference)
    ]
    difference = predicted - reference
    return FrameMetricValues(
        ssim=float(np.mean(ssim_values)),
        mse_times_1000=float(np.mean(np.square(difference)) * ERROR_SCALE),
        mae_times_1000=float(np.mean(np.abs(difference)) * ERROR_SCALE),
    )


def compute_frame_metrics(predicted: np.ndarray, reference: np.ndarray) -> FrameMetricSummary:

    pred, truth = _validate_frames(predicted, reference)
    per_horizon = tuple(
        _aggregate(pred[:, horizon, 0], truth[:, horizon, 0])
        for horizon in range(FORECAST_STEPS)
    )
    pooled_pred = pred[:, :, 0].reshape(-1, 128, 128)
    pooled_truth = truth[:, :, 0].reshape(-1, 128, 128)
    return FrameMetricSummary(
        overall=_aggregate(pooled_pred, pooled_truth),
        per_horizon=per_horizon,
    )
