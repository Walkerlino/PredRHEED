from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_structural_similarity

from .datasets import _load_npz_arrays, _validate_frames
from .schema import SchemaValidationError


FRAMES_PER_SECOND: Final[int] = 30
INPUT_STEPS: Final[int] = 15
FORECAST_STEPS: Final[int] = 5
TOTAL_STEPS: Final[int] = INPUT_STEPS + FORECAST_STEPS
PREDICTOR_FRAME_SHAPE: Final[tuple[int, int]] = (128, 128)
CLASSIFIER_SIZE_WH: Final[tuple[int, int]] = (64, 32)

SSIM_KEEP_THRESHOLD: Final[float] = 0.75
PIXEL_STD_LOWER: Final[float] = 0.03
PIXEL_STD_UPPER: Final[float] = 0.06

_ADAPTATION_KEYS = frozenset(
    {
        "frames",
        "labels",
        "origin_tags",
        "second_indices",
        "validation_frames",
        "validation_labels",
    }
)


class AdaptationContractError(ValueError):
    pass


def _validated_seconds(
    values: Sequence[int] | np.ndarray,
    name: str,
    *,
    require_unique: bool,
) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise AdaptationContractError(
            f"{name} must be a one-dimensional sequence of integers"
        ) from error
    seconds: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise AdaptationContractError(
                f"{name} must contain only non-Boolean integers"
            )
        second = int(value)
        if second < 0:
            raise AdaptationContractError(f"{name} must contain only non-negative seconds")
        if second > np.iinfo(np.int64).max:
            raise AdaptationContractError(f"{name} values must fit in int64")
        seconds.append(second)
    if require_unique and len(set(seconds)) != len(seconds):
        raise AdaptationContractError(f"{name} must not contain duplicate seconds")
    return tuple(seconds)


def _validated_windows(windows: np.ndarray) -> np.ndarray:
    validated = np.asarray(windows)
    if validated.ndim != 2 or validated.shape[1] != TOTAL_STEPS:
        raise AdaptationContractError(f"windows must have shape (N, {TOTAL_STEPS})")
    if not np.issubdtype(validated.dtype, np.integer):
        raise AdaptationContractError("windows must contain integer second indices")
    if validated.size:
        if int(validated.min()) < 0:
            raise AdaptationContractError("windows must not contain negative second indices")
        if int(validated.max()) > np.iinfo(np.int64).max:
            raise AdaptationContractError("window second indices must fit in int64")
    return validated.astype(np.int64, copy=False)


def _validate_label_array(
    labels: np.ndarray, length: int, name: str, *, num_classes: int = 3
) -> np.ndarray:
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise SchemaValidationError(f"{name} must be a one-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise SchemaValidationError(f"{name} must have an integer dtype")
    if len(labels) != length:
        raise SchemaValidationError(f"{name} length must match its frame array")
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise SchemaValidationError(
            f"{name} must be zero-based in [0, {num_classes}) — the source "
            "artifacts stored 1-based labels which must be shifted down once"
        )
    return labels.astype(np.int64)


@dataclass(frozen=True, slots=True)
class AdaptationBundle:

    frames: np.ndarray
    labels: np.ndarray
    origin_tags: np.ndarray
    second_indices: np.ndarray
    validation_frames: np.ndarray
    validation_labels: np.ndarray

    def __post_init__(self) -> None:
        frames = _validate_frames(self.frames, (None, 1, 32, 64), "adaptation training")
        labels = _validate_label_array(self.labels, len(frames), "adaptation labels")
        if not isinstance(self.origin_tags, np.ndarray) or self.origin_tags.ndim != 1:
            raise SchemaValidationError("origin_tags must be a one-dimensional array")
        if not np.issubdtype(self.origin_tags.dtype, np.integer):
            raise SchemaValidationError("origin_tags must have an integer dtype")
        if len(self.origin_tags) != len(frames):
            raise SchemaValidationError("origin_tags length must match frames")
        if not set(np.unique(self.origin_tags).tolist()) <= {0, 1, 2}:
            raise SchemaValidationError("origin_tags must contain only {0, 1, 2}")
        if not isinstance(self.second_indices, np.ndarray) or self.second_indices.ndim != 1:
            raise SchemaValidationError("second_indices must be a one-dimensional array")
        if not np.issubdtype(self.second_indices.dtype, np.integer):
            raise SchemaValidationError("second_indices must have an integer dtype")
        if len(self.second_indices) != len(frames):
            raise SchemaValidationError("second_indices length must match frames")
        if self.second_indices.size and self.second_indices.min() < 0:
            raise SchemaValidationError("second_indices must be non-negative")
        if (
            self.second_indices.size
            and int(self.second_indices.max()) > np.iinfo(np.int64).max
        ):
            raise SchemaValidationError("second_indices values must fit in int64")
        validation_frames = _validate_frames(
            self.validation_frames, (None, 1, 32, 64), "adaptation validation"
        )
        validation_labels = _validate_label_array(
            self.validation_labels, len(validation_frames), "adaptation validation labels"
        )
        object.__setattr__(self, "frames", frames.astype(np.float32))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "origin_tags", self.origin_tags.astype(np.uint8))
        object.__setattr__(self, "second_indices", self.second_indices.astype(np.int64))
        object.__setattr__(self, "validation_frames", validation_frames.astype(np.float32))
        object.__setattr__(self, "validation_labels", validation_labels)

    @classmethod
    def from_npz(cls, path: str | Path) -> "AdaptationBundle":
        arrays = _load_npz_arrays(Path(path), _ADAPTATION_KEYS, _ADAPTATION_KEYS)
        return cls(
            frames=arrays["frames"],
            labels=arrays["labels"],
            origin_tags=arrays["origin_tags"],
            second_indices=arrays["second_indices"],
            validation_frames=arrays["validation_frames"],
            validation_labels=arrays["validation_labels"],
        )

    def to_npz(self, path: str | Path) -> Path:

        resolved = Path(path).expanduser().resolve(strict=False)
        if resolved.exists() and resolved.is_dir():
            raise SchemaValidationError("bundle path must identify a file, not a directory")
        if resolved.suffix != ".npz":
            resolved = resolved.with_name(resolved.name + ".npz")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            resolved,
            frames=self.frames,
            labels=self.labels,
            origin_tags=self.origin_tags,
            second_indices=self.second_indices,
            validation_frames=self.validation_frames,
            validation_labels=self.validation_labels,
        )
        return resolved


def assert_no_test_second_leakage(
    second_indices: Sequence[int] | np.ndarray,
    test_seconds: Sequence[int] | np.ndarray,
) -> None:

    pool = set(
        _validated_seconds(second_indices, "second_indices", require_unique=False)
    )
    test = set(_validated_seconds(test_seconds, "test_seconds", require_unique=True))
    leak = pool & test
    if leak:
        raise AdaptationContractError(
            f"training-pool leakage into test: {len(leak)} seconds overlap"
        )


def build_sliding_windows(seconds: Sequence[int]) -> np.ndarray:

    secs = sorted(_validated_seconds(seconds, "seconds", require_unique=True))
    contiguous: list[tuple[int, int]] = []
    if secs:
        run_start = previous = secs[0]
        for second in secs[1:]:
            if second == previous + 1:
                previous = second
            else:
                contiguous.append((run_start, previous))
                run_start = previous = second
        contiguous.append((run_start, previous))
    windows: list[list[int]] = []
    for run_start, run_end in contiguous:
        run_length = run_end - run_start + 1
        if run_length >= TOTAL_STEPS:
            for start in range(run_start, run_end - TOTAL_STEPS + 2):
                windows.append(list(range(start, start + TOTAL_STEPS)))
    return np.asarray(windows, dtype=np.int64).reshape(-1, TOTAL_STEPS)


def build_target_per_frame(
    windows: np.ndarray, frame_labels_30fps: np.ndarray
) -> np.ndarray:

    windows = _validated_windows(windows)
    labels = np.asarray(frame_labels_30fps)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise AdaptationContractError(
            "frame_labels_30fps must be a one-dimensional integer array"
        )
    if labels.size and (labels.min() < 0 or labels.max() > 2):
        raise AdaptationContractError(
            "frame_labels_30fps must contain only the classes {0, 1, 2}"
        )
    if len(windows):
        required = (int(windows[:, INPUT_STEPS:].max()) + 1) * FRAMES_PER_SECOND
        if len(labels) < required:
            raise AdaptationContractError(
                "frame_labels_30fps must cover every forecast second at 30 fps"
            )
    out = np.zeros(
        (len(windows), FORECAST_STEPS, FRAMES_PER_SECOND), dtype=np.int64
    )
    for i, window in enumerate(windows):
        for h in range(FORECAST_STEPS):
            second = int(window[INPUT_STEPS + h])
            out[i, h] = labels[
                second * FRAMES_PER_SECOND : (second + 1) * FRAMES_PER_SECOND
            ]
    return out


def generate_predicted_frames(
    windows: np.ndarray,
    frames_1hz: np.ndarray,
    predictor: torch.nn.Module,
    *,
    device: str | torch.device,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:

    windows = _validated_windows(windows)
    frames = np.asarray(frames_1hz)
    if frames.ndim != 3 or frames.shape[1:] != PREDICTOR_FRAME_SHAPE:
        raise AdaptationContractError(
            f"frames_1hz must have shape (S, {PREDICTOR_FRAME_SHAPE[0]}, "
            f"{PREDICTOR_FRAME_SHAPE[1]})"
        )
    if len(windows) and int(windows.max()) >= len(frames):
        raise AdaptationContractError("windows index a second outside frames_1hz")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise AdaptationContractError("batch_size must be a positive integer")

    resolved_device = torch.device(device)
    predictor.to(resolved_device)
    predictor.eval()
    total = len(windows)
    predicted = np.zeros(
        (total, FORECAST_STEPS, 1, *PREDICTOR_FRAME_SHAPE), dtype=np.float16
    )
    true_future = np.zeros(
        (total, FORECAST_STEPS, 1, *PREDICTOR_FRAME_SHAPE), dtype=np.float16
    )
    with torch.no_grad():
        for i0 in range(0, total, batch_size):
            i1 = min(total, i0 + batch_size)
            history = np.zeros(
                (i1 - i0, INPUT_STEPS, 1, *PREDICTOR_FRAME_SHAPE), dtype=np.float32
            )
            for k in range(i1 - i0):
                indices = windows[i0 + k]
                for t in range(INPUT_STEPS):
                    history[k, t, 0] = frames[indices[t]]
            output = predictor(
                torch.from_numpy(history).to(resolved_device),
                future_seq=FORECAST_STEPS,
            )
            expected = (i1 - i0, FORECAST_STEPS, 1, *PREDICTOR_FRAME_SHAPE)
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
                raise AdaptationContractError(
                    f"predictor output must have shape {expected}"
                )
            predicted[i0:i1] = output.detach().cpu().numpy().astype(np.float16)
            for k in range(i1 - i0):
                indices = windows[i0 + k]
                for t in range(FORECAST_STEPS):
                    true_future[i0 + k, t, 0] = frames[indices[INPUT_STEPS + t]]
    return predicted, true_future


def predicted_frame_ssim(
    predicted: np.ndarray, true_future: np.ndarray
) -> np.ndarray:

    predicted = np.asarray(predicted)
    true_future = np.asarray(true_future)
    if predicted.shape != true_future.shape:
        raise AdaptationContractError("predicted and true frames must share a shape")
    if predicted.ndim != 5 or predicted.shape[1] != FORECAST_STEPS or predicted.shape[2] != 1:
        raise AdaptationContractError(
            f"predicted frames must have shape (N, {FORECAST_STEPS}, 1, H, W)"
        )
    count, horizon = predicted.shape[0], predicted.shape[1]
    ssim_flat = np.zeros(count * horizon, dtype=np.float32)
    for i in range(count):
        for h in range(horizon):
            frame_predicted = predicted[i, h, 0].astype(np.float32)
            frame_true = true_future[i, h, 0].astype(np.float32)
            ssim_flat[i * horizon + h] = skimage_structural_similarity(
                frame_true, frame_predicted, data_range=1.0
            )
    return ssim_flat


def gt_mode_seed_labels(target_per_frame: np.ndarray) -> np.ndarray:

    targets = np.asarray(target_per_frame)
    if targets.ndim != 3 or not np.issubdtype(targets.dtype, np.integer):
        raise AdaptationContractError(
            "target_per_frame must be an integer array of shape (N, 5, 30)"
        )
    if targets.size and targets.min() < 0:
        raise AdaptationContractError("target_per_frame labels must be zero-based")
    count, horizon = targets.shape[0], targets.shape[1]
    modes = np.zeros(count * horizon, dtype=np.int8)
    for i in range(count):
        for h in range(horizon):
            modes[i * horizon + h] = int(
                np.bincount(targets[i, h].astype(np.int64)).argmax()
            )
    return modes


def pixel_std_band_mask(
    frames: np.ndarray,
    *,
    lower: float = PIXEL_STD_LOWER,
    upper: float | None = PIXEL_STD_UPPER,
) -> np.ndarray:

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1] != 1:
        raise AdaptationContractError("frames must have shape (N, 1, H, W)")
    squeezed = frames[:, 0]
    std = squeezed.std(axis=(1, 2))
    keep = std >= lower
    if upper is not None:
        keep &= std < upper
    return keep


def _resize_kept_frames(predicted: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on environment
        raise AdaptationContractError(
            "OpenCV is required for source-compatible linear resize"
        ) from error

    count, horizon = predicted.shape[0], predicted.shape[1]
    total = count * horizon
    kept = int(keep_mask.sum())
    frames = np.zeros((kept, 1, 32, 64), dtype=np.float32)
    out_k = 0
    for flat_k in range(total):
        if not keep_mask[flat_k]:
            continue
        i = flat_k // horizon
        h = flat_k % horizon
        frame_128 = predicted[i, h, 0].astype(np.float32)
        frame_32x64 = cv2.resize(frame_128, CLASSIFIER_SIZE_WH, interpolation=cv2.INTER_LINEAR)
        frames[out_k, 0] = np.clip(frame_32x64, 0.0, 1.0)
        out_k += 1
    return frames


def build_adaptation_set(
    *,
    frames_1hz: np.ndarray,
    frame_labels_30fps: np.ndarray,
    train_seconds: Sequence[int],
    validation_seconds: Sequence[int],
    test_seconds: Sequence[int],
    original_train_frames: np.ndarray,
    original_train_labels: np.ndarray,
    original_train_frame_indices: np.ndarray,
    original_validation_frames: np.ndarray,
    original_validation_labels: np.ndarray,
    predictor: torch.nn.Module,
    predicted_frame_labels: np.ndarray | None = None,
    ssim_threshold: float = SSIM_KEEP_THRESHOLD,
    pixel_std_lower: float = PIXEL_STD_LOWER,
    pixel_std_upper: float | None = PIXEL_STD_UPPER,
    device: str | torch.device | None = None,
    predictor_batch_size: int = 4,
) -> AdaptationBundle:

    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    train_windows = build_sliding_windows(train_seconds)
    validation_windows = build_sliding_windows(validation_seconds)
    if len(validation_windows) == 0:
        windows = train_windows
    else:
        windows = np.concatenate([train_windows, validation_windows], axis=0)
    if len(windows) == 0:
        raise AdaptationContractError(
            "no sliding window fits inside the supplied train/validation seconds"
        )

    assert_no_test_second_leakage(windows.flatten(), test_seconds)

    predicted, true_future = generate_predicted_frames(
        windows,
        frames_1hz,
        predictor,
        device=resolved_device,
        batch_size=predictor_batch_size,
    )

    ssim_flat = predicted_frame_ssim(predicted, true_future)
    keep_mask = ssim_flat >= ssim_threshold

    second_index_all = windows[:, INPUT_STEPS:].flatten().astype(np.int64)
    kept_seconds = second_index_all[keep_mask]

    predicted_frames_32x64 = _resize_kept_frames(predicted, keep_mask)

    if predicted_frame_labels is None:
        target_per_frame = build_target_per_frame(windows, frame_labels_30fps)
        kept_labels = gt_mode_seed_labels(target_per_frame)[keep_mask].astype(np.int64)
    else:
        supplied = np.asarray(predicted_frame_labels)
        if supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.integer):
            raise AdaptationContractError(
                "predicted_frame_labels must be a one-dimensional integer array"
            )
        if len(supplied) != len(predicted_frames_32x64):
            raise AdaptationContractError(
                "predicted_frame_labels must align one-to-one with the SSIM-kept frames"
            )
        if supplied.size and (supplied.min() < 0 or supplied.max() > 2):
            raise AdaptationContractError(
                "predicted_frame_labels must be zero-based in {0, 1, 2}"
            )
        kept_labels = supplied.astype(np.int64)

    band_mask = pixel_std_band_mask(
        predicted_frames_32x64, lower=pixel_std_lower, upper=pixel_std_upper
    )
    predicted_frames_32x64 = predicted_frames_32x64[band_mask]
    kept_labels = kept_labels[band_mask]
    kept_seconds = kept_seconds[band_mask]

    original_train_frames = np.asarray(original_train_frames)
    original_validation_frames = np.asarray(original_validation_frames)
    frame_indices = np.asarray(original_train_frame_indices)
    if frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer):
        raise AdaptationContractError(
            "original_train_frame_indices must be a one-dimensional integer array"
        )
    if len(frame_indices) != len(original_train_frames):
        raise AdaptationContractError(
            "original_train_frame_indices must align with original_train_frames"
        )
    if frame_indices.size:
        if int(frame_indices.min()) < 0:
            raise AdaptationContractError(
                "original_train_frame_indices must be non-negative"
            )
        if int(frame_indices.max()) > np.iinfo(np.int64).max:
            raise AdaptationContractError(
                "original_train_frame_indices values must fit in int64"
            )
    original_train_seconds = (frame_indices // FRAMES_PER_SECOND).astype(np.int64)

    pool_seconds = np.concatenate([original_train_seconds, kept_seconds])
    assert_no_test_second_leakage(pool_seconds, test_seconds)

    frames = np.concatenate(
        [original_train_frames.astype(np.float32), predicted_frames_32x64], axis=0
    )
    labels = np.concatenate(
        [np.asarray(original_train_labels).astype(np.int64), kept_labels], axis=0
    )
    origin_tags = np.concatenate(
        [
            np.zeros(len(original_train_frames), dtype=np.uint8),
            np.ones(len(predicted_frames_32x64), dtype=np.uint8),
        ]
    )
    return AdaptationBundle(
        frames=frames,
        labels=labels,
        origin_tags=origin_tags,
        second_indices=pool_seconds,
        validation_frames=original_validation_frames.astype(np.float32),
        validation_labels=np.asarray(original_validation_labels).astype(np.int64),
    )


__all__ = [
    "AdaptationBundle",
    "AdaptationContractError",
    "CLASSIFIER_SIZE_WH",
    "FORECAST_STEPS",
    "FRAMES_PER_SECOND",
    "INPUT_STEPS",
    "PIXEL_STD_LOWER",
    "PIXEL_STD_UPPER",
    "PREDICTOR_FRAME_SHAPE",
    "SSIM_KEEP_THRESHOLD",
    "TOTAL_STEPS",
    "assert_no_test_second_leakage",
    "build_adaptation_set",
    "build_sliding_windows",
    "build_target_per_frame",
    "generate_predicted_frames",
    "gt_mode_seed_labels",
    "pixel_std_band_mask",
    "predicted_frame_ssim",
]
