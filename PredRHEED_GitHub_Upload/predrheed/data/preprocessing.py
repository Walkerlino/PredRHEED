from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .datasets import ClassificationBundle, SequenceBundle
from .schema import SampleRecord, SchemaValidationError



CROP_WIDTH = 800
CROP_HEIGHT = 400

CLASSIFICATION_HEIGHT = 32
CLASSIFICATION_WIDTH = 64

PREDICTION_FRAME_SIZE = (128, 128)

FRAMES_PER_SECOND = 30


class PreprocessingError(ValueError):
    pass


def _validate_bgr_frames(frames: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
    validated = list(frames)
    if not validated:
        raise PreprocessingError(f"{name} must contain at least one frame")
    for frame in validated:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise PreprocessingError(f"{name} must contain (height, width, 3) BGR arrays")
        if frame.dtype != np.uint8:
            raise PreprocessingError(f"{name} must contain uint8 BGR arrays")
    return validated


def _validate_uniform_bgr_frames(frames: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
    validated = _validate_bgr_frames(frames, name)
    first_shape = validated[0].shape
    if any(frame.shape != first_shape for frame in validated):
        raise PreprocessingError(f"{name} must all share one frame shape")
    return validated




def detect_bright_center(
    gray: np.ndarray, blur_ksize: int = 9
) -> tuple[int, int, int] | None:

    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise PreprocessingError(
            f"expected grayscale, got shape {getattr(gray, 'shape', None)}"
        )
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return int(round(cx)), int(round(cy)), int(round(r))


def crop_centered(
    frame: np.ndarray,
    cx: int,
    cy: int,
    crop_w: int,
    crop_h: int,
    pad_value: int = 0,
) -> np.ndarray:

    H, W = frame.shape[:2]
    half_w, half_h = crop_w // 2, crop_h // 2
    x1, y1 = cx - half_w, cy - half_h
    x2, y2 = x1 + crop_w, y1 + crop_h
    pl = max(0, -x1)
    pt = max(0, -y1)
    pr = max(0, x2 - W)
    pb = max(0, y2 - H)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    src = frame[sy1:sy2, sx1:sx2]
    if pl or pr or pt or pb:
        if frame.ndim == 3:
            out = np.full((crop_h, crop_w, frame.shape[2]), pad_value, dtype=frame.dtype)
        else:
            out = np.full((crop_h, crop_w), pad_value, dtype=frame.dtype)
        out[pt : pt + src.shape[0], pl : pl + src.shape[1]] = src
        return out
    return src.copy()


def estimate_center(
    frames: Sequence[np.ndarray],
    *,
    fps: float = float(FRAMES_PER_SECOND),
    sample_count: int = 10,
    inlier_tolerance_px: int = 50,
) -> tuple[int, int]:

    validated = _validate_bgr_frames(frames, "frames")
    total = len(validated)
    sample_secs = np.linspace(1, max(5, total / fps - 5), sample_count).astype(int)
    cxs: list[int] = []
    cys: list[int] = []
    for s in sample_secs:
        index = int(s * fps)
        if index < 0 or index >= total:
            continue
        gray = cv2.cvtColor(validated[index], cv2.COLOR_BGR2GRAY)
        d = detect_bright_center(gray)
        if d is None:
            continue
        cxs.append(d[0])
        cys.append(d[1])
    if not cxs:
        raise PreprocessingError("no centers detected")
    cxs_array = np.array(cxs)
    cys_array = np.array(cys)
    med_x, med_y = int(np.median(cxs_array)), int(np.median(cys_array))
    mask = (np.abs(cxs_array - med_x) <= inlier_tolerance_px) & (
        np.abs(cys_array - med_y) <= inlier_tolerance_px
    )
    if mask.sum() == 0:
        mask = np.ones(len(cxs_array), dtype=bool)
    return int(cxs_array[mask].mean()), int(cys_array[mask].mean())




@dataclass(frozen=True, slots=True)
class FrameQualityChecker:

    overexposure_ratio: float = 0.3
    underexposure_ratio: float = 0.55
    blur_threshold: float = 5
    brightness_range: tuple[float, float] = (15, 250)
    overexposure_pixel_value: int = 240
    underexposure_pixel_value: int = 30

    def __post_init__(self) -> None:
        for name in ("overexposure_ratio", "underexposure_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PreprocessingError(f"{name} must be a fraction in [0, 1]")
            if not 0.0 <= float(value) <= 1.0:
                raise PreprocessingError(f"{name} must be a fraction in [0, 1]")
        for name in ("overexposure_pixel_value", "underexposure_pixel_value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise PreprocessingError(f"{name} must be an integer in [0, 255]")
        if (
            isinstance(self.blur_threshold, bool)
            or not isinstance(self.blur_threshold, (int, float))
            or not np.isfinite(self.blur_threshold)
            or self.blur_threshold < 0
        ):
            raise PreprocessingError("blur_threshold must be a finite non-negative number")

    def check_overexposure(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overexposed = np.sum(gray > self.overexposure_pixel_value) / gray.size
        return bool(overexposed > self.overexposure_ratio)

    def check_underexposure(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        underexposed = np.sum(gray < self.underexposure_pixel_value) / gray.size
        return bool(underexposed > self.underexposure_ratio)

    def check_blur(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def get_brightness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def is_good_frame(self, frame: np.ndarray) -> tuple[bool, dict]:
        metrics = {
            "overexposed": self.check_overexposure(frame),
            "underexposed": self.check_underexposure(frame),
            "blur_score": self.check_blur(frame),
            "brightness": self.get_brightness(frame),
        }
        # Deliberately lenient source rule: a frame fails the exposure check
        # only when it is overexposed AND underexposed at the same time, and
        # brightness_range is recorded but not part of the decision.  This is
        # the rule that built the released averaged-frame dataset; do not
        # tighten it to an either/or check.
        is_good = (
            not (metrics["overexposed"] and metrics["underexposed"])
            and metrics["blur_score"] > self.blur_threshold
        )
        metrics["is_good"] = is_good
        return is_good, metrics




def _assess_frames(
    frames: Sequence[np.ndarray], checker: FrameQualityChecker
) -> tuple[list[dict], list[np.ndarray], list[int], list[int]]:
    frame_quality: list[dict] = []
    good_frames: list[np.ndarray] = []
    good_indices: list[int] = []
    bad_indices: list[int] = []
    for i, frame in enumerate(frames):
        is_good, metrics = checker.is_good_frame(frame)
        frame_quality.append({"index": i, "is_good": is_good, "metrics": metrics})
        if is_good:
            good_frames.append(frame)
            good_indices.append(i)
        else:
            bad_indices.append(i)
    return frame_quality, good_frames, good_indices, bad_indices


def _weighted_average_frame(
    frames: Sequence[np.ndarray],
    frame_quality: Sequence[dict],
    good_frames: Sequence[np.ndarray],
) -> np.ndarray:
    if len(good_frames) > 0:
        weights = np.array(
            [fq["metrics"]["blur_score"] for fq in frame_quality if fq["is_good"]]
        )
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            return np.mean(good_frames, axis=0).astype(np.uint8)
        weights = weights / total_weight
        average_frame = np.zeros_like(good_frames[0], dtype=np.float64)
        for frame, weight in zip(good_frames, weights):
            average_frame += frame.astype(np.float64) * weight
        return average_frame.astype(np.uint8)
    return np.mean(frames, axis=0).astype(np.uint8)


def _replacement_frame(
    bad_idx: int,
    frames: Sequence[np.ndarray],
    good_indices: Sequence[int],
    average_frame: np.ndarray,
    max_distance: int,
) -> np.ndarray:
    if not good_indices:
        return frames[bad_idx]
    distances = [abs(gi - bad_idx) for gi in good_indices]
    nearest_good_idx = good_indices[int(np.argmin(distances))]
    nearest_distance = min(distances)
    if nearest_distance <= max_distance:
        return frames[nearest_good_idx]
    prev_good = None
    next_good = None
    for gi in good_indices:
        if gi < bad_idx:
            prev_good = gi
        elif gi > bad_idx and next_good is None:
            next_good = gi
            break
    if prev_good is not None and next_good is not None:
        alpha = (bad_idx - prev_good) / (next_good - prev_good)
        return cv2.addWeighted(
            frames[prev_good], 1 - alpha, frames[next_good], alpha, 0
        )
    return average_frame


def process_second(
    frames: Sequence[np.ndarray],
    *,
    checker: FrameQualityChecker | None = None,
    frame_count: int = FRAMES_PER_SECOND,
    replacement_max_distance: int = 3,
) -> tuple[list[np.ndarray], np.ndarray, dict]:

    validated = _validate_uniform_bgr_frames(frames, "frames")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise PreprocessingError("frame_count must be a positive non-Boolean integer")
    if len(validated) != frame_count:
        raise PreprocessingError(
            f"expected {frame_count} frames, got {len(validated)}"
        )
    if checker is None:
        checker = FrameQualityChecker()

    frame_quality, good_frames, good_indices, bad_indices = _assess_frames(
        validated, checker
    )
    average_frame = _weighted_average_frame(validated, frame_quality, good_frames)

    processed_frames = list(validated)
    for bad_idx in bad_indices:
        processed_frames[bad_idx] = _replacement_frame(
            bad_idx, validated, good_indices, average_frame, replacement_max_distance
        )

    if not bad_indices:
        replacement_method = "no_replacement_needed"
    elif not good_indices:
        replacement_method = "no_good_frames_available"
    else:
        replacement_method = f"replaced_{len(bad_indices)}_frames"
    stats = {
        "total_frames": frame_count,
        "good_frames": len(good_frames),
        "bad_frames": len(bad_indices),
        "bad_frame_indices": bad_indices,
        "replacement_method": replacement_method,
    }
    return processed_frames, average_frame, stats


def average_second(
    frames: Sequence[np.ndarray],
    *,
    checker: FrameQualityChecker | None = None,
    frame_count: int = FRAMES_PER_SECOND,
) -> np.ndarray:

    validated = _validate_uniform_bgr_frames(frames, "frames")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise PreprocessingError("frame_count must be a positive non-Boolean integer")
    if len(validated) != frame_count:
        raise PreprocessingError(
            f"expected {frame_count} frames, got {len(validated)}"
        )
    if checker is None:
        checker = FrameQualityChecker()
    frame_quality, good_frames, _, _ = _assess_frames(validated, checker)
    return _weighted_average_frame(validated, frame_quality, good_frames)




def representative_frames(
    frames: Sequence[np.ndarray],
    *,
    checker: FrameQualityChecker | None = None,
    frames_per_second: int = FRAMES_PER_SECOND,
    output_size: tuple[int, int] = PREDICTION_FRAME_SIZE,
) -> np.ndarray:

    validated = _validate_uniform_bgr_frames(frames, "frames")
    if (
        isinstance(frames_per_second, bool)
        or not isinstance(frames_per_second, int)
        or frames_per_second <= 0
    ):
        raise PreprocessingError(
            "frames_per_second must be a positive non-Boolean integer"
        )
    total_seconds = len(validated) // frames_per_second
    if total_seconds == 0:
        raise PreprocessingError(
            f"need at least {frames_per_second} frames for one window"
        )
    width, height = output_size
    out = np.empty((total_seconds, 1, height, width), dtype=np.float32)
    for second_idx in range(total_seconds):
        start_frame = second_idx * frames_per_second
        window = validated[start_frame : start_frame + frames_per_second]
        average_frame = average_second(
            window, checker=checker, frame_count=frames_per_second
        )
        gray = cv2.cvtColor(average_frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        out[second_idx, 0] = resized.astype(np.float32) / 255.0
    return out




def classification_frames(
    frames: Sequence[np.ndarray],
    *,
    center: tuple[int, int] | None = None,
    crop_wh: tuple[int, int] = (CROP_WIDTH, CROP_HEIGHT),
    crop_rect: tuple[int, int, int, int] | None = None,
    apply_quality_replacement: bool = False,
    checker: FrameQualityChecker | None = None,
    frames_per_second: int = FRAMES_PER_SECOND,
    replacement_max_distance: int = 3,
) -> np.ndarray:

    validated = _validate_bgr_frames(frames, "frames")
    if center is not None and crop_rect is not None:
        raise PreprocessingError("center and crop_rect are mutually exclusive")

    if apply_quality_replacement:
        uniform = _validate_uniform_bgr_frames(validated, "frames")
        if (
            isinstance(frames_per_second, bool)
            or not isinstance(frames_per_second, int)
            or frames_per_second <= 0
        ):
            raise PreprocessingError(
                "frames_per_second must be a positive non-Boolean integer"
            )
        total_seconds = len(uniform) // frames_per_second
        if total_seconds == 0:
            raise PreprocessingError(
                f"need at least {frames_per_second} frames for one window"
            )
        replaced: list[np.ndarray] = []
        for second_idx in range(total_seconds):
            start_frame = second_idx * frames_per_second
            window = uniform[start_frame : start_frame + frames_per_second]
            processed, _, _ = process_second(
                window,
                checker=checker,
                frame_count=frames_per_second,
                replacement_max_distance=replacement_max_distance,
            )
            replaced.extend(processed)
        validated = replaced

    arr = np.empty(
        (len(validated), CLASSIFICATION_HEIGHT, CLASSIFICATION_WIDTH), dtype=np.float32
    )
    for idx, frame in enumerate(validated):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if center is not None:
            cx, cy = center
            gray = crop_centered(gray, int(cx), int(cy), crop_wh[0], crop_wh[1])
        elif crop_rect is not None:
            x1, y1, x2, y2 = crop_rect
            H, W = gray.shape
            xa = max(0, int(x1))
            ya = max(0, int(y1))
            xb = min(W, int(x2))
            yb = min(H, int(y2))
            cropped = gray[ya:yb, xa:xb]
            if cropped.size == 0:
                cropped = gray
            gray = cropped
        small = cv2.resize(
            gray,
            (CLASSIFICATION_WIDTH, CLASSIFICATION_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        arr[idx] = small.astype(np.float32) / 255.0
    X = arr[:, None, :, :]
    return np.repeat(X, 3, axis=1).astype(np.float32)




def _resolved_npz_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if resolved.suffix != ".npz":
        raise PreprocessingError("bundle path must use the .npz suffix")
    if resolved.exists() and resolved.is_dir():
        raise PreprocessingError("bundle path must identify a file, not a directory")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _atomic_write_npz(resolved: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, resolved)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_classification_bundle(
    path: str | Path,
    frames: np.ndarray,
    pattern_classes: Iterable[str],
    splits: Iterable[str],
) -> Path:

    resolved = _resolved_npz_path(path)
    bundle = ClassificationBundle(
        frames=np.asarray(frames),
        pattern_classes=tuple(pattern_classes),
        splits=tuple(splits),
    )
    _atomic_write_npz(
        resolved,
        {
            "frames": bundle.frames,
            "pattern_classes": np.asarray(bundle.pattern_classes),
            "splits": np.asarray(bundle.splits),
        },
    )
    return resolved


def write_sequence_bundle(
    path: str | Path,
    frames: np.ndarray,
    positions: Iterable[int],
    splits: Iterable[str],
    segment_ids: Iterable[str] | None = None,
) -> Path:

    resolved = _resolved_npz_path(path)
    position_array = np.asarray(tuple(positions))
    if position_array.ndim != 1 or position_array.dtype.kind not in {"i", "u"}:
        raise SchemaValidationError("positions must be a one-dimensional integer array")
    if position_array.size and int(position_array.max()) > np.iinfo(np.int64).max:
        raise SchemaValidationError("positions values must fit in int64")
    split_values = tuple(splits)
    if len(split_values) != len(position_array):
        raise SchemaValidationError("positions and splits must have matching lengths")
    segment_values = tuple(segment_ids) if segment_ids is not None else None
    if segment_values is not None and len(segment_values) != len(position_array):
        raise SchemaValidationError("segment_ids and positions must have matching lengths")
    if segment_values is not None and any(
        not isinstance(segment_id, str) or not segment_id
        for segment_id in segment_values
    ):
        raise SchemaValidationError("segment identifier must be a non-empty string")
    records = tuple(
        SampleRecord(
            int(position),
            split,
            segment_values[index] if segment_values is not None else None,
        )
        for index, (position, split) in enumerate(
            zip(position_array.tolist(), split_values)
        )
    )
    bundle = SequenceBundle(frames=np.asarray(frames), records=records)
    arrays: dict[str, np.ndarray] = {
        "frames": bundle.frames,
        "positions": np.asarray(
            [record.position for record in bundle.records], dtype=np.int64
        ),
        "splits": np.asarray([record.split for record in bundle.records]),
    }
    if segment_ids is not None:
        arrays["segment_ids"] = np.asarray(
            [record.segment_id for record in bundle.records]
        )
    _atomic_write_npz(resolved, arrays)
    return resolved
