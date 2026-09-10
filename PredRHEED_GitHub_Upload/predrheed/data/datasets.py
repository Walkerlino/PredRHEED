from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import BadZipFile

import numpy as np

from .schema import (
    PATTERN_CLASSES,
    PatternClass,
    SampleRecord,
    SchemaValidationError,
    SplitName,
    validate_pattern_class,
    validate_sample_records,
    validate_split_name,
)
from .windows import PredictionWindow, validate_window_split


_CLASSIFICATION_KEYS = frozenset({"frames", "pattern_classes", "splits"})
_SEQUENCE_REQUIRED_KEYS = frozenset({"frames", "positions", "splits"})
_SEQUENCE_OPTIONAL_KEYS = frozenset({"segment_ids"})


def _validate_frames(frames: np.ndarray, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    if not isinstance(frames, np.ndarray):
        raise SchemaValidationError(f"{name} frames must be a NumPy array")
    if frames.dtype == object or frames.dtype.kind not in {"f"}:
        raise SchemaValidationError(f"{name} frames must use a floating non-object dtype")
    if frames.ndim != len(shape) or any(
        (expected is not None and actual != expected)
        or (expected is None and actual <= 0)
        for actual, expected in zip(frames.shape, shape)
    ):
        raise SchemaValidationError(f"{name} frames have an invalid shape")
    if not np.isfinite(frames).all():
        raise SchemaValidationError(f"{name} frames must be finite")
    if np.any(frames < 0) or np.any(frames > 1):
        raise SchemaValidationError(f"{name} frames must be normalized to [0, 1]")
    return frames


def _string_values(values: np.ndarray, name: str) -> tuple[str, ...]:
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise SchemaValidationError(f"{name} must be a one-dimensional string array")
    if values.dtype == object or values.dtype.kind != "U":
        raise SchemaValidationError(f"{name} must be a Unicode non-object array")
    return tuple(str(value) for value in values.tolist())


def _load_npz_arrays(path: Path, allowed_keys: frozenset[str], required_keys: frozenset[str]) -> dict[str, np.ndarray]:
    resolved_path = Path(path).expanduser().resolve(strict=False)
    if not resolved_path.is_file():
        raise FileNotFoundError("input bundle does not exist")
    try:
        with resolved_path.open("rb") as source:
            bundle = np.load(source, allow_pickle=False)
            if not isinstance(bundle, np.lib.npyio.NpzFile):
                raise SchemaValidationError("could not load input bundle")
            with bundle:
                keys = frozenset(bundle.files)
                unknown = keys - allowed_keys
                missing = required_keys - keys
                if unknown:
                    raise SchemaValidationError(f"unknown npz fields: {sorted(unknown)}")
                if missing:
                    raise SchemaValidationError(f"missing npz fields: {sorted(missing)}")
                arrays = {key: bundle[key] for key in keys}
    except (BadZipFile, EOFError, OSError, ValueError) as error:
        if isinstance(error, SchemaValidationError):
            raise
        raise SchemaValidationError("could not load input bundle") from error
    if any(array.dtype == object for array in arrays.values()):
        raise SchemaValidationError("npz bundles must not contain object arrays")
    return arrays


@dataclass(frozen=True, slots=True)
class ClassificationBundle:

    frames: np.ndarray
    pattern_classes: tuple[PatternClass, ...]
    splits: tuple[SplitName, ...]

    def __post_init__(self) -> None:
        frames = _validate_frames(self.frames, (None, 3, 32, 64), "classification")
        pattern_classes = tuple(validate_pattern_class(value) for value in self.pattern_classes)
        splits = tuple(validate_split_name(value) for value in self.splits)
        if len(frames) != len(pattern_classes) or len(frames) != len(splits):
            raise SchemaValidationError("classification arrays must have matching lengths")
        object.__setattr__(self, "pattern_classes", pattern_classes)
        object.__setattr__(self, "splits", splits)

    @classmethod
    def from_npz(cls, path: Path) -> "ClassificationBundle":
        arrays = _load_npz_arrays(path, _CLASSIFICATION_KEYS, _CLASSIFICATION_KEYS)
        return cls(
            frames=arrays["frames"],
            pattern_classes=_string_values(arrays["pattern_classes"], "pattern_classes"),
            splits=_string_values(arrays["splits"], "splits"),
        )


@dataclass(frozen=True, slots=True)
class SequenceBundle:

    frames: np.ndarray
    records: tuple[SampleRecord, ...]

    def __post_init__(self) -> None:
        frames = _validate_frames(self.frames, (None, 1, None, None), "sequence")
        records = validate_sample_records(self.records)
        if len(frames) != len(records):
            raise SchemaValidationError("sequence frames and records must have matching lengths")
        object.__setattr__(self, "records", records)

    @classmethod
    def from_npz(cls, path: Path) -> "SequenceBundle":
        arrays = _load_npz_arrays(
            path,
            _SEQUENCE_REQUIRED_KEYS | _SEQUENCE_OPTIONAL_KEYS,
            _SEQUENCE_REQUIRED_KEYS,
        )
        positions = arrays["positions"]
        if positions.ndim != 1 or positions.dtype.kind not in {"i", "u"}:
            raise SchemaValidationError("positions must be a one-dimensional integer array")
        splits = _string_values(arrays["splits"], "splits")
        if len(positions) != len(splits):
            raise SchemaValidationError("positions and splits must have matching lengths")
        if "segment_ids" in arrays:
            segment_ids = _string_values(arrays["segment_ids"], "segment_ids")
            if len(segment_ids) != len(positions):
                raise SchemaValidationError("segment_ids and positions must have matching lengths")
        else:
            segment_ids = (None,) * len(positions)
        records = tuple(
            SampleRecord(int(position), split, segment_id)
            for position, split, segment_id in zip(positions.tolist(), splits, segment_ids)
        )
        return cls(frames=arrays["frames"], records=records)


class ClassificationDataset:

    def __init__(self, bundle: ClassificationBundle, split: SplitName) -> None:
        self._bundle = bundle
        self._split = validate_split_name(split)
        self._indices = tuple(
            index for index, item_split in enumerate(bundle.splits) if item_split == self._split
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        source_index = self._indices[index]
        pattern_class = self._bundle.pattern_classes[source_index]
        return self._bundle.frames[source_index], PATTERN_CLASSES.index(pattern_class)


class PredictionDataset:

    def __init__(self, bundle: SequenceBundle, windows: Iterable[PredictionWindow]) -> None:
        self._bundle = bundle
        self._windows = tuple(windows)
        self._frame_by_position = {
            record.position: bundle.frames[index]
            for index, record in enumerate(bundle.records)
        }
        for window in self._windows:
            validate_window_split(bundle.records, window)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        window = self._windows[index]
        observed = np.stack([self._frame_by_position[item] for item in window.input_indices])
        target = np.stack([self._frame_by_position[item] for item in window.target_indices])
        return observed, target
