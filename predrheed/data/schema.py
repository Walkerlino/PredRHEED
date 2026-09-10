from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, cast


PatternClass = Literal["streaky", "transition", "spotty"]
SplitName = Literal["train", "validation", "test"]

PATTERN_CLASSES: tuple[PatternClass, ...] = ("streaky", "transition", "spotty")
ALLOWED_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SampleRecord:

    position: int
    split: SplitName
    segment_id: str | None = None


def validate_pattern_class(value: str) -> PatternClass:

    if value not in PATTERN_CLASSES:
        raise SchemaValidationError(f"unknown pattern class: {value!r}")
    return cast(PatternClass, value)


def validate_split_name(value: str) -> SplitName:

    if value not in ALLOWED_SPLITS:
        raise SchemaValidationError(f"unknown split: {value!r}")
    return cast(SplitName, value)


def validate_sample_records(records: Iterable[SampleRecord]) -> tuple[SampleRecord, ...]:

    validated = tuple(records)
    if not validated:
        raise SchemaValidationError("sample records must not be empty")

    previous_position: int | None = None
    for record in validated:
        if not isinstance(record, SampleRecord):
            raise SchemaValidationError("sample records must be SampleRecord instances")
        if isinstance(record.position, bool) or not isinstance(record.position, int):
            raise SchemaValidationError("sample position must be a non-Boolean integer")
        if record.position < 0:
            raise SchemaValidationError("sample position must be non-negative")
        if not isinstance(record.split, str):
            raise SchemaValidationError("sample split must be a string")
        validate_split_name(record.split)
        if record.segment_id is not None:
            if not isinstance(record.segment_id, str):
                raise SchemaValidationError("segment identifier must be a string or None")
            if not record.segment_id:
                raise SchemaValidationError("segment identifier must not be empty")
        if previous_position is not None:
            if record.position <= previous_position:
                raise SchemaValidationError("sample positions must be strictly increasing")
            if record.position != previous_position + 1:
                raise SchemaValidationError("sample positions must be consecutive without gaps")
        previous_position = record.position
    return validated
