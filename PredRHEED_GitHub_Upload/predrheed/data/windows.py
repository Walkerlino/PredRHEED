from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import SampleRecord, SchemaValidationError, SplitName, validate_sample_records, validate_split_name


class NoValidWindowsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PredictionWindow:

    start_position: int
    assigned_split: SplitName
    input_indices: tuple[int, ...]
    target_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.start_position, bool) or not isinstance(
            self.start_position, int
        ):
            raise SchemaValidationError(
                "window start_position must be a non-Boolean integer"
            )
        if self.start_position < 0:
            raise SchemaValidationError("window start_position must be non-negative")
        validate_split_name(self.assigned_split)
        try:
            input_indices = tuple(self.input_indices)
            target_indices = tuple(self.target_indices)
        except TypeError as error:
            raise SchemaValidationError(
                "window input_indices and target_indices must be iterable"
            ) from error
        if not input_indices or not target_indices:
            raise SchemaValidationError(
                "window input_indices and target_indices must both be non-empty"
            )
        positions = input_indices + target_indices
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int):
                raise SchemaValidationError(
                    "window positions must be non-Boolean integers"
                )
            if position < 0:
                raise SchemaValidationError("window positions must be non-negative")
        if positions[0] != self.start_position:
            raise SchemaValidationError(
                "window start_position must equal the first input position"
            )
        if any(right != left + 1 for left, right in zip(positions, positions[1:])):
            raise SchemaValidationError(
                "window positions must be strictly increasing and consecutive"
            )
        object.__setattr__(self, "input_indices", input_indices)
        object.__setattr__(self, "target_indices", target_indices)


def validate_window_split(
    records: Iterable[SampleRecord], window: PredictionWindow
) -> None:

    by_position = {record.position: record for record in validate_sample_records(records)}
    positions = window.input_indices + window.target_indices
    if not positions or len(set(positions)) != len(positions):
        raise SchemaValidationError("window positions must be non-empty and unique")
    try:
        splits = {by_position[position].split for position in positions}
    except KeyError as error:
        raise SchemaValidationError("window position is not present in sample records") from error
    if splits != {window.assigned_split}:
        raise SchemaValidationError(
            "every input and target item must belong to the assigned split"
        )


def build_prediction_windows(
    records: Iterable[SampleRecord],
    *,
    assigned_split: SplitName,
    input_steps: int,
    target_steps: int,
    stride_steps: int,
    allow_empty: bool = False,
) -> tuple[PredictionWindow, ...]:

    validated = validate_sample_records(records)
    validate_split_name(assigned_split)
    for field_name, value in (
        ("input_steps", input_steps),
        ("target_steps", target_steps),
        ("stride_steps", stride_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SchemaValidationError(
                f"{field_name} must be a positive non-Boolean integer"
            )

    window_steps = input_steps + target_steps
    retained: list[PredictionWindow] = []
    for offset in range(0, len(validated) - window_steps + 1, stride_steps):
        candidate = validated[offset : offset + window_steps]
        positions = tuple(record.position for record in candidate)
        if {record.split for record in candidate} != {assigned_split}:
            continue
        window = PredictionWindow(
            start_position=positions[0],
            assigned_split=assigned_split,
            input_indices=positions[:input_steps],
            target_indices=positions[input_steps:],
        )
        retained.append(window)
    if not retained and not allow_empty:
        raise NoValidWindowsError("no candidate satisfies the split invariant")
    return tuple(retained)
