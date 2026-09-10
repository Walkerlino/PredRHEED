from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from predrheed.data.datasets import PredictionDataset, SequenceBundle
from predrheed.data.windows import PredictionWindow, build_prediction_windows
from predrheed.evaluation.decisions import PRIMARY_PROPORTION, UNRESOLVED_LABEL
from predrheed.models.seq2label import DirectSeq2Label, DirectSeq2LabelConfig

from predrheed.training.classification import LabelSmoothingCrossEntropy
from predrheed.training.common import (
    OptimizerContract,
    PAPER_SEED,
    TrainingContractError,
    require_nonempty_loader as _require_nonempty_loader,
    resolve_device,
    save_state_dict,
    set_reproducible_seeds,
)
from predrheed.training.prediction import (
    BLOCKED_RUNNER_AUGMENTATION,
    PredictionAugmentationContract,
    PredictionSchedulerContract,
    _opencv_resize_frame,
    augment_prediction_sequence,
    build_predictor_scheduler,
)


FRAMES_PER_SECOND: Final[int] = 30


@dataclass(frozen=True, slots=True)
class Seq2LabelTrainingContract:

    source_name: str
    batch_size: int
    max_epochs: int
    optimizer: OptimizerContract
    scheduler: PredictionSchedulerContract
    early_stopping_patience: int
    gradient_clip_norm: float
    num_workers: int
    num_classes: int = 3
    label_smoothing: float = 0.1
    horizons: int = 5
    input_steps: int = 15

    def __post_init__(self) -> None:
        for field_name in (
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "num_classes",
            "horizons",
            "input_steps",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingContractError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise TrainingContractError("num_workers must be a nonnegative integer")
        if not 0 <= self.label_smoothing < 1:
            raise TrainingContractError("label_smoothing must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise TrainingContractError("gradient_clip_norm must be positive")


_SEQ2LABEL_ADAMW = OptimizerContract(
    name="AdamW",
    learning_rate=5e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)


SEQ2LABEL_TRAINING = Seq2LabelTrainingContract(
    source_name="direct_seq2label",
    batch_size=4,
    max_epochs=100,
    optimizer=_SEQ2LABEL_ADAMW,
    scheduler=PredictionSchedulerContract(),
    early_stopping_patience=10,
    gradient_clip_norm=1.0,
    num_workers=0,
    num_classes=3,
    label_smoothing=0.1,
    horizons=5,
    input_steps=15,
)


def threshold_label(
    frame_labels: np.ndarray,
    *,
    proportion: float = PRIMARY_PROPORTION,
    number_of_classes: int = 3,
) -> int:

    labels = np.asarray(frame_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise TrainingContractError(
            "frame_labels must be a nonempty one-dimensional array"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TrainingContractError("frame_labels must have an integer dtype")
    if (
        isinstance(number_of_classes, bool)
        or not isinstance(number_of_classes, int)
        or number_of_classes <= 1
    ):
        raise TrainingContractError(
            "number_of_classes must be an integer greater than one"
        )
    if isinstance(proportion, bool) or not isinstance(proportion, (int, float)):
        raise TrainingContractError("proportion must be a finite numeric value")
    threshold = float(proportion)
    if not np.isfinite(threshold) or not 0.5 < threshold <= 1.0:
        raise TrainingContractError(
            "proportion must be greater than 0.5 and at most 1.0"
        )
    if int(labels.min()) < 0 or int(labels.max()) >= number_of_classes:
        raise TrainingContractError(
            "frame_labels are outside the configured class range"
        )
    counts = np.bincount(labels, minlength=number_of_classes)
    eligible = np.flatnonzero(counts >= threshold * labels.size)
    return int(eligible[0]) if eligible.size else UNRESOLVED_LABEL


def zero_based_frame_labels(labels: np.ndarray) -> np.ndarray:

    values = np.asarray(labels)
    if values.size == 0:
        raise TrainingContractError("labels must not be empty")
    if not np.issubdtype(values.dtype, np.integer):
        raise TrainingContractError("labels must have an integer dtype")
    return (values - 1 if values.min() == 1 else values).astype(np.int64)


def build_labels5(
    frame_labels: np.ndarray,
    *,
    start_position: int = 0,
    input_steps: int = 15,
    horizons: int = 5,
    frames_per_second: int = FRAMES_PER_SECOND,
    proportion: float = PRIMARY_PROPORTION,
    number_of_classes: int = 3,
) -> dict[int, tuple[int, ...]]:

    if (
        isinstance(start_position, bool)
        or not isinstance(start_position, int)
        or start_position < 0
    ):
        raise TrainingContractError("start_position must be a nonnegative integer")
    for name, value in (
        ("input_steps", input_steps),
        ("horizons", horizons),
        ("frames_per_second", frames_per_second),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TrainingContractError(f"{name} must be a positive integer")
    labels = np.asarray(frame_labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise TrainingContractError(
            "frame_labels must be a one-dimensional integer array"
        )
    if labels.size == 0 or labels.size % frames_per_second != 0:
        raise TrainingContractError(
            "frame_labels length must be a positive multiple of frames_per_second"
        )
    if int(labels.min()) < 0:
        raise TrainingContractError(
            "frame_labels must be zero-based; see zero_based_frame_labels"
        )
    total_seconds = labels.size // frames_per_second
    total_length = input_steps + horizons
    if total_seconds < total_length:
        raise TrainingContractError(
            "frame_labels must cover at least input_steps + horizons seconds"
        )
    labels5: dict[int, tuple[int, ...]] = {}
    for start in range(0, total_seconds - total_length + 1):
        labels5[start_position + start] = tuple(
            threshold_label(
                labels[
                    (start + input_steps + horizon)
                    * frames_per_second : (start + input_steps + horizon + 1)
                    * frames_per_second
                ],
                proportion=proportion,
                number_of_classes=number_of_classes,
            )
            for horizon in range(horizons)
        )
    return labels5


def balanced_seq2label_weights(
    train_targets: np.ndarray,
    *,
    num_classes: int = 3,
) -> torch.Tensor:

    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
        raise TrainingContractError("num_classes must be an integer greater than one")
    targets = np.asarray(train_targets)
    if targets.ndim != 1 or targets.size == 0:
        raise TrainingContractError(
            "train_targets must be a nonempty one-dimensional array"
        )
    if not np.issubdtype(targets.dtype, np.integer):
        raise TrainingContractError("train_targets must have an integer dtype")
    if int(targets.min()) < UNRESOLVED_LABEL or int(targets.max()) >= num_classes:
        raise TrainingContractError("train_targets contains an out-of-range label")
    resolved_targets = targets[targets != UNRESOLVED_LABEL]
    if resolved_targets.size == 0:
        raise TrainingContractError("train_targets contains no resolved labels")
    counts = np.bincount(resolved_targets, minlength=num_classes)
    if np.any(counts == 0):
        raise TrainingContractError(
            "every class must be present in the training targets"
        )
    weights = resolved_targets.size / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


class _Seq2LabelTrainingDataset(Dataset):

    def __init__(
        self,
        dataset: PredictionDataset,
        windows: Sequence[PredictionWindow],
        labels5: Mapping[int, Sequence[int]],
        *,
        input_steps: int,
        horizons: int,
        augmentation_profile: PredictionAugmentationContract | None,
    ) -> None:
        if len(dataset) != len(windows):
            raise TrainingContractError(
                "dataset and windows must describe the same window sequence"
            )
        for window in windows:
            labels = labels5.get(window.start_position)
            if labels is None:
                raise TrainingContractError(
                    "labels5 is missing an entry for a window start position"
                )
            if len(labels) != horizons:
                raise TrainingContractError(
                    "labels5 entries must contain exactly one label per horizon"
                )
        self._dataset = dataset
        self._windows = tuple(windows)
        self._labels5 = labels5
        self.input_steps = input_steps
        self.horizons = horizons
        self.augmentation_profile = augmentation_profile

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        observed, target = self._dataset[index]
        if observed.shape != (self.input_steps, 1, 128, 128):
            raise TrainingContractError(
                "observed sequence must have shape (15,1,128,128)"
            )
        if target.shape != (self.horizons, 1, 128, 128):
            raise TrainingContractError(
                "target sequence must contain one 128x128 frame per horizon"
            )
        if self.augmentation_profile is not None:
            combined = np.concatenate((observed, target), axis=0)[:, 0]
            augmented = augment_prediction_sequence(
                combined,
                self.augmentation_profile,
                resize_frame=_opencv_resize_frame,
            )
            observed = augmented[: self.input_steps, np.newaxis]
        observed_array = np.ascontiguousarray(observed, dtype=np.float32)
        labels = self._labels5[self._windows[index].start_position]
        return (
            torch.from_numpy(observed_array),
            torch.tensor(tuple(labels), dtype=torch.long),
        )


def build_seq2label_optimizer(
    model: torch.nn.Module,
    profile: Seq2LabelTrainingContract,
) -> torch.optim.AdamW:

    contract = profile.optimizer
    if contract.name != "AdamW":
        raise TrainingContractError("seq2label optimizer must be AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        betas=contract.betas,
        eps=contract.eps,
        weight_decay=contract.weight_decay,
    )


@dataclass(frozen=True, slots=True)
class Seq2LabelEpochMetrics:
    train_loss: float
    validation_mean_accuracy: float
    validation_per_horizon_accuracy: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Seq2LabelFitSummary:
    best_epoch: int
    epochs_run: int
    best_validation_mean_accuracy: float
    history: tuple[Seq2LabelEpochMetrics, ...]
    best_checkpoint_path: Path
    final_checkpoint_path: Path
    parameter_count: int


def _evaluate_mean_horizon_accuracy(
    model: torch.nn.Module,
    loader: object,
    *,
    device: str | torch.device,
    horizons: int,
) -> tuple[float, tuple[float, ...]]:

    _require_nonempty_loader(loader, "evaluation")
    resolved_device = torch.device(device)
    model.eval()
    correct = np.zeros(horizons)
    total = 0
    with torch.no_grad():
        for observed, target in loader:  # type: ignore[union-attr]
            logits = model(observed.to(resolved_device))
            predicted = logits.argmax(-1).cpu()
            correct += (predicted == target).float().sum(0).numpy()
            total += len(target)
    per_horizon = correct / total
    return float(np.mean(per_horizon)), tuple(float(value) for value in per_horizon)


def _resolved_seq2label_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: LabelSmoothingCrossEntropy,
    *,
    horizons: int,
) -> torch.Tensor | None:

    losses: list[torch.Tensor] = []
    for horizon in range(horizons):
        resolved = target[:, horizon] != UNRESOLVED_LABEL
        if torch.any(resolved):
            losses.append(
                criterion(logits[resolved, horizon], target[resolved, horizon])
            )
    return torch.stack(losses).mean() if losses else None


def _fit_seq2label_configured(
    model: torch.nn.Module,
    training_loader: object,
    validation_loader: object,
    *,
    device: str | torch.device,
    profile: Seq2LabelTrainingContract,
    class_weights: torch.Tensor,
    checkpoint_dir: str | Path,
) -> Seq2LabelFitSummary:

    configured_device = torch.device(device)
    checkpoint_directory = Path(checkpoint_dir)
    _require_nonempty_loader(training_loader, "training")
    _require_nonempty_loader(validation_loader, "validation")

    model.to(configured_device)
    criterion = LabelSmoothingCrossEntropy(
        num_classes=profile.num_classes,
        smoothing=profile.label_smoothing,
        class_weights=class_weights,
    )
    optimizer = build_seq2label_optimizer(model, profile)
    scheduler = build_predictor_scheduler(optimizer, profile)  # type: ignore[arg-type]

    history: list[Seq2LabelEpochMetrics] = []
    best_validation_mean_accuracy = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint: Path | None = None

    for epoch_index in range(profile.max_epochs):
        scheduler.prepare_epoch(epoch_index)
        model.train()
        sample_weighted_loss = 0.0
        resolved_sample_count = 0
        for observed, target in training_loader:  # type: ignore[union-attr]
            observed = observed.to(configured_device)
            target = target.to(configured_device)
            optimizer.zero_grad()
            logits = model(observed)
            loss = _resolved_seq2label_loss(
                logits,
                target,
                criterion,
                horizons=profile.horizons,
            )
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), profile.gradient_clip_norm
            )
            optimizer.step()
            resolved_in_batch = int(
                torch.any(target != UNRESOLVED_LABEL, dim=1).sum().item()
            )
            sample_weighted_loss += float(loss.item()) * resolved_in_batch
            resolved_sample_count += resolved_in_batch

        if resolved_sample_count == 0:
            raise TrainingContractError(
                "training epoch contains no resolved target-second labels"
            )

        validation_mean, validation_per_horizon = _evaluate_mean_horizon_accuracy(
            model,
            validation_loader,
            device=configured_device,
            horizons=profile.horizons,
        )
        history.append(
            Seq2LabelEpochMetrics(
                train_loss=sample_weighted_loss / resolved_sample_count,
                validation_mean_accuracy=validation_mean,
                validation_per_horizon_accuracy=validation_per_horizon,
            )
        )

        if validation_mean > best_validation_mean_accuracy:
            best_validation_mean_accuracy = validation_mean
            best_epoch = epoch_index + 1
            best_checkpoint = save_state_dict(
                model, checkpoint_directory / "best_model.pth"
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= profile.early_stopping_patience:
            break
        scheduler.step_after_epoch(epoch_index)

    final_checkpoint = save_state_dict(
        model, checkpoint_directory / "final_epoch_model.pth"
    )
    if best_checkpoint is None:
        raise RuntimeError("seq2label training produced no best checkpoint")
    return Seq2LabelFitSummary(
        best_epoch=best_epoch,
        epochs_run=len(history),
        best_validation_mean_accuracy=best_validation_mean_accuracy,
        history=tuple(history),
        best_checkpoint_path=best_checkpoint,
        final_checkpoint_path=final_checkpoint,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def _validated_frame_labels(
    frame_labels: np.ndarray,
    *,
    num_classes: int,
    frames_per_second: int,
) -> np.ndarray:
    labels = np.asarray(frame_labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise TrainingContractError(
            "frame_labels must be a one-dimensional integer array"
        )
    if labels.size == 0 or labels.size % frames_per_second != 0:
        raise TrainingContractError(
            "frame_labels length must be a positive multiple of frames_per_second"
        )
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise TrainingContractError(
            "frame_labels must be zero-based and below num_classes; "
            "see zero_based_frame_labels"
        )
    return labels.astype(np.int64, copy=False)


def fit_direct_seq2label(
    *,
    bundle: SequenceBundle,
    frame_labels: np.ndarray,
    checkpoint_dir: str | Path,
    profile: Seq2LabelTrainingContract = SEQ2LABEL_TRAINING,
    augmentation_profile: PredictionAugmentationContract = BLOCKED_RUNNER_AUGMENTATION,
    frames_per_second: int = FRAMES_PER_SECOND,
    device: str | torch.device | None = None,
    seed: int = PAPER_SEED,
) -> Seq2LabelFitSummary:

    if bundle.frames.shape[2:] != (128, 128):
        raise TrainingContractError("seq2label frames must be exactly 128x128")
    if (
        isinstance(frames_per_second, bool)
        or not isinstance(frames_per_second, int)
        or frames_per_second <= 0
    ):
        raise TrainingContractError("frames_per_second must be a positive integer")
    labels = _validated_frame_labels(
        frame_labels,
        num_classes=profile.num_classes,
        frames_per_second=frames_per_second,
    )
    represented_seconds = len(bundle.records)
    if labels.size // frames_per_second != represented_seconds:
        raise TrainingContractError(
            "frame_labels must cover exactly every represented bundle second at "
            "frames_per_second labels per second"
        )
    resolved_device = resolve_device(device)
    labels5 = build_labels5(
        labels,
        start_position=bundle.records[0].position,
        input_steps=profile.input_steps,
        horizons=profile.horizons,
        frames_per_second=frames_per_second,
        proportion=PRIMARY_PROPORTION,
        number_of_classes=profile.num_classes,
    )

    set_reproducible_seeds(seed)
    training_windows = build_prediction_windows(
        bundle.records,
        assigned_split="train",
        input_steps=profile.input_steps,
        target_steps=profile.horizons,
        stride_steps=1,
    )
    validation_windows = build_prediction_windows(
        bundle.records,
        assigned_split="validation",
        input_steps=profile.input_steps,
        target_steps=profile.horizons,
        stride_steps=1,
    )
    training_dataset = _Seq2LabelTrainingDataset(
        PredictionDataset(bundle, training_windows),
        training_windows,
        labels5,
        input_steps=profile.input_steps,
        horizons=profile.horizons,
        augmentation_profile=augmentation_profile,
    )
    validation_dataset = _Seq2LabelTrainingDataset(
        PredictionDataset(bundle, validation_windows),
        validation_windows,
        labels5,
        input_steps=profile.input_steps,
        horizons=profile.horizons,
        augmentation_profile=None,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=profile.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=profile.num_workers,
    )

    training_targets = np.concatenate(
        [
            np.asarray(labels5[window.start_position], dtype=np.int64)
            for window in training_windows
        ]
    )
    class_weights = balanced_seq2label_weights(
        training_targets,
        num_classes=profile.num_classes,
    )

    set_reproducible_seeds(seed)
    model = DirectSeq2Label(
        config=DirectSeq2LabelConfig(
            num_classes=profile.num_classes,
            horizons=profile.horizons,
        )
    )
    return _fit_seq2label_configured(
        model,
        training_loader,
        validation_loader,
        device=resolved_device,
        profile=profile,
        class_weights=class_weights,
        checkpoint_dir=checkpoint_dir,
    )
