from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from predrheed.data.datasets import ClassificationBundle, ClassificationDataset
from predrheed.models.classification import CNNTransformer

from predrheed.training.common import (
    OptimizerContract,
    PAPER_SEED,
    TrainingContractError,
    require_nonempty_loader as _require_nonempty_loader,
    resolve_device,
    save_state_dict,
    set_reproducible_seeds,
)


ClassifierModelName = Literal["cnn_transformer"]


@dataclass(frozen=True, slots=True)
class WarmupCosineClassifierSchedulerContract:
    kind: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = 5
    start_factor: float = 0.1
    eta_min: float = 1e-6
    step_unit: Literal["batch"] = "batch"


@dataclass(frozen=True, slots=True)
class ClassifierTrainingContract:

    source_name: Literal["cnn_transformer"]
    batch_size: int
    max_epochs: int
    optimizer: OptimizerContract
    scheduler: WarmupCosineClassifierSchedulerContract
    label_smoothing: float
    gradient_clip_norm: float
    early_stopping_patience: int
    num_workers: int = 0
    pin_memory: bool = False
    num_classes: int = 3

    def __post_init__(self) -> None:
        for field_name in (
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "num_classes",
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


_CLASSIFIER_ADAM = OptimizerContract(
    name="Adam",
    learning_rate=5e-4,
    betas=(0.9, 0.999),
    eps=1e-7,
    weight_decay=0.0,
)

CNN_TRANSFORMER_TRAINING = ClassifierTrainingContract(
    source_name="cnn_transformer",
    batch_size=32,
    max_epochs=130,
    optimizer=_CLASSIFIER_ADAM,
    scheduler=WarmupCosineClassifierSchedulerContract(),
    label_smoothing=0.1,
    gradient_clip_norm=1.0,
    early_stopping_patience=35,
    num_workers=0,
    pin_memory=True,
)


class LabelSmoothingCrossEntropy(torch.nn.Module):

    def __init__(
        self,
        *,
        num_classes: int = 3,
        smoothing: float,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TrainingContractError("num_classes must be an integer")
        if num_classes < 2:
            raise TrainingContractError("num_classes must be at least two")
        if not 0 <= smoothing < 1:
            raise TrainingContractError("smoothing must be in [0, 1)")
        if class_weights is not None and class_weights.shape != (num_classes,):
            raise TrainingContractError(
                f"class_weights must have shape ({num_classes},)"
            )
        self.num_classes = num_classes
        self.smoothing = float(smoothing)
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise TrainingContractError(
                f"logits must have shape (N, {self.num_classes})"
            )
        if target.ndim != 1 or target.shape[0] != logits.shape[0]:
            raise TrainingContractError("target must have shape (N,)")
        if target.numel() == 0:
            raise TrainingContractError("target must not be empty")
        if target.dtype != torch.long:
            raise TrainingContractError("target must use torch.long labels")
        if target.min().item() < 0 or target.max().item() >= self.num_classes:
            raise TrainingContractError("target contains an out-of-range label")

        with torch.no_grad():
            distribution = torch.zeros_like(logits)
            distribution.fill_(self.smoothing / (self.num_classes - 1))
            distribution.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -distribution * F.log_softmax(logits, dim=1)
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)[target]
            loss = loss.sum(dim=1) * weights
        else:
            loss = loss.sum(dim=1)
        return loss.mean()


class _ClassificationTrainingDataset(Dataset):

    def __init__(self, dataset: ClassificationDataset, *, training: bool) -> None:
        self._dataset = dataset
        self.training = training

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frame, label = self._dataset[index]
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        return tensor, label


def _build_classifier_model(model_name: ClassifierModelName) -> torch.nn.Module:
    if model_name == "cnn_transformer":
        return CNNTransformer()
    raise TrainingContractError(f"unknown classifier model: {model_name!r}")


def _expected_classifier_profile(
    model_name: ClassifierModelName,
) -> ClassifierTrainingContract:
    if model_name == "cnn_transformer":
        return CNN_TRANSFORMER_TRAINING
    raise TrainingContractError(f"unknown classifier model: {model_name!r}")


def build_classifier_optimizer(
    model: torch.nn.Module,
    profile: ClassifierTrainingContract,
) -> torch.optim.Adam:

    contract = profile.optimizer
    if contract.name != "Adam":
        raise TrainingContractError("classifier optimizer must be Adam")
    return torch.optim.Adam(
        model.parameters(),
        lr=contract.learning_rate,
        betas=contract.betas,
        eps=contract.eps,
        weight_decay=contract.weight_decay,
    )


@dataclass(frozen=True, slots=True)
class ClassifierScheduler:
    scheduler: object
    step_unit: Literal["batch"]


def build_classifier_scheduler(
    optimizer: torch.optim.Optimizer,
    profile: ClassifierTrainingContract,
    *,
    steps_per_epoch: int,
) -> ClassifierScheduler:

    if isinstance(steps_per_epoch, bool) or steps_per_epoch <= 0:
        raise TrainingContractError("steps_per_epoch must be positive")
    contract = profile.scheduler
    if isinstance(contract, WarmupCosineClassifierSchedulerContract):
        warmup_steps = contract.warmup_epochs * steps_per_epoch
        cosine_steps = (profile.max_epochs - contract.warmup_epochs) * steps_per_epoch
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=contract.start_factor,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=contract.eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        return ClassifierScheduler(scheduler=scheduler, step_unit="batch")

    raise TrainingContractError(f"unsupported classifier scheduler: {contract!r}")


@dataclass(frozen=True, slots=True)
class ClassificationEpochMetrics:
    loss: float
    accuracy: float
    recall_by_class: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ClassificationFitSummary:
    best_epoch: int
    epochs_run: int
    best_validation_accuracy: float
    training_history: tuple[ClassificationEpochMetrics, ...]
    validation_history: tuple[ClassificationEpochMetrics, ...]
    checkpoint_path: Path


def train_classification_epoch(
    model: torch.nn.Module,
    loader: object,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device,
    gradient_clip_norm: float,
    batch_scheduler: object | None = None,
) -> ClassificationEpochMetrics:

    batch_count = _require_nonempty_loader(loader, "training")
    resolved_device = torch.device(device)
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for inputs, target in loader:  # type: ignore[union-attr]
        inputs = inputs.to(resolved_device)
        target = target.to(resolved_device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()

        total_loss += float(loss.item())
        predicted = logits.argmax(dim=1)
        sample_count += int(target.shape[0])
        correct += int(predicted.eq(target).sum().item())

    return ClassificationEpochMetrics(
        loss=total_loss / batch_count,
        accuracy=correct / sample_count,
    )


def evaluate_classifier(
    model: torch.nn.Module,
    loader: object,
    loss_fn: torch.nn.Module,
    *,
    device: str | torch.device,
    num_classes: int = 3,
) -> ClassificationEpochMetrics:

    _require_nonempty_loader(loader, "validation")
    resolved_device = torch.device(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, target in loader:  # type: ignore[union-attr]
            inputs = inputs.to(resolved_device)
            target = target.to(resolved_device)
            logits = model(inputs)
            loss = loss_fn(logits, target)
            batch_sample_count = int(target.shape[0])
            total_loss += float(loss.item()) * batch_sample_count
            predicted = logits.argmax(dim=1)
            sample_count += batch_sample_count
            correct += int(predicted.eq(target).sum().item())
            for class_index in range(num_classes):
                class_mask = target == class_index
                class_total[class_index] += int(class_mask.sum().item())
                class_correct[class_index] += int(
                    ((predicted == class_index) & class_mask).sum().item()
                )

    recalls = tuple(
        class_correct[index] / max(class_total[index], 1)
        for index in range(num_classes)
    )
    return ClassificationEpochMetrics(
        loss=total_loss / sample_count,
        accuracy=correct / sample_count,
        recall_by_class=recalls,
    )


def _fit_classifier_configured(
    model: torch.nn.Module,
    training_loader: object,
    validation_loader: object,
    *,
    device: str | torch.device,
    profile: ClassifierTrainingContract,
    checkpoint_path: str | Path,
    class_weights: torch.Tensor | None = None,
) -> ClassificationFitSummary:

    configured_device = torch.device(device)
    checkpoint_target = Path(checkpoint_path)

    steps_per_epoch = _require_nonempty_loader(training_loader, "training")
    _require_nonempty_loader(validation_loader, "validation")
    model.to(configured_device)
    loss_fn = LabelSmoothingCrossEntropy(
        num_classes=profile.num_classes,
        smoothing=profile.label_smoothing,
        class_weights=class_weights,
    )
    optimizer = build_classifier_optimizer(model, profile)
    scheduler = build_classifier_scheduler(
        optimizer,
        profile,
        steps_per_epoch=steps_per_epoch,
    )

    training_history: list[ClassificationEpochMetrics] = []
    validation_history: list[ClassificationEpochMetrics] = []
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    patience_counter = 0
    resolved_checkpoint: Path | None = None

    for epoch_index in range(profile.max_epochs):
        training_metrics = train_classification_epoch(
            model,
            training_loader,
            loss_fn,
            optimizer,
            device=configured_device,
            gradient_clip_norm=profile.gradient_clip_norm,
            batch_scheduler=(
                scheduler.scheduler if scheduler.step_unit == "batch" else None
            ),
        )
        validation_metrics = evaluate_classifier(
            model,
            validation_loader,
            loss_fn,
            device=configured_device,
            num_classes=profile.num_classes,
        )
        training_history.append(training_metrics)
        validation_history.append(validation_metrics)

        if validation_metrics.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_metrics.accuracy
            best_epoch = epoch_index + 1
            resolved_checkpoint = save_state_dict(model, checkpoint_target)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= profile.early_stopping_patience:
            break

    if resolved_checkpoint is None:
        raise RuntimeError("classifier training produced no checkpoint")
    return ClassificationFitSummary(
        best_epoch=best_epoch,
        epochs_run=len(training_history),
        best_validation_accuracy=best_validation_accuracy,
        training_history=tuple(training_history),
        validation_history=tuple(validation_history),
        checkpoint_path=resolved_checkpoint,
    )


def fit_classifier(
    *,
    model_name: ClassifierModelName,
    bundle: ClassificationBundle,
    profile: ClassifierTrainingContract | None = None,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    seed: int = PAPER_SEED,
) -> ClassificationFitSummary:

    if profile is None:
        profile = _expected_classifier_profile(model_name)
    set_reproducible_seeds(seed)
    resolved_device = resolve_device(device)
    model = _build_classifier_model(model_name)
    training_dataset = _ClassificationTrainingDataset(
        ClassificationDataset(bundle, "train"),
        training=True,
    )
    validation_dataset = _ClassificationTrainingDataset(
        ClassificationDataset(bundle, "validation"),
        training=False,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
    )
    return _fit_classifier_configured(
        model,
        training_loader,
        validation_loader,
        device=resolved_device,
        profile=profile,
        checkpoint_path=checkpoint_path,
    )
