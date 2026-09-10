from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from predrheed.data.adaptation import AdaptationBundle
from predrheed.models.classification import CNNTransformer
from predrheed.training.classification import LabelSmoothingCrossEntropy
from predrheed.training.common import (
    OptimizerContract,
    PAPER_SEED,
    TrainingContractError,
    load_state_dict,
    resolve_device,
    save_state_dict,
    set_reproducible_seeds,
)


@dataclass(frozen=True, slots=True)
class AdaptationAugmentationContract:

    brightness_range: tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.02
    erase_probability: float = 0.5
    erase_area_range: tuple[float, float] = (0.05, 0.15)
    erase_aspect_range: tuple[float, float] = (0.3, 3.3)
    clip_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("brightness_range", "erase_area_range", "erase_aspect_range", "clip_range"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not all(isinstance(item, (int, float)) for item in value)
                or value[0] > value[1]
            ):
                raise TrainingContractError(f"{name} must be an ordered (low, high) tuple")
        if not isinstance(self.noise_std, (int, float)) or self.noise_std < 0:
            raise TrainingContractError("noise_std must be nonnegative")
        if not 0 <= self.erase_probability <= 1:
            raise TrainingContractError("erase_probability must be in [0, 1]")


ADAPTATION_AUGMENTATION = AdaptationAugmentationContract()


def augment_adaptation_image(
    image: np.ndarray,
    rng: np.random.Generator,
    profile: AdaptationAugmentationContract = ADAPTATION_AUGMENTATION,
) -> np.ndarray:

    if image.ndim < 2:
        raise TrainingContractError("adaptation image must have at least two dimensions")
    low, high = profile.brightness_range
    image = image * rng.uniform(low, high)
    image = image + rng.normal(0.0, profile.noise_std, size=image.shape).astype(np.float32)
    if rng.random() < profile.erase_probability:
        height, width = image.shape[-2], image.shape[-1]
        area_low, area_high = profile.erase_area_range
        aspect_low, aspect_high = profile.erase_aspect_range
        area = rng.uniform(area_low, area_high) * height * width
        aspect = rng.uniform(aspect_low, aspect_high)
        erase_height = int(np.sqrt(area * aspect))
        erase_width = int(np.sqrt(area / aspect))
        if 1 <= erase_height < height and 1 <= erase_width < width:
            y0 = int(rng.integers(0, height - erase_height))
            x0 = int(rng.integers(0, width - erase_width))
            image[..., y0 : y0 + erase_height, x0 : x0 + erase_width] = float(rng.random())
    clip_low, clip_high = profile.clip_range
    return np.clip(image, clip_low, clip_high).astype(np.float32)


class AdaptationDataset(Dataset):

    def __init__(
        self,
        frames: np.ndarray,
        labels: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator,
        augmentation: AdaptationAugmentationContract = ADAPTATION_AUGMENTATION,
    ) -> None:
        frames = np.asarray(frames)
        labels = np.asarray(labels)
        if frames.ndim != 4 or frames.shape[1] != 1 or frames.shape[2] != 32 or frames.shape[3] != 64:
            raise TrainingContractError("adaptation frames must have shape (N, 1, 32, 64)")
        if frames.dtype not in (np.float32, np.float64):
            raise TrainingContractError("adaptation frames must be float32 or float64")
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise TrainingContractError("adaptation labels must be a one-dimensional integer array")
        if len(labels) != len(frames):
            raise TrainingContractError("adaptation frames and labels must have matching lengths")
        if labels.size and labels.min() < 0:
            raise TrainingContractError("adaptation labels must be zero-based")
        self.frames = frames.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.training = bool(training)
        self.rng = rng
        self._worker_rng: np.random.Generator | None = None
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.frames[index].copy()
        if self.training:
            rng = self.rng
            worker_info = get_worker_info()
            if worker_info is not None:
                if self._worker_rng is None:
                    self._worker_rng = np.random.default_rng(worker_info.seed)
                rng = self._worker_rng
            image = augment_adaptation_image(image, rng, self.augmentation)
        image3 = np.repeat(image, 3, axis=0)
        return torch.from_numpy(image3), int(self.labels[index])


_ADAPTATION_ADAM = OptimizerContract(
    name="Adam",
    learning_rate=5e-5,
    betas=(0.9, 0.999),
    eps=1e-7,
    weight_decay=0.0,
)


@dataclass(frozen=True, slots=True)
class AdaptationTrainingContract:

    source_name: str = "predicted_frame_adaptation"
    batch_size: int = 32
    max_epochs: int = 50
    optimizer: OptimizerContract = _ADAPTATION_ADAM
    warmup_epochs: int = 3
    minimum_learning_rate: float = 5e-7
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    freeze_cnn_epochs: int = 10
    early_stopping_patience: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    num_classes: int = 3
    validation_batch_size: int = 128

    def __post_init__(self) -> None:
        for field_name in ("batch_size", "max_epochs", "num_classes", "validation_batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingContractError(f"{field_name} must be a positive integer")
        for field_name in ("warmup_epochs", "freeze_cnn_epochs", "num_workers"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrainingContractError(f"{field_name} must be a nonnegative integer")
        if self.warmup_epochs > self.max_epochs:
            raise TrainingContractError("warmup_epochs must not exceed max_epochs")
        if not 0 <= self.label_smoothing < 1:
            raise TrainingContractError("label_smoothing must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise TrainingContractError("gradient_clip_norm must be positive")
        if not 0 < self.minimum_learning_rate <= self.optimizer.learning_rate:
            raise TrainingContractError(
                "minimum_learning_rate must be positive and at most the learning rate"
            )
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience <= 0
        ):
            raise TrainingContractError(
                "early_stopping_patience must be None or a positive integer"
            )


ADAPTATION_TRAINING = AdaptationTrainingContract()


def adaptation_learning_rate(
    step: int,
    *,
    steps_per_epoch: int,
    profile: AdaptationTrainingContract = ADAPTATION_TRAINING,
) -> float:

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TrainingContractError("step must be a nonnegative integer")
    if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
        raise TrainingContractError("steps_per_epoch must be a positive integer")
    learning_rate = profile.optimizer.learning_rate
    minimum = profile.minimum_learning_rate
    warmup_steps = profile.warmup_epochs * steps_per_epoch
    cosine_steps = (profile.max_epochs - profile.warmup_epochs) * steps_per_epoch
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, cosine_steps)
    t = min(1.0, max(0.0, t))
    return minimum + 0.5 * (learning_rate - minimum) * (1 + np.cos(np.pi * t))


def adaptation_class_weights(
    labels: np.ndarray,
    *,
    num_classes: int = 3,
) -> torch.Tensor:

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or not np.issubdtype(label_array.dtype, np.integer):
        raise TrainingContractError("labels must be a one-dimensional integer array")
    if label_array.size == 0:
        raise TrainingContractError("labels must not be empty")
    if label_array.min() < 0 or label_array.max() >= num_classes:
        raise TrainingContractError("labels must be zero-based within the class range")
    label_tensor = torch.as_tensor(label_array, dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes)
    if torch.any(counts == 0).item():
        raise TrainingContractError("every class must be present in the adaptation pool")
    return label_tensor.numel() / (num_classes * counts.to(torch.float32))


def set_cnn_trainable(model: torch.nn.Module, trainable: bool) -> None:

    for parameter in model.cnn.parameters():
        parameter.requires_grad = trainable


def build_adaptation_optimizer(
    model: torch.nn.Module,
    profile: AdaptationTrainingContract,
    *,
    learning_rate: float | None = None,
) -> torch.optim.Adam:

    contract = profile.optimizer
    if contract.name != "Adam":
        raise TrainingContractError("adaptation optimizer must be Adam")
    return torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=contract.learning_rate if learning_rate is None else learning_rate,
        betas=contract.betas,
        eps=contract.eps,
        weight_decay=contract.weight_decay,
    )


def evaluate_adaptation_split(
    model: torch.nn.Module,
    loader: object,
    *,
    device: str | torch.device,
) -> tuple[float, float]:

    resolved_device = torch.device(device)
    model.eval()
    sample_count = 0
    correct = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for inputs, target in loader:  # type: ignore[union-attr]
            inputs = inputs.to(resolved_device, non_blocking=True)
            target = target.to(resolved_device, non_blocking=True)
            logits = model(inputs)
            _require_finite_logits(logits)
            loss_sum += float(criterion(logits, target).item())
            predicted = logits.argmax(dim=-1)
            correct += int((predicted == target).sum().item())
            sample_count += target.numel()
    accuracy = correct / sample_count if sample_count else 0.0
    mean_loss = loss_sum / sample_count if sample_count else 0.0
    return accuracy, mean_loss


DEFAULT_DOMINANT_PROPORTION = 0.60
PAPER_TEST_WINDOW_COUNT = 30
PAPER_FORECAST_STEPS = 5
PAPER_FRAME_LABELS_PER_SECOND = 30


def _require_finite_logits(logits: torch.Tensor) -> None:
    if not bool(torch.isfinite(logits).all()):
        raise TrainingContractError("model logits must contain only finite values")


def dominant_class_mask(
    frame_labels: np.ndarray,
    *,
    proportion: float = DEFAULT_DOMINANT_PROPORTION,
    num_classes: int = 3,
) -> np.ndarray:

    labels = np.asarray(frame_labels)
    if labels.ndim < 2 or not np.issubdtype(labels.dtype, np.integer):
        raise TrainingContractError(
            "frame_labels must be an integer array with a trailing frame axis"
        )
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise TrainingContractError("frame_labels are outside the class range")
    if not isinstance(proportion, (int, float)) or not 0 < float(proportion) <= 1:
        raise TrainingContractError("proportion must lie in (0, 1]")
    target_flat = labels.reshape(-1, labels.shape[-1])
    mask = np.zeros((target_flat.shape[0], num_classes), dtype=bool)
    for index, row in enumerate(target_flat):
        frequency = (
            np.bincount(row.astype(np.int64), minlength=num_classes)[:num_classes]
            / row.size
        )
        mask[index] = frequency >= proportion
    return mask


def prepare_predicted_frames_for_epoch_evaluation(
    predicted_frames: np.ndarray,
    frame_labels: np.ndarray,
    *,
    proportion: float = DEFAULT_DOMINANT_PROPORTION,
) -> tuple[torch.Tensor, np.ndarray]:

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on environment
        raise TrainingContractError(
            "OpenCV is required for the classifier-input resize"
        ) from error

    frames = np.asarray(predicted_frames).astype(np.float32)
    targets = np.asarray(frame_labels)
    if frames.ndim != 5 or frames.shape[2] != 1:
        raise TrainingContractError(
            "predicted_frames must have shape (K, horizons, 1, H, W)"
        )
    if targets.ndim != 3 or frames.shape[:2] != targets.shape[:2]:
        raise TrainingContractError(
            f"shape mismatch: frames={frames.shape}, targets={targets.shape}"
        )
    flat = frames.reshape(-1, frames.shape[-2], frames.shape[-1])
    resized = np.empty((flat.shape[0], 1, 32, 64), dtype=np.float32)
    for index, image in enumerate(flat):
        # INTER_LINEAR matches the final cascade's classifier preparation, so
        # epoch selection scores frames prepared identically to deployment.
        resized[index, 0] = cv2.resize(image, (64, 32), interpolation=cv2.INTER_LINEAR)
    resized3 = np.repeat(resized, 3, axis=1)
    mask = dominant_class_mask(targets, proportion=proportion)
    return torch.from_numpy(resized3), mask


def evaluate_predicted_frames_dominant(
    model: torch.nn.Module,
    prepared_frames: torch.Tensor,
    dominant_mask: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int = 128,
    forecast_steps: int = 5,
) -> tuple[float, tuple[float, ...]]:

    if prepared_frames.shape[0] != dominant_mask.shape[0]:
        raise TrainingContractError("prepared frames and dominance mask must align")
    if prepared_frames.shape[0] % forecast_steps != 0:
        raise TrainingContractError(
            "prepared frame count must be a multiple of forecast_steps"
        )
    resolved_device = torch.device(device)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, prepared_frames.shape[0], batch_size):
            batch = prepared_frames[start : start + batch_size].to(
                resolved_device, non_blocking=True
            )
            logits = model(batch)
            _require_finite_logits(logits)
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    predicted = np.concatenate(predictions)
    correct = dominant_mask[np.arange(len(predicted)), predicted]
    overall = float(correct.mean() * 100.0)
    per_horizon = tuple(
        float(correct.reshape(-1, forecast_steps)[:, h].mean() * 100.0)
        for h in range(forecast_steps)
    )
    return overall, per_horizon


TestCascadeEpochEvaluation = Callable[
    [torch.nn.Module, torch.device], tuple[float, tuple[float, ...]]
]


def dominant_proportion_evaluation(
    predicted_frames: np.ndarray,
    frame_labels: np.ndarray,
    *,
    proportion: float = DEFAULT_DOMINANT_PROPORTION,
    batch_size: int = 128,
) -> TestCascadeEpochEvaluation:

    prepared, mask = prepare_predicted_frames_for_epoch_evaluation(
        predicted_frames, frame_labels, proportion=proportion
    )
    forecast_steps = int(np.asarray(predicted_frames).shape[1])

    def _evaluate(
        model: torch.nn.Module, device: torch.device
    ) -> tuple[float, tuple[float, ...]]:
        return evaluate_predicted_frames_dominant(
            model,
            prepared,
            mask,
            device=device,
            batch_size=batch_size,
            forecast_steps=forecast_steps,
        )

    return _evaluate


def paper_test_cascade_evaluation(
    predicted_frames: np.ndarray,
    frame_labels: np.ndarray,
    *,
    batch_size: int = 128,
) -> TestCascadeEpochEvaluation:
    frames = np.asarray(predicted_frames)
    labels = np.asarray(frame_labels)
    if frames.ndim != 5 or labels.ndim != 3:
        raise TrainingContractError(
            "paper test arrays must have five frame and three label dimensions"
        )
    if (
        frames.shape[0] != PAPER_TEST_WINDOW_COUNT
        or labels.shape[0] != PAPER_TEST_WINDOW_COUNT
    ):
        raise TrainingContractError(
            "paper test evaluation requires exactly 30 windows"
        )
    if (
        frames.shape[1] != PAPER_FORECAST_STEPS
        or labels.shape[1] != PAPER_FORECAST_STEPS
    ):
        raise TrainingContractError(
            "paper test evaluation requires exactly five forecast horizons"
        )
    if labels.shape[2] != PAPER_FRAME_LABELS_PER_SECOND:
        raise TrainingContractError(
            "paper test evaluation requires 30 frame labels per target second"
        )
    return dominant_proportion_evaluation(
        frames,
        labels,
        proportion=DEFAULT_DOMINANT_PROPORTION,
        batch_size=batch_size,
    )


@dataclass(frozen=True, slots=True)
class AdaptationEpochRecord:

    epoch: int
    learning_rate: float
    training_loss: float
    training_accuracy: float
    validation_loss: float
    validation_accuracy: float
    cnn_frozen: bool
    test_cascade_accuracy: float | None
    test_per_horizon_accuracy: tuple[float, ...] | None
    checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class AdaptationFitSummary:

    epochs_run: int
    best_validation_epoch: int
    best_validation_accuracy: float
    peak_test_epoch: int | None
    peak_test_accuracy: float | None
    peak_test_checkpoint_path: Path | None
    history: tuple[AdaptationEpochRecord, ...]


def _validate_test_epoch_accuracy(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingContractError(
            f"test epoch evaluation {name} must be a real number"
        )
    validated = float(value)
    if not math.isfinite(validated):
        raise TrainingContractError(
            f"test epoch evaluation {name} must be finite"
        )
    if not 0.0 <= validated <= 100.0:
        raise TrainingContractError(
            f"test epoch evaluation {name} must lie in [0, 100]"
        )
    return validated


def _validate_test_epoch_result(
    result: object,
) -> tuple[float, tuple[float, ...]]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TrainingContractError(
            "test epoch evaluation must return (overall, per_horizon)"
        )
    overall, per_horizon = result
    if not isinstance(per_horizon, tuple):
        raise TrainingContractError(
            "test epoch evaluation per_horizon must be a tuple"
        )
    if len(per_horizon) != PAPER_FORECAST_STEPS:
        raise TrainingContractError(
            "test epoch evaluation per_horizon must contain exactly five values"
        )
    return (
        _validate_test_epoch_accuracy(overall, "overall accuracy"),
        tuple(
            _validate_test_epoch_accuracy(value, f"horizon {index} accuracy")
            for index, value in enumerate(per_horizon, start=1)
        ),
    )


def _fit_adaptation_configured(
    model: torch.nn.Module,
    training_loader: object,
    validation_loader: object,
    *,
    device: str | torch.device,
    profile: AdaptationTrainingContract,
    class_weights: torch.Tensor,
    checkpoint_dir: str | Path | None,
    test_epoch_evaluation: TestCascadeEpochEvaluation | None,
) -> AdaptationFitSummary:

    if test_epoch_evaluation is not None and checkpoint_dir is None:
        raise TrainingContractError(
            "checkpoint_dir is required when test_epoch_evaluation is provided"
        )
    resolved_device = torch.device(device)
    try:
        steps_per_epoch = max(1, len(training_loader))  # type: ignore[arg-type]
    except TypeError as error:
        raise TrainingContractError("training loader must define its length") from error

    criterion = LabelSmoothingCrossEntropy(
        num_classes=profile.num_classes,
        smoothing=profile.label_smoothing,
        class_weights=class_weights,
    )
    optimizer = build_adaptation_optimizer(model, profile)
    set_cnn_trainable(model, False)
    cnn_frozen = True

    history: list[AdaptationEpochRecord] = []
    best_validation_accuracy = -1.0
    best_validation_epoch = -1
    patience_left = profile.early_stopping_patience
    global_step = 0

    for epoch in range(profile.max_epochs):
        if cnn_frozen and epoch >= profile.freeze_cnn_epochs:
            set_cnn_trainable(model, True)
            cnn_frozen = False
            optimizer = build_adaptation_optimizer(
                model,
                profile,
                learning_rate=optimizer.param_groups[0]["lr"],
            )

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        current_lr = optimizer.param_groups[0]["lr"]
        for inputs, target in training_loader:  # type: ignore[union-attr]
            inputs = inputs.to(resolved_device, non_blocking=True)
            target = target.to(resolved_device, non_blocking=True)

            current_lr = adaptation_learning_rate(
                global_step, steps_per_epoch=steps_per_epoch, profile=profile
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            _require_finite_logits(logits)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                profile.gradient_clip_norm,
            )
            optimizer.step()
            global_step += 1

            total_loss += float(loss.item()) * target.numel()
            total_correct += int((logits.argmax(dim=-1) == target).sum().item())
            total_count += target.numel()

        if total_count == 0:
            raise TrainingContractError("training loader must not be empty")
        training_loss = total_loss / total_count
        training_accuracy = total_correct / total_count
        validation_accuracy, validation_loss = evaluate_adaptation_split(
            model, validation_loader, device=resolved_device
        )

        predicted_accuracy: float | None = None
        per_horizon: tuple[float, ...] | None = None
        if test_epoch_evaluation is not None:
            predicted_accuracy, per_horizon = _validate_test_epoch_result(
                test_epoch_evaluation(model, resolved_device)
            )

        checkpoint_path: Path | None = None
        if checkpoint_dir is not None:
            checkpoint_path = save_state_dict(
                model, Path(checkpoint_dir) / f"adaptation_epoch_{epoch + 1:03d}.pth"
            )

        history.append(
            AdaptationEpochRecord(
                epoch=epoch + 1,
                learning_rate=float(current_lr),
                training_loss=training_loss,
                training_accuracy=training_accuracy,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                cnn_frozen=cnn_frozen,
                test_cascade_accuracy=predicted_accuracy,
                test_per_horizon_accuracy=per_horizon,
                checkpoint_path=checkpoint_path,
            )
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_validation_epoch = epoch + 1
            if patience_left is not None:
                patience_left = profile.early_stopping_patience
        elif patience_left is not None:
            patience_left -= 1
        if patience_left is not None and patience_left <= 0:
            break

    evaluated = [
        record for record in history if record.test_cascade_accuracy is not None
    ]
    if evaluated:
        peak_test_record = max(
            evaluated, key=lambda record: record.test_cascade_accuracy
        )
        peak_test_epoch: int | None = peak_test_record.epoch
        peak_test_accuracy: float | None = peak_test_record.test_cascade_accuracy
        peak_test_checkpoint_path = peak_test_record.checkpoint_path
    else:
        peak_test_epoch = None
        peak_test_accuracy = None
        peak_test_checkpoint_path = None

    return AdaptationFitSummary(
        epochs_run=len(history),
        best_validation_epoch=best_validation_epoch,
        best_validation_accuracy=best_validation_accuracy,
        peak_test_epoch=peak_test_epoch,
        peak_test_accuracy=peak_test_accuracy,
        peak_test_checkpoint_path=peak_test_checkpoint_path,
        history=tuple(history),
    )


def fit_adapted_classifier(
    *,
    bundle: AdaptationBundle,
    initial_checkpoint: str | Path,
    profile: AdaptationTrainingContract = ADAPTATION_TRAINING,
    augmentation: AdaptationAugmentationContract = ADAPTATION_AUGMENTATION,
    checkpoint_dir: str | Path | None = None,
    class_weights: torch.Tensor | None = None,
    test_epoch_evaluation: TestCascadeEpochEvaluation | None = None,
    device: str | torch.device | None = None,
    seed: int = PAPER_SEED,
) -> AdaptationFitSummary:

    if not isinstance(bundle, AdaptationBundle):
        raise TrainingContractError("bundle must be an AdaptationBundle")
    if not isinstance(profile, AdaptationTrainingContract):
        raise TrainingContractError("profile must be an AdaptationTrainingContract")
    if not isinstance(augmentation, AdaptationAugmentationContract):
        raise TrainingContractError(
            "augmentation must be an AdaptationAugmentationContract"
        )
    if checkpoint_dir is None:
        raise TrainingContractError(
            "checkpoint_dir is required for the paper adaptation workflow"
        )
    if test_epoch_evaluation is None:
        raise TrainingContractError(
            "test_epoch_evaluation is required for the paper adaptation workflow"
        )

    set_reproducible_seeds(seed)
    resolved_device = resolve_device(device)

    model = CNNTransformer(
        num_classes=profile.num_classes,
        dr=0.3,
        projection_dim=64,
        num_heads=4,
        transformer_layers=8,
        input_shape=(3, 32, 64),
    )
    load_state_dict(model, initial_checkpoint, device="cpu")
    model.to(resolved_device)

    training_dataset = AdaptationDataset(
        bundle.frames,
        bundle.labels,
        training=True,
        rng=np.random.default_rng(seed),
        augmentation=augmentation,
    )
    validation_dataset = AdaptationDataset(
        bundle.validation_frames,
        bundle.validation_labels,
        training=False,
        rng=np.random.default_rng(seed + 1),
        augmentation=augmentation,
    )
    pin_memory = profile.pin_memory and resolved_device.type == "cuda"
    training_loader = DataLoader(
        training_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=profile.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=profile.validation_batch_size,
        shuffle=False,
        num_workers=profile.num_workers,
        pin_memory=pin_memory,
    )
    if class_weights is None:
        class_weights = adaptation_class_weights(
            bundle.labels, num_classes=profile.num_classes
        )
    return _fit_adaptation_configured(
        model,
        training_loader,
        validation_loader,
        device=resolved_device,
        profile=profile,
        class_weights=class_weights,
        checkpoint_dir=checkpoint_dir,
        test_epoch_evaluation=test_epoch_evaluation,
    )


__all__ = [
    "ADAPTATION_AUGMENTATION",
    "ADAPTATION_TRAINING",
    "AdaptationAugmentationContract",
    "AdaptationDataset",
    "TestCascadeEpochEvaluation",
    "AdaptationEpochRecord",
    "AdaptationFitSummary",
    "AdaptationTrainingContract",
    "DEFAULT_DOMINANT_PROPORTION",
    "PAPER_FORECAST_STEPS",
    "PAPER_FRAME_LABELS_PER_SECOND",
    "PAPER_TEST_WINDOW_COUNT",
    "adaptation_class_weights",
    "adaptation_learning_rate",
    "augment_adaptation_image",
    "build_adaptation_optimizer",
    "dominant_class_mask",
    "dominant_proportion_evaluation",
    "evaluate_adaptation_split",
    "evaluate_predicted_frames_dominant",
    "fit_adapted_classifier",
    "paper_test_cascade_evaluation",
    "prepare_predicted_frames_for_epoch_evaluation",
    "set_cnn_trainable",
]
