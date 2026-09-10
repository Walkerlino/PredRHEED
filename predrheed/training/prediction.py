from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import mean_squared_error as skimage_mean_squared_error
from skimage.metrics import structural_similarity as skimage_structural_similarity
from torch.utils.data import DataLoader, Dataset

from predrheed.data.datasets import PredictionDataset, SequenceBundle
from predrheed.data.windows import build_prediction_windows
from predrheed.models.prediction import MSAMConvLSTM, SAConvLSTM, SimVP

from predrheed.training.common import (
    OptimizerContract,
    PAPER_SEED,
    TrainingContractError,
    require_nonempty_loader as _require_nonempty_loader,
    resolve_device,
    save_state_dict,
    set_reproducible_seeds,
)


@dataclass(frozen=True, slots=True)
class PredictionAugmentationContract:

    source_name: Literal["direct_v3", "blocked_runner"]
    horizontal_flip_probability: float
    brightness_range: tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.01
    crop_size: int = 112
    crop_probability: float = 0.5
    clamp_range: tuple[float, float] = (0.0, 1.0)


DIRECT_V3_AUGMENTATION = PredictionAugmentationContract(
    source_name="direct_v3",
    horizontal_flip_probability=0.5,
)

BLOCKED_RUNNER_AUGMENTATION = PredictionAugmentationContract(
    source_name="blocked_runner",
    horizontal_flip_probability=0.0,
)


PredictorModelName = Literal[
    "msam_convlstm",
    "sa_convlstm",
    "simvp",
]


ResizeFrame = Callable[[np.ndarray, tuple[int, int]], np.ndarray]


def augment_prediction_sequence(
    sequence: np.ndarray,
    profile: PredictionAugmentationContract,
    *,
    resize_frame: ResizeFrame,
) -> np.ndarray:

    if sequence.ndim != 3:
        raise TrainingContractError("prediction sequence must have shape (T, H, W)")
    if not np.issubdtype(sequence.dtype, np.floating):
        raise TrainingContractError("prediction sequence must use a floating dtype")
    augmented = sequence.astype(np.float32, copy=True)

    if (
        profile.horizontal_flip_probability > 0
        and np.random.rand() < profile.horizontal_flip_probability
    ):
        augmented = augmented[:, :, ::-1].copy()

    brightness = np.random.uniform(*profile.brightness_range)
    low, high = profile.clamp_range
    augmented = np.clip(augmented * brightness, low, high)

    noise = np.random.normal(0.0, profile.noise_std, augmented.shape).astype(
        np.float32
    )
    augmented = np.clip(augmented + noise, low, high)

    if np.random.rand() < profile.crop_probability:
        time_steps, height, width = augmented.shape
        if height < profile.crop_size or width < profile.crop_size:
            raise TrainingContractError(
                "sequence dimensions must be at least as large as crop_size"
            )
        y_start = np.random.randint(0, height - profile.crop_size + 1)
        x_start = np.random.randint(0, width - profile.crop_size + 1)
        cropped = augmented[
            :,
            y_start : y_start + profile.crop_size,
            x_start : x_start + profile.crop_size,
        ]
        resized = np.zeros_like(augmented)
        for time_index in range(time_steps):
            resized_frame = resize_frame(cropped[time_index], (width, height))
            if resized_frame.shape != (height, width):
                raise TrainingContractError(
                    "resize_frame must return an array with the requested shape"
                )
            resized[time_index] = resized_frame
        augmented = resized

    return augmented.astype(np.float32, copy=False)


def _opencv_resize_frame(
    frame: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:

    import cv2

    resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    expected_shape = (size[1], size[0])
    if resized.shape != expected_shape:
        raise TrainingContractError(
            "OpenCV resize did not return the requested frame shape"
        )
    return resized


@dataclass(frozen=True, slots=True)
class PredictionSchedulerContract:
    kind: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = 5
    warmup_initial_lr: float = 1e-5
    eta_min: float = 1e-6
    step_unit: Literal["epoch"] = "epoch"


@dataclass(frozen=True, slots=True)
class PredictorTrainingContract:

    source_name: str
    forecast_horizon_seconds: int
    batch_size: int
    gradient_accumulation_steps: int
    max_epochs: int
    optimizer: OptimizerContract
    scheduler: PredictionSchedulerContract
    early_stopping_patience: int
    gradient_clip_norm: float
    num_workers: int

    def __post_init__(self) -> None:
        for field_name in (
            "forecast_horizon_seconds",
            "batch_size",
            "gradient_accumulation_steps",
            "max_epochs",
            "early_stopping_patience",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingContractError(f"{field_name} must be a positive integer")
        if self.gradient_clip_norm <= 0:
            raise TrainingContractError("gradient_clip_norm must be positive")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise TrainingContractError("num_workers must be a nonnegative integer")


_PREDICTOR_ADAMW = OptimizerContract(
    name="AdamW",
    learning_rate=5e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)


def _predictor_profile(
    *,
    source_name: str,
    forecast_horizon_seconds: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> PredictorTrainingContract:
    return PredictorTrainingContract(
        source_name=source_name,
        forecast_horizon_seconds=forecast_horizon_seconds,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_epochs=100,
        optimizer=_PREDICTOR_ADAMW,
        scheduler=PredictionSchedulerContract(),
        early_stopping_patience=10,
        gradient_clip_norm=1.0,
        num_workers=0,
    )


PREDICTOR_5S_TRAINING = _predictor_profile(
    source_name="predictor_5s",
    forecast_horizon_seconds=5,
    batch_size=4,
    gradient_accumulation_steps=1,
)

PREDICTOR_15S_TRAINING = _predictor_profile(
    source_name="predictor_15s",
    forecast_horizon_seconds=15,
    batch_size=4,
    gradient_accumulation_steps=1,
)

PREDICTOR_30S_TRAINING = _predictor_profile(
    source_name="predictor_30s",
    forecast_horizon_seconds=30,
    batch_size=2,
    gradient_accumulation_steps=2,
)


class _PredictionTrainingDataset(Dataset):

    def __init__(
        self,
        dataset: PredictionDataset,
        *,
        input_steps: int,
        target_steps: int,
        augmentation_profile: PredictionAugmentationContract | None,
    ) -> None:
        self._dataset = dataset
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.augmentation_profile = augmentation_profile

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        observed, target = self._dataset[index]
        if observed.shape != (self.input_steps, 1, 128, 128):
            raise TrainingContractError("observed sequence must have shape (15,1,128,128)")
        if target.shape != (self.target_steps, 1, 128, 128):
            raise TrainingContractError(
                "target sequence does not match the configured forecast horizon"
            )
        if self.augmentation_profile is not None:
            combined = np.concatenate((observed, target), axis=0)[:, 0]
            augmented = augment_prediction_sequence(
                combined,
                self.augmentation_profile,
                resize_frame=_opencv_resize_frame,
            )
            observed = augmented[: self.input_steps, np.newaxis]
            target = augmented[self.input_steps :, np.newaxis]
        observed_array = np.ascontiguousarray(observed, dtype=np.float32)
        target_array = np.ascontiguousarray(target, dtype=np.float32)
        return torch.from_numpy(observed_array), torch.from_numpy(target_array)


def _build_predictor_model(model_name: PredictorModelName) -> torch.nn.Module:
    if model_name == "msam_convlstm":
        return MSAMConvLSTM()
    if model_name == "sa_convlstm":
        return SAConvLSTM()
    if model_name == "simvp":
        return SimVP()
    raise TrainingContractError(f"unknown predictor model: {model_name!r}")


def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    if predicted.shape != target.shape:
        raise TrainingContractError("predicted and target tensors must have equal shapes")
    if predicted.ndim < 2 or predicted.shape[1] <= 0:
        raise TrainingContractError("prediction tensors must include target steps")
    base_mse = F.mse_loss(predicted, target)
    per_step_mse = torch.mean(
        torch.stack(
            [
                F.mse_loss(predicted[:, step], target[:, step])
                for step in range(predicted.shape[1])
            ]
        )
    )
    return base_mse + 0.1 * per_step_mse


def build_predictor_optimizer(
    model: torch.nn.Module,
    profile: PredictorTrainingContract,
) -> torch.optim.AdamW:

    contract = profile.optimizer
    if contract.name != "AdamW":
        raise TrainingContractError("predictor optimizer must be AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        betas=contract.betas,
        eps=contract.eps,
        weight_decay=contract.weight_decay,
    )


def build_predictor_scheduler(
    optimizer: torch.optim.Optimizer,
    profile: PredictorTrainingContract,
) -> "PredictorScheduler":

    contract = profile.scheduler
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=profile.max_epochs - contract.warmup_epochs,
        eta_min=contract.eta_min,
    )
    return PredictorScheduler(
        optimizer,
        cosine_scheduler=cosine,
        learning_rate=profile.optimizer.learning_rate,
        warmup_epochs=contract.warmup_epochs,
        warmup_initial_lr=contract.warmup_initial_lr,
    )


@dataclass(slots=True)
class PredictorScheduler:

    optimizer: torch.optim.Optimizer
    cosine_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    learning_rate: float
    warmup_epochs: int
    warmup_initial_lr: float

    def prepare_epoch(self, epoch_index: int) -> None:
        if epoch_index < self.warmup_epochs:
            if self.warmup_epochs == 1:
                learning_rate = self.learning_rate
            else:
                learning_rate = self.warmup_initial_lr + (
                    self.learning_rate - self.warmup_initial_lr
                ) * epoch_index / (self.warmup_epochs - 1)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate

    def step_after_epoch(self, epoch_index: int) -> None:
        cosine_start = self.warmup_epochs - 1
        if epoch_index >= cosine_start:
            self.cosine_scheduler.step()


@dataclass(frozen=True, slots=True)
class PredictionEpochMetrics:
    loss: float
    ssim: float | None = None
    mse_scaled: float | None = None
    mae_scaled: float | None = None


@dataclass(frozen=True, slots=True)
class PredictionFitSummary:
    best_epoch: int
    epochs_run: int
    best_validation_ssim: float
    training_history: tuple[PredictionEpochMetrics, ...]
    validation_history: tuple[PredictionEpochMetrics, ...]
    checkpoint_path: Path


def _validate_prediction_batch(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    expected_forecast_steps: int,
) -> None:
    if predicted.shape != target.shape:
        raise TrainingContractError("model output and target shapes must match")
    if target.ndim < 2 or target.shape[1] != expected_forecast_steps:
        raise TrainingContractError(
            f"target must contain exactly {expected_forecast_steps} forecast steps"
        )


def calculate_prediction_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float, float]:

    if predicted.shape != target.shape:
        raise TrainingContractError("predicted and target tensors must have equal shapes")
    if predicted.ndim != 5 or predicted.shape[2] != 1:
        raise TrainingContractError(
            "prediction metrics require shape (N, T, 1, H, W)"
        )
    predicted_array = predicted.detach().cpu().numpy()
    target_array = target.detach().cpu().numpy()
    ssim_scores: list[float] = []
    mse_scores: list[float] = []
    mae_scores: list[float] = []
    for batch_index in range(predicted_array.shape[0]):
        for step_index in range(predicted_array.shape[1]):
            predicted_frame = predicted_array[batch_index, step_index, 0]
            target_frame = target_array[batch_index, step_index, 0]
            ssim_scores.append(
                float(
                    skimage_structural_similarity(
                        target_frame,
                        predicted_frame,
                        data_range=1.0,
                    )
                )
            )
            mse_scores.append(
                float(skimage_mean_squared_error(target_frame, predicted_frame))
            )
            mae_scores.append(float(np.mean(np.abs(target_frame - predicted_frame))))
    return (
        float(np.mean(ssim_scores)),
        float(np.mean(mse_scores) * 1000),
        float(np.mean(mae_scores) * 1000),
    )


def train_prediction_epoch(
    model: torch.nn.Module,
    loader: object,
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device,
    gradient_clip_norm: float,
    gradient_accumulation_steps: int,
    expected_forecast_steps: int,
) -> PredictionEpochMetrics:

    batch_count = _require_nonempty_loader(loader, "training")
    resolved_device = torch.device(device)
    if gradient_accumulation_steps <= 0:
        raise TrainingContractError("gradient_accumulation_steps must be positive")
    model.train()
    total_loss = 0.0
    total_sample_count = 0
    accumulated_sample_count = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (observed, target) in enumerate(loader, start=1):  # type: ignore[union-attr]
        observed = observed.to(resolved_device)
        target = target.to(resolved_device)
        predicted = model(observed, future_seq=target.shape[1])
        _validate_prediction_batch(
            predicted,
            target,
            expected_forecast_steps=expected_forecast_steps,
        )
        loss = prediction_loss(predicted, target)
        batch_sample_count = int(target.shape[0])
        if batch_sample_count <= 0:
            raise TrainingContractError("prediction batches must not be empty")
        (loss * batch_sample_count).backward()
        accumulated_sample_count += batch_sample_count

        if (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == batch_count
        ):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_sample_count)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_sample_count = 0
        total_loss += float(loss.item()) * batch_sample_count
        total_sample_count += batch_sample_count

    return PredictionEpochMetrics(loss=total_loss / total_sample_count)


def evaluate_prediction_loss(
    model: torch.nn.Module,
    loader: object,
    *,
    device: str | torch.device,
    expected_forecast_steps: int,
) -> PredictionEpochMetrics:

    _require_nonempty_loader(loader, "validation")
    resolved_device = torch.device(device)
    model.eval()
    total_loss = 0.0
    total_sample_count = 0
    all_predicted: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    with torch.no_grad():
        for observed, target in loader:  # type: ignore[union-attr]
            observed = observed.to(resolved_device)
            target = target.to(resolved_device)
            predicted = model(observed, future_seq=target.shape[1])
            _validate_prediction_batch(
                predicted,
                target,
                expected_forecast_steps=expected_forecast_steps,
            )
            batch_sample_count = int(target.shape[0])
            total_loss += (
                float(prediction_loss(predicted, target).item())
                * batch_sample_count
            )
            total_sample_count += batch_sample_count
            all_predicted.append(predicted)
            all_target.append(target)
    ssim, mse_scaled, mae_scaled = calculate_prediction_metrics(
        torch.cat(all_predicted, dim=0),
        torch.cat(all_target, dim=0),
    )
    return PredictionEpochMetrics(
        loss=total_loss / total_sample_count,
        ssim=ssim,
        mse_scaled=mse_scaled,
        mae_scaled=mae_scaled,
    )


def _fit_predictor_configured(
    model: torch.nn.Module,
    training_loader: object,
    validation_loader: object,
    *,
    device: str | torch.device,
    profile: PredictorTrainingContract,
    checkpoint_path: str | Path,
) -> PredictionFitSummary:

    configured_device = torch.device(device)
    checkpoint_target = Path(checkpoint_path)
    _require_nonempty_loader(training_loader, "training")
    _require_nonempty_loader(validation_loader, "validation")

    model.to(configured_device)
    optimizer = build_predictor_optimizer(model, profile)
    scheduler = build_predictor_scheduler(optimizer, profile)
    training_history: list[PredictionEpochMetrics] = []
    validation_history: list[PredictionEpochMetrics] = []
    best_validation_ssim = float("-inf")
    best_epoch = 0
    patience_counter = 0
    resolved_checkpoint: Path | None = None

    for epoch_index in range(profile.max_epochs):
        scheduler.prepare_epoch(epoch_index)
        training_metrics = train_prediction_epoch(
            model,
            training_loader,
            optimizer,
            device=configured_device,
            gradient_clip_norm=profile.gradient_clip_norm,
            gradient_accumulation_steps=profile.gradient_accumulation_steps,
            expected_forecast_steps=profile.forecast_horizon_seconds,
        )
        validation_metrics = evaluate_prediction_loss(
            model,
            validation_loader,
            device=configured_device,
            expected_forecast_steps=profile.forecast_horizon_seconds,
        )
        training_history.append(training_metrics)
        validation_history.append(validation_metrics)

        if validation_metrics.ssim is None:
            raise RuntimeError("validation SSIM was not calculated")
        if validation_metrics.ssim > best_validation_ssim:
            best_validation_ssim = validation_metrics.ssim
            best_epoch = epoch_index + 1
            resolved_checkpoint = save_state_dict(model, checkpoint_target)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= profile.early_stopping_patience:
            break
        scheduler.step_after_epoch(epoch_index)

    if resolved_checkpoint is None:
        raise RuntimeError("predictor training produced no checkpoint")
    return PredictionFitSummary(
        best_epoch=best_epoch,
        epochs_run=len(training_history),
        best_validation_ssim=best_validation_ssim,
        training_history=tuple(training_history),
        validation_history=tuple(validation_history),
        checkpoint_path=resolved_checkpoint,
    )


def fit_predictor(
    *,
    model_name: PredictorModelName = "msam_convlstm",
    bundle: SequenceBundle,
    profile: PredictorTrainingContract,
    augmentation_profile: PredictionAugmentationContract,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    seed: int = PAPER_SEED,
) -> PredictionFitSummary:

    if bundle.frames.shape[2:] != (128, 128):
        raise TrainingContractError("paper predictor frames must be exactly 128x128")
    set_reproducible_seeds(seed)
    resolved_device = resolve_device(device)
    model = _build_predictor_model(model_name)
    training_windows = build_prediction_windows(
        bundle.records,
        assigned_split="train",
        input_steps=15,
        target_steps=profile.forecast_horizon_seconds,
        stride_steps=1,
    )
    validation_windows = build_prediction_windows(
        bundle.records,
        assigned_split="validation",
        input_steps=15,
        target_steps=profile.forecast_horizon_seconds,
        stride_steps=1,
    )
    training_dataset = _PredictionTrainingDataset(
        PredictionDataset(bundle, training_windows),
        input_steps=15,
        target_steps=profile.forecast_horizon_seconds,
        augmentation_profile=augmentation_profile,
    )
    validation_dataset = _PredictionTrainingDataset(
        PredictionDataset(bundle, validation_windows),
        input_steps=15,
        target_steps=profile.forecast_horizon_seconds,
        augmentation_profile=None,
    )
    training_loader_kwargs: dict[str, object] = {
        "batch_size": profile.batch_size,
        "shuffle": True,
        "num_workers": profile.num_workers,
    }
    if augmentation_profile == BLOCKED_RUNNER_AUGMENTATION:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(seed)
        training_loader_kwargs["generator"] = shuffle_generator
    training_loader = DataLoader(training_dataset, **training_loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=profile.num_workers,
    )
    return _fit_predictor_configured(
        model,
        training_loader,
        validation_loader,
        device=resolved_device,
        profile=profile,
        checkpoint_path=checkpoint_path,
    )
