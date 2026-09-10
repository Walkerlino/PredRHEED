from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from predrheed.data.datasets import PredictionDataset, SequenceBundle
from predrheed.data.windows import build_prediction_windows
from predrheed.evaluation.frame_metrics import (
    FrameMetricSummary,
    compute_frame_metrics,
)
from predrheed.models.chunked_attention import (
    patch_model_chunked,
    run_equivalence_check,
)
from predrheed.models.prediction import MSAMConvLSTM, MSAMConvLSTMConfig
from predrheed.training.common import (
    PAPER_SEED,
    TrainingContractError,
    load_state_dict,
    resolve_device,
    set_reproducible_seeds,
)
from predrheed.training.prediction import (
    BLOCKED_RUNNER_AUGMENTATION,
    PREDICTOR_5S_TRAINING,
    PredictionFitSummary,
    _PredictionTrainingDataset,
    _fit_predictor_configured,
    _predictor_profile,
    calculate_prediction_metrics,
)


AblationVariantName = Literal[
    "full_repro",
    "no_downsample",
    "no_attn_dropout",
    "no_mem_clip",
]

MemoryProfileName = Literal["batch4", "batch2_accum2", "batch1_accum4"]


ABLATION_SEEDS: tuple[int, int, int] = (2025, 2026, 2027)

ABLATION_VARIANTS: dict[AblationVariantName, MSAMConvLSTMConfig] = {
    "full_repro": MSAMConvLSTMConfig(),
    "no_downsample": MSAMConvLSTMConfig(spatial_reduction=1),
    "no_attn_dropout": MSAMConvLSTMConfig(attention_dropout=0.0),
    "no_mem_clip": MSAMConvLSTMConfig(memory_max_norm=None),
}

MEMORY_PROFILES: dict[MemoryProfileName, tuple[int, int]] = {
    "batch4": (4, 1),
    "batch2_accum2": (2, 2),
    "batch1_accum4": (1, 4),
}

_G7_FIELDS = ("spatial_reduction", "attention_dropout", "memory_max_norm")

_G7_EXPECTED_DELTA = {
    "no_downsample": "spatial_reduction",
    "no_attn_dropout": "attention_dropout",
    "no_mem_clip": "memory_max_norm",
}

_INPUT_STEPS = 15
_TARGET_STEPS = 5
_STRIDE_STEPS = 1

_EVALUATION_BATCH_SIZE = 4


@dataclass(frozen=True, slots=True)
class VariantConfigEcho:

    spatial_reduction: int
    attention_dropout: float
    memory_max_norm: float | None


def variant_config_echo(config: MSAMConvLSTMConfig) -> VariantConfigEcho:

    if not isinstance(config, MSAMConvLSTMConfig):
        raise TrainingContractError("config must be an MSAMConvLSTMConfig")
    return VariantConfigEcho(
        spatial_reduction=config.spatial_reduction,
        attention_dropout=config.attention_dropout,
        memory_max_norm=config.memory_max_norm,
    )


def assert_single_delta(
    variant: str,
    echo: VariantConfigEcho,
    full_echo: VariantConfigEcho,
) -> None:

    diffs = [
        field
        for field in _G7_FIELDS
        if getattr(echo, field) != getattr(full_echo, field)
    ]
    if variant == "full_repro":
        if diffs:
            raise TrainingContractError(
                f"G7 FAILED: full variant has deltas {diffs}"
            )
        return
    if variant not in _G7_EXPECTED_DELTA:
        raise TrainingContractError(
            f"unknown variant {variant!r}, expected one of "
            f"{tuple(ABLATION_VARIANTS)}"
        )
    expected = _G7_EXPECTED_DELTA[variant]
    if diffs != [expected]:
        raise TrainingContractError(
            f"G7 FAILED: variant {variant} deltas {diffs}, expected [{expected}]"
        )


@dataclass(frozen=True, slots=True)
class AblationRunRecord:

    variant: AblationVariantName
    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    config_echo: VariantConfigEcho
    deviations: tuple[str, ...]
    chunked_attention: bool
    g8_equivalence: dict | None
    diverged: bool
    fit_summary: PredictionFitSummary | None
    test_ssim: float | None
    test_mse_times_1000: float | None
    test_mae_times_1000: float | None
    per_horizon: FrameMetricSummary | None


def _evaluate_ablation_test_metrics(
    model: torch.nn.Module,
    bundle: SequenceBundle,
    *,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[tuple[float, float, float], FrameMetricSummary] | None:

    test_windows = build_prediction_windows(
        bundle.records,
        assigned_split="test",
        input_steps=_INPUT_STEPS,
        target_steps=_TARGET_STEPS,
        stride_steps=_STRIDE_STEPS,
        allow_empty=True,
    )
    if not test_windows:
        return None
    test_dataset = _PredictionTrainingDataset(
        PredictionDataset(bundle, test_windows),
        input_steps=_INPUT_STEPS,
        target_steps=_TARGET_STEPS,
        augmentation_profile=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=_EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    load_state_dict(model, checkpoint_path, device=device)
    model.eval()
    predicted_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for observed, target in test_loader:
            predicted_batches.append(
                model(observed.to(device), future_seq=target.shape[1]).cpu()
            )
            target_batches.append(target)
    predicted = torch.cat(predicted_batches, dim=0)
    target = torch.cat(target_batches, dim=0)
    flat_metrics = calculate_prediction_metrics(predicted, target)
    per_horizon = compute_frame_metrics(predicted.numpy(), target.numpy())
    return flat_metrics, per_horizon


def fit_msam_ablation(
    *,
    variant: AblationVariantName,
    bundle: SequenceBundle,
    checkpoint_path: str | Path,
    seed: int = PAPER_SEED,
    memory_profile: MemoryProfileName = "batch4",
    device: str | torch.device | None = None,
) -> AblationRunRecord:

    if variant not in ABLATION_VARIANTS:
        raise TrainingContractError(
            f"unknown variant {variant!r}, expected one of "
            f"{tuple(ABLATION_VARIANTS)}"
        )
    if memory_profile not in MEMORY_PROFILES:
        raise TrainingContractError(
            f"unknown memory profile {memory_profile!r}, expected one of "
            f"{tuple(MEMORY_PROFILES)}"
        )
    if bundle.frames.shape[2:] != (128, 128):
        raise TrainingContractError("paper predictor frames must be exactly 128x128")

    config = ABLATION_VARIANTS[variant]
    echo = variant_config_echo(config)
    full_echo = variant_config_echo(MSAMConvLSTMConfig())
    assert_single_delta(variant, echo, full_echo)

    batch_size, accumulation_steps = MEMORY_PROFILES[memory_profile]
    if memory_profile == "batch4":
        profile = PREDICTOR_5S_TRAINING
        deviations: tuple[str, ...] = ()
    else:
        profile = _predictor_profile(
            source_name=f"ablation_{variant}_memforced",
            forecast_horizon_seconds=_TARGET_STEPS,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
        )
        deviations = (
            f"memory-forced: batch {batch_size} x accum {accumulation_steps} "
            "(BatchNorm micro-batch statistics differ from batch-4 training)",
        )

    use_chunked = variant == "no_downsample"
    g8_results: dict | None = None
    if use_chunked:
        g8_results = run_equivalence_check()

    set_reproducible_seeds(seed)
    resolved_device = resolve_device(device)
    model = MSAMConvLSTM(config=config)
    if use_chunked:
        patch_model_chunked(model)

    training_windows = build_prediction_windows(
        bundle.records,
        assigned_split="train",
        input_steps=_INPUT_STEPS,
        target_steps=_TARGET_STEPS,
        stride_steps=_STRIDE_STEPS,
    )
    validation_windows = build_prediction_windows(
        bundle.records,
        assigned_split="validation",
        input_steps=_INPUT_STEPS,
        target_steps=_TARGET_STEPS,
        stride_steps=_STRIDE_STEPS,
    )
    training_dataset = _PredictionTrainingDataset(
        PredictionDataset(bundle, training_windows),
        input_steps=_INPUT_STEPS,
        target_steps=_TARGET_STEPS,
        augmentation_profile=BLOCKED_RUNNER_AUGMENTATION,
    )
    validation_dataset = _PredictionTrainingDataset(
        PredictionDataset(bundle, validation_windows),
        input_steps=_INPUT_STEPS,
        target_steps=_TARGET_STEPS,
        augmentation_profile=None,
    )
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=profile.num_workers,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=_EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=profile.num_workers,
    )

    try:
        summary = _fit_predictor_configured(
            model,
            training_loader,
            validation_loader,
            device=resolved_device,
            profile=profile,
            checkpoint_path=checkpoint_path,
        )
    except RuntimeError as error:
        if str(error) != "predictor training produced no checkpoint":
            raise
        return AblationRunRecord(
            variant=variant,
            seed=seed,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            config_echo=echo,
            deviations=deviations,
            chunked_attention=use_chunked,
            g8_equivalence=g8_results,
            diverged=True,
            fit_summary=None,
            test_ssim=None,
            test_mse_times_1000=None,
            test_mae_times_1000=None,
            per_horizon=None,
        )

    test_result = _evaluate_ablation_test_metrics(
        model,
        bundle,
        checkpoint_path=summary.checkpoint_path,
        device=resolved_device,
    )
    if test_result is None:
        test_ssim: float | None = None
        test_mse_times_1000: float | None = None
        test_mae_times_1000: float | None = None
        per_horizon: FrameMetricSummary | None = None
    else:
        (test_ssim, test_mse_times_1000, test_mae_times_1000), per_horizon = test_result

    return AblationRunRecord(
        variant=variant,
        seed=seed,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        config_echo=echo,
        deviations=deviations,
        chunked_attention=use_chunked,
        g8_equivalence=g8_results,
        diverged=False,
        fit_summary=summary,
        test_ssim=test_ssim,
        test_mse_times_1000=test_mse_times_1000,
        test_mae_times_1000=test_mae_times_1000,
        per_horizon=per_horizon,
    )


__all__ = [
    "ABLATION_SEEDS",
    "ABLATION_VARIANTS",
    "AblationRunRecord",
    "AblationVariantName",
    "MEMORY_PROFILES",
    "MemoryProfileName",
    "VariantConfigEcho",
    "assert_single_delta",
    "fit_msam_ablation",
    "variant_config_echo",
]
