from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

from predrheed.evaluation import (
    PRIMARY_PROPORTION,
    SENSITIVITY_PROPORTIONS,
    LeaveOneSegmentOutSummary,
    cluster_bootstrap_accuracy,
    compute_frame_metrics,
    leave_one_segment_out_accuracy,
    score_predicted_classes,
)


_ANALYSIS_KEYS = (
    "predicted_frames",
    "reference_frames",
    "predicted_labels",
    "reference_frame_labels",
    "cluster_ids",
    "segment_ids",
)

@dataclass(frozen=True, slots=True)
class _OutputTarget:
    path: Path
    protected_paths: tuple[Path, ...]
    initial_identity: tuple[int, int] | None
    force: bool


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _load_npz_arrays(
    path: Path,
    required_keys: Sequence[str],
    *,
    input_name: str,
) -> dict[str, np.ndarray]:
    with Path(path).open("rb") as source:
        archive = np.load(source, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(f"{input_name} must be an NPZ archive")
        with archive:
            for key in required_keys:
                if key not in archive.files:
                    raise ValueError(f"missing required array: {key}")
            return {key: archive[key] for key in required_keys}


def _load_analysis_arrays(path: Path) -> dict[str, np.ndarray]:
    return _load_npz_arrays(path, _ANALYSIS_KEYS, input_name="input")


def _validate_analysis_arrays(arrays: dict[str, np.ndarray]) -> None:
    sample_counts = {
        value.shape[0]
        for value in arrays.values()
        if value.ndim >= 1
    }
    if any(value.ndim == 0 for value in arrays.values()) or len(sample_counts) != 1:
        raise ValueError("analysis arrays must share the same sample count")
    sample_count = sample_counts.pop()
    expected_shapes = {
        "predicted_frames": (sample_count, 5, 1, 128, 128),
        "reference_frames": (sample_count, 5, 1, 128, 128),
        "predicted_labels": (sample_count, 5),
        "reference_frame_labels": (sample_count, 5, 30),
        "cluster_ids": (sample_count,),
        "segment_ids": (sample_count,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in ("cluster_ids", "segment_ids"):
        if arrays[name].dtype.kind not in {"i", "u", "U"}:
            raise ValueError(f"{name} must contain integer or string values")
        if np.unique(arrays[name]).size < 2:
            raise ValueError(f"{name} must contain at least two unique values")


def _segment_summary_payload(
    summary: LeaveOneSegmentOutSummary,
) -> dict[str, object]:
    return {
        "segment_ids": summary.cluster_ids,
        "accuracies": summary.accuracies,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
    }


def _resolve_path(path: Path) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=False)
    except RuntimeError as error:
        raise ValueError(f"unable to resolve path {path}: {error}") from error


def _paths_refer_to_same_file(left: Path, right: Path) -> bool:
    resolved_left = _resolve_path(left)
    resolved_right = _resolve_path(right)
    if resolved_left == resolved_right:
        return True
    try:
        return resolved_left.samefile(resolved_right)
    except OSError:
        return False


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _validate_output_target(
    output: Path | None,
    *,
    protected_paths: Sequence[Path],
    force: bool,
) -> _OutputTarget | None:
    if output is None:
        return None
    resolved_output = _resolve_path(output)
    resolved_protected_paths = tuple(_resolve_path(path) for path in protected_paths)
    if any(
        _paths_refer_to_same_file(resolved_output, protected)
        for protected in resolved_protected_paths
    ):
        raise ValueError("--output must not refer to an input or checkpoint path")
    if resolved_output.exists() and not force:
        raise ValueError("output already exists; pass --force to replace it")
    return _OutputTarget(
        path=resolved_output,
        protected_paths=resolved_protected_paths,
        initial_identity=_file_identity(resolved_output),
        force=force,
    )


def _revalidate_output_target(target: _OutputTarget) -> None:
    if any(
        _paths_refer_to_same_file(target.path, protected)
        for protected in target.protected_paths
    ):
        raise ValueError("--output must not refer to an input or checkpoint path")
    if _file_identity(target.path) != target.initial_identity:
        raise ValueError("output target changed after validation")


def _write_json(
    payload: object,
    target: _OutputTarget | None,
) -> None:
    content = json.dumps(
        _json_ready(payload),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    if target is None:
        print(content)
        return
    output = target.path
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        _revalidate_output_target(target)
        if target.force:
            temporary_path.replace(output)
        else:
            try:
                os.link(temporary_path, output)
            except FileExistsError as error:
                raise ValueError(
                    "output already exists; pass --force to replace it"
                ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _run_analysis(args: argparse.Namespace) -> int:
    output = _validate_output_target(
        args.output,
        protected_paths=(args.input,),
        force=args.force,
    )
    arrays = _load_analysis_arrays(args.input)
    _validate_analysis_arrays(arrays)
    frame_metrics = compute_frame_metrics(
        arrays["predicted_frames"],
        arrays["reference_frames"],
    )
    decisions: dict[str, object] = {}
    for proportion in (PRIMARY_PROPORTION, *SENSITIVITY_PROPORTIONS):
        reference, summary = score_predicted_classes(
            arrays["predicted_labels"],
            arrays["reference_frame_labels"],
            proportion=proportion,
        )
        decisions[f"{proportion:.2f}"] = {
            "summary": summary,
            "cluster_bootstrap": cluster_bootstrap_accuracy(
                arrays["predicted_labels"],
                reference,
                arrays["cluster_ids"],
                replicates=args.bootstrap_replicates,
                confidence_level=args.confidence_level,
                seed=args.seed,
            ),
            "leave_one_segment_out": _segment_summary_payload(
                leave_one_segment_out_accuracy(
                    arrays["predicted_labels"],
                    reference,
                    arrays["segment_ids"],
                )
            ),
        }
    _write_json(
        {
            "schema_version": 1,
            "command": "analyze",
            "frame_metrics": frame_metrics,
            "decisions": decisions,
        },
        output,
    )
    return 0


def _run_ablation(args: argparse.Namespace) -> int:
    from predrheed.data.datasets import SequenceBundle
    from predrheed.training.ablation import fit_msam_ablation

    output = _validate_output_target(
        args.output,
        protected_paths=(args.bundle, args.checkpoint),
        force=args.force,
    )
    result = fit_msam_ablation(
        variant=args.variant,
        bundle=SequenceBundle.from_npz(args.bundle),
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        memory_profile=args.memory_profile,
        device=args.device,
    )
    _write_json(
        {"schema_version": 1, "command": "ablation", "result": result},
        output,
    )
    return 0


def _run_seq2label(args: argparse.Namespace) -> int:
    from predrheed.data.datasets import SequenceBundle
    from predrheed.training.seq2label import fit_direct_seq2label

    output = _validate_output_target(
        args.output,
        protected_paths=(
            args.bundle,
            args.labels,
            args.checkpoint_dir,
            args.checkpoint_dir / "best_model.pth",
            args.checkpoint_dir / "final_epoch_model.pth",
        ),
        force=args.force,
    )
    frame_labels = _load_npz_arrays(
        args.labels,
        ("frame_labels",),
        input_name="labels input",
    )["frame_labels"]
    result = fit_direct_seq2label(
        bundle=SequenceBundle.from_npz(args.bundle),
        frame_labels=frame_labels,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )
    _write_json(
        {"schema_version": 1, "command": "seq2label", "result": result},
        output,
    )
    return 0


def _device_argument(value: str) -> str:
    import torch

    try:
        resolved = torch.device(value)
        torch.empty(0, device=resolved)
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid device or unavailable device {value!r}: {error}"
        ) from error
    return value


def build_parser() -> argparse.ArgumentParser:
    from predrheed.training.ablation import ABLATION_VARIANTS, MEMORY_PROFILES

    parser = argparse.ArgumentParser(
        prog="predrheed-revision",
        description="Run PredRHEED revision analyses on caller-supplied inputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser(
        "analyze",
        help="Compute frame, decision, bootstrap, and leave-one-segment-out metrics.",
    )
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output JSON file",
    )
    analyze.add_argument("--bootstrap-replicates", type=int, default=1000)
    analyze.add_argument("--confidence-level", type=float, default=0.95)
    analyze.add_argument("--seed", type=int, default=2025)
    analyze.set_defaults(handler=_run_analysis)

    ablation = subparsers.add_parser(
        "ablation",
        help="Run one MSAM-ConvLSTM component-ablation configuration.",
    )
    ablation.add_argument("--bundle", type=Path, required=True)
    ablation.add_argument("--variant", choices=tuple(ABLATION_VARIANTS), required=True)
    ablation.add_argument("--checkpoint", type=Path, required=True)
    ablation.add_argument(
        "--memory-profile",
        choices=tuple(MEMORY_PROFILES),
        default="batch4",
    )
    ablation.add_argument("--seed", type=int, default=2025)
    ablation.add_argument("--device", type=_device_argument)
    ablation.add_argument("--output", type=Path)
    ablation.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output JSON file",
    )
    ablation.set_defaults(handler=_run_ablation)

    seq2label = subparsers.add_parser(
        "seq2label",
        help="Train the direct sequence-to-label comparison model.",
    )
    seq2label.add_argument("--bundle", type=Path, required=True)
    seq2label.add_argument("--labels", type=Path, required=True)
    seq2label.add_argument("--checkpoint-dir", type=Path, required=True)
    seq2label.add_argument("--seed", type=int, default=2025)
    seq2label.add_argument("--device", type=_device_argument)
    seq2label.add_argument("--output", type=Path)
    seq2label.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output JSON file",
    )
    seq2label.set_defaults(handler=_run_seq2label)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (BadZipFile, EOFError) as error:
        parser.error(f"invalid NPZ archive: {error}")
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
