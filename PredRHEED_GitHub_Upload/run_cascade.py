from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Sequence
from zipfile import BadZipFile

import numpy as np
import torch

from predrheed.data.preprocessing import _atomic_write_npz
from predrheed.evaluation.cascade import _validate_history, run_five_second_cascade
from predrheed.models import CNNTransformer, MSAMConvLSTM
from predrheed.revision_cli import (
    _device_argument,
    _revalidate_output_target,
    _validate_output_target,
)
from predrheed.training.common import load_state_dict, resolve_device


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=_device_argument)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != ".npz":
            raise ValueError("--output must use the .npz suffix")
        target = _validate_output_target(
            args.output,
            protected_paths=(
                args.history, args.predictor_checkpoint, args.classifier_checkpoint
            ),
            force=False,
        )
        with args.history.open("rb") as source:
            history = np.load(source, allow_pickle=False)
            if not isinstance(history, np.ndarray):
                history.close()
                raise ValueError("--history must be a NPY array")
        observed = torch.from_numpy(history)
        _validate_history(observed)
        device = resolve_device(args.device)
        predictor = MSAMConvLSTM().to(device)
        classifier = CNNTransformer().to(device)
        load_state_dict(predictor, args.predictor_checkpoint, device=device)
        load_state_dict(classifier, args.classifier_checkpoint, device=device)
        result = run_five_second_cascade(
            predictor, classifier, observed.to(device=device, dtype=torch.float32)
        )
        arrays = {
            "predicted_frames": result.predicted_frames.detach().cpu().numpy(),
            "predicted_labels": result.predicted_labels,
            "classifier_logits": result.classifier_logits.detach().cpu().numpy(),
        }
        target.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.path.parent) as staging:
            staged = Path(staging) / "predictions.npz"
            _atomic_write_npz(staged, arrays)
            _revalidate_output_target(target)
            os.link(staged, target.path)
    except FileExistsError:
        parser.error("output already exists; choose a new --output path")
    except (BadZipFile, EOFError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = str(error).replace(
            "; pass --force to replace it", "; choose a new --output path"
        )
        parser.error(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
