from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

from predrheed.data.adaptation import AdaptationBundle
from predrheed.revision_cli import (
    _device_argument,
    _paths_refer_to_same_file,
    _resolve_path,
    _write_json,
)
from predrheed.training.adaptation import (
    fit_adapted_classifier,
    paper_test_cascade_evaluation,
)
from predrheed.training.common import PAPER_SEED


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PAPER_SEED)
    parser.add_argument("--device", type=_device_argument)
    args = parser.parse_args(argv)
    try:
        checkpoint_dir = _resolve_path(args.checkpoint_dir)
        inputs = (
            args.bundle, args.initial_checkpoint, args.test_predictions, args.test_labels
        )
        if any(_paths_refer_to_same_file(checkpoint_dir, path) for path in inputs):
            raise ValueError("--checkpoint-dir must not refer to an input path")
        if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
            raise ValueError("--checkpoint-dir must identify a directory")
        if any(checkpoint_dir.glob("adaptation_epoch_*.pth")):
            raise ValueError("checkpoint directory already contains adaptation epochs")

        bundle = AdaptationBundle.from_npz(args.bundle)
        arrays = []
        for path in (args.test_predictions, args.test_labels):
            with path.open("rb") as source:
                array = np.load(source, allow_pickle=False)
                if not isinstance(array, np.ndarray):
                    array.close()
                    raise ValueError(f"{path} must be a NPY array")
                arrays.append(array)
        evaluation = paper_test_cascade_evaluation(*arrays)
        summary = fit_adapted_classifier(
            bundle=bundle,
            initial_checkpoint=args.initial_checkpoint,
            checkpoint_dir=checkpoint_dir,
            test_epoch_evaluation=evaluation,
            seed=args.seed,
            device=args.device,
        )
        _write_json(summary, None)
    except (BadZipFile, EOFError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
