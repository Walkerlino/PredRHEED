import argparse
from pathlib import Path
from typing import get_args

from predrheed.data.datasets import SequenceBundle
from predrheed.revision_cli import _device_argument, _validate_output_target, _write_json
from predrheed.training.common import PAPER_SEED
from predrheed.training.prediction import (
    BLOCKED_RUNNER_AUGMENTATION,
    PREDICTOR_5S_TRAINING,
    PREDICTOR_15S_TRAINING,
    PREDICTOR_30S_TRAINING,
    PredictorModelName,
    fit_predictor,
)


def main(argv=None):
    profiles = {5: PREDICTOR_5S_TRAINING, 15: PREDICTOR_15S_TRAINING, 30: PREDICTOR_30S_TRAINING}
    parser = argparse.ArgumentParser(description="Train a PredRHEED future-frame predictor.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=get_args(PredictorModelName), default="msam_convlstm")
    parser.add_argument("--horizon", type=int, choices=tuple(profiles), default=5)
    parser.add_argument("--seed", type=int, default=PAPER_SEED)
    parser.add_argument("--device", type=_device_argument)
    args = parser.parse_args(argv)
    try:
        _validate_output_target(args.checkpoint, protected_paths=(args.bundle,), force=False)
        result = fit_predictor(
            model_name=args.model,
            bundle=SequenceBundle.from_npz(args.bundle),
            profile=profiles[args.horizon],
            augmentation_profile=BLOCKED_RUNNER_AUGMENTATION,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            device=args.device,
        )
        _write_json(result, None)
    except (OSError, ValueError) as error:
        parser.error(str(error).replace("--output", "--checkpoint").replace(
            "pass --force to replace it", "choose a new checkpoint path"
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
