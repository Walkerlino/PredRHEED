import argparse
from pathlib import Path
from typing import get_args

from predrheed.data.datasets import ClassificationBundle
from predrheed.revision_cli import _device_argument, _validate_output_target, _write_json
from predrheed.training.classification import ClassifierModelName, fit_classifier
from predrheed.training.common import PAPER_SEED


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a PredRHEED pattern classifier.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=get_args(ClassifierModelName), default="cnn_transformer")
    parser.add_argument("--seed", type=int, default=PAPER_SEED)
    parser.add_argument("--device", type=_device_argument)
    args = parser.parse_args(argv)
    try:
        _validate_output_target(args.checkpoint, protected_paths=(args.bundle,), force=False)
        result = fit_classifier(
            model_name=args.model,
            bundle=ClassificationBundle.from_npz(args.bundle),
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
