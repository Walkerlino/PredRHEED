# Running the experiments

Install the dependencies in the [README](../README.md) and run commands from
the repository root. Replace example paths with your inputs and outputs.
See the [data contract](data_contract.md) for input formats and splits.
Use each script's `--help` for options.

## 1. Train the main models

```powershell
python train_classifier.py --bundle ../inputs/classification.npz --checkpoint ../outputs/classifier.pth
python train_predictor.py --bundle ../inputs/sequences.npz --checkpoint ../outputs/predictor_5s.pth
```

Main training selects checkpoints using validation metrics. The predictor
defaults to a 5 s forecast. Its `--horizon 15` and `--horizon 30` options
apply only to frame prediction.

## 2. Adapt the classifier

```powershell
python adapt_classifier.py --bundle ../inputs/adaptation.npz --initial-checkpoint ../outputs/classifier.pth --test-predictions ../inputs/test_predictions.npy --test-labels ../inputs/test_frame_labels.npy --checkpoint-dir ../outputs/adaptation
```

Adaptation runs 50 epochs without early stopping, freezing the CNN for
epochs 1–10 and unfreezing it at epoch 11. Every epoch is saved and evaluated
on the same 30 test windows. Test samples do not enter the adaptation loss.
A peak on this test trajectory is not validation-based checkpoint selection.

## 3. Run the five-second cascade

```powershell
python run_cascade.py --history ../inputs/history.npy --predictor-checkpoint ../outputs/predictor_5s.pth --classifier-checkpoint ../outputs/adapted_classifier.pth --output ../outputs/cascade_predictions.npz
```

Replace `adapted_classifier.pth` with the intended adaptation epoch's
tensor-only `state_dict` checkpoint. Cascade output contains predictions
only. Add matching reference frames, true frame labels, and grouping IDs
to prepare the [analysis NPZ](data_contract.md#analysis-npz).

## 4. Run revision comparisons and analysis

```powershell
python run_analysis.py ablation --bundle ../inputs/sequences.npz --variant no_downsample --checkpoint ../outputs/no_downsample.pth --output ../outputs/ablation.json
python run_analysis.py seq2label --bundle ../inputs/sequences.npz --labels ../inputs/frame_labels.npz --checkpoint-dir ../outputs/seq2label --output ../outputs/seq2label.json
python run_analysis.py analyze --input ../inputs/analysis.npz --output ../outputs/analysis.json
```

Ablation runs one variant and seed per invocation. `full_repro` is the
comparator. The three ablations are `no_downsample`, `no_attn_dropout`,
and `no_mem_clip`. `seq2label` trains the direct sequence-to-label comparator.
`analyze` consumes the analysis NPZ and reports frame metrics, decision
metrics, cluster-bootstrap intervals, and leave-one-segment-out accuracies.
Use `python run_analysis.py SUBCOMMAND --help` for options. These commands
provide individual run records and analyses.

## Shared evaluation rules

The dominant-label threshold is 0.60, with 0.65 and 0.70 for sensitivity
analysis. Unresolved seconds count as incorrect and remain in the overall
accuracy denominator. Coverage and resolved-only accuracy are separate.

Frame metrics are pooled and per-horizon SSIM with `data_range=1.0`,
MSE × 1000, and MAE × 1000. Cluster bootstrap and leave-one-segment-out
analysis use caller-supplied grouping IDs.
