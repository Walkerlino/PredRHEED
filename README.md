# PredRHEED

Code for **PredRHEED**, a cascaded deep-learning framework for in situ RHEED
monitoring during ScAlN molecular beam epitaxy. `MSAMConvLSTM` predicts future
frames, and `CNNTransformer` classifies them as `streaky`, `transition`, or
`spotty`. The evaluated end-to-end forecast horizon is 5 seconds. The 15 s
and 30 s configurations concern frame prediction only.

The comparison models are `CNNOnly`, `TransformerOnly`, `SAConvLSTM`,
`SimVP`, and the non-generative `DirectSeq2Label` baseline.

```text
PredRHEED-Model/
├── train_predictor.py
├── train_classifier.py
├── adapt_classifier.py
├── run_cascade.py
├── run_analysis.py
├── predrheed/          # Models, data preparation, training, and evaluation
├── docs/               # Data formats and experiment instructions
└── requirements.txt
```

## Installation

From the repository root, install the recorded direct dependencies:

```powershell
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) targets Windows x64 and CPython 3.10.
The reported paper environment is Windows 11 Pro 64-bit, Python 3.10.19,
PyTorch 2.9.1 with CUDA 13.0 and cuDNN 9.12.0, and one RTX 5090 with 32 GB
of memory. To install the [hash-checked runtime closure](requirements-lock-win_amd64-cp310-cu130.txt)
instead of the direct-dependency list:

```powershell
python -m pip install --require-hashes -r requirements-lock-win_amd64-cp310-cu130.txt
```

The lock covers runtime dependencies, excluding the operating system, NVIDIA
driver, build tools, and research inputs and outputs. The scripts run directly
with `python` from this directory. Package installation is optional.

## Run order

| Step | Script | Purpose |
|---|---|---|
| 1 | `train_classifier.py`, `train_predictor.py` | Train the classifier and predictor from supplied bundles |
| 2 | `adapt_classifier.py` | Adapt the classifier using the prepared adaptation bundle and test-monitoring inputs |
| 3 | `run_cascade.py` | Load both checkpoints and predict five future pattern labels |
| 4 | `run_analysis.py` | Run `ablation`, `seq2label`, or `analyze` with the required inputs |

[Commands and evaluation rules](docs/revision_analysis.md) give complete
examples. Use `python SCRIPT.py --help` for options. Training defaults to
seed 2025, with cuDNN deterministic mode enabled and benchmark mode disabled.
CUDA is used when available, with CPU fallback.

## Data and weights

The recorded [split assignments](docs/split_manifest.csv) and
[30 cascade test windows](docs/cascade_test_windows.csv) are included.
Supply experimental frames, labels, and checkpoints using the
[data contract](docs/data_contract.md). Keep these inputs and generated
outputs outside this repository. Experimental images, labels, trained
weights, and stored numerical results are not included. Cascade outputs
must be combined with real reference data and grouping IDs before `analyze`.

## Citation

P. Gao, M. M. H. Tanim, W. Wang, G. Baker, J. Chen, Z. Wang, J. Liu, Y. Wei,
D. Wang, Q. Qu, Z. Mi. Deep Learning Assisted Prediction of Reflection
High-Energy Electron Diffraction Patterns During Heteroepitaxy.
