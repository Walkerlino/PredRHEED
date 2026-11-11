# PredRHEED

Code for **PredRHEED**, a cascaded deep-learning framework for in situ RHEED
monitoring during ScAlN molecular beam epitaxy. `MSAMConvLSTM` predicts future
frames, and `CNNTransformer` classifies them as `streaky`, `transition`, or
`spotty`. The evaluated end-to-end forecast horizon is 5 seconds.

## Installation

From the repository root, install the recorded direct dependencies:

```powershell
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) targets Windows x64 and CPython 3.10.
The environment is Windows 11 Pro 64-bit, Python 3.10.19,
PyTorch 2.9.1 with CUDA 13.0 and cuDNN 9.12.0, and one RTX 5090 with 32 GB
of memory. 

## PS
All rights reserved: reuse requires written permission from the corresponding authors.
