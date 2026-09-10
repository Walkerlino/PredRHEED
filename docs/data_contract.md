# Data formats

Supply experimental frames, labels, and checkpoints externally. Frames must be finite
floating-point arrays in `[0, 1]`. Use Unicode arrays for strings and integer
arrays for labels and indices; object arrays are not supported.
Class indices are `0 = streaky`, `1 = transition`, `2 = spotty`.
Split names are `train`, `validation`, and `test`.

## Splits and windows

Use the recorded 24 s segment partition for classification/cascade and the
separate 60 s block partition for predictor benchmarks. Prediction windows
use 15 one-second inputs, a 1 s stride, and 5/15/30 targets. The complete
input-target window must stay within one split; adjacent segments in that
split may share a window.

[split_manifest.csv](split_manifest.csv) contains the recorded block assignments
from `split_v1_reproduction.json` and `split_v2_blocked.json`. The `partition`
values are `cascade_24s` and `predictor_60s`. Second indices start at zero
within the 737 s analyzed sequence, and all end indices are exclusive.
Expand each block's assignment over its second range to form the Sequence
NPZ `splits` array. At 30 fps, second `s` covers raw-frame indices
`[30*s, 30*(s+1))`.

[cascade_test_windows.csv](cascade_test_windows.csv) lists the 30 cascade
test windows reconstructed from the recorded partition using the 15+5 s
window rule. Rows follow increasing input-start time; `window_id` is the
zero-based row index. Use `block_id` for both `cluster_ids` and `segment_ids`
in the cascade analysis, and align input and reference arrays to this order.
These files record data membership and window order independently of model
architecture and checkpoints.

## Classification NPZ

| Field | Shape / values |
|---|---|
| `frames` | `[N, 3, 32, 64]` |
| `pattern_classes` | `[N]`, `streaky`, `transition`, or `spotty` |
| `splits` | `[N]`, split names |

## Sequence NPZ

| Field | Shape / values |
|---|---|
| `frames` | `[N, 1, 128, 128]` for paper training |
| `positions` | `[N]`, consecutive nonnegative second indices |
| `splits` | `[N]`, split names |
| `segment_ids` | Optional `[N]`, nonempty Unicode identifiers |

## Adaptation NPZ

| Field | Shape / values |
|---|---|
| `frames`, `validation_frames` | `[N, 1, 32, 64]`, `[M, 1, 32, 64]` |
| `labels`, `validation_labels` | `[N]`, `[M]`, class indices |
| `origin_tags` | `[N]`, `0` original frames, `1` filtered predictions; `2` also accepted for ground-truth 1 Hz frames |
| `second_indices` | `[N]`, nonnegative second indices |

## Additional script inputs

| Input | Format / shape |
|---|---|
| `adapt_classifier.py --test-predictions` | NPY, `[30, 5, 1, 128, 128]` |
| `adapt_classifier.py --test-labels` | NPY, `[30, 5, 30]`, class indices |
| `run_cascade.py --history` | NPY, `[N, 15, 1, 128, 128]` |
| `run_analysis.py seq2label --labels` | NPZ field `frame_labels`, `[30 * N]` class indices, 30 per bundle second in bundle order |

## Analysis NPZ

| Field | Shape / values |
|---|---|
| `predicted_frames`, `reference_frames` | `[N, 5, 1, 128, 128]` each |
| `predicted_labels` | `[N, 5]`, class indices |
| `reference_frame_labels` | `[N, 5, 30]`, class indices |
| `cluster_ids`, `segment_ids` | `[N]` each, integer or Unicode IDs with at least two distinct values |

Align all arrays to the same windows. `cluster_ids` groups bootstrap samples;
`segment_ids` groups leave-one-segment-out analysis. Add real references and
grouping IDs to cascade predictions to form this input. See the
[commands and evaluation rules](revision_analysis.md).
