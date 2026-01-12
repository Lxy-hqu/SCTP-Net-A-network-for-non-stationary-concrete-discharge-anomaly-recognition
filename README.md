# Open-Source Version

This directory provides a minimal, reproducible code skeleton for multi-stage video (process sequence) classification, including configuration, model definitions, the CFC/LTC module, and a minimal forward-pass smoke test.

## Directory Layout

```
open_source_version/
  src/
    resnet18_multistage_lnn_logits.py
    model.py
    config.py
    torch_cfc.py
    smoke_test.py
```

## Files

- `src/config.py`
  - The extracted `CONFIG` dictionary from the training script (data, model, training, distillation [used during early debugging; disabled in final training], system, and paths).
- `src/model.py`
  - The extracted model definitions from the training script:
    - `ResNet18WithECA`: ResNet18 backbone + ECA attention
    - `TemporalAttention`: adaptive attention fusion across stages
    - `MultiStageVideoClassifier`: multi-stage feature extraction + cross-stage modeling + continuous dynamics + classifier head
- `src/torch_cfc.py`
  - Implementation of the CFC/LTC continuous-time neural network module (imported by `model.py`).
- `src/smoke_test.py`
  - Minimal import check and forward-pass smoke test (does not require the dataset; runs a forward pass with random tensors).

## Dataset Layout (`base_training_6`)

The training script expects the dataset root directory to be `base_training_6` by default, with the following structure:

```
base_training_6/
  train/
    dry/
      process_0001/
        0001.jpg
        0002.jpg
        ...
    less_accure/
    less_bleeding/
    normal/
    no_accure/
    severe_bleeding/
  validation/
    dry/
      process_0001/
        0001.jpg
        ...
    less_accure/
    less_bleeding/
    normal/
    no_accure/
    severe_bleeding/
  test/
    dry/
      process_0001/
        0001.jpg
        ...
    less_accure/
    less_bleeding/
    normal/
    no_accure/
    severe_bleeding/
```

Key data-loading assumptions (see `MultiStageConcreteDataset` in the training script):

- Each class folder contains multiple `process_XXXX/` subfolders, where each subfolder represents one process-sequence sample.
- Frames inside `process_XXXX/` are named in temporal order (e.g., `0001.jpg`, `0002.jpg`, ...).
- Frames are sorted by filename to preserve temporal order, then the sequence is split into `num_stages` stages; each stage takes `max_frames // num_stages` frames).

## Quick Smoke Test (No Dataset Required)

Run in `open_source_version/src`:

```bash
python smoke_test.py
```

Common arguments:

```bash
python smoke_test.py --device cpu --num-stages 4 --seq-len 8 --batch-size 4
```

## Training Paths

Default dataset paths in `src/config.py` are relative (e.g., `./base_training_6/train`). For reproduction, ensure either:

- Your current working directory contains `base_training_6/`, or
- You update `CONFIG["data"]["train_dir"] / val_dir / test_dir` to absolute paths.
