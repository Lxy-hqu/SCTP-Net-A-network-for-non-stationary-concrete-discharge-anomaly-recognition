# Open-Source Version

This directory provides a minimal, reproducible code skeleton for multi-stage video (process sequence) classification, including configuration, model definitions, the CFC/LTC module, and a minimal forward-pass smoke test.

## 欢迎使用

欢迎大家使用本方法！本框架不仅适用于混凝土出料异常识别，同样欢迎将其应用于更广泛的**非平稳工程过程长序列预测**任务。非平稳工程过程往往具有时变统计特性、复杂的阶段性动态及长程依赖，本方法通过多阶段特征提取与连续时间神经网络建模，能够有效捕捉此类序列的动态演变规律。如果您正在研究工业过程监控、设备状态预测、生产质量溯源等非平稳长序列场景，欢迎参考、使用并进一步拓展本工作。期待与各位研究者共同推动非平稳工程过程智能分析的发展！

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
  - Original reference repository: https://github.com/raminmh/CfC.git
- `src/smoke_test.py`
  - Minimal import check and forward-pass smoke test (does not require the dataset; runs a forward pass with random tensors).

## Environment & Dependencies

The following versions are from the current development environment used to run `src/model.py` and `src/smoke_test.py`:

- OS: Windows-10-10.0.26100-SP0
- Python: 3.10.16 (Anaconda, MSC v.1929 64 bit)
- PyTorch: 2.8.0+cu129 (CUDA runtime 12.9; `torch.cuda.is_available() == True`)
- torchvision: 0.23.0+cu129
- NumPy: 2.0.1 (required by `src/torch_cfc.py`)

Minimal required packages to run `src/model.py`:

- torch
- torchvision
- numpy

## Citation

If you use the CfC/LTC module (`src/torch_cfc.py`) in academic work, cite the original paper:

```bibtex
@article{hasani_closed-form_2022,
    title = {Closed-form continuous-time neural networks},
    journal = {Nature Machine Intelligence},
    author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and Teschl, Gerald and Rus, Daniela},
    issn = {2522-5839},
    month = nov,
    year = {2022},
}
```

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

Run from the repository root:

```bash
python -m src.smoke_test
```

Common arguments:

```bash
python -m src.smoke_test --device cpu --num-stages 4 --seq-len 8 --batch-size 4
```

## Training Paths

Default dataset paths in `src/config.py` are relative (e.g., `./base_training_6/train`). For reproduction, ensure either:

- Your current working directory contains `base_training_6/`, or
- You update `CONFIG["data"]["train_dir"] / val_dir / test_dir` to absolute paths.
