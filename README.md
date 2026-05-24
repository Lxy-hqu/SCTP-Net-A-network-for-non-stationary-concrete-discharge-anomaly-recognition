# Open-Source Version

This directory provides a minimal, reproducible code skeleton for multi-stage video (process sequence) classification, including configuration, model definitions, the CFC/LTC module, and a minimal forward-pass smoke test.

## Welcome

We warmly welcome researchers to transfer the temporal modeling ideas in this work to a wide range of non-stationary time-series modeling scenarios. We also look forward to feedback, suggestions, and application experience from different engineering fields. The core value of SCTP-Net is not limited to concrete discharge anomaly recognition. Instead, it provides a modeling perspective for non-stationary processes: decomposing a long temporal process into stage-aware segments, modeling continuous dynamic evolution within each stage, and propagating and accumulating discriminative evidence across stages.

It should be noted that the stage partitioning in this work is mainly based on engineering experience and process observation, rather than a strict explicit formulation of the underlying physical mechanism. Therefore, when adapting this framework to other applications, researchers are encouraged not to mechanically reuse the stage partitioning strategy or network architecture. Instead, they should start from their own engineering problems and analyze whether the temporal process contains stage-wise behavior, whether key evidence drifts over time, whether local observations are misaligned with global labels, and whether information needs to be transferred across stages. Researchers should also choose an appropriate sequence length according to their process characteristics and computing resources, and perform sensitivity analysis on the number of stages by following the spirit of this work. These considerations can guide a more task-specific model design.

For different application scenarios, the upper-level feature extraction network should be replaced with a network that better matches the data modality and task objective, such as specialized feature extractors for images, sensor signals, point clouds, text logs, or multimodal industrial data. This repository is intended to provide a stage-aware, continuous-time, cross-stage evidence propagation paradigm, rather than a fixed network template that must be copied unchanged.

In industrial settings, many anomalies do not appear suddenly at a single time point. Instead, they often emerge through gradual evolution, stage transition, and evidence accumulation. Therefore, instead of simply aggregating an entire sequence uniformly, it is often more important to study when discriminative evidence appears, how it evolves, and how it can be inherited and strengthened by later stages. This is one of the main reasons why the temporal modeling idea in this work may be transferable to other industrial scenarios.

We are especially interested in seeing researchers further incorporate physical mechanisms, process dynamics, or domain constraints into this modeling framework. For example, future extensions may design stage partitioning, state propagation, and fusion strategies based on real process stages, state-transition rules, conservation relationships, equipment control logic, or interpretable time scales. Such extensions may help the model learn not only statistical correlations from data, but also dynamic regularities of engineering processes, thereby improving stability, interpretability, and cross-scenario generalization. We look forward to hearing your good news in the pull request section /issues.

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

## Data Availability

For access to the original images, please contact xuyang@stu.hqu.edu.cn.

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

## Citation

If you use SCTP-Net in academic work, please cite:

```bibtex
@article{zhou_sctp-net_2026,
    title = {SCTP-Net: a multi-stage continuous-time propagation network for non-stationary concrete discharge anomaly recognition},
    author = {Zhou, Xuejin and Li, Xuyang and Yang, Jianhong and Fang, Huaiying and Tu, Ran and Zeng, Yi and Zhong, Jinjin},
    journal = {Advanced Engineering Informatics},
    volume = {75},
    pages = {104843},
    year = {2026},
    doi = {10.1016/j.aei.2026.104843},
}
```
