# S2.1 — Training Configuration

## Objective

Define the reproducible training policy for the S1 custom CNN before implementing the training loop.

S2.1 does **not** execute training, create checkpoints, or select a best checkpoint. Those responsibilities belong to later Sprint 2 tasks.

## Configuration File

The authoritative configuration is:

`configs/training.yaml`

## Training Policy

| Area | Decision |
|---|---|
| Seed | 42 |
| Determinism | Enabled |
| Device | CUDA when available, CPU fallback |
| Batch size | 32 |
| Workers | 2 |
| Pin memory | Enabled |
| Epochs | 30 |
| Loss | CrossEntropyLoss |
| Label smoothing | 0.0 |
| Class weighting | Enabled, square-root inverse frequency |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau |
| Scheduler monitor | Validation loss |
| Gradient clipping | Max norm 1.0 |
| Augmentation | Mild training-only augmentation |
| Input | RGB, 224x224, [0,1] |
| Pretrained normalization | Disabled |

## Why These Choices

### Batch Size

A batch size of 32 is the initial baseline for the compact 422,788-parameter CNN. It is deliberately conservative for the project's local 4 GB GPU constraint while still providing useful batch statistics for BatchNorm.

The training loop must remain capable of falling back to CPU when CUDA is unavailable.

### Loss

The classification head produces four raw logits, so CrossEntropyLoss is the direct supervised objective.

Label smoothing is disabled in the initial baseline to keep the training signal simple and interpretable.

### Class Weighting

The training split is imbalanced:

| Class | Train images |
|---|---:|
| bracelet | 706 |
| earring_best | 2630 |
| necklace | 1390 |
| ring_best | 187 |

A square-root inverse-frequency weighting policy is used instead of full inverse-frequency weighting. This gives the minority classes additional influence without allowing the smallest class to dominate the classification objective.

The configured normalized weights are:

- bracelet: 0.958352
- earring_best: 0.496534
- necklace: 0.682999
- ring_best: 1.862115

These weights are derived only from the training split. Validation and test distributions are not used to construct the training loss.

Classification remains a supporting training signal; retrieval quality will be evaluated separately in Sprint 4.

### Optimizer

AdamW with a 1e-3 learning rate and 1e-4 weight decay provides a simple baseline for the small custom network without introducing unnecessary optimizer complexity.

### Scheduler

ReduceLROnPlateau monitors validation loss and halves the learning rate after three validation intervals without sufficient improvement.

Checkpoint creation and best-checkpoint selection are intentionally not defined here; S2.3 and S2.6 will own those decisions.

### Augmentation

The baseline uses mild training-only augmentation:

- horizontal flip with probability 0.5;
- rotation within ±10 degrees;
- limited brightness/contrast/saturation/hue variation.

The objective is to introduce moderate visual variation without changing jewelry category semantics. Validation and test preprocessing remain deterministic and unaugmented.

### Gradient Clipping

A maximum gradient norm of 1.0 is included as a lightweight numerical-stability safeguard. It is not intended to compensate for an unstable architecture.

## Reproducibility

The training configuration fixes seed 42 and requests deterministic execution. The training implementation in S2.2 must apply these settings consistently to Python, NumPy, PyTorch, CUDA when available, and DataLoader workers.

Deterministic execution can reduce throughput on some CUDA operations; reproducibility is preferred for this proof-of-concept baseline.

## Scope Boundary

S2.1 defines:

- training hyperparameters;
- loss policy;
- optimizer;
- scheduler;
- device and DataLoader policy;
- augmentation policy;
- reproducibility policy;
- monitoring signals.

S2.1 does not implement:

- training loop;
- checkpoint saving;
- checkpoint selection;
- retrieval embedding generation;
- Top-K evaluation;
- retrieval-quality claims.

## Next Task

**S2.2 — Training Loop**

The next task should consume `configs/training.yaml` and implement the training/validation execution without changing the model architecture.
