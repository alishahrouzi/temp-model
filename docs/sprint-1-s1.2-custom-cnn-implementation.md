# S1.2 — Custom CNN Implementation

## Objective

Translate the S1.1 architecture specification into an explicit PyTorch implementation without introducing a pretrained backbone or additional model complexity.

## Implemented Components

- Four custom convolutional blocks:
  - 3 → 32
  - 32 → 64
  - 64 → 128
  - 128 → 256
- Each block:
  - Conv2d 3×3, stride 1, padding 1
  - BatchNorm2d
  - ReLU
  - MaxPool2d 2×2
- Adaptive global average pooling to 1×1
- 256 → 128 projection with ReLU
- L2-normalized 128-dimensional retrieval embedding
- 128 → 4 classification head
- No pretrained weights or external vision backbone

## Public Model Paths

### Training path

`forward(x)` returns:

1. normalized 128-dimensional embedding
2. four-class logits

### Retrieval path

`encode(x)` returns only the L2-normalized 128-dimensional embedding.

### Raw representation

`forward_features(x)` exposes the 128-dimensional pre-normalization representation for inspection and later experimentation.

## Input Contract

The model expects:

- shape: `[B, 3, 224, 224]`
- dtype: floating-point tensor
- value range: supplied by S0.5 preprocessing as `[0, 1]`

The model does not perform image resizing, augmentation, or pretrained normalization.

## Parameter Count

The implemented architecture has exactly **422,788 trainable parameters**, matching the S1.1 design estimate.

Breakdown:

| Component | Trainable parameters |
|---|---:|
| Convolutional layers | 388,416 |
| BatchNorm parameters | 960 |
| Embedding projection | 32,896 |
| Classification head | 516 |
| **Total** | **422,788** |

## Scope Boundary

S1.2 implements the model architecture only.

It does not include:

- training loop;
- optimizer or learning-rate configuration;
- loss computation;
- checkpointing;
- data augmentation;
- training-time normalization;
- retrieval index;
- similarity calculation;
- evaluation.

Those remain assigned to later sprint tasks.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| Custom PyTorch CNN implemented | Pass |
| No pretrained backbone | Pass |
| Four convolutional stages implemented | Pass |
| 128-d retrieval representation implemented | Pass |
| L2 retrieval normalization implemented | Pass |
| Four-class classifier implemented | Pass |
| Input contract enforced | Pass |
| Exact parameter count matches S1.1 | Pass |
| Training/retrieval paths separated | Pass |
| Training pipeline excluded | Pass |
