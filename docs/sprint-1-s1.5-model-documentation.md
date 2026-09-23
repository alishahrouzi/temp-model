# S1.5 — Model Documentation

## Objective

Document the final custom CNN implemented during Sprint 1, including its architecture, public interface, tensor contracts, parameter count, retrieval/classification separation, and S1.4 validation evidence.

## Final Architecture

```text
Input [B, 3, 224, 224]
        |
ConvBlock 1: 3 -> 32
        |
[B, 32, 112, 112]
        |
ConvBlock 2: 32 -> 64
        |
[B, 64, 56, 56]
        |
ConvBlock 3: 64 -> 128
        |
[B, 128, 28, 28]
        |
ConvBlock 4: 128 -> 256
        |
[B, 256, 14, 14]
        |
AdaptiveAvgPool2d(1,1)
        |
[B, 256]
        |
Linear 256 -> 128 + ReLU
        |
[B, 128]
        |
L2 normalization
        |
        +------> retrieval embedding
        |
Linear 128 -> 4
        |
[B, 4] classification logits
```

Each convolutional block contains:

`Conv2d(3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)`

Channel progression:

`3 -> 32 -> 64 -> 128 -> 256`

## Model Components

### ConvBlock

Reusable feature-extraction stage containing convolution, batch normalization, activation, and spatial downsampling.

### CustomCNN

Main model containing four convolutional feature blocks, adaptive global average pooling, a 128-dimensional embedding projection, and an explicit classification head.

### ClassificationHead

A separate `Linear(128, 4)` layer.

Class order:

1. `bracelet`
2. `earring_best`
3. `necklace`
4. `ring_best`

The head returns raw logits. Softmax is intentionally not part of the model.

## Public Interface

### `forward(x)`

Input:

`[B, 3, 224, 224]`

Returns:

`(normalized_embedding, classification_logits)`

with shapes:

`[B, 128]` and `[B, 4]`.

### `encode(x)`

Returns only the L2-normalized 128-dimensional embedding. This is the retrieval interface and is independent of the classification head.

### `forward_features(x)`

Returns the 128-dimensional representation before L2 normalization.

### `classify(embedding)`

Maps a 128-dimensional embedding to four raw class logits.

### `count_trainable_parameters(model)`

Returns the number of trainable parameters.

## Input Contract

| Property | Contract |
|---|---|
| Tensor shape | `[B, 3, 224, 224]` |
| Channels | RGB / 3 |
| Spatial size | 224 × 224 |
| Value preparation | S0.5 tensor pipeline, currently [0, 1] |
| Output embedding | 128 dimensions |
| Output classes | 4 |

The model itself does not resize images or perform augmentation.

## Parameter Count

The final implementation contains exactly **422,788 trainable parameters**.

| Component | Parameters |
|---|---:|
| Convolutional layers | 388,416 |
| BatchNorm parameters | 960 |
| Embedding projection | 32,896 |
| Classification head | 516 |
| **Total** | **422,788** |

## S1.4 Validation Evidence

S1.4 was executed successfully with a batch of two random inputs.

```text
Input shape:          (2, 3, 224, 224)
Feature Block 1:      (2, 32, 112, 112)
Feature Block 2:      (2, 64, 56, 56)
Feature Block 3:      (2, 128, 28, 28)
Feature Block 4:      (2, 256, 14, 14)
Global Pool:          (2, 256, 1, 1)
Embedding:            (2, 128)
Logits:               (2, 4)
Embedding L2 norm:    approximately 1.0
Trainable parameters: 422,788
Numerical outputs:    finite
```

The validation script is `scripts/test_model_forward.py`.

S1.4 validates model wiring and tensor contracts only. It does not measure classification or retrieval quality.

## Retrieval Boundary

The classification head is not part of the future similarity calculation.

The intended retrieval path is:

```text
query image
    -> preprocessing
    -> CustomCNN.encode()
    -> 128-d normalized embedding
    -> similarity calculation
    -> ranking
    -> Top-K results
```

Sprint 2 will train the model. Later sprints will define similarity and retrieval evaluation.

## Training Boundary

Sprint 1 defines and validates the model architecture only.

Deferred decisions:

- loss function;
- optimizer;
- learning rate;
- training batch size;
- augmentation policy;
- training epochs;
- checkpoint strategy;
- classification accuracy;
- embedding quality evaluation;
- similarity metric;
- Top-K retrieval evaluation.

## Related Files

| File | Responsibility |
|---|---|
| `src/temp_model/model.py` | Custom CNN implementation |
| `scripts/test_model_forward.py` | S1.4 forward-pass validation |
| `configs/model.yaml` | Model configuration |
| `docs/sprint-1-s1.1-model-design.md` | Architecture specification |
| `docs/sprint-1-s1.2-custom-cnn-implementation.md` | Implementation record |
| `docs/sprint-1-s1.3-classification-head.md` | Classification head record |
| `docs/sprint-1-s1.4-forward-pass-test.md` | Forward-pass validation record |
| `docs/sprint-1-s1.5-model-documentation.md` | Final model documentation |

## Sprint 1 Completion Criteria

| Task | Status |
|---|---|
| S1.1 Model Design | Complete |
| S1.2 Custom CNN Implementation | Complete |
| S1.3 Classification Head | Complete |
| S1.4 Forward Pass Test | Complete |
| S1.5 Model Documentation | Complete |
