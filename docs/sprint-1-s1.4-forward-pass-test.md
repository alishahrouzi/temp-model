# S1.4 — Forward Pass Test

## Objective

Validate the structural forward pass of the custom CNN before training begins.

This task verifies tensor shapes, feature-stage transitions, embedding normalization, classification output dimensions, parameter count, and numerical stability. It does not evaluate model quality.

## Test Scope

The validation uses a deterministic random input with:

- batch size: 2
- channels: 3
- spatial size: 224 × 224

Expected feature-stage shapes:

| Stage | Expected shape |
|---|---|
| Input | `[2, 3, 224, 224]` |
| ConvBlock 1 | `[2, 32, 112, 112]` |
| ConvBlock 2 | `[2, 64, 56, 56]` |
| ConvBlock 3 | `[2, 128, 28, 28]` |
| ConvBlock 4 | `[2, 256, 14, 14]` |
| Global average pooling | `[2, 256, 1, 1]` |
| Retrieval embedding | `[2, 128]` |
| Classification logits | `[2, 4]` |

Additional assertions:

- embedding rows have L2 norm approximately 1;
- all intermediate and final tensors are finite;
- trainable parameter count is exactly 422,788;
- the classification head is a linear layer producing raw logits;
- `CustomCNN.forward()` returns the same normalized embedding produced by `encode()`.

## Implementation

The test is implemented in:

`scripts/test_model_forward.py`

It runs the actual PyTorch model in evaluation mode and checks each feature stage instead of testing only the final output. This makes failures localizable to the corresponding model component.

## Scope Boundary

S1.4 does not include:

- training;
- loss functions;
- optimizer;
- accuracy;
- retrieval metrics;
- similarity ranking;
- checkpointing;
- dataset-level evaluation.

Those belong to later sprints/tasks.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| Correct input contract | Pass |
| Feature block shapes validated | Pass |
| Global pooling shape validated | Pass |
| 128-d embedding shape validated | Pass |
| Embedding normalization validated | Pass |
| 4-class logits shape validated | Pass |
| Raw linear classification head validated | Pass |
| NaN/Inf checks | Pass |
| Parameter count validated | Pass |
| Retrieval and classification paths both exercised | Pass |
