# S1.3 — Classification Head

## Objective

Complete the classification component of the custom CNN as an explicit, independently addressable module while preserving the retrieval representation as the primary model output.

## Implementation

The model now exposes:

`ClassificationHead`

with:

- input dimension: 128
- output dimension: 4
- operation: `Linear(128, 4)`
- raw outputs: four class logits
- no softmax layer

Class order is fixed to the dataset categories:

1. `bracelet`
2. `earring_best`
3. `necklace`
4. `ring_best`

The ordering is stored in `CLASS_NAMES` and exposed by the classification head.

## Model Integration

The head is attached to `CustomCNN` as:

`classification_head`

and is reached through:

`CustomCNN.classify(embedding)`

The complete forward path remains:

`image → normalized embedding → classification logits`

The retrieval path remains independent:

`image → encode() → 128-d normalized embedding`

Therefore the classifier is not part of the later similarity calculation.

## Why No Softmax

The classification head returns raw logits rather than probabilities. This keeps the model compatible with standard multi-class classification losses such as cross-entropy, which expect logits and internally apply the required normalization.

Probability conversion, loss selection, class weighting, and training configuration are intentionally deferred to Sprint 2.

## Parameter Impact

The classification head contains:

- weights: 128 × 4 = 512
- bias: 4
- total: **516 trainable parameters**

The complete model remains at **422,788 trainable parameters**.

## Scope Boundary

S1.3 does not include:

- training;
- loss implementation;
- optimizer;
- class weighting;
- accuracy evaluation;
- forward-pass testing;
- checkpointing.

Those belong to later tasks.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| Explicit classification head implemented | Pass |
| 128 → 4 mapping | Pass |
| Dataset class order defined | Pass |
| Raw logits returned | Pass |
| No softmax added to model | Pass |
| Head separated from retrieval embedding | Pass |
| Retrieval path remains independent | Pass |
| Classification training deferred | Pass |
