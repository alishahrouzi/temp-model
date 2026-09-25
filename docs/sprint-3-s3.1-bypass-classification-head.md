# S3.1 — Remove/Bypass Classification Head

## Objective

Establish an explicit retrieval forward path that produces the learned image
embedding without invoking the classification head.

The classification head is retained in the model because Sprint 2 training
checkpoints contain its parameters and the head remains part of the
classification training path. It is not used by retrieval.

## Implementation

The custom CNN now exposes:

- `forward_features(x)`: convolutional feature extraction and embedding projection.
- `encode(x)`: L2-normalized 128-dimensional retrieval embedding.
- `forward_retrieval(x)`: explicit retrieval entry point that calls `encode(x)`
  and therefore bypasses `classification_head`.
- `classify(embedding)`: classification-only path retained for training compatibility.
- `forward(x)`: existing training-compatible path remains unchanged.

This separation avoids removing the classification parameters from existing
checkpoints while making the retrieval path explicit and independent of the classifier.

## Retrieval Data Flow

```text
Input Image
    |
    v
Feature Extractor
    |
    v
Global Average Pooling
    |
    v
128-d Embedding Projection
    |
    v
L2 Normalization
    |
    v
Retrieval Embedding
    |
    +----> Similarity / Ranking (later Sprint 3 tasks)
```

The classification head is outside this retrieval path.

## Validation

Run:

```bash
python scripts/test_retrieval_path.py
```

The validation confirms:

1. Retrieval output shape is `[B, 128]`.
2. Retrieval embeddings are finite.
3. Retrieval embeddings are L2-normalized.
4. The retrieval path produces the same embedding as `encode()`.
5. The classification method is deliberately made to fail during the test;
   retrieval still succeeds, proving that the classification head is not invoked.
6. The classification head remains present for Sprint 2 checkpoint/training compatibility.

## Scope

S3.1 does not:

- generate or store dataset embeddings;
- calculate similarity;
- rank images;
- implement Top-K retrieval;
- modify the trained checkpoint;
- retrain the model.

Those responsibilities belong to subsequent Sprint 3 and Sprint 4 tasks.
