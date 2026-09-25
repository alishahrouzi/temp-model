# S4.1 — Retrieval Pipeline

## Objective

Compose the completed Sprint 3 retrieval primitives into one retrieval
operation:

```
query embedding
    -> cosine similarity
    -> ranking
    -> ranked candidate results
```

The pipeline operates on a query embedding and an existing
`EmbeddingStore`.

## Interface

`retrieve(query_embedding, store)`

Input:

- one L2-normalized query embedding with dimension 128;
- a validated `EmbeddingStore`.

Output:

- a list of `RetrievalResult` objects;
- one result for every candidate in the store;
- results ordered by descending cosine similarity;
- each result preserves its store index and metadata:
  `image_path`, `class_name`, and `description`.

## Pipeline Composition

S4.1 does not introduce a new similarity metric or ranking algorithm.

It composes:

1. S3.4 `cosine_similarity`
2. S3.5 `rank_by_similarity`

The store already guarantees normalized, finite 128-dimensional embeddings and
metadata row alignment.

## Scope Boundary

S4.1 intentionally retrieves and ranks **all** candidates.

It does not:

- select Top-K;
- exclude the query image;
- calculate retrieval metrics;
- generate query embeddings from image files;
- convert similarity to a percentage;
- provide a UI.

These concerns are handled by subsequent Sprint 4 tasks.

## Validation

The runtime test verifies:

- similarity and ranking are composed correctly;
- descending result order;
- exact-score tie stability;
- score preservation;
- metadata alignment;
- invalid store rejection.

The test uses a small in-memory `EmbeddingStore` and does not require dataset
files or a trained checkpoint.
