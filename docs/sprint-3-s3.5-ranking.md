# S3.5 — Ranking

## Objective

Order all candidate images by their similarity score so the highest-scoring
candidate appears first.

## Interface

`rank_by_similarity(similarity_scores)`

Input:

- one-dimensional NumPy array with shape `[N]`;
- one similarity score per candidate.

Output:

- one-dimensional NumPy array of candidate indices with shape `[N]`;
- indices are ordered by descending similarity.

The original candidate order is preserved for exact score ties through a stable
sort.

## Relationship to S3.4

S3.4 calculates cosine similarity scores:

```
scores = cosine_similarity(query_embedding, candidate_embeddings)
```

S3.5 consumes those scores and produces an ordering:

```
similarity scores -> ranked candidate indices
```

The ranking function does not recalculate similarity.

## Validation

The function validates:

- one-dimensional input;
- finite similarity scores;
- empty input.

Runtime validation also checks:

- descending ordering;
- stable tie handling;
- index dtype;
- rejection of invalid shapes;
- rejection of NaN values.

## Scope Boundary

S3.5 ranks all candidates only.

It does not:

- select Top-K results;
- exclude the query image;
- load an embedding store;
- generate embeddings;
- calculate similarity;
- calculate Top-1/Top-5/Top-10 metrics;
- convert similarity to a percentage.

Those concerns remain separate retrieval/evaluation tasks.

## Design Rationale

Ranking is intentionally kept as a small independent operation. This preserves
the separation between similarity calculation and result selection, and allows
later retrieval tasks to decide whether they need all ranked candidates, query
exclusion, or Top-K slicing.
