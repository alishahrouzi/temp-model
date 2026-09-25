# S3.4 — Similarity Function

## Objective

Define the similarity metric used to compare the 128-dimensional retrieval embeddings produced by Temp Model.

## Selected Metric

**Cosine similarity** is the selected metric.

For embeddings q and x:

    cosine(q,x) = (q · x) / (||q||₂ ||x||₂)

Temp Model retrieval embeddings are L2-normalized by CustomCNN.encode() / forward_retrieval(). Therefore, for stored embeddings:

    ||q||₂ = ||x||₂ = 1

and cosine similarity reduces numerically to the dot product:

    cosine(q,x) = q · x

This makes the calculation simple and efficient while preserving the intended cosine metric.

## Interface

cosine_similarity(query_embedding, candidate_embeddings)

Supported inputs:

- query: [128]
- one candidate: [128]
- multiple candidates: [N, 128]

Returns:

- float for one candidate;
- numpy.ndarray[N] for multiple candidates.

The function validates embedding dimensionality, finite values, L2 normalization, and exactly one query vector.

## Scope Boundary

S3.4 only defines the pairwise similarity calculation.

It does not rank candidates, select Top-K, exclude the query image, load an embedding store, calculate retrieval metrics, or produce a similarity percentage.

Those concerns remain separate tasks in the retrieval/evaluation pipeline.

## Interpretation

Cosine similarity ranges from -1 to 1 in the general case:

- 1: identical embedding direction;
- 0: orthogonal embedding directions;
- -1: opposite embedding directions.

A similarity score is not classification accuracy and is not converted to a percentage in this task.
