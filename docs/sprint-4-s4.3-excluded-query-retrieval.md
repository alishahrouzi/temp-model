# S4.3 — Excluded Query Retrieval

## Objective

Evaluate retrieval with the query image explicitly excluded from the candidate
set.

The query is still taken from the persisted embedding store, but its own store
row is removed before ranking. This isolates retrieval of the other dataset
images and prepares the pipeline for later Top-K evaluation.

## Procedure

1. Load the persisted `EmbeddingStore`.
2. Select one store row as the query.
3. Use that row's normalized embedding as the query embedding.
4. Exclude the query row by its store index.
5. Compute cosine similarity against the remaining candidates.
6. Rank the remaining candidates using the existing S3.5 ranking logic.
7. Verify that the query index is absent from the results.

## Implementation

S4.3 extends the existing `retrieve()` operation with an optional
`exclude_index` parameter.

When `exclude_index=None`, S4.1 behavior is unchanged.

When an index is supplied:

- the index must be an integer;
- it must be within the store bounds;
- exactly that candidate row is removed;
- all remaining candidates retain their original store indices;
- similarity scores and metadata alignment are unchanged;
- ranking remains the existing stable descending ranking from S3.5.

No new similarity metric or ranking algorithm is introduced.

## Runtime

Prepare the local runtime store if required:

    python scripts/prepare_retrieval_store.py --dataset-root "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset\dataset"

Run the S4.3 evaluation:

    python scripts/excluded_query_retrieval.py

A specific query can be selected with:

    python scripts/excluded_query_retrieval.py --query-index 10

The runtime store remains local and gitignored.

## Scope Boundary

S4.3 does not:

- calculate Top-1/Top-5/Top-10 metrics;
- select a fixed Top-K result count;
- generate embeddings from a raw image file;
- convert similarity into a percentage;
- evaluate the full test set;
- measure category consistency.

These are handled by later Sprint 4 tasks.

## Acceptance Criteria

The task passes when:

- the store loads successfully;
- retrieval returns exactly `N-1` candidates;
- the query index is absent from the result list;
- remaining candidates are ranked by the existing similarity logic;
- metadata and original store indices remain aligned;
- invalid exclusion indices are rejected.
