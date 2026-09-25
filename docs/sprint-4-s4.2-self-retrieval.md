# S4.2 — Self Retrieval

## Objective

Verify the fundamental retrieval invariant that an image retrieves itself as
the highest-ranked candidate when its own persisted embedding is used as the
query.

## Procedure

1. Load the persisted `EmbeddingStore`.
2. Select one candidate row by index.
3. Use that row's normalized embedding as the query embedding.
4. Run the S4.1 retrieval pipeline against the same store.
5. Verify that the selected query index is ranked first.
6. Verify that its similarity score is approximately `1.0`.

The test uses the stored embedding directly. It does not regenerate the
embedding from the image file, so this task isolates retrieval correctness from
image preprocessing and model inference.

## Scope Boundary

S4.2 is a self-retrieval validation only.

It does not:

- exclude the query image;
- calculate Top-1/Top-5/Top-10 metrics over the dataset;
- evaluate unseen/excluded queries;
- measure category consistency;
- convert similarity to a percentage;
- modify the retrieval ranking logic.

Query exclusion is intentionally disabled here because the expected self-match
is the query itself. Exclusion is handled by S4.3.

## Runtime

Default command:

```bash
python scripts/self_retrieval.py
```

A specific store row can be selected with:

```bash
python scripts/self_retrieval.py --query-index 10
```

## Acceptance Criteria

The task passes when:

- the store loads successfully;
- retrieval returns candidates;
- the selected query index is rank 1;
- the self-similarity score is approximately `1.0`.
