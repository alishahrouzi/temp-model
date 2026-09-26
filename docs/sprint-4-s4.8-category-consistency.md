# S4.8 — Category Consistency Analysis

## Objective

Analyze whether retrieved candidates remain consistent with the query category across Top-1, Top-5, and Top-10 retrieval.

S4.4-S4.6 measure whether at least one same-category result appears in Top-K. S4.8 adds category purity: the fraction of returned Top-K candidates that belong to the query category.

## Definitions

For each test query, after excluding its own image:

- Top-1 consistency: 1 if the Top-1 result has the same category, otherwise 0.
- Top-5 consistency: same-category results among Top-5 divided by the number of returned Top-5 results.
- Top-10 consistency: same-category results among Top-10 divided by the number of returned Top-10 results.

Overall consistency is the mean of the per-query values.

S4.8 also produces a Top-1 confusion matrix:

- rows = query/true category;
- columns = retrieved Top-1/predicted category;
- each cell = number of queries with that true/predicted category pair.

## Why this is separate from S4.4-S4.6

Top-K accuracy only asks whether at least one relevant result appears in Top-K. For example, a query with one correct result and four wrong results is a Top-5 hit but has only 20% Top-5 category consistency.

S4.8 therefore measures category purity of the returned list without changing the retrieval algorithm, model, embeddings, or similarity calculation.

## Implementation

Files:

- src/temp_model/category_consistency.py
- scripts/test_category_consistency.py
- scripts/analyze_category_consistency.py

The analysis reuses the existing S4.3 excluded-query retrieval pipeline.

## Runtime

Run the validation test:

    python scripts/test_category_consistency.py

Run the real test-set analysis:

    python scripts/analyze_category_consistency.py

The JSON report is written to:

    results/evaluation/s4.8-category-consistency.json

## Scope Boundary

S4.8 does not:

- retrain the model;
- change embeddings;
- change cosine similarity;
- change ranking;
- introduce a new retrieval algorithm;
- convert similarity scores into percentages;
- replace Top-K accuracy from S4.4-S4.6.

It only analyzes the category composition of already-ranked retrieval results.

## Acceptance Criteria

The task passes when:

- every test query is evaluated with self-exclusion;
- Top-1/5/10 consistency is computed from ranked candidates;
- per-category results are reported;
- the Top-1 confusion matrix is internally consistent;
- the JSON report is generated;
- no persisted embeddings or model checkpoints are modified.
