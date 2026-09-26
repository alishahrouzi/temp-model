# S4.7 — Similarity Score Analysis

## Objective

Analyze the cosine similarity scores produced by the existing excluded-query retrieval pipeline. This task evaluates the score distribution itself; it does not redefine similarity or accuracy.

## What Is Measured

For every test query:

1. Use the persisted 128-dimensional embedding.
2. Exclude the query's own store row.
3. Run the existing S4.3 retrieval and ranking pipeline.
4. Collect all remaining cosine similarity scores.
5. Summarize all scores and the scores appearing in the Top-1, Top-5, and Top-10 result ranges.
6. Separately summarize Top-1 scores where the retrieved class matches or differs from the query class.

For each score collection, the report contains:

- count
- minimum / maximum
- mean
- median
- standard deviation
- 5th percentile
- 25th percentile
- 75th percentile
- 95th percentile

## Interpretation

Cosine similarity is the retrieval score and remains in its natural range of approximately -1 to 1 for normalized embeddings.

A high similarity score does **not** mean the query is correct. Correctness is still determined by the class-consistency criteria used in S4.4–S4.6.

The separate same-class/different-class Top-1 summaries are included to examine whether incorrect Top-1 retrievals also receive high similarity scores. This is descriptive analysis, not a new decision threshold.

## Implementation

- `src/temp_model/similarity_analysis.py`
- `scripts/test_similarity_analysis.py`
- `scripts/analyze_similarity_scores.py`

The implementation reuses S4.3 retrieval and does not modify the model, embeddings, similarity function, or ranking algorithm.

## Runtime

Prepare the local store if necessary:

    python scripts/prepare_retrieval_store.py --dataset-root "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset\dataset"

Run validation:

    python scripts/test_similarity_analysis.py

Run the real analysis:

    python scripts/analyze_similarity_scores.py

The real run writes a JSON report to:

    results/evaluation/s4.7-similarity-score-analysis.json

The result directory is intended for runtime evaluation artifacts and should remain outside source-controlled model code.

## Scope Boundary

S4.7 does not:

- change the similarity formula;
- introduce a score threshold;
- convert similarity to a percentage;
- change Top-1/Top-5/Top-10 accuracy;
- retrain the model;
- change embeddings or ranking.

## Acceptance Criteria

- Every test query is analyzed with self-exclusion enabled.
- All remaining candidate scores are included.
- Top-1/Top-5/Top-10 score summaries are produced.
- Same-class and different-class Top-1 score groups are separated when available.
- Results are finite and internally count-consistent.
- The analysis does not modify persisted embeddings.
