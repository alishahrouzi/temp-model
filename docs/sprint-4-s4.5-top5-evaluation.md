# S4.5 — Top-5 Evaluation

## Objective

Measure Top-5 retrieval accuracy over the persisted test embedding store using excluded-query retrieval.

## Metric Definition

For every test image:

1. Use its persisted embedding as the query.
2. Exclude the query's own store index.
3. Rank the remaining candidates with the existing S4.3 retrieval pipeline.
4. Inspect the five highest-ranked candidates.
5. Count the query as a Top-5 hit when at least one of those candidates belongs to the same dataset class as the query.

The metric is:

`Top-5 accuracy = queries with at least one correct class in Top-5 / total queries`

Here, "correct" means **same class**, not identical image.

The similarity score remains a separate retrieval score and must not be interpreted as accuracy.

## Implementation

The evaluation is implemented in:

- `src/temp_model/evaluation.py`
- `scripts/evaluate_top5.py`
- `scripts/test_top5_evaluation.py`

The evaluator reuses the existing S4.3 `retrieve(..., exclude_index=...)` operation and takes the first five ranked results. It does not introduce another similarity metric or ranking method.

## Runtime

Prepare the local store if necessary:

    python scripts/prepare_retrieval_store.py --dataset-root "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset\dataset"

Run the validation test:

    python scripts/test_top5_evaluation.py

Run the real test-set evaluation:

    python scripts/evaluate_top5.py

## Scope Boundary

S4.5 does not:

- calculate Top-1 or Top-10;
- calculate category consistency as a separate metric;
- change the model or embeddings;
- retrain the model;
- convert similarity scores into percentages;
- introduce a new ranking algorithm.

Top-1 was evaluated in S4.4 and Top-10 is handled by S4.6.

## Acceptance Criteria

The task passes when:

- every test query is evaluated;
- each query excludes itself;
- up to five highest-ranked candidates are inspected;
- correct/incorrect counts are internally consistent;
- Top-5 accuracy equals `correct_count / query_count`;
- runtime evaluation completes without changing the persisted store.
