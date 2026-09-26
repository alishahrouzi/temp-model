# S4.4 — Top-1 Evaluation

## Objective

Measure the Top-1 retrieval accuracy over the persisted test embedding store
using excluded-query retrieval.

## Metric Definition

For every test image:

1. Use its persisted embedding as the query.
2. Exclude the query's own store index.
3. Rank the remaining candidates with the existing S4.3 retrieval pipeline.
4. Inspect the highest-ranked candidate.
5. Count the query as a Top-1 hit when the candidate belongs to the same
   dataset class as the query.

The metric is:

`Top-1 accuracy = correct Top-1 queries / total queries`

Here, "correct" means **same class**, not identical image.

The similarity score of the Top-1 result remains a separate retrieval score and
must not be interpreted as accuracy.

## Implementation

The evaluation is implemented in:

- `src/temp_model/evaluation.py`
- `scripts/evaluate_top1.py`
- `scripts/test_top1_evaluation.py`

The evaluator reuses the existing S4.3 `retrieve(..., exclude_index=...)`
operation. It does not introduce another similarity metric or ranking method.

## Runtime

Prepare the local store if necessary:

    python scripts/prepare_retrieval_store.py --dataset-root "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset\dataset"

Run the validation test:

    python scripts/test_top1_evaluation.py

Run the real test-set evaluation:

    python scripts/evaluate_top1.py

## Scope Boundary

S4.4 does not:

- calculate Top-5 or Top-10;
- calculate category consistency as a separate metric;
- change the model or embeddings;
- retrain the model;
- convert similarity scores into percentages;
- introduce a new ranking algorithm.

Top-5 and Top-10 are handled by S4.5 and S4.6.

## Acceptance Criteria

The task passes when:

- every test query is evaluated;
- each query excludes itself;
- exactly one Top-1 candidate is inspected per query;
- correct/incorrect counts are internally consistent;
- Top-1 accuracy equals `correct_count / query_count`;
- runtime evaluation completes without changing the persisted store.
