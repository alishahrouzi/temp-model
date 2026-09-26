"""Runtime validation for S4.7 similarity score analysis."""

from __future__ import annotations

import math
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.similarity_analysis import (
    SimilarityScoreAnalysis,
    analyze_similarity_scores,
)


EMBEDDING_DIMENSION = 128


def unit_vector(index: int, dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def main() -> None:
    embeddings = np.stack(
        [
            unit_vector(0),
            unit_vector(0),
            unit_vector(1),
            unit_vector(1),
        ]
    )
    metadata = [
        {"image_path": "a.jpg", "class": "bracelet", "description": "A"},
        {"image_path": "b.jpg", "class": "bracelet", "description": "B"},
        {"image_path": "c.jpg", "class": "ring_best", "description": "C"},
        {"image_path": "d.jpg", "class": "ring_best", "description": "D"},
    ]
    store = EmbeddingStore(
        embeddings=embeddings,
        metadata_records=metadata,
        split="test",
        checkpoint_epoch=27,
    )

    result = analyze_similarity_scores(store)

    if not isinstance(result, SimilarityScoreAnalysis):
        raise AssertionError("Unexpected analysis result type.")
    if result.query_count != 4 or result.candidate_count_per_query != 3:
        raise AssertionError("Unexpected query/candidate counts.")
    if result.all_scores.count != 12:
        raise AssertionError("All-score count must equal queries × candidates.")
    if result.top1_scores.count != 4:
        raise AssertionError("Top-1 score count must equal query count.")
    if result.top5_scores.count != 12:
        raise AssertionError("Top-5 score count should use all available candidates.")
    if result.top10_scores.count != 12:
        raise AssertionError("Top-10 score count should use all available candidates.")
    if result.top1_same_class_scores.count != 4:
        raise AssertionError("All Top-1 matches should be same-class in this fixture.")
    if result.top1_different_class_scores.count != 0:
        raise AssertionError("Fixture should have no different-class Top-1 results.")
    if not math.isclose(result.all_scores.minimum, 0.0):
        raise AssertionError("Expected zero minimum cosine similarity.")
    if not math.isclose(result.all_scores.maximum, 1.0):
        raise AssertionError("Expected one maximum cosine similarity.")
    if not math.isclose(result.top1_scores.minimum, 1.0):
        raise AssertionError("Expected perfect Top-1 similarity in fixture.")

    try:
        analyze_similarity_scores(object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Invalid store type should be rejected.")

    print("S4.7 similarity score analysis validation passed.")
    print(f"  Query count: {result.query_count}")
    print(f"  Candidate count/query: {result.candidate_count_per_query}")
    print(f"  All scores: {result.all_scores.count}")
    print(f"  Top-1 mean: {result.top1_scores.mean:.6f}")


if __name__ == "__main__":
    main()
