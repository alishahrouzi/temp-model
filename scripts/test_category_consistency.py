"""Runtime validation for S4.8 category consistency analysis."""

from __future__ import annotations

import math
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.category_consistency import (
    CategoryConsistencyAnalysis,
    analyze_category_consistency,
)
from src.temp_model.embedding_store import EmbeddingStore


EMBEDDING_DIMENSION = 128


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
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

    result = analyze_category_consistency(store)

    if not isinstance(result, CategoryConsistencyAnalysis):
        raise AssertionError("Unexpected analysis result type.")
    if result.query_count != 4:
        raise AssertionError("Unexpected query count.")
    if result.categories != ("bracelet", "ring_best"):
        raise AssertionError("Unexpected category ordering.")
    if not math.isclose(result.top1_consistency, 1.0):
        raise AssertionError("Expected perfect Top-1 category consistency.")
    if not math.isclose(result.top5_consistency, 1.0):
        raise AssertionError("Expected perfect Top-5 category consistency.")
    if not math.isclose(result.top10_consistency, 1.0):
        raise AssertionError("Expected perfect Top-10 category consistency.")

    for category in result.categories:
        summary = result.per_category[category]
        if summary.query_count != 2:
            raise AssertionError("Each category should have two queries.")
        if not math.isclose(summary.top1_consistency, 1.0):
            raise AssertionError("Expected perfect per-category Top-1 consistency.")
        if not math.isclose(summary.top5_consistency, 1.0):
            raise AssertionError("Expected perfect per-category Top-5 consistency.")
        if not math.isclose(summary.top10_consistency, 1.0):
            raise AssertionError("Expected perfect per-category Top-10 consistency.")

    expected_confusion = {
        "bracelet": {"bracelet": 2, "ring_best": 0},
        "ring_best": {"bracelet": 0, "ring_best": 2},
    }
    if result.top1_confusion_matrix != expected_confusion:
        raise AssertionError("Unexpected Top-1 confusion matrix.")

    try:
        analyze_category_consistency(object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Invalid store type should be rejected.")

    print("S4.8 category consistency validation passed.")
    print(f"  Query count: {result.query_count}")
    print(f"  Categories: {', '.join(result.categories)}")
    print(f"  Top-1 consistency: {result.top1_consistency:.6f}")
    print(f"  Top-5 consistency: {result.top5_consistency:.6f}")
    print(f"  Top-10 consistency: {result.top10_consistency:.6f}")


if __name__ == "__main__":
    main()
