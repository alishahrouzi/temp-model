"""Runtime validation for S4.4 Top-1 evaluation."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.evaluation import Top1Evaluation, evaluate_top1


def unit_vector(index: int, dimension: int = 128) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def main() -> None:
    embeddings = np.stack(
        [
            unit_vector(0),
            unit_vector(0),
            unit_vector(1),
            unit_vector(2),
        ]
    )
    metadata = [
        {"image_path": "a.jpg", "class": "bracelet", "description": "A"},
        {"image_path": "b.jpg", "class": "bracelet", "description": "B"},
        {"image_path": "c.jpg", "class": "earring_best", "description": "C"},
        {"image_path": "d.jpg", "class": "ring_best", "description": "D"},
    ]
    store = EmbeddingStore(
        embeddings=embeddings,
        metadata_records=metadata,
        split="test",
        checkpoint_epoch=27,
    )

    result = evaluate_top1(store)

    expected = Top1Evaluation(
        query_count=4,
        correct_count=2,
        incorrect_count=2,
        accuracy=0.5,
    )

    if result != expected:
        raise AssertionError(f"Unexpected Top-1 result: {result} != {expected}")

    if not 0.0 <= result.accuracy <= 1.0:
        raise AssertionError("Top-1 accuracy must be between 0 and 1.")

    try:
        evaluate_top1(object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Invalid store type should be rejected.")

    print("S4.4 Top-1 evaluation validation passed.")
    print(f"  Query count: {result.query_count}")
    print(f"  Correct: {result.correct_count}")
    print(f"  Incorrect: {result.incorrect_count}")
    print(f"  Top-1 accuracy: {result.accuracy:.6f}")


if __name__ == "__main__":
    main()
