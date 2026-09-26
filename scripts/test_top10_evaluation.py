"""Runtime validation for S4.6 Top-10 evaluation."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.evaluation import Top10Evaluation, evaluate_top10


def unit_vector(index: int, dimension: int = 128) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def main() -> None:
    embeddings = np.stack(
        [unit_vector(0)] * 3
        + [unit_vector(1)]
        + [unit_vector(2)]
        + [unit_vector(3)]
        + [unit_vector(4)]
        + [unit_vector(5)]
        + [unit_vector(6)]
        + [unit_vector(7)]
        + [unit_vector(8)]
        + [unit_vector(9)]
    )
    metadata = [
        {"image_path": "a.jpg", "class": "bracelet", "description": "A"},
        {"image_path": "b.jpg", "class": "bracelet", "description": "B"},
        {"image_path": "c.jpg", "class": "bracelet", "description": "C"},
        {"image_path": "d.jpg", "class": "earring_best", "description": "D"},
        {"image_path": "e.jpg", "class": "ring_best", "description": "E"},
        {"image_path": "f.jpg", "class": "necklace", "description": "F"},
        {"image_path": "g.jpg", "class": "earring_best", "description": "G"},
        {"image_path": "h.jpg", "class": "ring_best", "description": "H"},
        {"image_path": "i.jpg", "class": "necklace", "description": "I"},
        {"image_path": "j.jpg", "class": "earring_best", "description": "J"},
        {"image_path": "k.jpg", "class": "ring_best", "description": "K"},
        {"image_path": "l.jpg", "class": "necklace", "description": "L"},
    ]
    store = EmbeddingStore(
        embeddings=embeddings,
        metadata_records=metadata,
        split="test",
        checkpoint_epoch=27,
    )

    result = evaluate_top10(store)

    expected = Top10Evaluation(
        query_count=12,
        correct_count=12,
        incorrect_count=0,
        accuracy=1.0,
    )

    if result != expected:
        raise AssertionError(f"Unexpected Top-10 result: {result} != {expected}")

    if not 0.0 <= result.accuracy <= 1.0:
        raise AssertionError("Top-10 accuracy must be between 0 and 1.")

    try:
        evaluate_top10(object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Invalid store type should be rejected.")

    print("S4.6 Top-10 evaluation validation passed.")
    print(f"  Query count: {result.query_count}")
    print(f"  Correct: {result.correct_count}")
    print(f"  Incorrect: {result.incorrect_count}")
    print(f"  Top-10 accuracy: {result.accuracy:.6f}")


if __name__ == "__main__":
    main()
