"""Runtime validation for S4.1 retrieval pipeline."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.retrieval import RetrievalResult, retrieve


def unit_vector(index: int, dimension: int = 128) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def main() -> None:
    embeddings = np.stack(
        [
            unit_vector(0),
            unit_vector(1),
            unit_vector(0),
            unit_vector(2),
        ]
    )
    metadata = [
        {"image_path": "a.jpg", "class": "bracelet", "description": "A"},
        {"image_path": "b.jpg", "class": "earring_best", "description": "B"},
        {"image_path": "c.jpg", "class": "necklace", "description": "C"},
        {"image_path": "d.jpg", "class": "ring_best", "description": "D"},
    ]
    store = EmbeddingStore(
        embeddings=embeddings,
        metadata_records=metadata,
        split="test",
        checkpoint_epoch=27,
    )

    query = unit_vector(0)
    results = retrieve(query, store)

    expected_indices = [0, 2, 1, 3]
    actual_indices = [result.index for result in results]
    if actual_indices != expected_indices:
        raise AssertionError(
            f"Unexpected retrieval order: {actual_indices} != {expected_indices}"
        )

    if not all(isinstance(result, RetrievalResult) for result in results):
        raise AssertionError("All retrieval outputs must be RetrievalResult instances.")

    expected_scores = [1.0, 1.0, 0.0, 0.0]
    actual_scores = [result.score for result in results]
    if not np.allclose(actual_scores, expected_scores):
        raise AssertionError(
            f"Unexpected scores: {actual_scores} != {expected_scores}"
        )

    if [result.image_path for result in results] != ["a.jpg", "c.jpg", "b.jpg", "d.jpg"]:
        raise AssertionError("Metadata alignment was not preserved.")

    if [result.class_name for result in results] != [
        "bracelet",
        "necklace",
        "earring_best",
        "ring_best",
    ]:
        raise AssertionError("Class metadata alignment was not preserved.")

    try:
        retrieve(query, object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Invalid store type should be rejected.")

    print("S4.1 retrieval-pipeline validation passed.")
    print("  Candidate count: 4")
    print(f"  Ranked indices: {actual_indices}")
    print(f"  Ranked scores: {actual_scores}")
    print("  Components: cosine similarity -> descending ranking")
    print("  Top-K selection: not implemented")
    print("  Query exclusion: not implemented")


if __name__ == "__main__":
    main()
