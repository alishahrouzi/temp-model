"""Runtime validation for S4.3 excluded-query retrieval."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.retrieval import retrieve


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

    query_index = 0
    query = store.embeddings[query_index]
    results = retrieve(query, store, exclude_index=query_index)

    expected_indices = [2, 1, 3]
    actual_indices = [result.index for result in results]
    if actual_indices != expected_indices:
        raise AssertionError(
            f"Unexpected excluded-query order: {actual_indices} != {expected_indices}"
        )

    if query_index in actual_indices:
        raise AssertionError("Excluded query index is still present in results.")

    expected_scores = [1.0, 0.0, 0.0]
    actual_scores = [result.score for result in results]
    if not np.allclose(actual_scores, expected_scores):
        raise AssertionError(
            f"Unexpected scores: {actual_scores} != {expected_scores}"
        )

    if [result.image_path for result in results] != ["c.jpg", "b.jpg", "d.jpg"]:
        raise AssertionError("Metadata alignment was not preserved.")

    if len(results) != store.sample_count - 1:
        raise AssertionError("Excluded retrieval must return N-1 candidates.")

    try:
        retrieve(query, store, exclude_index=store.sample_count)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range exclusion index should be rejected.")

    try:
        retrieve(query, store, exclude_index="0")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Non-integer exclusion index should be rejected.")

    print("S4.3 excluded-query retrieval validation passed.")
    print(f"  Query index: {query_index}")
    print(f"  Candidate count: {len(results)}")
    print(f"  Excluded index: {query_index}")
    print(f"  Ranked indices: {actual_indices}")
    print(f"  Ranked scores: {actual_scores}")
    print("  Query present in results: no")
    print("  Top-K selection: not implemented")


if __name__ == "__main__":
    main()
