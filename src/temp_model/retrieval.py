"""Retrieval pipeline for Temp Model embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.similarity import cosine_similarity
from src.temp_model.ranking import rank_by_similarity


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked candidate returned by the retrieval pipeline."""

    index: int
    score: float
    image_path: str
    class_name: str
    description: str


def retrieve(
    query_embedding: np.ndarray,
    store: EmbeddingStore,
    exclude_index: int | None = None,
) -> list[RetrievalResult]:
    """Compare a query embedding against the store and rank candidates.

    When exclude_index is provided, that store row is removed before ranking
    results. This supports excluded-query evaluation while preserving the
    S4.1 retrieval behavior when no exclusion is requested.
    """
    if not isinstance(store, EmbeddingStore):
        raise TypeError("store must be an EmbeddingStore instance.")

    if exclude_index is not None:
        if not isinstance(exclude_index, (int, np.integer)):
            raise TypeError("exclude_index must be an integer or None.")
        if exclude_index < 0 or exclude_index >= store.sample_count:
            raise ValueError(
                f"exclude_index must be in [0, {store.sample_count - 1}], "
                f"got {exclude_index}."
            )

    scores = cosine_similarity(query_embedding, store.embeddings)

    if exclude_index is not None:
        candidate_mask = np.ones(store.sample_count, dtype=bool)
        candidate_mask[exclude_index] = False
        candidate_indices = np.flatnonzero(candidate_mask)
        ranked_local_indices = rank_by_similarity(scores[candidate_indices])
        ranked_indices = candidate_indices[ranked_local_indices]
    else:
        ranked_indices = rank_by_similarity(scores)

    results: list[RetrievalResult] = []
    for index in ranked_indices:
        candidate_index = int(index)
        record = store.metadata_records[candidate_index]
        results.append(
            RetrievalResult(
                index=candidate_index,
                score=float(scores[candidate_index]),
                image_path=str(record["image_path"]),
                class_name=str(record["class"]),
                description=str(record["description"]),
            )
        )

    return results
