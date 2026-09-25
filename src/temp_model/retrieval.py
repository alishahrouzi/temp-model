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
) -> list[RetrievalResult]:
    """Compare a query embedding against the store and rank all candidates.

    The pipeline composes S3.4 cosine similarity and S3.5 ranking. It returns
    every candidate in descending similarity order. Query exclusion and Top-K
    selection are intentionally left to later Sprint 4 tasks.
    """
    if not isinstance(store, EmbeddingStore):
        raise TypeError("store must be an EmbeddingStore instance.")

    scores = cosine_similarity(query_embedding, store.embeddings)
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
