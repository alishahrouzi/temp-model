"""Similarity functions for normalized Temp Model embeddings."""

from __future__ import annotations

import numpy as np

from src.temp_model.model import EMBEDDING_DIM


def _validate_embedding_array(embedding: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"{name} must have shape [D] or [N, D]; got {array.shape}."
        )
    if array.shape[-1] != EMBEDDING_DIM:
        raise ValueError(
            f"{name} must have embedding dimension {EMBEDDING_DIM}; "
            f"got {array.shape[-1]}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    norms = np.linalg.norm(array, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} must contain L2-normalized embeddings.")
    return array


def cosine_similarity(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> float | np.ndarray:
    """Return cosine similarity between one query and one or more candidates.

    Temp Model embeddings are L2-normalized at model output, so cosine
    similarity reduces to the dot product. The function intentionally does
    not rank or select candidates; that belongs to later retrieval tasks.
    """
    query = _validate_embedding_array(query_embedding, name="query_embedding")
    candidates = _validate_embedding_array(
        candidate_embeddings, name="candidate_embeddings"
    )

    if query.ndim != 1:
        raise ValueError(
            f"query_embedding must be a single [D] embedding; got {query.shape}."
        )

    scores = candidates @ query
    if candidates.ndim == 1:
        return float(scores)
    return scores.astype(np.float32, copy=False)
