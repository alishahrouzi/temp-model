"""Ranking utilities for Temp Model similarity scores."""

from __future__ import annotations

import numpy as np


def rank_by_similarity(similarity_scores: np.ndarray) -> np.ndarray:
    """Return candidate indices ordered by descending similarity score.

    The function ranks every candidate and intentionally does not perform
    Top-K selection or query exclusion; those concerns belong to later
    retrieval-pipeline tasks.
    """
    scores = np.asarray(similarity_scores, dtype=np.float32)

    if scores.ndim != 1:
        raise ValueError(
            f"similarity_scores must have shape [N]; got {scores.shape}."
        )
    if scores.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(scores).all():
        raise ValueError("similarity_scores contains NaN or Inf values.")

    # Stable descending sort keeps original candidate order for exact ties.
    return np.argsort(-scores, kind="stable").astype(np.int64, copy=False)
