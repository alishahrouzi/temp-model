"""Runtime validation for S3.5 similarity ranking."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.ranking import rank_by_similarity


def main() -> None:
    scores = np.array([0.20, 0.90, 0.50, 0.90, -0.10], dtype=np.float32)
    ranked_indices = rank_by_similarity(scores)

    expected = np.array([1, 3, 2, 0, 4], dtype=np.int64)
    if not np.array_equal(ranked_indices, expected):
        raise AssertionError(
            f"Unexpected ranking: {ranked_indices.tolist()} != {expected.tolist()}"
        )

    ranked_scores = scores[ranked_indices]
    if not np.all(ranked_scores[:-1] >= ranked_scores[1:]):
        raise AssertionError("Ranked scores are not in descending order.")

    if ranked_indices.dtype != np.int64:
        raise AssertionError("Ranking indices must use int64 dtype.")

    if not np.array_equal(
        rank_by_similarity(np.empty(0, dtype=np.float32)),
        np.empty(0, dtype=np.int64),
    ):
        raise AssertionError("Empty input should return an empty int64 index array.")

    try:
        rank_by_similarity(np.array([[0.1, 0.2]], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("2D similarity scores should be rejected.")

    try:
        rank_by_similarity(np.array([0.1, np.nan], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN similarity scores should be rejected.")

    print("S3.5 ranking validation passed.")
    print("  Input candidate count: 5")
    print(f"  Ranking indices: {ranked_indices.tolist()}")
    print(f"  Ranked scores: {[float(score) for score in ranked_scores]}")
    print("  Order: descending similarity")
    print("  Tie handling: stable (original candidate order)")
    print("  Top-K selection: not implemented")
    print("  Query exclusion: not implemented")


if __name__ == "__main__":
    main()
