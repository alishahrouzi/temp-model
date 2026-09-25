"""Validate the S3.4 cosine similarity function."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.similarity import cosine_similarity  # noqa: E402


def main() -> None:
    basis = np.eye(128, dtype=np.float32)
    identical = basis[0]
    orthogonal = basis[1]
    opposite = -basis[0]

    assert np.isclose(cosine_similarity(identical, identical), 1.0)
    assert np.isclose(cosine_similarity(identical, orthogonal), 0.0)
    assert np.isclose(cosine_similarity(identical, opposite), -1.0)

    candidates = np.stack([identical, orthogonal, opposite])
    scores = cosine_similarity(identical, candidates)

    assert scores.shape == (3,)
    assert np.allclose(scores, np.array([1.0, 0.0, -1.0], dtype=np.float32))
    assert np.isfinite(scores).all()

    print("S3.4 similarity-function validation passed.")
    print("  Metric: cosine similarity")
    print(f"  Query dimension: {identical.shape[0]}")
    print(f"  Candidate count: {len(candidates)}")
    print(f"  Identical score: {scores[0]:.6f}")
    print(f"  Orthogonal score: {scores[1]:.6f}")
    print(f"  Opposite score: {scores[2]:.6f}")
    print("  Ranking: not implemented")
    print("  Top-K selection: not implemented")


if __name__ == "__main__":
    main()
