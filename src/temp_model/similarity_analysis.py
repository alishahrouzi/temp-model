"""Similarity score analysis for excluded-query retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.retrieval import retrieve


@dataclass(frozen=True)
class ScoreSummary:
    """Descriptive statistics for a collection of similarity scores."""

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float
    p05: float
    p25: float
    p75: float
    p95: float


@dataclass(frozen=True)
class SimilarityScoreAnalysis:
    """Aggregate similarity-score analysis over excluded test queries."""

    query_count: int
    candidate_count_per_query: int
    all_scores: ScoreSummary
    top1_scores: ScoreSummary
    top5_scores: ScoreSummary
    top10_scores: ScoreSummary
    top1_same_class_scores: ScoreSummary
    top1_different_class_scores: Optional[ScoreSummary]


def _summary(values: list[float]) -> Optional[ScoreSummary]:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Similarity scores must be finite.")
    percentiles = np.percentile(array, [5, 25, 75, 95])
    return ScoreSummary(
        count=int(array.size),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=float(np.std(array)),
        p05=float(percentiles[0]),
        p25=float(percentiles[1]),
        p75=float(percentiles[2]),
        p95=float(percentiles[3]),
    )


def analyze_similarity_scores(store: EmbeddingStore) -> SimilarityScoreAnalysis:
    """Analyze similarity scores for every test query with self excluded."""
    if not isinstance(store, EmbeddingStore):
        raise TypeError("store must be an EmbeddingStore instance.")
    if store.sample_count < 2:
        raise ValueError("Similarity analysis requires at least two samples.")

    all_scores: list[float] = []
    top1_scores: list[float] = []
    top5_scores: list[float] = []
    top10_scores: list[float] = []
    top1_same: list[float] = []
    top1_different: list[float] = []

    for query_index, query_record in enumerate(store.metadata_records):
        results = retrieve(
            store.embeddings[query_index],
            store,
            exclude_index=query_index,
        )
        if not results:
            raise AssertionError(
                f"Query index {query_index} produced no candidates."
            )

        scores = [float(result.score) for result in results]
        all_scores.extend(scores)
        top1_scores.append(scores[0])
        top5_scores.extend(scores[:5])
        top10_scores.extend(scores[:10])

        query_class = str(query_record["class"])
        if results[0].class_name == query_class:
            top1_same.append(scores[0])
        else:
            top1_different.append(scores[0])

    return SimilarityScoreAnalysis(
        query_count=store.sample_count,
        candidate_count_per_query=store.sample_count - 1,
        all_scores=_summary(all_scores),
        top1_scores=_summary(top1_scores),
        top5_scores=_summary(top5_scores),
        top10_scores=_summary(top10_scores),
        top1_same_class_scores=_summary(top1_same),
        top1_different_class_scores=_summary(top1_different),
    )
