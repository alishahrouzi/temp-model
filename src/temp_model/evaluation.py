"""Top-1 retrieval evaluation for Temp Model."""

from __future__ import annotations

from dataclasses import dataclass

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.retrieval import retrieve


@dataclass(frozen=True)
class Top1Evaluation:
    """Aggregate Top-1 retrieval evaluation result."""

    query_count: int
    correct_count: int
    incorrect_count: int
    accuracy: float


def evaluate_top1(store: EmbeddingStore) -> Top1Evaluation:
    """Evaluate class-consistent Top-1 retrieval over every store query.

    Each query uses its persisted embedding and excludes its own store row.
    A query is counted as correct when the highest-ranked remaining candidate
    belongs to the same class as the query.
    """
    if not isinstance(store, EmbeddingStore):
        raise TypeError("store must be an EmbeddingStore instance.")

    if store.sample_count < 2:
        raise ValueError("Top-1 evaluation requires at least two samples.")

    correct_count = 0

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

        if results[0].class_name == str(query_record["class"]):
            correct_count += 1

    query_count = store.sample_count
    incorrect_count = query_count - correct_count
    accuracy = correct_count / query_count

    return Top1Evaluation(
        query_count=query_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        accuracy=accuracy,
    )


@dataclass(frozen=True)
class Top5Evaluation:
    """Aggregate Top-5 retrieval evaluation result."""

    query_count: int
    correct_count: int
    incorrect_count: int
    accuracy: float


def evaluate_top5(store: EmbeddingStore) -> Top5Evaluation:
    """Evaluate class-consistent Top-5 retrieval over every store query.

    Each query uses its persisted embedding and excludes its own store row.
    A query is counted as correct when at least one of the five highest-ranked
    remaining candidates belongs to the same class as the query.
    """
    if not isinstance(store, EmbeddingStore):
        raise TypeError("store must be an EmbeddingStore instance.")

    if store.sample_count < 2:
        raise ValueError("Top-5 evaluation requires at least two samples.")

    correct_count = 0

    for query_index, query_record in enumerate(store.metadata_records):
        results = retrieve(
            store.embeddings[query_index],
            store,
            exclude_index=query_index,
        )

        top5_results = results[:5]
        if not top5_results:
            raise AssertionError(
                f"Query index {query_index} produced no candidates."
            )

        query_class = str(query_record["class"])
        if any(result.class_name == query_class for result in top5_results):
            correct_count += 1

    query_count = store.sample_count
    incorrect_count = query_count - correct_count
    accuracy = correct_count / query_count

    return Top5Evaluation(
        query_count=query_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        accuracy=accuracy,
    )


@dataclass(frozen=True)
class Top10Evaluation:
    """Aggregate Top-10 retrieval evaluation result."""

    query_count: int
    correct_count: int
    incorrect_count: int
    accuracy: float


def evaluate_top10(store: EmbeddingStore) -> Top10Evaluation:
    """Evaluate class-consistent Top-10 retrieval over every store query.

    Each query uses its persisted embedding and excludes its own store row.
    A query is counted as correct when at least one of the ten highest-ranked
    remaining candidates belongs to the same class as the query.
    """
    if not isinstance(store, EmbeddingStore):
        raise TypeError("store must be an EmbeddingStore instance.")

    if store.sample_count < 2:
        raise ValueError("Top-10 evaluation requires at least two samples.")

    correct_count = 0

    for query_index, query_record in enumerate(store.metadata_records):
        results = retrieve(
            store.embeddings[query_index],
            store,
            exclude_index=query_index,
        )

        top10_results = results[:10]
        if not top10_results:
            raise AssertionError(
                f"Query index {query_index} produced no candidates."
            )

        query_class = str(query_record["class"])
        if any(result.class_name == query_class for result in top10_results):
            correct_count += 1

    query_count = store.sample_count
    incorrect_count = query_count - correct_count
    accuracy = correct_count / query_count

    return Top10Evaluation(
        query_count=query_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        accuracy=accuracy,
    )
