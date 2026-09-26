"""Category consistency analysis for excluded-query retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.retrieval import retrieve


@dataclass(frozen=True)
class CategoryConsistency:
    """Consistency of retrieved categories for one query category."""

    query_count: int
    top1_consistency: float
    top5_consistency: float
    top10_consistency: float


@dataclass(frozen=True)
class CategoryConsistencyAnalysis:
    """Aggregate category consistency over excluded test queries."""

    query_count: int
    categories: tuple[str, ...]
    top1_consistency: float
    top5_consistency: float
    top10_consistency: float
    per_category: dict[str, CategoryConsistency]
    top1_confusion_matrix: dict[str, dict[str, int]]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate a mean from an empty collection.")
    return sum(values) / len(values)


def analyze_category_consistency(
    store: EmbeddingStore,
) -> CategoryConsistencyAnalysis:
    """Analyze same-category purity across Top-1, Top-5, and Top-10 results."""
    if not isinstance(store, EmbeddingStore):
        raise TypeError("store must be an EmbeddingStore instance.")
    if store.sample_count < 2:
        raise ValueError(
            "Category consistency analysis requires at least two samples."
        )

    categories = tuple(
        sorted({str(record["class"]) for record in store.metadata_records})
    )
    per_category_values = {
        category: {"top1": [], "top5": [], "top10": []}
        for category in categories
    }
    confusion = {
        true_class: {predicted_class: 0 for predicted_class in categories}
        for true_class in categories
    }

    top1_values: list[float] = []
    top5_values: list[float] = []
    top10_values: list[float] = []

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

        query_class = str(query_record["class"])
        top5 = results[:5]
        top10 = results[:10]

        top1_consistency = float(results[0].class_name == query_class)
        top5_consistency = sum(
            result.class_name == query_class for result in top5
        ) / len(top5)
        top10_consistency = sum(
            result.class_name == query_class for result in top10
        ) / len(top10)

        top1_values.append(top1_consistency)
        top5_values.append(top5_consistency)
        top10_values.append(top10_consistency)

        per_category_values[query_class]["top1"].append(top1_consistency)
        per_category_values[query_class]["top5"].append(top5_consistency)
        per_category_values[query_class]["top10"].append(top10_consistency)

        confusion[query_class][results[0].class_name] += 1

    per_category = {
        category: CategoryConsistency(
            query_count=len(values["top1"]),
            top1_consistency=_mean(values["top1"]),
            top5_consistency=_mean(values["top5"]),
            top10_consistency=_mean(values["top10"]),
        )
        for category, values in per_category_values.items()
    }

    return CategoryConsistencyAnalysis(
        query_count=store.sample_count,
        categories=categories,
        top1_consistency=_mean(top1_values),
        top5_consistency=_mean(top5_values),
        top10_consistency=_mean(top10_values),
        per_category=per_category,
        top1_confusion_matrix=confusion,
    )
