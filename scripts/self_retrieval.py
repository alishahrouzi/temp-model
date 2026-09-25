"""Evaluate self-retrieval on a persisted embedding store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.retrieval import retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S4.2 self-retrieval.")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path("embeddings/stores/test"),
        help="Path to the persisted embedding store.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Zero-based store index used as the query.",
    )
    return parser.parse_args()


def run_self_retrieval(store_dir: Path, query_index: int) -> dict[str, object]:
    store = EmbeddingStore.load(store_dir)

    if query_index < 0 or query_index >= store.sample_count:
        raise ValueError(
            f"query_index must be in [0, {store.sample_count - 1}], "
            f"got {query_index}."
        )

    query_embedding = store.embeddings[query_index]
    results = retrieve(query_embedding, store)

    if not results:
        raise AssertionError("Self-retrieval returned no candidates.")

    top_result = results[0]
    query_record = store.metadata_records[query_index]

    if top_result.index != query_index:
        raise AssertionError(
            "Self-retrieval failed: query image was not ranked first. "
            f"Expected index {query_index}, got {top_result.index}."
        )

    if not np.isclose(top_result.score, 1.0, atol=1e-5, rtol=1e-5):
        raise AssertionError(
            "Self-retrieval failed: query-to-itself similarity is not 1.0. "
            f"Got {top_result.score:.8f}."
        )

    return {
        "store_dir": str(store_dir),
        "query_index": query_index,
        "query_image_path": str(query_record["image_path"]),
        "query_class": str(query_record["class"]),
        "candidate_count": len(results),
        "top_result_index": top_result.index,
        "top_result_score": top_result.score,
        "top_result_image_path": top_result.image_path,
        "top_result_class": top_result.class_name,
        "passed": True,
    }


def main() -> None:
    args = parse_args()
    result = run_self_retrieval(args.store_dir, args.query_index)

    print("S4.2 self-retrieval validation passed.")
    print(f"  Query index: {result['query_index']}")
    print(f"  Query image: {result['query_image_path']}")
    print(f"  Query class: {result['query_class']}")
    print(f"  Candidate count: {result['candidate_count']}")
    print(f"  Top-1 index: {result['top_result_index']}")
    print(f"  Top-1 score: {result['top_result_score']:.6f}")
    print(f"  Top-1 image: {result['top_result_image_path']}")
    print("  Query exclusion: disabled (intentional for self-retrieval)")


if __name__ == "__main__":
    main()
