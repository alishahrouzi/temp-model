"""Run S4.7 similarity score analysis on a persisted embedding store."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.similarity_analysis import analyze_similarity_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S4.7 similarity score analysis."
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path("embeddings/stores/test"),
        help="Path to the persisted embedding store.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation/s4.7-similarity-score-analysis.json"),
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = EmbeddingStore.load(args.store_dir)
    result = analyze_similarity_scores(store)

    print("S4.7 similarity score analysis completed.")
    print(f"  Query count: {result.query_count}")
    print(f"  Candidate count/query: {result.candidate_count_per_query}")
    print(f"  All-score count: {result.all_scores.count}")
    print(f"  All-score mean: {result.all_scores.mean:.6f}")
    print(f"  All-score median: {result.all_scores.median:.6f}")
    print(
        f"  All-score min/max: "
        f"{result.all_scores.minimum:.6f}/{result.all_scores.maximum:.6f}"
    )
    print(f"  Top-1 mean: {result.top1_scores.mean:.6f}")
    print(f"  Top-5 mean: {result.top5_scores.mean:.6f}")
    print(f"  Top-10 mean: {result.top10_scores.mean:.6f}")
    print(f"  Top-1 same-class mean: {result.top1_same_class_scores.mean:.6f}")
    if result.top1_different_class_scores.count:
        print(
            "  Top-1 different-class mean: "
            f"{result.top1_different_class_scores.mean:.6f}"
        )
    else:
        print("  Top-1 different-class mean: n/a (no incorrect Top-1 results)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    print(f"  JSON report: {args.output}")


if __name__ == "__main__":
    main()
