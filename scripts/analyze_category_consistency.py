"""Run S4.8 category consistency analysis on a persisted embedding store."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.category_consistency import analyze_category_consistency
from src.temp_model.embedding_store import EmbeddingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S4.8 category consistency analysis."
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
        default=Path("results/evaluation/s4.8-category-consistency.json"),
        help="JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = EmbeddingStore.load(args.store_dir)
    result = analyze_category_consistency(store)

    print("S4.8 category consistency analysis completed.")
    print(f"  Query count: {result.query_count}")
    print(f"  Categories: {', '.join(result.categories)}")
    print(f"  Top-1 consistency: {result.top1_consistency:.6f}")
    print(f"  Top-5 consistency: {result.top5_consistency:.6f}")
    print(f"  Top-10 consistency: {result.top10_consistency:.6f}")

    print("  Per-category consistency:")
    for category in result.categories:
        summary = result.per_category[category]
        print(
            f"    {category}: queries={summary.query_count}, "
            f"Top-1={summary.top1_consistency:.6f}, "
            f"Top-5={summary.top5_consistency:.6f}, "
            f"Top-10={summary.top10_consistency:.6f}"
        )

    print("  Top-1 confusion matrix:")
    for true_class in result.categories:
        row = result.top1_confusion_matrix[true_class]
        formatted = ", ".join(
            f"{predicted_class}={row[predicted_class]}"
            for predicted_class in result.categories
        )
        print(f"    {true_class}: {formatted}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    print(f"  JSON report: {args.output}")


if __name__ == "__main__":
    main()
