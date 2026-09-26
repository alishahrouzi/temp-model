"""Run Top-5 retrieval evaluation on a persisted embedding store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore
from src.temp_model.evaluation import evaluate_top5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S4.5 Top-5 retrieval evaluation."
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path("embeddings/stores/test"),
        help="Path to the persisted embedding store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = EmbeddingStore.load(args.store_dir)
    result = evaluate_top5(store)

    print("S4.5 Top-5 evaluation completed.")
    print(f"  Query count: {result.query_count}")
    print(f"  Correct Top-5 matches: {result.correct_count}")
    print(f"  Incorrect Top-5 matches: {result.incorrect_count}")
    print(f"  Top-5 accuracy: {result.accuracy:.6f}")
    print("  Query exclusion: enabled")
    print("  Relevance criterion: at least one query/result class match in Top-5")


if __name__ == "__main__":
    main()
