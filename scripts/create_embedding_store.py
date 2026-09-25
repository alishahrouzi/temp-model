"""Build a versioned persistent embedding store from S3.2 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.embedding_store import EmbeddingStore  # noqa: E402
from src.temp_model.model import EMBEDDING_DIM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a persistent Temp Model embedding store."
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument(
        "--input-dir", type=Path, default=REPO_ROOT / "embeddings"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "embeddings" / "stores",
    )
    args = parser.parse_args()

    embedding_path = args.input_dir / f"{args.split}.npy"
    generation_path = args.input_dir / f"{args.split}_generation.json"

    if not embedding_path.is_file():
        raise FileNotFoundError(f"S3.2 embedding file not found: {embedding_path}")
    if not generation_path.is_file():
        raise FileNotFoundError(f"S3.2 generation metadata not found: {generation_path}")

    embeddings = np.load(embedding_path, allow_pickle=False)
    with generation_path.open("r", encoding="utf-8") as handle:
        generation = json.load(handle)

    records = generation.get("metadata_records")
    if not isinstance(records, list):
        raise ValueError("S3.2 metadata must contain a metadata_records list.")
    if generation.get("embedding_dimension") != EMBEDDING_DIM:
        raise ValueError("S3.2 embedding dimension does not match the model contract.")
    if generation.get("sample_count") != len(records):
        raise ValueError("S3.2 sample count does not match metadata record count.")
    if embeddings.shape[0] != len(records):
        raise ValueError("Embedding rows do not match S3.2 metadata records.")

    checkpoint_epoch = generation.get("checkpoint_epoch")
    store_dir = args.output_dir / args.split
    store = EmbeddingStore(
        embeddings=embeddings,
        metadata_records=records,
        split=args.split,
        checkpoint_epoch=checkpoint_epoch,
    )
    store.save(store_dir)

    loaded = EmbeddingStore.load(store_dir)
    if not np.array_equal(loaded.embeddings, store.embeddings):
        raise RuntimeError("Storage round-trip changed embedding values.")
    if loaded.metadata_records != store.metadata_records:
        raise RuntimeError("Storage round-trip changed metadata records.")

    print("S3.3 embedding storage completed.")
    print(f"  Split: {args.split}")
    print(f"  Samples: {loaded.sample_count}")
    print(f"  Embedding shape: {loaded.embeddings.shape}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Checkpoint epoch: {loaded.checkpoint_epoch}")
    print(f"  Store: {store_dir}")


if __name__ == "__main__":
    main()
