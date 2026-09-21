"""Download and persist the configured Hugging Face dataset.

This task intentionally handles acquisition only. Dataset QA, duplicate analysis,
splitting, and preprocessing belong to later Sprint 0 tasks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import DatasetDict, load_dataset

DEFAULT_DATASET_ID = "sidd707/jewelry-design-dataset"
DEFAULT_OUTPUT_DIR = Path("data/raw/jewelry-design-dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Temp Model dataset from Hugging Face."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_metadata(dataset_id: str, dataset: DatasetDict, output_dir: Path) -> dict:
    return {
        "dataset_id": dataset_id,
        "source": "Hugging Face Datasets",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_dir.as_posix(),
        "splits": {name: len(split_data) for name, split_data in dataset.items()},
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)

    print(f"Persisting dataset to: {args.output_dir}")
    dataset.save_to_disk(str(args.output_dir))

    metadata = build_metadata(args.dataset_id, dataset, args.output_dir)
    metadata_path = args.output_dir / "acquisition_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Dataset acquisition completed.")
    for split_name, split_dataset in dataset.items():
        print(f"  - {split_name}: {len(split_dataset)} samples")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
