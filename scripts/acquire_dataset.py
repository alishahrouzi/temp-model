"""Acquire the project dataset from Hugging Face or an existing local copy.

S0.1 intentionally handles acquisition only. Dataset QA, duplicate analysis,
splitting, and preprocessing belong to later Sprint 0 tasks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

DEFAULT_DATASET_ID = "sidd707/jewelry-design-dataset"
DEFAULT_OUTPUT_DIR = Path("data/raw/jewelry-design-dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire the Temp Model dataset from Hugging Face or a local copy."
    )
    parser.add_argument("--source", choices=("huggingface", "local"), default="local")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Existing local dataset directory. Required when --source local.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination used only for Hugging Face acquisition.",
    )
    return parser.parse_args()


def _split_sizes(dataset: DatasetDict | Dataset) -> dict[str, int]:
    if isinstance(dataset, DatasetDict):
        return {name: len(split_data) for name, split_data in dataset.items()}
    return {"default": len(dataset)}


def build_metadata(
    dataset_id: str,
    source: str,
    dataset: DatasetDict | Dataset,
    location: Path,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source": source,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "location": str(location),
        "splits": _split_sizes(dataset),
    }


def load_local_dataset(input_dir: Path) -> DatasetDict | Dataset:
    if not input_dir.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Local dataset path is not a directory: {input_dir}")

    try:
        return load_from_disk(str(input_dir))
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise RuntimeError(
            "The local path is not a Hugging Face dataset saved with save_to_disk. "
            "S0.1 currently expects the existing local copy to be a Hugging Face "
            "Dataset/DatasetDict directory. Do not alter the dataset before S0.2."
        ) from exc


def acquire_from_huggingface(
    dataset_id: str, output_dir: Path
) -> tuple[DatasetDict | Dataset, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset from Hugging Face: {dataset_id}")
    dataset = load_dataset(dataset_id)
    print(f"Persisting dataset to: {output_dir}")
    dataset.save_to_disk(str(output_dir))
    return dataset, output_dir


def main() -> None:
    args = parse_args()

    if args.source == "local":
        if args.input_dir is None:
            raise SystemExit("--input-dir is required when --source local.")

        dataset = load_local_dataset(args.input_dir)
        metadata_location = args.input_dir
    else:
        dataset, metadata_location = acquire_from_huggingface(
            args.dataset_id, args.output_dir
        )

    metadata = build_metadata(
        dataset_id=args.dataset_id,
        source=args.source,
        dataset=dataset,
        location=metadata_location,
    )
    metadata_path = metadata_location / "acquisition_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Dataset acquisition completed.")
    for split_name, size in metadata["splits"].items():
        print(f"  - {split_name}: {size} samples")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
