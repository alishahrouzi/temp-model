"""Acquire the Temp Model dataset from Hugging Face or an existing local copy.

S0.1 handles dataset acquisition and basic source registration only.
Dataset QA, duplicate analysis, splitting, and preprocessing belong to
subsequent Sprint 0 tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_DATASET_ID = "sidd707/jewelry-design-dataset"
DEFAULT_OUTPUT_DIR = Path("data/raw/jewelry-design-dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire the Temp Model dataset from Hugging Face or a local copy."
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "local"),
        default="local",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Existing local dataset directory. It may be either the dataset "
            "repository root or the nested directory containing class folders "
            "and dataset_labels.csv. Required when --source local."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Destination for Hugging Face acquisition and acquisition metadata. "
            "Local mode does not copy raw images."
        ),
    )
    return parser.parse_args()


def _split_sizes(dataset: DatasetDict | Dataset) -> dict[str, int]:
    if isinstance(dataset, DatasetDict):
        return {name: len(split_data) for name, split_data in dataset.items()}
    return {"default": len(dataset)}


def _normalise_relative_path(value: str) -> Path:
    """Convert Windows-style CSV paths to a platform-independent relative path."""
    parts = PureWindowsPath(value).parts
    return Path(*parts)


def _resolve_local_dataset_root(input_dir: Path) -> Path:
    """Resolve either the repository root or the actual dataset directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Local dataset path is not a directory: {input_dir}")

    if (input_dir / "dataset_labels.csv").is_file():
        return input_dir

    nested_dataset = input_dir / "dataset"
    if (nested_dataset / "dataset_labels.csv").is_file():
        return nested_dataset

    raise RuntimeError(
        "The local dataset structure is not recognized. Expected either "
        "<input>/dataset_labels.csv with class directories, or "
        "<input>/dataset/dataset_labels.csv with class directories."
    )


def _local_metadata(
    dataset_id: str,
    input_dir: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    labels_path = dataset_root / "dataset_labels.csv"

    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"No data rows found in labels file: {labels_path}")

    fieldnames = set(rows[0].keys())
    required_fields = {"image_path", "description"}
    missing_fields = required_fields - fieldnames
    if missing_fields:
        raise RuntimeError(
            f"Labels file is missing required columns: {sorted(missing_fields)}"
        )

    missing_paths: list[str] = []
    for row in rows:
        relative_path = row["image_path"]
        image_path = dataset_root.joinpath(_normalise_relative_path(relative_path))
        if not image_path.is_file():
            missing_paths.append(relative_path)

    class_directories = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )

    return {
        "dataset_id": dataset_id,
        "source": "local",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_location": str(input_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "labels_file": "dataset_labels.csv",
        "labels_record_count": len(rows),
        "class_directories": class_directories,
        "resolvable_image_paths": len(rows) - len(missing_paths),
        "missing_image_path_count": len(missing_paths),
        "missing_image_paths": missing_paths,
    }


def acquire_from_local(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset_root = _resolve_local_dataset_root(input_dir)
    metadata = _local_metadata(dataset_id, input_dir, dataset_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata["metadata_location"] = str((output_dir / "acquisition_metadata.json").resolve())
    return metadata


def acquire_from_huggingface(
    dataset_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset from Hugging Face: {dataset_id}")
    dataset = load_dataset(dataset_id)
    print(f"Persisting dataset to: {output_dir}")
    dataset.save_to_disk(str(output_dir))

    return {
        "dataset_id": dataset_id,
        "source": "huggingface",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_location": dataset_id,
        "dataset_root": str(output_dir.resolve()),
        "splits": _split_sizes(dataset),
        "metadata_location": str(
            (output_dir / "acquisition_metadata.json").resolve()
        ),
    }


def main() -> None:
    args = parse_args()

    if args.source == "local":
        if args.input_dir is None:
            raise SystemExit("--input-dir is required when --source local.")

        metadata = acquire_from_local(
            dataset_id=args.dataset_id,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    else:
        metadata = acquire_from_huggingface(
            dataset_id=args.dataset_id,
            output_dir=args.output_dir,
        )

    metadata_path = args.output_dir / "acquisition_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Dataset acquisition completed.")
    print(f"  - Source: {metadata['source']}")
    print(f"  - Dataset: {metadata['dataset_id']}")
    print(f"  - Metadata: {metadata_path}")

    if metadata["source"] == "local":
        print(f"  - CSV records: {metadata['labels_record_count']}")
        print(
            "  - Resolvable image paths: "
            f"{metadata['resolvable_image_paths']}"
        )
        print(
            "  - Missing image paths: "
            f"{metadata['missing_image_path_count']}"
        )
        print(
            "  - Raw images copied: no "
            "(local mode registers the existing dataset in place)"
        )
    else:
        for split_name, size in metadata["splits"].items():
            print(f"  - {split_name}: {size} samples")


if __name__ == "__main__":
    main()
