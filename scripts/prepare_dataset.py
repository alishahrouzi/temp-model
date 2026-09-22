"""Validate and expose the S0.5 dataset preparation pipeline.

S0.5 consumes the manifests produced by S0.4 and provides a reusable PyTorch
Dataset with deterministic preprocessing. It never copies, moves, rewrites,
or augments source images.
"""

from __future__ import annotations

import argparse
import csv
import sys
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from PIL import Image, UnidentifiedImageError

# Allow direct execution from the repository root without requiring package installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.dataset import DEFAULT_IMAGE_SIZE, JewelrySplitDataset


DEFAULT_SPLIT_DIR = Path("results/dataset-split")
DEFAULT_OUTPUT_DIR = Path("results/dataset-preparation")
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate S0.4 manifests and the S0.5 deterministic preprocessing pipeline."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    return parser.parse_args()


def resolve_dataset_root(input_dir: Path) -> Path:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {input_dir}")
    if (input_dir / "dataset_labels.csv").is_file():
        return input_dir
    nested = input_dir / "dataset"
    if (nested / "dataset_labels.csv").is_file():
        return nested
    raise RuntimeError(
        "Unrecognized dataset structure. Expected dataset_labels.csv at input root "
        "or input/dataset."
    )


def normalize_path(value: str) -> str:
    return str(Path(*PureWindowsPath(value).parts))


def validate_manifest(manifest_path: Path, dataset_root: Path) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "class", "description"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Invalid manifest columns in {manifest_path}")
        rows = list(reader)

    missing = []
    unreadable = []
    classes = Counter()
    normalized_paths = set()

    for row in rows:
        relative = normalize_path(row["image_path"])
        normalized_paths.add(relative)
        classes[row["class"]] += 1
        path = dataset_root / Path(relative)

        if not path.is_file():
            missing.append(relative)
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append(
                {"path": relative, "error": f"{type(exc).__name__}: {exc}"}
            )

    return {
        "manifest": manifest_path.name,
        "image_count": len(rows),
        "class_distribution": dict(sorted(classes.items())),
        "missing_images": missing,
        "unreadable_images": unreadable,
        "duplicate_manifest_paths": len(rows) - len(normalized_paths),
    }


def validate_transformed_sample(dataset: JewelrySplitDataset) -> dict:
    sample = dataset[0]
    tensor = sample["image"]
    return {
        "sample_image_path": sample["image_path"],
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_min": float(tensor.min().item()),
        "tensor_max": float(tensor.max().item()),
        "class": sample["class"],
    }


def write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_preparation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# S0.5 — Dataset Preparation Pipeline Report",
        "",
        "## Objective",
        "Validate the S0.4 manifests and provide a reusable PyTorch dataset "
        "with deterministic preprocessing for later training and retrieval work.",
        "",
        "## Preprocessing",
        f"- Image size: {report['preprocessing']['image_size']} x {report['preprocessing']['image_size']}",
        "- Color conversion: RGB",
        "- Tensor conversion: float tensor in [0, 1]",
        "- Augmentation: none",
        "- Learned/pretrained normalization: none",
        "- Source images: unchanged",
        "",
        "## Split Validation",
        "",
        "| Split | Images | Missing | Unreadable | Duplicate manifest paths |",
        "|---|---:|---:|---:|---:|",
    ]

    for split in SPLITS:
        result = report["splits"][split]
        lines.append(
            f"| {split} | {result['image_count']} | {len(result['missing_images'])} | "
            f"{len(result['unreadable_images'])} | {result['duplicate_manifest_paths']} |"
        )

    lines.extend(
        [
            "",
            "## Pipeline Sample Validation",
            "",
            f"- Dataset item tensor shape: {report['sample_validation']['tensor_shape']}",
            f"- Tensor dtype: {report['sample_validation']['tensor_dtype']}",
            f"- Tensor range: [{report['sample_validation']['tensor_min']:.6f}, {report['sample_validation']['tensor_max']:.6f}]",
            f"- Sample path: {report['sample_validation']['sample_image_path']}",
            "",
            "## Scope Boundary",
            "- No source image copying, moving, renaming, or rewriting.",
            "- No augmentation in S0.5.",
            "- No model training.",
            "- No pretrained model or pretrained normalization statistics.",
            "- Model-specific augmentation and normalization remain open decisions for Sprint 1/2.",
        ]
    )

    (output_dir / "dataset_preparation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")

    dataset_root = resolve_dataset_root(args.input_dir)
    split_results = {}

    for split in SPLITS:
        split_results[split] = validate_manifest(
            args.split_dir / f"{split}.csv", dataset_root
        )
        if (
            split_results[split]["missing_images"]
            or split_results[split]["unreadable_images"]
        ):
            raise RuntimeError(f"S0.5 validation failed for {split} split.")

    datasets = {
        split: JewelrySplitDataset(
            args.split_dir / f"{split}.csv",
            dataset_root,
            image_size=args.image_size,
        )
        for split in SPLITS
    }

    sample_validation = validate_transformed_sample(datasets["train"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "split_directory": str(args.split_dir),
        "preprocessing": {
            "image_size": args.image_size,
            "color_mode": "RGB",
            "tensor_range": "[0, 1]",
            "augmentation": False,
            "normalization": None,
        },
        "splits": split_results,
        "sample_validation": sample_validation,
        "validation": {
            "all_split_images_resolve": True,
            "all_split_images_readable": True,
            "all_manifest_paths_unique_within_split": all(
                result["duplicate_manifest_paths"] == 0
                for result in split_results.values()
            ),
        },
    }

    write_report(args.output_dir, report)

    print("Dataset preparation validation completed.")
    for split in SPLITS:
        print(f"  - {split}: {split_results[split]['image_count']} images")
    print(
        f"  - Preprocessing: RGB -> resize {args.image_size}x{args.image_size} -> tensor [0,1]"
    )
    print("  - Augmentation: none")
    print(f"  - Report: {args.output_dir}")


if __name__ == "__main__":
    main()
