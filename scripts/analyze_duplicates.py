"""Analyze exact and perceptual duplicate images for Sprint 0.3.

The source dataset is never modified. Exact duplicates are identified with
SHA-256. Near duplicates are identified with a 64-bit perceptual hash (pHash)
implemented with Pillow + NumPy, followed by Hamming-distance threshold
analysis.

This task intentionally reports evidence for the later split decision; it does
not delete, rename, or move images and does not choose a final deduplication
threshold automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_DATASET_ID = "sidd707/jewelry-design-dataset"
DEFAULT_OUTPUT_DIR = Path("results/duplicate-analysis")
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"
}
IGNORED_DIRECTORY_NAMES = {".cache", ".git", "__pycache__"}
DEFAULT_THRESHOLDS = (4, 6, 8, 10, 12)
PHASH_SIZE = 32
PHASH_LOW_FREQUENCY_SIZE = 8
PHASH_BITS = 64
HAMMING_BYTE_TABLE = np.array(
    [int(i).bit_count() for i in range(256)], dtype=np.uint8
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exact and perceptual duplicates without modifying the dataset."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="pHash Hamming-distance thresholds to evaluate.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Number of hashes compared per vectorized block.",
    )
    return parser.parse_args()


def normalise_relative_path(value: str) -> Path:
    return Path(*PureWindowsPath(value).parts)


def resolve_dataset_root(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {input_dir}")
    if (input_dir / "dataset_labels.csv").is_file():
        return input_dir
    nested = input_dir / "dataset"
    if (nested / "dataset_labels.csv").is_file():
        return nested
    raise RuntimeError(
        "Unrecognized dataset structure. Expected dataset_labels.csv either "
        "at the input root or under input/dataset."
    )


def read_labels_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    for encoding in CSV_ENCODINGS:
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(encoding)))), encoding
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path} with {CSV_ENCODINGS}")


def discover_image_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRECTORY_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path.relative_to(root))
    return sorted(files, key=lambda value: str(value).lower())


def class_from_path(path: Path) -> str:
    return path.parts[0] if len(path.parts) > 1 else "__root__"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dct_matrix(size: int, low_frequency_size: int) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64)
    frequencies = np.arange(low_frequency_size, dtype=np.float64)[:, None]
    matrix = np.cos(
        (np.pi / size) * (coords[None, :] + 0.5) * frequencies
    )
    matrix[0] *= 1.0 / np.sqrt(2.0)
    matrix *= np.sqrt(2.0 / size)
    return matrix


DCT_BASIS = dct_matrix(PHASH_SIZE, PHASH_LOW_FREQUENCY_SIZE)


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize(
            (PHASH_SIZE, PHASH_SIZE),
            Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(image, dtype=np.float64)

    coefficients = DCT_BASIS @ pixels @ DCT_BASIS.T
    low_frequency = coefficients[:PHASH_LOW_FREQUENCY_SIZE, :PHASH_LOW_FREQUENCY_SIZE]

    # Exclude the DC coefficient because it mainly captures overall brightness.
    values = low_frequency.flatten()[1:]
    median = float(np.median(values))
    bits = (values > median).astype(np.uint8)

    # 63 AC coefficients + one deterministic final bit = 64 bits.
    bits = np.concatenate([bits, np.array([0], dtype=np.uint8)])
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def exact_duplicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["sha256"]].append(record)

    duplicates = []
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        duplicates.append(
            {
                "hash": digest,
                "size": len(members),
                "classes": sorted({member["class"] for member in members}),
                "cross_class": len({member["class"] for member in members}) > 1,
                "paths": [member["path"] for member in members],
            }
        )
    return sorted(duplicates, key=lambda item: (-item["size"], item["paths"]))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def perceptual_pairs(
    hashes: np.ndarray,
    threshold: int,
    block_size: int,
) -> list[tuple[int, int, int]]:
    """Return all unique i<j pairs at or below the Hamming threshold."""
    count = len(hashes)
    pairs: list[tuple[int, int, int]] = []

    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        block = hashes[start:end]
        xor = block[:, None] ^ hashes[None, :]
        byte_view = xor.view(np.uint8).reshape(end - start, count, 8)
        distances = HAMMING_BYTE_TABLE[byte_view].sum(axis=2)

        for local_index in range(end - start):
            global_index = start + local_index
            valid = np.flatnonzero(
                (distances[local_index] <= threshold)
                & (np.arange(count) > global_index)
            )
            pairs.extend(
                (global_index, int(other), int(distances[local_index, other]))
                for other in valid
            )
    return pairs


def build_perceptual_groups(
    records: list[dict[str, Any]],
    pairs: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    union_find = UnionFind(len(records))
    for left, right, _distance in pairs:
        union_find.union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        root = union_find.find(index)
        groups[root].append(index)

    output = []
    for members in groups.values():
        if len(members) < 2:
            continue
        paths = [records[index]["path"] for index in members]
        classes = sorted({records[index]["class"] for index in members})
        distances = [
            distance
            for left, right, distance in pairs
            if left in members and right in members
        ]
        output.append(
            {
                "size": len(members),
                "classes": classes,
                "cross_class": len(classes) > 1,
                "min_pair_hamming_distance": min(distances) if distances else None,
                "max_pair_hamming_distance": max(distances) if distances else None,
                "paths": sorted(paths),
            }
        )
    return sorted(output, key=lambda item: (-item["size"], item["paths"]))


def threshold_summary(
    records: list[dict[str, Any]],
    pairs: list[tuple[int, int, int]],
) -> dict[str, Any]:
    class_by_index = {index: record["class"] for index, record in enumerate(records)}
    same_class = sum(
        class_by_index[left] == class_by_index[right] for left, right, _ in pairs
    )
    cross_class = len(pairs) - same_class
    distances = [distance for _, _, distance in pairs]

    return {
        "pair_count": len(pairs),
        "same_class_pair_count": same_class,
        "cross_class_pair_count": cross_class,
        "min_hamming_distance": min(distances) if distances else None,
        "max_hamming_distance": max(distances) if distances else None,
        "mean_hamming_distance": (
            float(np.mean(distances)) if distances else None
        ),
    }


def analyze_dataset(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
    thresholds: list[int],
    block_size: int,
) -> dict[str, Any]:
    if not thresholds or any(value < 0 or value > PHASH_BITS for value in thresholds):
        raise ValueError("Thresholds must be between 0 and 64.")
    thresholds = sorted(set(thresholds))

    root = resolve_dataset_root(input_dir)
    rows, csv_encoding = read_labels_csv(root / "dataset_labels.csv")
    if not rows:
        raise RuntimeError("No data rows found in dataset_labels.csv")

    discovered = discover_image_files(root)
    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []

    for relative_path in discovered:
        absolute_path = root / relative_path
        try:
            with Image.open(absolute_path) as image:
                image.verify()
            digest = sha256_file(absolute_path)
            phash = perceptual_hash(absolute_path)
            records.append(
                {
                    "path": str(relative_path),
                    "class": class_from_path(relative_path),
                    "sha256": digest,
                    "phash": f"{phash:016x}",
                }
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append(
                {
                    "path": str(relative_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    exact_groups = exact_duplicate_groups(records)
    hash_values = np.array(
        [int(record["phash"], 16) for record in records], dtype=np.uint64
    )

    threshold_results: dict[str, Any] = {}
    for threshold in thresholds:
        pairs = perceptual_pairs(hash_values, threshold, block_size)
        groups = build_perceptual_groups(records, pairs)
        threshold_results[str(threshold)] = {
            "summary": threshold_summary(records, pairs),
            "groups": groups,
            "pairs": [
                {
                    "left": records[left]["path"],
                    "right": records[right]["path"],
                    "left_class": records[left]["class"],
                    "right_class": records[right]["class"],
                    "hamming_distance": distance,
                }
                for left, right, distance in pairs
            ],
        }

    exact_cross_class = sum(group["cross_class"] for group in exact_groups)
    exact_member_count = sum(group["size"] for group in exact_groups)

    report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root.resolve()),
        "labels_encoding": csv_encoding,
        "scope": {
            "source_modified": False,
            "images_deleted": False,
            "images_renamed": False,
            "images_moved": False,
            "final_deduplication_threshold_selected": False,
            "note": (
                "Perceptual thresholds are reported as analysis candidates. "
                "No threshold is automatically adopted for dataset splitting."
            ),
        },
        "configuration": {
            "phash_algorithm": "64-bit pHash using 32x32 grayscale image and 8x8 low-frequency DCT",
            "phash_bits": PHASH_BITS,
            "thresholds": thresholds,
            "block_size": block_size,
        },
        "summary": {
            "discovered_image_files": len(discovered),
            "analyzed_images": len(records),
            "unreadable_images": len(unreadable),
            "exact_duplicate_group_count": len(exact_groups),
            "exact_duplicate_member_count": exact_member_count,
            "exact_cross_class_group_count": exact_cross_class,
            "perceptual_thresholds": {
                threshold: threshold_results[str(threshold)]["summary"]
                for threshold in thresholds
            },
        },
        "exact_duplicate_groups": exact_groups,
        "perceptual_analysis": threshold_results,
        "unreadable_images": unreadable,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "duplicate_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# S0.3 — Duplicate Analysis Report",
        "",
        "## Objective",
        "Identify exact and near-duplicate images before the dataset split. "
        "The source dataset is not modified.",
        "",
        "## Dataset Summary",
        f"- Discovered image files: **{len(discovered)}**",
        f"- Images analyzed: **{len(records)}**",
        f"- Unreadable images during duplicate analysis: **{len(unreadable)}**",
        f"- Exact duplicate groups: **{len(exact_groups)}**",
        f"- Exact duplicate members: **{exact_member_count}**",
        f"- Exact duplicate groups crossing classes: **{exact_cross_class}**",
        "",
        "## Exact Duplicate Detection",
        "Exact duplicates are grouped by SHA-256 of the complete file bytes.",
        "",
    ]

    if exact_groups:
        for index, group in enumerate(exact_groups, start=1):
            lines.append(
                f"### Group {index} — {group['size']} files "
                f"(cross_class={group['cross_class']})"
            )
            lines.extend(f"- {path}" for path in group["paths"])
            lines.append("")
    else:
        lines.append("- No exact duplicate groups found.")
        lines.append("")

    lines.extend(
        [
            "## Perceptual Duplicate Analysis",
            "Near duplicates are compared using 64-bit pHash Hamming distance. "
            "Multiple thresholds are measured so the split policy can be chosen "
            "from observed data rather than a hard-coded assumption.",
            "",
            "| Threshold | Pairs | Same-class pairs | Cross-class pairs | Groups |",
            "|---:|---:|---:|---:|---:|",
        ]
    )

    for threshold in thresholds:
        result = threshold_results[str(threshold)]
        summary = result["summary"]
        lines.append(
            f"| {threshold} | {summary['pair_count']} | "
            f"{summary['same_class_pair_count']} | "
            f"{summary['cross_class_pair_count']} | "
            f"{len(result['groups'])} |"
        )

    lines.extend(
        [
            "",
            "### Threshold Interpretation",
            "No final perceptual-deduplication threshold is selected by S0.3. "
            "S0.4 should use these results to keep duplicate/near-duplicate "
            "groups together when constructing train/validation/test splits.",
            "",
            "### Perceptual Groups",
        ]
    )

    for threshold in thresholds:
        groups = threshold_results[str(threshold)]["groups"]
        lines.append(f"#### Hamming threshold ≤ {threshold}")
        if not groups:
            lines.append("- No groups found.")
            continue
        for index, group in enumerate(groups, start=1):
            lines.append(
                f"- Group {index}: {group['size']} files; "
                f"classes={', '.join(group['classes'])}; "
                f"cross_class={group['cross_class']}; "
                f"distance={group['min_pair_hamming_distance']}–"
                f"{group['max_pair_hamming_distance']}"
            )
            lines.extend(f"  - {path}" for path in group["paths"])

    lines.extend(
        [
            "",
            "## Scope Boundary",
            "- Read-only analysis.",
            "- No image deletion, renaming, moving, or rewriting.",
            "- No final train/validation/test split.",
            "- No final perceptual threshold selection.",
            "- No model training.",
        ]
    )
    (output_dir / "duplicate_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    report = analyze_dataset(
        dataset_id=args.dataset_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        block_size=args.block_size,
    )
    summary = report["summary"]
    print("Duplicate analysis completed.")
    print(f"  - Images analyzed: {summary['analyzed_images']}")
    print(
        "  - Exact duplicate groups: "
        f"{summary['exact_duplicate_group_count']}"
    )
    print(
        "  - Exact duplicate members: "
        f"{summary['exact_duplicate_member_count']}"
    )
    for threshold in args.thresholds:
        result = summary["perceptual_thresholds"][threshold]
        print(
            f"  - pHash <= {threshold}: "
            f"{result['pair_count']} pairs, "
            f"{result['cross_class_pair_count']} cross-class"
        )
    print(f"  - Report: {args.output_dir}")


if __name__ == "__main__":
    main()
