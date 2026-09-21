"""Create a leakage-aware train/validation/test split for Sprint 0.4.

The split is built from the evidence produced by S0.2 and S0.3.

Rules:
- only readable discovered images are eligible;
- exact duplicate groups are indivisible;
- pHash groups at the conservative S0.3 threshold (default: <= 4) are also
  indivisible;
- cross-class perceptual groups are kept intact rather than forcing a class;
- no source images are copied, moved, renamed, or modified;
- the split manifests reference the original relative image paths.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

DEFAULT_DATASET_ID = "sidd707/jewelry-design-dataset"
DEFAULT_DUPLICATE_REPORT = Path("results/duplicate-analysis/duplicate_analysis.json")
DEFAULT_OUTPUT_DIR = Path("results/dataset-split")
DEFAULT_SPLITS = {"train": 0.80, "validation": 0.10, "test": 0.10}
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a leakage-aware dataset split from the S0.3 report."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--duplicate-report", type=Path, default=DEFAULT_DUPLICATE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=4,
        help="S0.3 pHash group threshold used as an indivisible split constraint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=256)
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


def read_labels_csv(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    raw = path.read_bytes()
    for encoding in CSV_ENCODINGS:
        try:
            rows = list(csv.DictReader(io.StringIO(raw.decode(encoding)))
            if not rows:
                raise RuntimeError("dataset_labels.csv contains no records.")
            required = {"image_path", "description"}
            missing = required - set(rows[0])
            if missing:
                raise RuntimeError(f"Missing required CSV columns: {sorted(missing)}")
            result: dict[str, dict[str, str]] = {}
            for row in rows:
                relative = normalise_relative_path(row["image_path"])
                result[str(relative)] = row
            return result, encoding
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path} with {CSV_ENCODINGS}")


def discover_image_files(root: Path) -> list[Path]:
    ignored = {".cache", ".git", "__pycache__"}
    extensions = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"
    }
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() in extensions:
            paths.append(path.relative_to(root))
    return sorted(paths, key=lambda value: str(value).lower())


def class_from_path(path: Path) -> str:
    return path.parts[0] if len(path.parts) > 1 else "__root__"


def load_duplicate_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"S0.3 duplicate report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if "exact_duplicate_groups" not in report or "perceptual_analysis" not in report:
        raise RuntimeError("Invalid S0.3 report: required duplicate sections are missing.")
    return report


def build_constraints(
    records: list[dict[str, Any]],
    duplicate_report: dict[str, Any],
    phash_threshold: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    path_to_index = {record["path"]: index for index, record in enumerate(records)}
    parent = list(range(len(records)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    def add_group(paths: list[str]) -> None:
        indices = [path_to_index[path] for path in paths if path in path_to_index]
        if len(indices) < 2:
            return
        first = indices[0]
        for index in indices[1:]:
            union(first, index)

    exact_groups = duplicate_report["exact_duplicate_groups"]
    for group in exact_groups:
        add_group(group["paths"])

    threshold_data = duplicate_report["perceptual_analysis"].get(str(phash_threshold))
    if threshold_data is None:
        available = sorted(int(value) for value in duplicate_report["perceptual_analysis"])
        raise ValueError(
            f"pHash threshold {phash_threshold} is not present in S0.3 report. "
            f"Available thresholds: {available}"
        )

    perceptual_groups = threshold_data["groups"]
    for group in perceptual_groups:
        add_group(group["paths"])

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[find(index)].append(index)

    groups = list(grouped.values())
    groups.sort(
        key=lambda members: (
            -len(members),
            min(records[index]["path"].lower() for index in members),
        )
    )

    exact_member_paths = {path for group in exact_groups for path in group["paths"]}
    phash_member_paths = {path for group in perceptual_groups for path in group["paths"]}

    metadata = {
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_members_in_split": sum(
            1 for path in exact_member_paths if path in path_to_index
        ),
        "perceptual_group_count": len(perceptual_groups),
        "perceptual_group_members_in_split": sum(
            1 for path in phash_member_paths if path in path_to_index
        ),
        "phash_threshold": phash_threshold,
        "combined_constraint_group_count": len(groups),
        "combined_multi_image_group_count": sum(len(g) > 1 for g in groups),
        "combined_constrained_images": sum(len(g) for g in groups if len(g) > 1),
    }
    return groups, metadata


def group_statistics(
    group: list[int], records: list[dict[str, Any]]
) -> tuple[int, Counter[str]]:
    return len(group), Counter(records[index]["class"] for index in group)


def split_objective(
    counts: dict[str, Counter[str]],
    totals: Counter[str],
    split_sizes: Counter[str],
    target_ratios: dict[str, float],
) -> float:
    total_images = sum(totals.values())
    score = 0.0
    for split, ratio in target_ratios.items():
        target_size = total_images * ratio
        score += 2.0 * abs(split_sizes[split] - target_size) / total_images
        for class_name, total in totals.items():
            target = total * ratio
            score += 4.0 * abs(counts[split][class_name] - target) / total_images
    return score


def assign_groups(
    groups: list[list[int]],
    records: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
    attempts: int,
) -> tuple[dict[int, str], float]:
    split_names = tuple(ratios)
    totals = Counter(records[index]["class"] for index in range(len(records)))
    rng = random.Random(seed)
    prepared = [(group, *group_statistics(group, records)) for group in groups]

    best_assignment: dict[int, str] | None = None
    best_score = float("inf")

    for attempt in range(max(1, attempts)):
        ordered = prepared.copy()
        if attempt:
            rng.shuffle(ordered)
        ordered.sort(
            key=lambda item: (
                -item[1],
                -max(item[2].values()),
                rng.random() if attempt else 0.0,
            )
        )

        counts = {split: Counter() for split in split_names}
        split_sizes = Counter()
        assignment: dict[int, str] = {}

        for group, size, class_counts in ordered:
            candidates = []
            for split in split_names:
                counts[split].update(class_counts)
                split_sizes[split] += size
                score = split_objective(counts, totals, split_sizes, ratios)
                counts[split].subtract(class_counts)
                split_sizes[split] -= size
                candidates.append((score, split))

            _, selected = min(candidates, key=lambda item: (item[0], item[1]))
            counts[selected].update(class_counts)
            split_sizes[selected] += size
            for index in group:
                assignment[index] = selected

        score = split_objective(counts, totals, split_sizes, ratios)
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not construct a dataset split.")
    return best_assignment, best_score


def validate_split(
    assignment: dict[int, str],
    groups: list[list[int]],
    records: list[dict[str, Any]],
) -> None:
    expected = set(range(len(records)))
    assigned = set(assignment)
    if assigned != expected:
        raise RuntimeError(
            f"Split assignment does not cover dataset exactly. "
            f"missing={len(expected - assigned)}, extra={len(assigned - expected)}"
        )

    for group in groups:
        split_names = {assignment[index] for index in group}
        if len(split_names) != 1:
            raise RuntimeError(
                "Leakage constraint violated: a duplicate/perceptual group "
                f"was assigned to multiple splits: {[records[i]['path'] for i in group]}"
            )


def build_report(
    dataset_id: str,
    root: Path,
    records: list[dict[str, Any]],
    assignment: dict[int, str],
    groups: list[list[int]],
    constraint_metadata: dict[str, Any],
    duplicate_report: dict[str, Any],
    objective_score: float,
    seed: int,
    attempts: int,
    csv_encoding: str,
) -> dict[str, Any]:
    split_order = ("train", "validation", "test")
    class_names = sorted({record["class"] for record in records})
    total = len(records)
    split_summary = {}

    for split in split_order:
        items = [
            records[index] for index, value in assignment.items() if value == split
        ]
        counts = Counter(item["class"] for item in items)
        split_summary[split] = {
            "image_count": len(items),
            "target_ratio": DEFAULT_SPLITS[split],
            "actual_ratio": len(items) / total,
            "class_distribution": {
                class_name: counts.get(class_name, 0) for class_name in class_names
            },
        }

    exact_groups = duplicate_report["exact_duplicate_groups"]
    threshold_data = duplicate_report["perceptual_analysis"][
        str(constraint_metadata["phash_threshold"])
    ]

    return {
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root.resolve()),
        "labels_encoding": csv_encoding,
        "source_image_count": total,
        "split_configuration": DEFAULT_SPLITS,
        "split_algorithm": {
            "type": "deterministic greedy group assignment with repeated seeded candidates",
            "seed": seed,
            "attempts": attempts,
            "objective_score": objective_score,
        },
        "leakage_constraints": {
            "exact_duplicate_groups": len(exact_groups),
            "phash_threshold": constraint_metadata["phash_threshold"],
            "phash_group_count": len(threshold_data["groups"]),
            "combined_constraint_group_count": len(groups),
            "combined_multi_image_group_count": constraint_metadata[
                "combined_multi_image_group_count"
            ],
            "policy": (
                "Every exact duplicate group and every S0.3 perceptual group "
                "at the selected threshold is assigned to exactly one split. "
                "Cross-class perceptual groups are kept intact."
            ),
        },
        "split_summary": split_summary,
        "validation": {
            "all_images_assigned_once": True,
            "no_constraint_group_crosses_splits": True,
        },
    }


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    records: list[dict[str, Any]],
    assignment: dict[int, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in DEFAULT_SPLITS:
        rows = [
            records[index] for index, value in assignment.items() if value == split
        ]
        rows.sort(key=lambda record: record["path"].lower())
        with (output_dir / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "class", "description"])
            for record in rows:
                writer.writerow([record["path"], record["class"], record["description"]])

    (output_dir / "dataset_split.json").write_text(
        json.dumps(
            {
                "report": report,
                "assignments": {
                    records[index]["path"]: value for index, value in assignment.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# S0.4 — Dataset Split Report",
        "",
        "## Objective",
        "Create train/validation/test manifests while preventing exact and "
        "S0.3 near-duplicate leakage across splits.",
        "",
        "## Split Configuration",
        "- Train: 80% target",
        "- Validation: 10% target",
        "- Test: 10% target",
        f"- pHash constraint threshold: <= {report['leakage_constraints']['phash_threshold']}",
        "",
        "## Dataset",
        f"- Source images: {report['source_image_count']}",
        "- Source dataset files were not modified.",
        "- CSV manifests reference the original relative image paths.",
        "",
        "## Results",
        "",
        "| Split | Images | Actual ratio | Target ratio |",
        "|---|---:|---:|---:|",
    ]

    for split, summary in report["split_summary"].items():
        lines.append(
            f"| {split} | {summary['image_count']} | "
            f"{summary['actual_ratio']:.4f} | {summary['target_ratio']:.2f} |"
        )

    lines.extend(["", "### Class Distribution", ""])
    lines.append("| Class | Train | Validation | Test |")
    lines.append("|---|---:|---:|---:|")
    classes = sorted(
        {
            class_name
            for summary in report["split_summary"].values()
            for class_name in summary["class_distribution"]
        }
    )
    for class_name in classes:
        lines.append(
            f"| {class_name} | "
            f"{report['split_summary']['train']['class_distribution'].get(class_name, 0)} | "
            f"{report['split_summary']['validation']['class_distribution'].get(class_name, 0)} | "
            f"{report['split_summary']['test']['class_distribution'].get(class_name, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Leakage Control",
            f"- Exact duplicate groups constrained: {report['leakage_constraints']['exact_duplicate_groups']}",
            f"- pHash groups constrained: {report['leakage_constraints']['phash_group_count']}",
            f"- Combined multi-image constraint groups: {report['leakage_constraints']['combined_multi_image_group_count']}",
            "- Exact duplicates never cross splits.",
            "- pHash candidate groups at the selected threshold never cross splits.",
            "- Cross-class perceptual groups remain intact; they are not relabeled.",
            "",
            "## Important Interpretation",
            "The pHash threshold is a split-leakage control derived from S0.3, "
            "not a claim that every perceptual group represents the same physical "
            "jewelry product. The split intentionally prefers conservative leakage "
            "control over perfectly exact 80/10/10 counts.",
            "",
            "## Output Files",
            "- train.csv",
            "- validation.csv",
            "- test.csv",
            "- dataset_split.json",
            "- dataset_split.md",
            "",
            "## Scope Boundary",
            "- No source image deletion.",
            "- No source image copying or moving.",
            "- No image rewriting or augmentation.",
            "- No model training.",
            "- No use of CSV descriptions as model labels; directory names are the class source.",
        ]
    )

    (output_dir / "dataset_split.md").write_text("\n".join(lines), encoding="utf-8")


def create_split(
    dataset_id: str,
    input_dir: Path,
    duplicate_report_path: Path,
    output_dir: Path,
    phash_threshold: int,
    seed: int,
    attempts: int,
) -> dict[str, Any]:
    root = resolve_dataset_root(input_dir)
    labels, csv_encoding = read_labels_csv(root / "dataset_labels.csv")
    duplicate_report = load_duplicate_report(duplicate_report_path)

    discovered = discover_image_files(root)
    records: list[dict[str, Any]] = []
    skipped_unreadable = []

    from PIL import Image, UnidentifiedImageError

    for relative in discovered:
        path = root / relative
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            skipped_unreadable.append(
                {"path": str(relative), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        key = str(relative)
        records.append(
            {
                "path": key,
                "class": class_from_path(relative),
                "description": labels.get(key, {}).get("description", ""),
            }
        )

    if skipped_unreadable:
        raise RuntimeError(
            "Unreadable images were discovered during S0.4. Fix the dataset "
            f"before splitting. Found {len(skipped_unreadable)} unreadable image(s)."
        )

    report_analyzed = duplicate_report.get("summary", {}).get("analyzed_images")
    if report_analyzed != len(records):
        raise RuntimeError(
            "Dataset changed after S0.3: S0.3 analyzed "
            f"{report_analyzed} images, but S0.4 found {len(records)} readable images."
        )

    groups, constraint_metadata = build_constraints(
        records, duplicate_report, phash_threshold
    )
    assignment, objective_score = assign_groups(
        groups=groups,
        records=records,
        ratios=DEFAULT_SPLITS,
        seed=seed,
        attempts=attempts,
    )
    validate_split(assignment, groups, records)

    report = build_report(
        dataset_id=dataset_id,
        root=root,
        records=records,
        assignment=assignment,
        groups=groups,
        constraint_metadata=constraint_metadata,
        duplicate_report=duplicate_report,
        objective_score=objective_score,
        seed=seed,
        attempts=attempts,
        csv_encoding=csv_encoding,
    )
    report["skipped_unreadable_images"] = skipped_unreadable
    write_outputs(output_dir, report, records, assignment)
    return report


def main() -> None:
    args = parse_args()
    report = create_split(
        dataset_id=args.dataset_id,
        input_dir=args.input_dir,
        duplicate_report_path=args.duplicate_report,
        output_dir=args.output_dir,
        phash_threshold=args.phash_threshold,
        seed=args.seed,
        attempts=args.attempts,
    )

    print("Dataset split completed.")
    print(f"  - Images assigned: {report['source_image_count']}")
    for split, summary in report["split_summary"].items():
        print(f"  - {split}: {summary['image_count']} ({summary['actual_ratio']:.4f})")
    print(
        "  - Leakage constraint groups: "
        f"{report['leakage_constraints']['combined_multi_image_group_count']}"
    )
    print(f"  - pHash threshold: {report['leakage_constraints']['phash_threshold']}")
    print(f"  - Report: {args.output_dir}")


if __name__ == "__main__":
    main()
