"""Prepare the local retrieval embedding store for Sprint 4 evaluation.

Generated retrieval artifacts remain local and are intentionally excluded from Git.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local Temp Model retrieval embedding store."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument(
        "--manifest-dir", type=Path, default=REPO_ROOT / "results" / "dataset-split"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "results" / "training" / "best-checkpoint" / "best.pt",
    )
    parser.add_argument("--embeddings-dir", type=Path, default=REPO_ROOT / "embeddings")
    parser.add_argument(
        "--stores-dir", type=Path, default=REPO_ROOT / "embeddings" / "stores"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate S3.2 embeddings before rebuilding the store.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    display = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
    print(f"Running: {display}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    manifest_path = args.manifest_dir / f"{args.split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {args.checkpoint}. "
            "Run the Sprint 2 training/checkpoint selection workflow first."
        )

    embedding_path = args.embeddings_dir / f"{args.split}.npy"
    generation_path = args.embeddings_dir / f"{args.split}_generation.json"
    store_dir = args.stores_dir / args.split
    store_manifest = store_dir / "store.json"
    store_embeddings = store_dir / "embeddings.npy"
    store_metadata = store_dir / "metadata.json"

    if args.force or not embedding_path.is_file() or not generation_path.is_file():
        run([
            PYTHON,
            str(REPO_ROOT / "scripts" / "generate_embeddings.py"),
            "--dataset-root", str(args.dataset_root),
            "--split", args.split,
            "--manifest-dir", str(args.manifest_dir),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(args.embeddings_dir),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
        ])
    else:
        print("S3.2 artifacts already exist; reusing them.")

    if (
        args.force
        or not store_manifest.is_file()
        or not store_embeddings.is_file()
        or not store_metadata.is_file()
    ):
        run([
            PYTHON,
            str(REPO_ROOT / "scripts" / "create_embedding_store.py"),
            "--split", args.split,
            "--input-dir", str(args.embeddings_dir),
            "--output-dir", str(args.stores_dir),
        ])
    else:
        print("S3.3 store artifacts already exist; reusing them.")

    if not (
        store_manifest.is_file()
        and store_embeddings.is_file()
        and store_metadata.is_file()
    ):
        raise RuntimeError(f"Retrieval store preparation failed: {store_dir}")

    print("Retrieval store preparation completed.")
    print(f"  Split: {args.split}")
    print(f"  Store: {store_dir}")
    print("  Runtime artifacts remain local and are intentionally gitignored.")


if __name__ == "__main__":
    main()
