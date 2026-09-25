"""Select the best Temp Model training checkpoint from validation metrics.

S2.6 selects a checkpoint using validation-only evidence. The test split is
never used, and no retraining is performed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_history(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Training history must contain a non-empty 'history' list.")
    required = {
        "epoch", "validation_loss", "validation_accuracy",
        "train_loss", "train_accuracy", "learning_rate",
    }
    for index, row in enumerate(history, start=1):
        if not required.issubset(row):
            missing = sorted(required.difference(row))
            raise ValueError(f"History row {index} is missing fields: {missing}")
    return payload


def select_best(history: list[dict[str, Any]]) -> dict[str, Any]:
    # Primary: minimum validation loss; secondary: maximum validation accuracy;
    # final deterministic tie-breaker: earliest epoch.
    return min(
        history,
        key=lambda row: (
            float(row["validation_loss"]),
            -float(row["validation_accuracy"]),
            int(row["epoch"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best Temp Model checkpoint using validation metrics."
    )
    parser.add_argument(
        "--history", type=Path,
        default=REPO_ROOT / "results" / "training" / "training_history.json",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=REPO_ROOT / "checkpoints" / "s2.3",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results" / "training" / "best-checkpoint",
    )
    args = parser.parse_args()

    payload = load_history(args.history)
    selected = select_best(payload["history"])
    epoch = int(selected["epoch"])

    source = args.checkpoint_dir / f"epoch_{epoch:03d}.pt"
    if not source.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint does not exist: {source}. "
            "Run the completed training/checkpointing workflow first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "best.pt"
    shutil.copy2(source, destination)

    selection = {
        "selection_version": 1,
        "criterion": {
            "primary": "minimum_validation_loss",
            "secondary": "maximum_validation_accuracy",
            "tie_breaker": "earliest_epoch",
        },
        "selected_epoch": epoch,
        "source_checkpoint": str(source),
        "selected_checkpoint": str(destination),
        "metrics": selected,
        "data_scope": {
            "train_samples": payload["train_samples"],
            "validation_samples": payload["validation_samples"],
            "test_split_used": False,
        },
    }

    metadata_path = args.output_dir / "selection.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    print(f"Selected epoch: {epoch}")
    print(f"Validation loss: {selected['validation_loss']:.6f}")
    print(f"Validation accuracy: {selected['validation_accuracy']:.6f}")
    print(f"Best checkpoint copied to: {destination}")
    print(f"Selection metadata written to: {metadata_path}")
    print("S2.6 best-checkpoint selection completed.")


if __name__ == "__main__":
    main()
