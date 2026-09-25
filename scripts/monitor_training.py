"""Generate monitoring artifacts from a Temp Model training history.

S2.4 visualizes the existing epoch-level training history without retraining
the model or selecting a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_history(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Training history must contain a non-empty 'history' list.")
    required = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "learning_rate",
    }
    for index, row in enumerate(history, start=1):
        if not required.issubset(row):
            missing = sorted(required.difference(row))
            raise ValueError(f"History row {index} is missing fields: {missing}")
    return payload


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload["history"]
    best_val_loss = min(history, key=lambda row: row["validation_loss"])
    best_val_accuracy = max(history, key=lambda row: row["validation_accuracy"])
    final = history[-1]

    return {
        "epochs_completed": payload["epochs_completed"],
        "device": payload["device"],
        "train_samples": payload["train_samples"],
        "validation_samples": payload["validation_samples"],
        "best_validation_loss": {
            "epoch": best_val_loss["epoch"],
            "value": best_val_loss["validation_loss"],
        },
        "best_validation_accuracy": {
            "epoch": best_val_accuracy["epoch"],
            "value": best_val_accuracy["validation_accuracy"],
        },
        "final_epoch": {
            "epoch": final["epoch"],
            "train_loss": final["train_loss"],
            "train_accuracy": final["train_accuracy"],
            "validation_loss": final["validation_loss"],
            "validation_accuracy": final["validation_accuracy"],
            "learning_rate": final["learning_rate"],
        },
    }


def create_plots(payload: dict[str, Any], output_dir: Path) -> None:
    history = payload["history"]
    epochs = [row["epoch"] for row in history]

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    plt.plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Train Accuracy",
    )
    plt.plot(
        epochs,
        [row["validation_accuracy"] for row in history],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, [row["learning_rate"] for row in history], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Temp Model training monitoring artifacts."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=REPO_ROOT / "results" / "training" / "training_history.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "training" / "monitoring",
    )
    args = parser.parse_args()

    payload = load_history(args.history)
    summary = build_summary(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    create_plots(payload, args.output_dir)

    print(f"Monitoring summary written to: {args.output_dir / 'training_summary.json'}")
    print(f"Monitoring plots written to: {args.output_dir}")
    print("S2.4 training monitoring completed.")


if __name__ == "__main__":
    main()
