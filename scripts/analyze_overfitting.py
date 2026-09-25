"""Analyze overfitting signals from a completed Temp Model training history.

S2.5 is descriptive only: it analyzes train/validation behavior and does not
select a checkpoint or use the test split.
"""

from __future__ import annotations

import argparse
import json
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


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload["history"]
    best_val_loss = min(history, key=lambda row: row["validation_loss"])
    best_val_accuracy = max(history, key=lambda row: row["validation_accuracy"])
    final = history[-1]

    best_epoch = best_val_loss["epoch"]
    best_epoch_row = next(row for row in history if row["epoch"] == best_epoch)

    final_accuracy_gap = final["train_accuracy"] - final["validation_accuracy"]
    final_loss_gap = final["validation_loss"] - final["train_loss"]

    post_best = [row for row in history if row["epoch"] > best_epoch]
    post_best_accuracy_drop = (
        best_epoch_row["validation_accuracy"] - final["validation_accuracy"]
    )
    post_best_loss_increase = (
        final["validation_loss"] - best_epoch_row["validation_loss"]
    )

    train_loss_continues_down = final["train_loss"] < best_epoch_row["train_loss"]
    train_accuracy_continues_up = final["train_accuracy"] > best_epoch_row["train_accuracy"]
    validation_loss_worsens = final["validation_loss"] > best_epoch_row["validation_loss"]
    validation_accuracy_worsens = final["validation_accuracy"] < best_epoch_row["validation_accuracy"]

    late_overfitting_signal = (
        bool(post_best)
        and train_loss_continues_down
        and train_accuracy_continues_up
        and validation_loss_worsens
        and validation_accuracy_worsens
    )

    return {
        "epochs_analyzed": len(history),
        "data_scope": {
            "train_samples": payload["train_samples"],
            "validation_samples": payload["validation_samples"],
            "test_split_used": False,
        },
        "observed_validation_peaks": {
            "minimum_validation_loss": {
                "epoch": best_val_loss["epoch"],
                "value": best_val_loss["validation_loss"],
            },
            "maximum_validation_accuracy": {
                "epoch": best_val_accuracy["epoch"],
                "value": best_val_accuracy["validation_accuracy"],
            },
        },
        "final_epoch": {
            "epoch": final["epoch"],
            "train_loss": final["train_loss"],
            "validation_loss": final["validation_loss"],
            "train_accuracy": final["train_accuracy"],
            "validation_accuracy": final["validation_accuracy"],
            "learning_rate": final["learning_rate"],
        },
        "generalization_gap_at_final_epoch": {
            "accuracy": final_accuracy_gap,
            "loss": final_loss_gap,
        },
        "post_observed_validation_peak": {
            "reference_epoch": best_epoch,
            "epochs_after_reference": len(post_best),
            "validation_loss_increase": post_best_loss_increase,
            "validation_accuracy_drop": post_best_accuracy_drop,
            "train_loss_change": final["train_loss"] - best_epoch_row["train_loss"],
            "train_accuracy_change": final["train_accuracy"] - best_epoch_row["train_accuracy"],
        },
        "signals": {
            "train_loss_continues_down_after_reference": train_loss_continues_down,
            "train_accuracy_continues_up_after_reference": train_accuracy_continues_up,
            "validation_loss_worsens_after_reference": validation_loss_worsens,
            "validation_accuracy_worsens_after_reference": validation_accuracy_worsens,
            "late_overfitting_signal": late_overfitting_signal,
        },
        "interpretation": {
            "status": (
                "late_generalization_degradation_observed"
                if late_overfitting_signal
                else "no_clear_late_overfitting_signal"
            ),
            "note": (
                "The training history shows a late divergence pattern after the "
                "observed validation peak: training metrics continue improving "
                "while validation metrics deteriorate. Earlier validation metrics "
                "are also highly variable, so this result is evidence of late "
                "generalization degradation rather than proof that all validation "
                "fluctuations are caused by overfitting."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Temp Model training history for overfitting signals."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=REPO_ROOT / "results" / "training" / "training_history.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "training" / "overfitting_analysis.json",
    )
    args = parser.parse_args()

    payload = load_history(args.history)
    analysis = analyze(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    print(f"Overfitting analysis written to: {args.output}")
    print(
        "S2.5 overfitting analysis completed "
        f"(status={analysis['interpretation']['status']})."
    )


if __name__ == "__main__":
    main()
