"""Train and validate the Temp Model custom CNN.

S2.2 implements only the training loop. Checkpoint creation and selection
belong to S2.3/S2.6 and are intentionally not performed here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.dataset import (  # noqa: E402
    JewelrySplitDataset,
    build_preprocessing_transform,
    build_training_transform,
)
from src.temp_model.model import CLASS_NAMES, CustomCNN  # noqa: E402


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


def resolve_device(config: dict[str, Any]) -> torch.device:
    strategy = config["device"]["strategy"]
    preferred = config["device"]["preferred"]
    if strategy != "auto":
        return torch.device(strategy)
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(config["device"]["fallback"])


def build_transforms(config: dict[str, Any]):
    image_size = int(config["input"]["image_size"])
    aug = config["augmentation"]
    if not aug["enabled"]:
        return build_preprocessing_transform(image_size), build_preprocessing_transform(image_size)
    jitter = aug["color_jitter"]
    train_transform = build_training_transform(
        image_size=image_size,
        horizontal_flip_probability=float(aug["horizontal_flip_probability"]),
        rotation_degrees=float(aug["rotation_degrees"]),
        brightness=float(jitter["brightness"]),
        contrast=float(jitter["contrast"]),
        saturation=float(jitter["saturation"]),
        hue=float(jitter["hue"]),
    )
    return train_transform, build_preprocessing_transform(image_size)


def make_loader(dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    dl = config["dataloader"]
    workers = int(dl["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(dl["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(dl["pin_memory"]) and torch.cuda.is_available(),
        drop_last=bool(dl["drop_last"]),
        persistent_workers=bool(dl["persistent_workers"]) and workers > 0,
    )


def class_weights(config: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    policy = config["loss"]["class_weighting"]
    if not policy["enabled"]:
        return None
    values = policy["classes"]
    try:
        weights = [float(values[name]) for name in CLASS_NAMES]
    except KeyError as exc:
        raise ValueError(f"Missing class weight for {exc.args[0]}") from exc
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: CustomCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = torch.tensor(
                [class_to_index[name] for name in batch["class"]],
                dtype=torch.long,
                device=device,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            _, logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_grad_norm
                    )
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += batch_size

    if total == 0:
        raise RuntimeError("DataLoader produced no samples.")
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Temp Model custom CNN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "training.yaml",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing the four dataset class folders.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset-split",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=REPO_ROOT / "results" / "training" / "training_history.json",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["training"]

    set_seed(int(config["seed"]), bool(config["deterministic"]))
    device = resolve_device(config)
    train_transform, eval_transform = build_transforms(config)

    train_dataset = JewelrySplitDataset(
        args.split_dir / "train.csv",
        args.dataset_root,
        image_size=int(config["input"]["image_size"]),
        transform=train_transform,
    )
    validation_dataset = JewelrySplitDataset(
        args.split_dir / "validation.csv",
        args.dataset_root,
        image_size=int(config["input"]["image_size"]),
        transform=eval_transform,
    )

    train_loader = make_loader(train_dataset, config, shuffle=True)
    validation_loader = make_loader(validation_dataset, config, shuffle=False)

    model = CustomCNN().to(device)
    weights = class_weights(config, device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(config["loss"]["label_smoothing"]),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
        betas=tuple(config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"]["mode"],
        factor=float(config["scheduler"]["factor"]),
        patience=int(config["scheduler"]["patience"]),
        threshold=float(config["scheduler"]["threshold"]),
        min_lr=float(config["scheduler"]["min_lr"]),
    )

    history: list[dict[str, float | int]] = []
    epochs = int(config["epochs"])
    max_grad_norm = float(config["gradient"]["max_norm"])

    print(f"Training device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            max_grad_norm,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            max_grad_norm=None,
        )
        scheduler.step(validation_loss)

        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": learning_rate,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.4f} | "
            f"lr={learning_rate:.6g}"
        )

    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(args.config),
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "epochs_completed": epochs,
        "history": history,
    }
    with args.history_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Training history written to: {args.history_output}")
    print("S2.2 training loop completed. No checkpoint was created or selected.")


if __name__ == "__main__":
    main()
