"""Dataset loading and deterministic preprocessing for Temp Model S0.5."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_IMAGE_SIZE = 224


class JewelrySplitDataset(Dataset[dict[str, Any]]):
    """Load one S0.4 split manifest with deterministic image preprocessing.

    The dataset keeps the source images untouched. Each item is loaded from its
    original path, converted to RGB, resized to a fixed square, and converted
    to a float tensor in [0, 1]. No augmentation or learned/pretrained
    normalization is applied in S0.5; those choices belong to later model
    development and training work.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer")

        self.records = self._read_manifest()
        self.transform = build_preprocessing_transform(image_size)

    def _read_manifest(self) -> list[dict[str, str]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "class", "description"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(
                    f"Manifest must contain columns: {sorted(required)}"
                )
            records = list(reader)

        if not records:
            raise ValueError(f"Manifest contains no records: {self.manifest_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.dataset_root / Path(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image referenced by manifest not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor: Tensor = self.transform(image)

        return {
            "image": tensor,
            "class": record["class"],
            "image_path": record["image_path"],
            "description": record["description"],
        }


def build_preprocessing_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    """Return the S0.5 deterministic preprocessing transform."""
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )
