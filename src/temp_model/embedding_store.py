"""Persistent storage utilities for Temp Model retrieval embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.temp_model.model import EMBEDDING_DIM

STORAGE_VERSION = 1
EMBEDDING_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "store.json"


class EmbeddingStore:
    """Read and write a versioned embedding store.

    The store keeps the numeric embedding matrix separate from JSON metadata so
    retrieval can load the compact NumPy array without parsing image metadata.
    Row i in embeddings.npy corresponds exactly to metadata_records[i].
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata_records: list[dict[str, str]],
        *,
        split: str,
        checkpoint_epoch: int | None = None,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.metadata_records = metadata_records
        self.split = split
        self.checkpoint_epoch = checkpoint_epoch
        self.validate()

    def validate(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D; got {self.embeddings.shape}.")
        if self.embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension must be {EMBEDDING_DIM}; "
                f"got {self.embeddings.shape[1]}."
            )
        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embeddings contain NaN or Inf values.")
        if len(self.metadata_records) != self.embeddings.shape[0]:
            raise ValueError(
                "Metadata record count must match embedding row count: "
                f"{len(self.metadata_records)} != {self.embeddings.shape[0]}."
            )

        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("Embeddings must be L2-normalized.")

        required = {"image_path", "class", "description"}
        for index, record in enumerate(self.metadata_records):
            if not required.issubset(record):
                missing = sorted(required - set(record))
                raise ValueError(
                    f"Metadata record {index} is missing required fields: {missing}."
                )

    @property
    def sample_count(self) -> int:
        return int(self.embeddings.shape[0])

    def save(self, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self.validate()

        np.save(store_dir / EMBEDDING_FILENAME, self.embeddings)
        with (store_dir / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(
                {"metadata_records": self.metadata_records},
                handle,
                indent=2,
                ensure_ascii=False,
            )

        manifest: dict[str, Any] = {
            "storage_version": STORAGE_VERSION,
            "split": self.split,
            "sample_count": self.sample_count,
            "embedding_dimension": EMBEDDING_DIM,
            "dtype": str(self.embeddings.dtype),
            "normalized": True,
            "checkpoint_epoch": self.checkpoint_epoch,
            "embedding_file": EMBEDDING_FILENAME,
            "metadata_file": METADATA_FILENAME,
            "row_alignment": "metadata_records[i] corresponds to embeddings[i]",
        }
        with (store_dir / MANIFEST_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, store_dir: Path) -> "EmbeddingStore":
        store_dir = Path(store_dir)
        manifest_path = store_dir / MANIFEST_FILENAME
        metadata_path = store_dir / METADATA_FILENAME
        embeddings_path = store_dir / EMBEDDING_FILENAME

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Store manifest not found: {manifest_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Store metadata not found: {metadata_path}")
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Store embeddings not found: {embeddings_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if manifest.get("storage_version") != STORAGE_VERSION:
            raise ValueError(
                f"Unsupported storage version: {manifest.get('storage_version')!r}."
            )

        embeddings = np.load(embeddings_path, allow_pickle=False)
        records = metadata.get("metadata_records")
        if not isinstance(records, list):
            raise ValueError("Store metadata must contain a metadata_records list.")

        store = cls(
            embeddings=embeddings,
            metadata_records=records,
            split=str(manifest.get("split", "")),
            checkpoint_epoch=manifest.get("checkpoint_epoch"),
        )
        if store.sample_count != manifest.get("sample_count"):
            raise ValueError("Store sample count does not match its manifest.")
        if manifest.get("embedding_dimension") != EMBEDDING_DIM:
            raise ValueError("Store embedding dimension does not match the model contract.")
        return store
