"""Structural forward-pass validation for the Temp Model custom CNN.

S1.4 intentionally validates model wiring and tensor contracts only.
Training, accuracy, retrieval metrics, and checkpointing belong to later tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT 
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.temp_model.model import (  # noqa: E402
    CLASS_NAMES,
    EMBEDDING_DIM,
    INPUT_CHANNELS,
    INPUT_IMAGE_SIZE,
    NUM_CLASSES,
    CustomCNN,
    count_trainable_parameters,
)


BATCH_SIZE = 2
EXPECTED_FEATURE_SHAPES = (
    (BATCH_SIZE, 32, 112, 112),
    (BATCH_SIZE, 64, 56, 56),
    (BATCH_SIZE, 128, 28, 28),
    (BATCH_SIZE, 256, 14, 14),
)
EXPECTED_PARAMETER_COUNT = 422_788


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf values."


def main() -> None:
    torch.manual_seed(42)

    model = CustomCNN()
    model.eval()

    assert count_trainable_parameters(model) == EXPECTED_PARAMETER_COUNT
    assert model.embedding_dim == EMBEDDING_DIM
    assert model.num_classes == NUM_CLASSES
    assert model.classification_head.num_classes == NUM_CLASSES
    assert model.classification_head.class_names == CLASS_NAMES
    assert isinstance(model.classification_head.linear, nn.Linear)

    x = torch.randn(
        BATCH_SIZE,
        INPUT_CHANNELS,
        INPUT_IMAGE_SIZE,
        INPUT_IMAGE_SIZE,
    )

    with torch.no_grad():
        feature = x
        for index, block in enumerate(model.feature_extractor):
            feature = block(feature)
            expected_shape = EXPECTED_FEATURE_SHAPES[index]
            assert tuple(feature.shape) == expected_shape, (
                f"Feature block {index + 1}: expected {expected_shape}, "
                f"got {tuple(feature.shape)}."
            )
            assert_finite(f"feature block {index + 1}", feature)

        pooled = model.global_pool(feature)
        assert tuple(pooled.shape) == (BATCH_SIZE, 256, 1, 1)
        assert_finite("global pooled feature", pooled)

        flattened = torch.flatten(pooled, start_dim=1)
        projected = model.embedding_projection(flattened)
        assert tuple(projected.shape) == (BATCH_SIZE, EMBEDDING_DIM)
        assert_finite("projected embedding", projected)

        embedding = model.encode(x)
        assert tuple(embedding.shape) == (BATCH_SIZE, EMBEDDING_DIM)
        assert_finite("normalized embedding", embedding)

        embedding_norms = torch.linalg.vector_norm(embedding, ord=2, dim=1)
        assert torch.allclose(
            embedding_norms,
            torch.ones(BATCH_SIZE),
            atol=1e-5,
            rtol=1e-5,
        ), f"Embedding L2 norms are not approximately 1: {embedding_norms.tolist()}"

        returned_embedding, logits = model(x)
        assert torch.equal(returned_embedding, embedding)
        assert tuple(logits.shape) == (BATCH_SIZE, NUM_CLASSES)
        assert_finite("classification logits", logits)

    print("S1.4 forward-pass validation passed.")
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Feature shapes: {EXPECTED_FEATURE_SHAPES}")
    print(f"  Pooled shape: {(BATCH_SIZE, 256, 1, 1)}")
    print(f"  Embedding shape: {(BATCH_SIZE, EMBEDDING_DIM)}")
    print(f"  Logits shape: {(BATCH_SIZE, NUM_CLASSES)}")
    print("  Embedding L2 norm: approximately 1.0")
    print(f"  Trainable parameters: {EXPECTED_PARAMETER_COUNT}")
    print("  Numerical stability: finite outputs")


if __name__ == "__main__":
    main()
