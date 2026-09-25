"""Validate the S3.1 retrieval path bypasses the classification head.

S3.1 keeps the classification head for checkpoint/training compatibility, but
the retrieval path must produce embeddings without invoking that head.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.model import EMBEDDING_DIM, INPUT_CHANNELS, INPUT_IMAGE_SIZE, CustomCNN  # noqa: E402


BATCH_SIZE = 2


def main() -> None:
    torch.manual_seed(42)

    model = CustomCNN()
    model.eval()

    x = torch.randn(
        BATCH_SIZE,
        INPUT_CHANNELS,
        INPUT_IMAGE_SIZE,
        INPUT_IMAGE_SIZE,
    )

    with torch.no_grad():
        expected = model.encode(x)

        def fail_classification(_: torch.Tensor) -> torch.Tensor:
            raise AssertionError("Classification head was invoked by retrieval path.")

        model.classify = fail_classification  # type: ignore[method-assign]
        retrieval_embedding = model.forward_retrieval(x)

    assert tuple(retrieval_embedding.shape) == (BATCH_SIZE, EMBEDDING_DIM)
    assert torch.isfinite(retrieval_embedding).all()
    assert torch.allclose(
        retrieval_embedding,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )

    norms = torch.linalg.vector_norm(retrieval_embedding, ord=2, dim=1)
    assert torch.allclose(
        norms,
        torch.ones(BATCH_SIZE),
        atol=1e-5,
        rtol=1e-5,
    )

    assert hasattr(model, "classification_head")
    print("S3.1 retrieval-path validation passed.")
    print(f"  Retrieval embedding shape: {tuple(retrieval_embedding.shape)}")
    print("  Classification head invoked: no")
    print("  Embedding L2 norm: approximately 1.0")
    print("  Classification head retained: yes (training/checkpoint compatibility)")


if __name__ == "__main__":
    main()
