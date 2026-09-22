"""Custom lightweight CNN for Temp Model retrieval.

The network follows the S1.1 architecture specification:
ConvBlock x4 -> global average pooling -> 128-d projection -> 4-class head.

No pretrained components are used.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CLASS_NAMES = (
    "bracelet",
    "earring_best",
    "necklace",
    "ring_best",
)
NUM_CLASSES = len(CLASS_NAMES)
EMBEDDING_DIM = 128
INPUT_CHANNELS = 3
INPUT_IMAGE_SIZE = 224


class ConvBlock(nn.Module):
    """Single convolutional feature stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ClassificationHead(nn.Module):
    """Four-class linear classification head over the retrieval embedding."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        class_names: tuple[str, ...] = CLASS_NAMES,
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names must contain at least one class.")
        self.class_names = class_names
        self.input_dim = input_dim
        self.num_classes = len(class_names)
        self.linear = nn.Linear(input_dim, self.num_classes)

    def forward(self, embedding: Tensor) -> Tensor:
        """Return raw class logits with shape [B, num_classes]."""
        if embedding.ndim != 2:
            raise ValueError(
                "Expected a 2D embedding [B, D], "
                f"got shape {tuple(embedding.shape)}."
            )
        if embedding.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected embedding dimension {self.input_dim}, "
                f"got {embedding.shape[1]}."
            )
        return self.linear(embedding)


class CustomCNN(nn.Module):
    """Compact custom CNN with a retrieval embedding and classifier."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()

        if num_classes != NUM_CLASSES:
            raise ValueError(
                f"Temp Model expects exactly {NUM_CLASSES} classes: {CLASS_NAMES}."
            )

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.feature_extractor = nn.Sequential(
            ConvBlock(INPUT_CHANNELS, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_projection = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classification_head = ClassificationHead(
            input_dim=embedding_dim,
            class_names=CLASS_NAMES,
        )

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D input [B, C, H, W], got shape {tuple(x.shape)}."
            )
        if x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels, got {x.shape[1]}."
            )
        if x.shape[2] != INPUT_IMAGE_SIZE or x.shape[3] != INPUT_IMAGE_SIZE:
            raise ValueError(
                "Expected spatial size "
                f"{INPUT_IMAGE_SIZE}x{INPUT_IMAGE_SIZE}, "
                f"got {x.shape[2]}x{x.shape[3]}."
            )

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the unnormalized 128-d embedding representation."""
        self._validate_input(x)
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.embedding_projection(x)

    def encode(self, x: Tensor) -> Tensor:
        """Return an L2-normalized embedding for retrieval."""
        embedding = self.forward_features(x)
        return F.normalize(embedding, p=2, dim=1)

    def classify(self, embedding: Tensor) -> Tensor:
        """Map a 128-d embedding to the four class logits."""
        return self.classification_head(embedding)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (normalized_embedding, classification_logits)."""
        embedding = self.encode(x)
        logits = self.classify(embedding)
        return embedding, logits


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
