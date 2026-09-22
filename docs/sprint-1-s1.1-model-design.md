# S1.1 — Custom CNN Model Design

## 1. Objective

Define the exact architecture of the lightweight custom CNN that will be implemented in S1.2 and later trained in Sprint 2.

The model is designed for the Temp Model retrieval objective:

image → learned visual representation → embedding vector → similarity → ranked results

A classification head is included as a supervised training signal, but the learned feature representation is the primary artifact used by the later retrieval pipeline.

S1.1 is a design task. No training, checkpoint, embedding generation, or retrieval evaluation is performed here.

## 2. Design Constraints

- custom CNN implemented with PyTorch;
- no pretrained backbone;
- no transfer learning;
- compact enough for local development on limited GPU memory;
- suitable for 224×224 RGB input from S0.5;
- produce a dedicated embedding vector for retrieval;
- retain a classification head for supervised feature learning;
- keep feature extraction and classification logically separable;
- avoid unnecessary architectural complexity;
- remain easy to inspect and modify.

## 3. Proposed Architecture

Four convolutional feature stages are followed by global average pooling, a compact projection layer, and a four-class classification head.

High-level structure:

Input → ConvBlock1 → ConvBlock2 → ConvBlock3 → ConvBlock4 → Global Average Pooling → Embedding Projection → Classification Head

| Stage | Operation | Output Shape |
|---|---|---|
| Input | RGB image | [B, 3, 224, 224] |
| Block 1 | Conv 3×3, 32 → BatchNorm → ReLU → MaxPool 2×2 | [B, 32, 112, 112] |
| Block 2 | Conv 3×3, 64 → BatchNorm → ReLU → MaxPool 2×2 | [B, 64, 56, 56] |
| Block 3 | Conv 3×3, 128 → BatchNorm → ReLU → MaxPool 2×2 | [B, 128, 28, 28] |
| Block 4 | Conv 3×3, 256 → BatchNorm → ReLU → MaxPool 2×2 | [B, 256, 14, 14] |
| Pool | Adaptive Global Average Pooling | [B, 256, 1, 1] |
| Flatten | Flatten | [B, 256] |
| Projection | Linear 256 → 128 → ReLU | [B, 128] |
| Embedding | L2 normalization at feature-extraction time | [B, 128] |
| Classifier | Linear 128 → 4 | [B, 4] |

B denotes batch size.

The four classes are bracelet, earring_best, necklace, and ring_best.

## 4. Convolutional Blocks

Each stage uses:

Conv2d → BatchNorm2d → ReLU → MaxPool2d

Channel progression:

3 → 32 → 64 → 128 → 256

Each convolution uses kernel 3×3, stride 1, padding 1. Each stage ends with 2×2 max pooling.

The architecture deliberately uses one convolution per block. A deeper two-convolution-per-block design was not selected because this project currently prioritizes a compact and explainable baseline.

## 5. Embedding Design

After the final convolution:

[B, 256, 14, 14] → AdaptiveAvgPool2d(1) → [B, 256]

The representation is projected from 256 to 128 dimensions.

The 128-dimensional vector is the model's retrieval embedding. For retrieval, it will be L2-normalized before similarity calculation.

This creates the intended separation:

- convolutional layers learn visual features;
- projection produces the compact retrieval representation;
- classifier uses that representation for supervised training;
- retrieval later uses the representation without the classifier.

## 6. Classification Head

The classification head is:

128-dimensional embedding → Linear(128, 4)

The four logits correspond to the four dataset categories.

The classifier provides supervised learning signal during Sprint 2. It is not the retrieval output.

## 7. Parameter and Resource Target

Approximate trainable parameter count:

- convolutional layers: ~388K;
- projection layer: ~33K;
- classifier: ~0.5K;
- normalization parameters: ~1K;
- total: approximately 423K parameters.

The exact count will be confirmed by S1.2 implementation.

## 8. Why This Architecture

### Custom and transparent

Every feature-extraction layer is explicitly defined in the project. There is no pretrained vision backbone.

### Compact

The 32 → 64 → 128 → 256 progression provides increasing feature capacity while remaining small enough for local experimentation.

### Appropriate spatial reduction

Four 2×2 pooling operations reduce 224 → 112 → 56 → 28 → 14 before global pooling.

### Global Average Pooling

Global pooling avoids a large fully connected layer over the complete spatial feature map. A flattened 256×14×14 map would contain 50,176 values, while global pooling reduces it to 256.

### Dedicated embedding

The explicit 128-dimensional projection creates a stable interface between model training and the future retrieval pipeline.

### Classification as auxiliary supervision

The four-class classifier gives the model a direct supervised objective during training without making classification the final product.

## 9. Normalization and Augmentation Boundary

S0.5 currently provides:

RGB → Resize(224,224) → Tensor [0,1]

S1.1 does not introduce pretrained ImageNet normalization.

Model/training normalization and augmentation remain training-pipeline decisions for S2.1, where they can be evaluated against the actual dataset.

The model expects a three-channel float tensor with spatial size 224×224.

## 10. Forward-Path Contract

S1.2 must expose two logically distinct paths.

Training path:

image → embedding → logits

Retrieval path:

image → embedding

The implementation should make the separation explicit, either through separate methods or a single forward API that returns both embedding and logits.

The exact Python API belongs to S1.2.

## 11. Expected Tensor Flow

For batch size B:

[B, 3, 224, 224]
↓
[B, 32, 112, 112]
↓
[B, 64, 56, 56]
↓
[B, 128, 28, 28]
↓
[B, 256, 14, 14]
↓
[B, 256, 1, 1]
↓
[B, 256]
↓
[B, 128]  ← retrieval embedding
↓
[B, 4]    ← classification logits

The 128-dimensional vector is the only representation intended to enter the future similarity pipeline.

## 12. Deferred Decisions

The following remain outside S1.1:

- optimizer;
- learning rate;
- batch size;
- number of epochs;
- loss weighting;
- augmentation strength;
- early stopping;
- checkpoint selection;
- similarity metric;
- retrieval Top-K configuration;
- excluded-query evaluation protocol.

These belong to Sprint 2 and Sprint 3/4.

## 13. Acceptance Criteria

| Criterion | Result |
|---|---|
| Custom CNN architecture defined | Pass |
| No pretrained backbone | Pass |
| Input/output tensor shapes defined | Pass |
| Four dataset classes represented | Pass |
| Dedicated retrieval embedding defined | Pass |
| Classification head defined | Pass |
| Feature/classification separation defined | Pass |
| Compact architecture target defined | Pass |
| Training configuration deferred appropriately | Pass |
| Retrieval metric deferred appropriately | Pass |
| S1.2 implementation-ready specification | Pass |

## 14. S1.1 Boundary

S1.1 is complete when the architecture can be implemented in S1.2 without an unresolved architectural decision.

No model code is intentionally added in S1.1. The next task, S1.2 — Custom CNN Implementation, will translate this specification into the PyTorch model and verify the actual parameter count and tensor shapes with a real forward pass.
