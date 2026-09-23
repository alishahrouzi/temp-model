# Temp Model

A lightweight, custom image-similarity proof of concept for jewelry images.

## Project Goal

Temp Model is an independent mini-project created to test a focused capability:

> Given one jewelry image, retrieve the most visually similar images from the same dataset and report a similarity score for each result.

The project uses the `sidd707/jewelry-design-dataset` dataset from Hugging Face and focuses on a small custom model rather than using a pretrained image model as the core similarity model.

## Scope

The project covers:

- Dataset validation and preparation
- A lightweight custom CNN
- Model training
- Image feature / embedding extraction
- Similarity calculation
- Top-1, Top-5, and Top-10 retrieval
- Evaluation and visual result inspection
- A reproducible local demo

The project does **not** aim to reproduce the architecture or production infrastructure of the frozen Gold Image Recognition Engine.

Out of scope for this mini-project:

- Production deployment
- API and full frontend
- Vector database
- General-purpose visual search
- Price, weight, purity, or product metadata recognition
- Object detection or segmentation
- Store-specific production data
- Full robustness testing across arbitrary real-world conditions

## Dataset

Source:

https://huggingface.co/datasets/sidd707/jewelry-design-dataset

The dataset contains four jewelry categories:

- bracelet
- earring_best
- necklace
- ring_best

The original dataset provides a `train` split. Train/validation/test subsets for this project are created during Sprint 0 while preserving class distribution as much as practical and preventing duplicate-group leakage.

## High-Level Pipeline

```text
Input Image
    |
    v
Preprocessing
    |
    v
Custom Lightweight CNN
    |
    v
128-d Normalized Embedding
    |
    v
Similarity Calculation
    |
    v
Ranking
    |
    v
Top-K Similar Images + Similarity Scores
```

## Project Status

Current phase: **Sprint 1 complete**

Completed:

- Sprint 0 — Dataset Validation & Preparation
- S1.1 — Custom CNN Model Design
- S1.2 — Custom CNN Implementation
- S1.3 — Classification Head
- S1.4 — Forward Pass Test
- S1.5 — Model Documentation

Next phase: **Sprint 2 — Model Training**

## Model Summary

The current custom CNN has:

- four convolutional blocks;
- channel progression `3 → 32 → 64 → 128 → 256`;
- adaptive global average pooling;
- 128-dimensional normalized retrieval embedding;
- separate four-class linear classification head;
- exactly **422,788 trainable parameters**;
- no pretrained backbone.

The retrieval interface is `CustomCNN.encode()`.

Detailed model documentation:

`docs/sprint-1-s1.5-model-documentation.md`

## Development Principles

- Keep the model small and focused.
- Prefer measurable results over unnecessary architecture.
- Separate training, feature extraction, retrieval, and evaluation concerns.
- Record assumptions and limitations explicitly.
- Do not interpret a similarity score as classification accuracy.

## Planned Deliverables

- Source code
- Trained custom model checkpoint
- Dataset preparation and split information
- Embedding generation pipeline
- Similarity retrieval pipeline
- Evaluation results
- Visual demo
- Technical documentation

## Python Environment

The project targets Python 3.12 and is intended to run locally during the MVP/PoC stage.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## License

This repository contains project code. Dataset usage remains subject to the dataset's own terms and license.
