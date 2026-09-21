# S0.5 — Dataset Preparation Pipeline

## Objective

Consume the leakage-aware manifests produced by S0.4 and expose a reusable PyTorch dataset/preprocessing pipeline without modifying the source dataset.

## Input

S0.5 consumes:

- results/dataset-split/train.csv
- results/dataset-split/validation.csv
- results/dataset-split/test.csv

Each manifest provides image_path, class, and description.

## Preparation Policy

For each image, the pipeline:

1. resolves the original relative path against the local dataset root;
2. opens the source image with Pillow;
3. converts it to RGB;
4. resizes it deterministically to 224 x 224;
5. converts it to a float tensor in [0, 1].

No source image is copied, moved, renamed, or rewritten.

## Deliberately Deferred Decisions

S0.5 does not introduce data augmentation or pretrained normalization statistics. Those choices depend on the custom CNN and training behavior and therefore remain decisions for Sprint 1/2.

This also prevents accidental coupling to a pretrained backbone, which is outside the Temp Model requirement.

## Validation

The preparation script validates all three manifests before exposing the PyTorch datasets:

- every referenced image exists;
- every referenced image is readable;
- no duplicate path occurs within a manifest;
- a real training sample can be transformed into the expected tensor format.

## Files

- src/temp_model/dataset.py — reusable PyTorch dataset and deterministic transform.
- scripts/prepare_dataset.py — local validation/reporting entry point.
- results/dataset-preparation/dataset_preparation.json — generated validation report.
- results/dataset-preparation/dataset_preparation.md — generated human-readable report.

## Execution

From the repository root:

    python scripts/prepare_dataset.py --input-dir "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset"

Optional image size override:

    python scripts/prepare_dataset.py --input-dir "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset" --image-size 256

The default 224 x 224 size is a pipeline default, not a final model architecture decision.

## Scope Boundary

S0.5 does not:

- train a model;
- augment images;
- calculate embeddings;
- calculate similarity;
- alter source images;
- create class labels from CSV descriptions.
