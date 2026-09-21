# S0.1 — Dataset Acquisition

## Objective

Acquire the project dataset from its declared Hugging Face source and persist a
local copy in a reproducible way.

## Dataset

- Dataset ID: sidd707/jewelry-design-dataset
- Source: Hugging Face Datasets
- Local raw-data directory: data/raw/jewelry-design-dataset/

## Implementation

The acquisition is implemented in:

scripts/acquire_dataset.py

Run:

    python scripts/acquire_dataset.py

Optional overrides:

    python scripts/acquire_dataset.py --dataset-id sidd707/jewelry-design-dataset --output-dir data/raw/jewelry-design-dataset

The script:
1. Loads the dataset using the Hugging Face datasets library.
2. Persists all available dataset splits with save_to_disk.
3. Writes acquisition_metadata.json containing the source, acquisition timestamp,
   output location, and split sizes.

## Scope Boundary

This task does not perform:
- dataset quality analysis
- corrupted-image detection
- duplicate detection
- class-distribution analysis
- train/validation/test splitting
- image preprocessing or augmentation

Those activities belong to subsequent Sprint 0 tasks.

## Git Policy

Raw dataset files are intentionally excluded by .gitignore and must not be committed.
