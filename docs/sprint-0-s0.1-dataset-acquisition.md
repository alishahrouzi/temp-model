# S0.1 — Dataset Acquisition

## Objective

Make the specified project dataset available to the Temp Model without
unnecessarily downloading a second copy.

## Dataset

- Dataset ID: sidd707/jewelry-design-dataset
- Preferred source for the current environment: existing local copy
- Local dataset path:
  E:/Privat File/Projects/Zargar Interview/dataset/jewelry-design-dataset

## Implementation

The acquisition script is:

scripts/acquire_dataset.py

The script supports two acquisition modes.

### 1. Existing local dataset

Use this when the dataset has already been downloaded and saved with the
Hugging Face datasets library:

    python scripts/acquire_dataset.py --source local --input-dir "E:/Privat File/Projects/Zargar Interview/dataset/jewelry-design-dataset"

This mode loads the existing dataset with load_from_disk and does not
download or duplicate the dataset.

### 2. Hugging Face acquisition

For a clean environment without a local copy:

    python scripts/acquire_dataset.py --source huggingface

This downloads the configured dataset and persists it under the configured
raw-data directory.

## Metadata

acquisition_metadata.json is generated from the dataset actually loaded by
the script. It records:

- dataset ID
- acquisition source
- acquisition timestamp
- dataset location
- available split sizes

The metadata is for reproducibility and traceability. It is not a dataset QA
report.

## Scope Boundary

This task does not perform:

- dataset quality analysis
- corrupted-image detection
- duplicate detection
- class-distribution analysis beyond reporting existing split sizes
- train/validation/test splitting
- image preprocessing or augmentation

Those activities belong to subsequent Sprint 0 tasks.

## Important Assumption

The current local mode expects the supplied directory to be a Hugging Face
dataset directory created by save_to_disk. If the local folder instead
contains ordinary image files/folders, S0.1 must be adapted to that structure
before execution; no transformation is performed automatically.

## Git Policy

Raw dataset files are intentionally excluded by .gitignore and must not be
committed.
