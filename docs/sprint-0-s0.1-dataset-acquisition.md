# S0.1 — Dataset Acquisition

## Objective

Register the specified project dataset for the Temp Model without
unnecessarily downloading or duplicating an existing local copy.

## Dataset

- Dataset ID: `sidd707/jewelry-design-dataset`
- Current environment: existing local copy
- Local repository root:

  `E:/Privat File/Projects/Zargar Interview/dataset/jewelry-design-dataset`

The actual image dataset is located under:

`dataset/`

and contains the class directories plus `dataset_labels.csv`.

## Implementation

The acquisition script is:

`scripts/acquire_dataset.py`

The script supports two acquisition modes.

### 1. Existing local dataset

Use the existing local copy:

```cmd
python scripts/acquire_dataset.py --source local --input-dir "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset"
```

The script recognizes both:

- the repository root containing `dataset/dataset_labels.csv`
- the direct dataset directory containing `dataset_labels.csv`

Local mode does **not** copy the image files and does not require the dataset
to have been created with Hugging Face `save_to_disk()`.

For the current dataset, the script registers:

- dataset source
- dataset root
- labels file
- number of CSV records
- class directories
- resolvable image paths
- missing image-path count and paths

### 2. Hugging Face acquisition

For a clean environment without a local copy:

```cmd
python scripts/acquire_dataset.py --source huggingface
```

This downloads the configured dataset and persists it under the configured
raw-data directory.

## Metadata

Local mode writes `acquisition_metadata.json` under the configured output
directory. It does not modify the source dataset.

The metadata is for acquisition traceability and reproducibility. The local
path-resolution check is intentionally limited to verifying that CSV image
paths can be resolved; it is not a complete dataset QA process.

## Current local dataset observation

During S0.1 validation of the supplied local copy:

- CSV records: 6156
- Resolvable image paths: 6140
- Missing image paths: 16

The 16 missing paths are recorded rather than deleted or repaired at this
stage. Their cause belongs to S0.2 dataset inspection.

The CSV descriptions are not used as class labels. Category information for
the model dataset is based on the class-directory structure
(`bracelet`, `earring_best`, `necklace`, `ring_best`). The text
descriptions remain metadata and are not model inputs in this project.

## Scope Boundary

This task does not perform:

- corrupted-image detection
- full dataset quality analysis
- duplicate detection
- final class-distribution analysis
- train/validation/test splitting
- image preprocessing or augmentation
- model training

Those activities belong to subsequent Sprint 0 and model-development tasks.

## Git Policy

Raw dataset files are intentionally excluded by `.gitignore` and must not
be committed.

Machine-specific absolute dataset paths are supplied at runtime and are not
stored in shared configuration.
