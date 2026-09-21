# S0.2 — Dataset Inspection

## Objective
Inspect the actual local dataset before split, preprocessing, augmentation, or model-development decisions.

## Implementation
Script: `scripts/inspect_dataset.py`

Run:
```cmd
python scripts/inspect_dataset.py --input-dir "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset"
```

## Checks
- CSV record count
- discovered image count
- CSV/file consistency
- missing and duplicate CSV paths
- unlisted image files
- class distribution from directories
- image formats/extensions
- readability/corruption
- width/height ranges
- unique and common resolutions
- per-image inspection records

## Data Interpretation
Directory names are the category source: `bracelet`, `earring_best`, `necklace`, `ring_best`.
The CSV `description` field is metadata only and is not a class label or model input.

## Output
- `results/dataset-inspection/dataset_inspection.json`
- `results/dataset-inspection/dataset_inspection.md`

## Scope Boundary
S0.2 is read-only. It does not delete, repair, rename, copy, split, resize, normalize, augment, deduplicate, or train.