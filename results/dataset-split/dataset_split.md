# S0.4 — Dataset Split Report

## Objective
Create train/validation/test manifests while preventing exact and S0.3 near-duplicate leakage across splits.

## Split Configuration
- Train: 80% target
- Validation: 10% target
- Test: 10% target
- pHash constraint threshold: <= 4

## Dataset
- Source images: 6141
- Source dataset files were not modified.
- CSV manifests reference the original relative image paths.

## Results

| Split | Images | Actual ratio | Target ratio |
|---|---:|---:|---:|
| train | 4913 | 0.8000 | 0.80 |
| validation | 614 | 0.1000 | 0.10 |
| test | 614 | 0.1000 | 0.10 |

### Class Distribution

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| bracelet | 706 | 88 | 88 |
| earring_best | 2630 | 329 | 329 |
| necklace | 1390 | 174 | 174 |
| ring_best | 187 | 23 | 23 |

## Leakage Control
- Exact duplicate groups constrained: 23
- pHash groups constrained: 195
- Combined multi-image constraint groups: 195
- Exact duplicates never cross splits.
- pHash candidate groups at the selected threshold never cross splits.
- Cross-class perceptual groups remain intact; they are not relabeled.

## Important Interpretation
The pHash threshold is a split-leakage control derived from S0.3, not a claim that every perceptual group represents the same physical jewelry product. The split intentionally prefers conservative leakage control over perfectly exact 80/10/10 counts.

## Output Files
- train.csv
- validation.csv
- test.csv
- dataset_split.json
- dataset_split.md

## Scope Boundary
- No source image deletion.
- No source image copying or moving.
- No image rewriting or augmentation.
- No model training.
- No use of CSV descriptions as model labels; directory names are the class source.