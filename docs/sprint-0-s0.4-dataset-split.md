# S0.4 — Dataset Split

## Objective

Create deterministic train/validation/test manifests from the S0.2/S0.3 validated dataset while preventing duplicate leakage across splits.

## Source of Truth

S0.4 does not recompute duplicate evidence. It consumes:

- S0.3: results/duplicate-analysis/duplicate_analysis.json
- S0.2 dataset discovery/readability behavior
- directory names as class labels
- CSV description only as metadata

The source dataset is never modified.

## Split Policy

Target proportions:

- Train: 80%
- Validation: 10%
- Test: 10%

Because S0.3 found substantial perceptual similarity, the split uses the conservative observed pHash threshold **<= 4** as a leakage constraint.

This is deliberately **not** a claim that every pHash group is the same physical jewelry product. It is a conservative rule for evaluation integrity.

### Indivisible Groups

The following are kept entirely inside one split:

1. every exact duplicate group from S0.3;
2. every perceptual group at pHash Hamming distance <= 4;
3. any combined/transitive group formed by overlap between those constraints.

Cross-class perceptual groups are not relabeled. Their directory classes remain unchanged.

## Assignment Algorithm

S0.4 uses a deterministic greedy group-assignment algorithm.

For each candidate assignment it minimizes a weighted objective based on:

- deviation from the 80/10/10 image-count targets;
- deviation from per-class target counts.

Multiple deterministic candidates are evaluated from a fixed seed, and the lowest-objective candidate is selected.

This is group-aware stratification rather than a naive random image-level split.

## Leakage Validation

Before writing outputs, the script verifies:

- every readable discovered image is assigned exactly once;
- no exact duplicate group crosses splits;
- no selected pHash group crosses splits.

If these checks fail, S0.4 aborts instead of producing a potentially misleading split.

## Outputs

Generated under results/dataset-split:

- train.csv
- validation.csv
- test.csv
- dataset_split.json
- dataset_split.md

The CSV manifests contain:

- image_path
- class
- description

The image files themselves are not copied into split directories. This avoids unnecessary duplication and keeps the source dataset unchanged.

## Execution

From the repository root:

```cmd
python scripts/create_dataset_split.py --input-dir "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset"
```

To explicitly use another threshold already present in S0.3:

```cmd
python scripts/create_dataset_split.py --input-dir "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset" --phash-threshold 6
```

The default for the project is **4** because S0.3 showed that larger thresholds rapidly increase cross-class pairs and group connectivity.

## Scope Boundary

S0.4 does not:

- delete images;
- rename images;
- move or copy source images;
- augment or resize images;
- train a model;
- select a final pHash deduplication threshold;
- use descriptions as classification labels.

The pHash threshold is used here only as a conservative split-leakage constraint.
