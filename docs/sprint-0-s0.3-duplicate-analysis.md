# S0.3 — Duplicate Analysis

## Objective

Detect exact and visually near-duplicate images before the dataset split. Duplicate analysis is required because placing the same or near-identical image in different splits can leak information and make retrieval/classification evaluation misleading.

## Implementation

Script:

`scripts/analyze_duplicates.py`

Run:

```cmd
python scripts/analyze_duplicates.py --input-dir "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset"
```

The script performs two levels of analysis:

1. **Exact duplicates** using SHA-256 of complete file bytes.
2. **Perceptual duplicates** using a compact 64-bit pHash and Hamming-distance thresholds.

Default perceptual thresholds:

- 4
- 6
- 8
- 10
- 12

The thresholds are intentionally evaluated together. S0.3 does **not** select a final deduplication threshold automatically.

## Why both methods?

An exact hash detects byte-for-byte copies only. The same visual image can exist as different files after format conversion, metadata changes, resizing, or mild image processing. A perceptual hash provides a second signal for these visually similar cases.

The pHash implementation is self-contained using Pillow and NumPy; no pretrained vision model is used.

## Outputs

Generated files are written under:

- `results/duplicate-analysis/duplicate_analysis.json`
- `results/duplicate-analysis/duplicate_analysis.md`

The JSON report contains:

- analyzed image records
- SHA-256 hashes
- pHash values
- exact duplicate groups
- perceptual duplicate pairs for every tested threshold
- perceptual groups for every tested threshold
- same-class vs cross-class pair counts
- unreadable files encountered during this task
- configuration and scope metadata

The Markdown report is the human-readable summary.

## Important interpretation rule

A perceptual group is a candidate near-duplicate group, not proof that two images depict the same physical jewelry product. A low Hamming distance can still occur for visually similar but distinct products.

Therefore, S0.4 should use S0.3 evidence to define the split strategy. Duplicate/near-duplicate groups should remain within a single split when appropriate, while ambiguous cases should be reviewed rather than blindly deleted.

## Scope Boundary

S0.3 is read-only:

- no image deletion
- no image renaming
- no image movement
- no source-dataset modification
- no final train/validation/test split
- no final deduplication threshold selection
- no model training

## Dataset Integrity

The source dataset remains the source of truth. The analysis reads the existing local dataset and stores only derived reports under the ignored `results/` directory.
