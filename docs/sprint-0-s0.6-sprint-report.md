# Sprint 0 — Dataset Validation & Preparation Report

## 1. Sprint Overview

**Project:** Temp Model  
**Sprint:** Sprint 0 — Dataset Validation & Preparation  
**Dataset:** `sidd707/jewelry-design-dataset`  
**Target:** Establish a validated, leakage-aware, reproducible dataset pipeline before custom CNN development.

Sprint 0 covered dataset acquisition, inspection, duplicate analysis, train/validation/test splitting, and deterministic PyTorch preparation. No model training or retrieval logic was introduced in this sprint.

The sprint is considered complete because all planned Sprint 0 tasks S0.1–S0.5 were implemented, executed, and validated, and the consolidated S0.6 report documents the resulting dataset state and decisions.

---

## 2. Sprint Scope

| Task | Description | Status |
|---|---|---|
| S0.1 | Dataset Acquisition | Complete |
| S0.2 | Dataset Inspection | Complete |
| S0.3 | Duplicate Analysis | Complete |
| S0.4 | Dataset Split | Complete |
| S0.5 | Dataset Preparation Pipeline | Complete |
| S0.6 | Sprint 0 Consolidated Report | Complete |

The sprint intentionally excludes:

- model architecture and training;
- embedding generation;
- similarity calculation;
- retrieval ranking;
- data augmentation;
- production API/frontend;
- vector database or FAISS infrastructure;
- modification or deletion of source images.

These activities belong to later sprints.

---

## 3. Planned vs Actual Execution

### Planned sequence

The approved Sprint 0 sequence was:

1. Acquire/register the dataset.
2. Inspect dataset integrity and composition.
3. Analyze exact and perceptual duplicates.
4. Create leakage-aware train/validation/test manifests.
5. Build a reusable deterministic preprocessing pipeline.
6. Consolidate the sprint results.

### Actual execution

The implementation followed the planned dependency order. S0.3 was completed before S0.4 so duplicate evidence could be used directly by the split algorithm. S0.5 then consumed the resulting manifests rather than independently rebuilding the split.

The Git history shows the main Sprint 0 implementation activity on **20–21 September 2026**. The final S0.5 validation report was committed on **21 September 2026 at 10:35 UTC**.

During S0.5, a direct-script import-path issue was discovered during real execution. The issue was fixed in a dedicated branch/PR and the script was rerun successfully. This did not require changing the dataset design or preprocessing policy.

---

## 4. Dataset Acquisition — S0.1

The project uses:

`sidd707/jewelry-design-dataset`

The current environment contains an existing local copy. S0.1 therefore registers and validates the local dataset instead of unnecessarily duplicating the image files.

The acquisition pipeline supports both:

- an existing local dataset;
- Hugging Face acquisition for a clean environment.

The current dataset uses the following category directories:

- `bracelet`
- `earring_best`
- `necklace`
- `ring_best`

The CSV file contains image paths and descriptions. The **directory name is the category source**. The CSV description is treated as metadata only and is not used as a class label or model input.

### Initial acquisition observations

| Metric | Result |
|---|---:|
| CSV records | 6,156 |
| Resolvable CSV image paths | 6,140 |
| Missing CSV image paths | 16 |

The missing references were recorded rather than silently deleted or repaired.

---

## 5. Dataset Inspection — S0.2

S0.2 performed a read-only inspection of the actual image files before any split or preprocessing operation.

### Dataset inventory

| Metric | Result |
|---|---:|
| CSV records | 6,156 |
| Discovered image files | 6,141 |
| Resolvable CSV images | 6,140 |
| Missing CSV image references | 16 |
| Duplicate CSV paths | 0 |
| Unlisted image files | 1 |
| Readable images | 6,141 |
| Unreadable images | 0 |

The difference between the CSV and discovered files is therefore explicitly preserved in the reports: one discovered image is not listed in the CSV, while 16 CSV references cannot be resolved.

### Class distribution

| Class | Images | Approx. share |
|---|---:|---:|
| bracelet | 882 | 14.4% |
| earring_best | 3,288 | 53.5% |
| necklace | 1,738 | 28.3% |
| ring_best | 233 | 3.8% |
| **Total** | **6,141** | **100%** |

The dataset is therefore class-imbalanced, with `earring_best` being the largest class and `ring_best` the smallest.

### Image properties

| Property | Result |
|---|---|
| JPEG | 3,529 |
| PNG | 2,610 |
| GIF | 2 |
| Unique resolutions | 192 |
| Width range | 90–3,872 px |
| Height range | 90–3,872 px |
| Readability failures | 0 |

No source images were modified during inspection.

---

## 6. Duplicate Analysis — S0.3

S0.3 used two independent signals:

1. **SHA-256** for byte-for-byte exact duplicates.
2. A self-contained **64-bit perceptual hash (pHash)** with Hamming-distance thresholds.

The perceptual thresholds evaluated were 4, 6, 8, 10, and 12.

### Exact duplicates

- Exact duplicate groups: **23**
- Exact duplicate members: **47**

These groups are strong evidence of repeated files and therefore must not be allowed to cross evaluation splits.

### Perceptual similarity

| pHash threshold | Similar pairs | Cross-class pairs |
|---:|---:|---:|
| ≤ 4 | 768 | 6 |
| ≤ 6 | 2,179 | 40 |
| ≤ 8 | 5,899 | 153 |
| ≤ 10 | 14,175 | 557 |
| ≤ 12 | 32,457 | 2,019 |

The results show that increasing the threshold rapidly increases the number of candidate near-duplicate relationships and cross-class relationships.

For Sprint 0, threshold **≤ 4** was selected as a conservative **split-leakage constraint**, not as a final deduplication threshold.

A pHash relationship is not treated as proof that two files represent the same physical jewelry product.

---

## 7. Leakage-Aware Dataset Split — S0.4

The final split target was:

- Train: 80%
- Validation: 10%
- Test: 10%

The split algorithm operates on indivisible constraint groups rather than independently assigning individual images.

### Leakage constraints

The following were kept inside a single split:

1. every exact duplicate group;
2. every S0.3 perceptual group at pHash Hamming distance ≤ 4;
3. transitive groups formed when exact and perceptual constraints overlap.

Cross-class perceptual groups were not relabeled.

### Assignment method

S0.4 uses a deterministic greedy group-assignment strategy with:

- seed: **42**;
- candidate attempts: **256**;
- objective based on deviation from total split targets and per-class targets.

### Final split

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| bracelet | 706 | 88 | 88 | 882 |
| earring_best | 2,630 | 329 | 329 | 3,288 |
| necklace | 1,390 | 174 | 174 | 1,738 |
| ring_best | 187 | 23 | 23 | 233 |
| **Total** | **4,913** | **614** | **614** | **6,141** |

Actual ratios:

- Train: **80.0033%**
- Validation: **9.9984%**
- Test: **9.9984%**

### Leakage validation

The generated split passed the following checks:

- all 6,141 readable images assigned exactly once;
- no constrained duplicate/perceptual group crosses splits;
- no unreadable image was included;
- manifests contain unique image paths within each split.

The resulting manifests reference the original source files. Image files are not copied into separate split directories.

---

## 8. Dataset Preparation Pipeline — S0.5

S0.5 introduced the reusable PyTorch dataset layer and deterministic preprocessing.

### Preparation flow

`manifest → source image → RGB → resize 224×224 → float tensor [0,1]`

The pipeline:

1. resolves the manifest's relative image path;
2. opens the original image with Pillow;
3. converts it to RGB;
4. resizes it to 224×224;
5. converts it to a `torch.float32` tensor in [0, 1].

### Validation result

| Split | Images | Missing | Unreadable | Duplicate manifest paths |
|---|---:|---:|---:|---:|
| train | 4,913 | 0 | 0 | 0 |
| validation | 614 | 0 | 0 | 0 |
| test | 614 | 0 | 0 | 0 |

A real training sample was successfully transformed with:

- tensor shape: **[3, 224, 224]**
- dtype: **torch.float32**
- observed range: **[0.070588, 1.000000]**

### Deliberate design boundaries

S0.5 does not introduce:

- augmentation;
- pretrained normalization statistics;
- model training;
- embedding generation;
- similarity calculation.

The 224×224 size is a pipeline default, not a final model architecture decision.

Model-specific augmentation and normalization remain open decisions for Sprint 1/2.

---

## 9. Final Sprint 0 Dataset State

At the end of Sprint 0, the dataset pipeline provides:

`6,141 readable images → leakage-aware 80/10/10 manifests → deterministic RGB/224×224 tensors`

The final usable image counts are:

- **4,913 train**
- **614 validation**
- **614 test**

The source dataset remains unchanged.

The project now has a reproducible dataset foundation suitable for custom CNN development.

---

## 10. Important Technical Decisions

### 10.1 Directory labels over CSV descriptions

The dataset descriptions contain natural-language metadata and can describe an item differently from its directory category. Therefore:

- directory name = category label;
- description = metadata;
- description is excluded from model input.

This prevents accidental label contamination.

### 10.2 Conservative leakage control

The pHash threshold of 4 is used to protect evaluation integrity. It is not treated as a semantic identity detector and does not trigger automatic deletion.

### 10.3 No source-data rewriting

The original dataset is preserved. All split and preprocessing information is represented through manifests and runtime transformations.

### 10.4 No pretrained vision backbone

Sprint 0 introduces no pretrained model or pretrained normalization statistics. This preserves the requirement that the later retrieval model be custom and keeps model-specific decisions in the appropriate sprint.

### 10.5 Retrieval remains the primary objective

Classification-related category information is retained because it is useful for later analysis, but Sprint 0 does not treat classification accuracy as the project's primary objective. The eventual success criteria are retrieval-oriented.

---

## 11. Repository and Git Traceability

Sprint 0 was implemented through isolated branches and pull requests rather than direct uncontrolled changes to the sprint branch.

Relevant milestones include:

- repository setup and project structure;
- S0.3 duplicate-analysis implementation;
- S0.4 leakage-aware split implementation;
- S0.5 dataset-preparation implementation;
- S0.5 import-path correction after real execution;
- S0.5 validation-result documentation;
- this S0.6 consolidated report.

The latest Sprint 0 branch before this report is:

`sprint/0-dataset-validation`

Latest pre-report commit:

`5e27fd78db934a6681bdc2a801481326aa85937d`

Commit message:

`doc(s0.5): report dataset preparation analysis`

The S0.5 execution issue was isolated to script import-path handling and was corrected in a dedicated fix PR before the successful validation run.

---

## 12. Acceptance Criteria

| Criterion | Result |
|---|---|
| Dataset registered and traceable | Pass |
| Dataset integrity inspected | Pass |
| All discovered images readable | Pass |
| Exact duplicate analysis completed | Pass |
| Perceptual duplicate analysis completed | Pass |
| Leakage-aware split created | Pass |
| All images assigned exactly once | Pass |
| Constraint groups kept within one split | Pass |
| Train/validation/test manifests generated | Pass |
| Reusable PyTorch dataset implemented | Pass |
| Deterministic preprocessing validated | Pass |
| Source images preserved | Pass |
| Model training performed in Sprint 0 | Intentionally deferred |

---

## 13. Known Limitations and Deferred Decisions

Sprint 0 establishes dataset integrity; it does not establish model quality.

The following remain open for later sprints:

- custom CNN architecture;
- embedding dimensionality;
- training loss and classification head design;
- augmentation policy;
- model-specific normalization;
- optimizer and learning-rate configuration;
- checkpoint selection;
- embedding similarity metric;
- retrieval ranking;
- Top-1/Top-5/Top-10 retrieval evaluation;
- self-retrieval vs excluded-query evaluation;
- similarity-score interpretation;
- visual result inspection.

The dataset itself is also not a production jewelry catalog. The eventual model should therefore be evaluated against the actual intended retrieval conditions before any production conclusions are made.

---

## 14. Sprint Conclusion

Sprint 0 successfully completed its objective: the project now has a validated and reproducible dataset foundation for the custom retrieval model.

The final dataset state is **6,141 readable images across four categories**, split into **4,913 train / 614 validation / 614 test** with duplicate-aware leakage constraints. The preparation layer provides deterministic RGB 224×224 tensors without modifying the source images.

No model-quality claim is made from Sprint 0. The next stage is **Sprint 1 — Custom CNN Development**, where the custom feature-learning architecture will be designed and implemented against this validated dataset foundation.
