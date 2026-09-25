# S3.2 — Dataset Embedding Generation

## Objective
Generate the normalized 128-dimensional retrieval embedding for every image in a selected dataset split using the best checkpoint selected in S2.6.

## Implementation
The `scripts/generate_embeddings.py` workflow loads a Sprint 0 split manifest, loads the S2.6 best checkpoint, applies deterministic preprocessing, runs `CustomCNN.forward_retrieval()`, generates one normalized 128-d embedding per image, validates the output, and writes runtime artifacts to the ignored `embeddings/` directory.

## Default Policy
The script defaults to the `test` split for compact runtime validation. `train`, `validation`, and `test` can all be generated explicitly.

Example:
```bash
python scripts/generate_embeddings.py --dataset-root "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset\dataset" --split test
```

## Output
```text
embeddings/
├── test.npy
└── test_generation.json
```
These are runtime artifacts and are intentionally ignored by Git. The final persistent embedding representation and metadata contract are deferred to S3.3.

## Validation
- sample count equals the manifest record count;
- embedding shape is `[N, 128]`;
- all values are finite;
- every embedding has L2 norm approximately equal to 1;
- the selected checkpoint is loaded before inference.

S3.2 does not calculate similarity, rank images, implement Top-K retrieval, or perform retrieval evaluation.