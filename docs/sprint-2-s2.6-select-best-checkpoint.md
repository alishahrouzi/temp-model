# S2.6 — Select Best Checkpoint

## Objective

Select one training checkpoint for downstream feature extraction and retrieval work using validation-only evidence.

S2.6 does not use the test split, retrain the model, or change training hyperparameters.

## Selection Policy

The selection rule is deterministic:

1. **Primary criterion:** minimum validation loss.
2. **Secondary criterion:** maximum validation accuracy.
3. **Final tie-breaker:** earliest epoch.

Validation loss is the primary criterion because the training configuration explicitly monitors `validation_loss`, and loss provides a continuous optimization signal. Validation accuracy remains a secondary supporting criterion.

## Current Training Run

The completed 30-epoch run has:

| Metric | Epoch | Value |
|---|---:|---:|
| Minimum validation loss | 27 | 0.53569 |
| Maximum validation accuracy | 27 | 84.53% |

Both criteria identify **epoch 27**, so no tie-breaking decision is required for this run.

Selected checkpoint source:

`checkpoints/s2.3/epoch_027.pt`

The selection script copies it to:

`results/training/best-checkpoint/best.pt`

and writes:

`results/training/best-checkpoint/selection.json`

## Scope

The selected checkpoint is intended for:

**Sprint 3 — Feature Extraction & Similarity**

The selection is based only on the train/validation run. The test split remains untouched and is reserved for later retrieval evaluation.

## Reproducibility

Run from the repository root:

```bash
python scripts/select_best_checkpoint.py
```

Optional paths:

```bash
python scripts/select_best_checkpoint.py --history "<path-to-training_history.json>" --checkpoint-dir "<path-to-s2.3-checkpoints>" --output-dir "<path-to-best-checkpoint-output>"
```

The script verifies that the selected epoch checkpoint exists before copying it.

## Important Boundary

The `best.pt` artifact is a copy of the selected training checkpoint; it is not a newly trained model.

S2.6 does **not**:

- evaluate the test split;
- calculate Top-1/Top-5/Top-10 retrieval;
- generate embeddings;
- calculate image similarity;
- modify model architecture;
- retrain the network.
