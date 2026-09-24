# S2.3 — Checkpointing

## Objective

Add controlled checkpoint creation to the existing S2.2 training loop without changing the training policy or introducing best-checkpoint selection.

## Checkpoint Policy

After every completed epoch, the training script writes:

- `epoch_001.pt`, `epoch_002.pt`, ..., one checkpoint per completed epoch;
- `last.pt`, overwritten after each epoch and representing the latest completed epoch.

The default output directory is:

`checkpoints/s2.3/`

Checkpoint artifacts are runtime outputs and are intentionally ignored by Git.

## Checkpoint Contents

Each checkpoint contains:

- checkpoint format version;
- completed epoch number;
- model `state_dict`;
- optimizer `state_dict`;
- scheduler `state_dict`;
- epoch metrics;
- resolved training configuration;
- Python random state;
- NumPy random state;
- PyTorch random state;
- CUDA random states when CUDA is available.

This provides the state required for a later resume-capable workflow without coupling S2.3 to model selection.

## Relationship to Training History

`results/training/training_history.json` remains the human-readable epoch history.

The checkpoint stores the model and optimizer/scheduler states that JSON cannot represent. The two outputs therefore serve different purposes.

## Best Checkpoint Boundary

S2.3 does **not** declare any checkpoint as the final or best model.

Although each checkpoint contains validation metrics, selecting the checkpoint to use for later retrieval is intentionally deferred to:

**S2.6 — Select Best Checkpoint**

This keeps checkpoint creation separate from model-quality selection and avoids leaking a selection decision into the infrastructure task.

## Runtime

The existing command remains valid:

```bash
python scripts/train.py --dataset-root "<path-to-dataset-root>"
```

An optional custom checkpoint directory can be supplied:

```bash
python scripts/train.py --dataset-root "<path-to-dataset-root>" --checkpoint-dir "<checkpoint-directory>"
```

No changes are required to the existing S2.1 configuration.

## Scope

S2.3 adds only checkpoint persistence.

It does not:

- change the CNN architecture;
- change optimizer, scheduler, loss, augmentation, or dataset splits;
- evaluate the test split;
- select the best checkpoint;
- build the retrieval pipeline;
- export an inference model.

## Validation

S2.3 should be runtime-validated locally by confirming that:

1. the configured number of epochs completes;
2. one `epoch_XXX.pt` file is produced for every completed epoch;
3. `last.pt` corresponds to the final completed epoch;
4. checkpoints can be loaded with PyTorch;
5. checkpoint metadata matches the corresponding history row.

## Next Task

**S2.4 — Training Monitoring**
