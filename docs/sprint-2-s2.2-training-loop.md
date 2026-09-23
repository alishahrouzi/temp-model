# S2.2 — Training Loop

## Objective

Implement the executable training and validation loop using the S2.1 configuration and the existing 422,788-parameter custom CNN.

## Implementation

The training entry point is:

`scripts/train.py`

The loop:

1. loads `configs/training.yaml`;
2. fixes random seeds and deterministic settings;
3. resolves CUDA/CPU automatically;
4. loads the existing train and validation manifests;
5. applies S2.1 augmentation only to the training split;
6. keeps validation preprocessing deterministic;
7. constructs the existing `CustomCNN`;
8. applies weighted CrossEntropyLoss;
9. trains with AdamW;
10. clips gradients to the configured maximum norm;
11. evaluates validation loss and accuracy after every epoch;
12. updates ReduceLROnPlateau from validation loss;
13. records epoch-level metrics and learning rate;
14. writes training history to the configured JSON output.

## Training/Evaluation Boundary

Training uses:

- train split: 4,913 images;
- mild random augmentation;
- shuffled batches;
- gradient updates.

Validation uses:

- validation split: 614 images;
- deterministic preprocessing;
- no gradient updates;
- no augmentation.

The test split is intentionally not touched by S2.2. It remains reserved for later evaluation.

## Classification Signal

The model returns:

- normalized 128-dimensional embedding;
- four-class logits.

S2.2 uses only the logits for the supervised classification loss. The embedding remains available for the later retrieval pipeline.

The class weights come from the S2.1 configuration and were derived from the training split only.

## Monitoring

Each completed epoch records:

- train loss;
- train accuracy;
- validation loss;
- validation accuracy;
- learning rate.

The history is written to:

`results/training/training_history.json`

This file contains metrics only. It is not a model checkpoint.

## Checkpoint Boundary

S2.2 deliberately does **not**:

- save model weights;
- resume training from a checkpoint;
- select the best epoch;
- export an inference model.

Checkpoint creation is S2.3 and best-checkpoint selection is S2.6.

## Runtime Command

From the repository root:

```bash
python scripts/train.py --dataset-root "<path-to-dataset-root>"
```

Example for the current local dataset layout:

```bash
python scripts/train.py --dataset-root "E:\Privat File\Projects\Zargar Interview\dataset\jewelry-design-dataset\dataset"
```

The training run is intentionally performed locally because the GitHub integration can modify repository source/configuration but does not execute the user's Windows/GPU environment.

## Expected Baseline Environment

The configuration targets the project's known local constraint of a 4 GB CUDA GPU:

- batch size: 32;
- image size: 224x224;
- compact custom CNN;
- 30 configured epochs.

If CUDA is unavailable, the loop falls back to CPU.

## Scope Boundary

S2.2 completes the executable training/validation mechanism.

It does not make any model-quality claim until an actual local training run has been executed and its history reviewed.

## Next Task

**S2.3 — Checkpointing**

The next task should add controlled checkpoint creation without changing the training loop's core behavior.
