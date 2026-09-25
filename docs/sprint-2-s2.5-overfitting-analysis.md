# S2.5 — Overfitting Analysis

## Objective

Analyze the completed S2.2/S2.3 training history for evidence of overfitting or late generalization degradation.

This task is **descriptive**. It does not select a checkpoint, modify training configuration, retrain the model, or use the test split. Checkpoint selection remains S2.6.

## Input

Source history:

`results/training/training_history.json`

Training run:

- Epochs: 30
- Train samples: 4,913
- Validation samples: 614
- Device: CPU
- Test split used: No

## Analysis Method

The analysis compares:

1. training loss and validation loss;
2. training accuracy and validation accuracy;
3. the epoch with minimum validation loss;
4. the epoch with maximum validation accuracy;
5. the behavior after the observed validation peak;
6. the final train/validation generalization gaps.

A late overfitting signal is recorded when, after the observed validation peak, training loss continues to decrease and training accuracy continues to increase while validation loss increases and validation accuracy decreases.

This is treated as evidence of **late generalization degradation**, not as proof that every validation fluctuation is caused by overfitting.

## Observed Results

The minimum validation loss and maximum validation accuracy both occur at **epoch 27**:

| Metric | Epoch | Value |
|---|---:|---:|
| Minimum validation loss | 27 | 0.53569 |
| Maximum validation accuracy | 27 | 84.53% |

At epoch 27:

- train loss: 0.27584
- train accuracy: 93.47%
- validation loss: 0.53569
- validation accuracy: 84.53%

At the final epoch (30):

- train loss: 0.27316
- train accuracy: 93.49%
- validation loss: 0.71650
- validation accuracy: 78.83%

## Late-Run Behavior

Between epochs 27 and 30:

- train loss decreased by approximately 0.00268;
- train accuracy increased by approximately 0.02 percentage points;
- validation loss increased by approximately 0.18081;
- validation accuracy decreased by approximately 5.70 percentage points.

This is a consistent train/validation divergence pattern: the model continues fitting the training data while validation performance deteriorates.

The final generalization gaps are:

- accuracy gap: approximately 14.66 percentage points;
- loss gap: approximately 0.44334.

## Interpretation

The 30-epoch history contains a clear **late generalization degradation signal** after epoch 27.

There is also substantial validation fluctuation earlier in training. Therefore, the analysis does not attribute those earlier oscillations to overfitting without additional evidence.

The important conclusion for the next task is that the training run contains multiple checkpoints with materially different validation behavior. This provides the evidence required for S2.6 to formally define and implement checkpoint selection.

## Scope Boundary

S2.5 does **not**:

- select the final/best checkpoint;
- evaluate the test split;
- change hyperparameters;
- retrain the model;
- evaluate retrieval quality;
- claim retrieval performance from classification accuracy.

Checkpoint selection is handled exclusively by **S2.6 — Select Best Checkpoint**.

## Reproducibility

Run from the repository root:

```bash
python scripts/analyze_overfitting.py
```

Optional paths:

```bash
python scripts/analyze_overfitting.py --history "<path-to-training_history.json>" --output "<path-to-analysis.json>"
```

Output:

`results/training/overfitting_analysis.json`
