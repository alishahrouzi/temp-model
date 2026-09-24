# S2.4 — Training Monitoring

## Objective

Provide reproducible monitoring artifacts for the completed training run without retraining the model or selecting a checkpoint.

## Inputs

The monitoring script reads:

`results/training/training_history.json`

The existing history contains, per epoch:

- train loss;
- train accuracy;
- validation loss;
- validation accuracy;
- learning rate.

## Implementation

Entry point:

`scripts/monitor_training.py`

The script produces:

- `training_summary.json`;
- `loss.png`;
- `accuracy.png`;
- `learning_rate.png`.

Default output directory:

`results/training/monitoring/`

## Summary

The generated summary records:

- completed epochs;
- device;
- train/validation sample counts;
- epoch with minimum validation loss;
- epoch with maximum validation accuracy;
- final epoch metrics and learning rate.

These values are descriptive monitoring results. They do not select the production/retrieval checkpoint.

## Visualization

### Loss

Train and validation loss are plotted together to expose convergence and divergence patterns.

### Accuracy

Train and validation accuracy are plotted together as supporting classification monitoring signals.

### Learning Rate

The configured scheduler's learning-rate changes are plotted on a logarithmic y-axis so reductions remain visible across the full run.

## Runtime

From repository root:

```bash
python scripts/monitor_training.py
```

Or with explicit paths:

```bash
python scripts/monitor_training.py --history "<path-to-training_history.json>" --output-dir "<output-directory>"
```

The script does not access the test split and does not modify checkpoints.

## Scope Boundary

S2.4 does **not**:

- retrain the model;
- change training hyperparameters;
- select the best checkpoint;
- evaluate retrieval quality;
- use the test split;
- perform overfitting analysis beyond making the monitoring curves available.

Interpretation of overfitting and checkpoint selection remain later tasks:

- **S2.5 — Overfitting Analysis**
- **S2.6 — Select Best Checkpoint**

## Validation

S2.4 should be runtime-validated by running the monitoring script against the existing 30-epoch history and confirming all four artifacts are generated and readable.
