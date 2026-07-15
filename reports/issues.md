# Backlog / Issues

GitHub CLI is installed, but the current GitHub authentication token is invalid.
Create these as GitHub issues after running `gh auth login`.

## Issue 1 - Set up project environment

Install dependencies in `.venv`, verify TensorFlow imports, and document the exact Python version used.

Acceptance criteria:
- `.venv` exists locally.
- `pip install -r requirements.txt` completes.
- `python -c "import tensorflow as tf; print(tf.__version__)"` works.

## Issue 2 - California Housing regression pipeline

Run phases 1 to 3: clean split/scaling, baseline regression model, and TensorBoard comparison.

Acceptance criteria:
- `scripts/phase1_pipeline_california.py` prints expected train/val/test shapes.
- `scripts/phase2_baseline_regression.py` reports final validation MAE and test MAE.
- `scripts/phase3_tensorboard_california.py` writes logs under `logs/fit`.

## Issue 3 - Pima Diabetes binary classification

Run phases 4 to 6: baseline, regularisation, and keras-tuner search.

Acceptance criteria:
- Baseline accuracy is compared to the majority-class baseline.
- Regularised model reports accuracy, precision, and recall.
- Keras-tuner produces a best trial summary.

## Issue 4 - Wine Quality multiclass classification

Run phases 7 and 8: baseline 3-class classifier and BatchNorm comparison.

Acceptance criteria:
- Aggregated class distribution is printed.
- Four BatchNorm configurations are compared.
- TensorBoard logs are written under `logs/wine`.

## Issue 5 - Cross-dataset synthesis and custom pipeline

Run phase 9 and complete the README with real metrics.

Acceptance criteria:
- The comparison table has no placeholder values.
- The three synthesis questions are answered.
- The post-mortem section is filled.
