# Compliance Review

This file maps the assignment requirements from `partie2_pmc_donnees_structurees.pdf`
to the repository implementation.

## Required deliverables

| Requirement | Status | Evidence |
|---|---|---|
| Git repository with frequent task tracking | Done | Git history plus GitHub issues #1-#10, all closed with validation comments. |
| Python environment and dependencies | Done | `.venv` created locally, `requirements.txt`, validated imports in README. |
| Three independent dataset scripts | Done | `scripts/california_housing_project.py`, `scripts/pima_diabetes_project.py`, `scripts/wine_quality_project.py`. |
| Phase-by-phase implementation | Done | `scripts/phase1_...py` through `scripts/phase9_...py`. |
| TensorBoard logs | Done locally | Generated under `logs/fit`, `logs/wine`, and `logs/custom`; intentionally ignored by Git because they are generated artifacts. |
| California Housing regression | Done | Phases 1-3, metrics documented in README. |
| Pima Diabetes binary classification | Done | Phases 4-6, baseline, regularization, tuner metrics documented in README. |
| Wine Quality multiclass classification | Done | Phases 7-8, class aggregation and BatchNorm comparison documented in README. |
| Cross-dataset comparison table | Done | README table filled with real run metrics. |
| Personal dataset pipeline | Done | `scripts/phase9_custom_pipeline.py` on Breast Cancer, Adam vs SGD documented. |
| Reflection and post-mortem | Done | README synthesis questions and post-mortem completed. |

## Validation commands used

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow as tf, pandas, sklearn, keras_tuner; print(tf.__version__, pandas.__version__, sklearn.__version__, keras_tuner.__version__)"
Get-ChildItem scripts -Filter *.py | ForEach-Object { .\.venv\Scripts\python.exe -m py_compile $_.FullName }
.\.venv\Scripts\python.exe scripts\phase1_pipeline_california.py
.\.venv\Scripts\python.exe scripts\phase2_baseline_regression.py
.\.venv\Scripts\python.exe scripts\phase3_tensorboard_california.py
.\.venv\Scripts\python.exe scripts\phase4_pima_baseline.py
.\.venv\Scripts\python.exe scripts\phase5_pima_regularisation.py
.\.venv\Scripts\python.exe scripts\phase6_pima_kerastuner.py
.\.venv\Scripts\python.exe scripts\phase7_wine_baseline.py
.\.venv\Scripts\python.exe scripts\phase8_wine_batchnorm.py
.\.venv\Scripts\python.exe scripts\phase9_custom_pipeline.py
```

## Notes

- `data/sklearn/`, `logs/`, and `kt_dir/` are ignored intentionally. They are caches or generated experiment artifacts.
- The PDF mentions committing often after phases. The GitHub issues preserve phase-level tracking; the final validation commit groups the run fixes and README results.
- The Wine target is treated as nominal 3-class classification, as justified in `scripts/phase7_wine_baseline.py`.
