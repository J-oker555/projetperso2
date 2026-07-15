# Jour 2 - PMC sur donnees structurees

Ce depot suit le document `partie2_pmc_donnees_structurees.pdf`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Architecture

```text
.
├── data/                    # CSV locaux optionnels
├── notebooks/               # Explorations ad hoc
├── reports/
│   └── issues.md            # Backlog local à convertir en GitHub issues
├── scripts/                 # Scripts exécutables des 9 phases
├── requirements.txt
└── README.md
```

## Scripts

- `scripts/phase1_pipeline_california.py` : split et normalisation propre de California Housing.
- `scripts/phase2_baseline_regression.py` : baseline PMC de regression.
- `scripts/phase3_tensorboard_california.py` : comparaison TensorBoard avec et sans normalisation.
- `scripts/phase4_pima_baseline.py` : baseline Pima Diabetes.
- `scripts/phase5_pima_regularisation.py` : L2, Dropout et Early Stopping sur Pima.
- `scripts/phase6_pima_kerastuner.py` : RandomSearch keras-tuner sur Pima.
- `scripts/phase7_wine_baseline.py` : baseline Wine Quality en 3 classes.
- `scripts/phase8_wine_batchnorm.py` : comparaison BatchNorm sur Wine.
- `scripts/phase9_custom_pipeline.py` : pipeline personnel sur Breast Cancer.

TensorBoard :

```bash
tensorboard --logdir logs
```

## Hypotheses avant tableau comparatif

1. Pima devrait profiter le plus de la regularisation, car le dataset est petit et legerement desequilibre.
2. California Housing devrait moins profiter des regularisations de classification, car le signal de regression est surtout controle par la normalisation et la capacite du modele.
3. Le desequilibre de classes devrait expliquer une grande partie de l'ecart entre Pima et Wine Quality.

## Tableau comparatif a remplir apres execution

| Dataset | Tache | val_accuracy / val_MAE | val_loss finale | Epochs reels | Meilleur lr | Meilleure taille couche | Regularisation |
|---|---|---:|---:|---:|---:|---:|---|
| California Housing | Regression | A remplir | A remplir | A remplir | N/A | 64/32 | Aucune baseline |
| Pima Diabetes | Classification binaire | A remplir | A remplir | A remplir | A remplir | A remplir | L2 + Dropout + Early Stopping |
| Wine Quality | Multiclassification | A remplir | A remplir | A remplir | N/A | 64/32 | L2 + BatchNorm |

## Questions de synthese

- Hyperparametre le plus impactant : a remplir avec les resultats des runs.
- Plus grand ecart train/validation : a remplir apres lecture TensorBoard.
- Early stopping le plus tot : a remplir apres comparaison des epochs reels.

## Post-mortem

1. Surprise du jour : a remplir.
2. Ce que je referais differemment : a remplir.
3. Ce que je n'ai pas eu le temps d'explorer : a remplir.
