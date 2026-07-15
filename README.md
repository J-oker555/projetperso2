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
|-- data/                    # CSV locaux optionnels
|-- notebooks/               # Explorations ad hoc
|-- reports/
|   |-- issues.md            # Backlog local
|   `-- github-issues/       # Corps des issues GitHub
|-- scripts/                 # Scripts executables des 9 phases
|-- requirements.txt
`-- README.md
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

## Tableau comparatif apres execution

| Dataset | Tache | val_accuracy / val_MAE | val_loss finale | Epochs reels | Meilleur lr | Meilleure taille couche | Regularisation |
|---|---|---:|---:|---:|---:|---:|---|
| California Housing | Regression | val_MAE 0.3561 avec scaling TensorBoard | 0.2992 baseline phase 2 | 100 | 0.001 Adam | 64/32 | Aucune baseline |
| Pima Diabetes | Classification binaire | val_accuracy 0.8618 tuner | 0.5250 test tuner | 136 phase 5 | 0.01 | 64/8 | L2 0.001 + Dropout 0.1 + Early Stopping |
| Wine Quality | Multiclassification | val_accuracy 0.8398 | 0.5463 | 54 | 0.001 Adam | 64/32 | L2 + BatchNorm avant activation |

## Questions de synthese

- Hyperparametre le plus impactant : le learning rate sur Pima. Le tuner place les meilleurs essais a 0.01 ou 0.005, avec un meilleur `val_accuracy` a 0.8618. Attention : le test tombe a 0.7273, donc la validation reste bruitee sur ce petit dataset.
- Plus grand ecart train/validation : Wine Quality. La baseline monte a environ 0.97 train accuracy en fin de run, pendant que la validation plafonne autour de 0.82. C'est le signal d'overfitting le plus net.
- Early stopping le plus tot : Wine avec BatchNorm + couche extra, stop a 45 epochs. Le dataset est petit et tres desequilibre, donc la validation plafonne vite.

## Resultats complementaires

- Setup : TensorFlow 2.21.0, pandas 3.0.3, scikit-learn 1.9.0, keras-tuner 1.4.8.
- California phase 1 : shapes `(13209, 8)`, `(3303, 8)`, `(4128, 8)`. Le test garde une moyenne non nulle apres scaling, ce qui confirme que le scaler est fit uniquement sur le train.
- California phase 2 : `test_mae=0.3759`, soit environ 37 587 dollars d'erreur moyenne.
- California phase 3 : run normalise `val_mae=0.3561`, run non normalise `val_mae=0.5853`.
- Pima phase 4 : baseline majoritaire 0.6515, baseline PMC `val_accuracy=0.8374`, `test_accuracy=0.7468`.
- Pima phase 5 : modele regularise `test_accuracy=0.7338`, `precision=0.6275`, `recall=0.5926`, stop a 136 epochs.
- Pima phase 6 : meilleur tuner `units=64`, `second_units=8`, `l2_lambda=0.001`, `dropout=0.1`, `learning_rate=0.01`, `val_accuracy=0.8618`, `test_accuracy=0.7273`.
- Wine phase 7 : distribution agregee `low=63`, `medium=1319`, `high=217`, baseline `val_accuracy=0.8281`, `test_accuracy=0.8625`.
- Wine phase 8 : meilleure config `BatchNorm avant activation`, `val_accuracy=0.8398`, stop a 54 epochs. La config sans BN finit a `val_accuracy=0.8281` en 63 epochs.
- Pipeline personnel Breast Cancer : Adam `test_accuracy=0.9649` en 179 epochs, SGD `test_accuracy=0.9649` en 200 epochs. Adam converge plus vite. L'entree extreme hors distribution predit `[0.]`, donc une confiance forte sur une entree absurde.

## Post-mortem

1. Surprise du jour : le tuner Pima optimise bien la validation mais ne garantit pas un meilleur test set, signe que le dataset est petit et sensible au split.
2. Ce que je referais differemment : ajouter plus tot des matrices de confusion pour Pima et Wine, car l'accuracy seule masque les classes minoritaires.
3. Ce que je n'ai pas eu le temps d'explorer : class weights, seuil de decision optimise pour Pima, et regression ordinale pour Wine Quality.
