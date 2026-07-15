## Objectif

Comparer quatre configurations BatchNorm sur Wine Quality.

## Contexte

La phase compare l'absence de BatchNorm, BatchNorm avant activation, BatchNorm après activation, et une variante plus profonde. L'intérêt principal est la vitesse et la stabilité de convergence dans TensorBoard, pas seulement l'accuracy finale.

## Tâches

- [ ] Exécuter `scripts/phase8_wine_batchnorm.py`.
- [ ] Vérifier les quatre configurations.
- [ ] Relever `val_accuracy`, `val_loss_final` et epochs réels.
- [ ] Ouvrir TensorBoard sur `logs/wine`.
- [ ] Comparer la pente des courbes `val_loss`.
- [ ] Reporter la meilleure configuration dans le README.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase8_wine_batchnorm.py
.\.venv\Scripts\tensorboard.exe --logdir logs/wine
```

## Critères d'acceptation

- Les quatre configs produisent un résultat.
- Les logs TensorBoard sont générés.
- Le README distingue convergence plus rapide et meilleure accuracy finale.
