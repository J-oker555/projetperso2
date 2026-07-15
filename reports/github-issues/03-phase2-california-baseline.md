## Objectif

Construire et entraîner une baseline PMC de régression sur California Housing.

## Contexte

La sortie doit être une couche `Dense(1)` sans activation, car la cible est une valeur continue. Une sigmoid serait une erreur conceptuelle car elle borne la prédiction entre 0 et 1.

## Tâches

- [ ] Exécuter `scripts/phase2_baseline_regression.py`.
- [ ] Vérifier le `model.summary()`.
- [ ] Relever la `val_mae` finale.
- [ ] Relever la `test_mae`.
- [ ] Reporter les métriques dans le tableau du README.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase2_baseline_regression.py
```

## Critères d'acceptation

- Le modèle s'entraîne sans erreur.
- La loss utilisée est `mse`.
- La métrique principale est `mae`.
- La couche de sortie n'a pas d'activation.
