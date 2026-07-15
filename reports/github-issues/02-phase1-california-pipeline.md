## Objectif

Implémenter et valider le pipeline de données California Housing pour la régression.

## Contexte

Le PDF demande de charger California Housing, faire un split propre train/validation/test, normaliser uniquement à partir du train, puis vérifier les shapes. Le point critique est d'éviter toute fuite de données.

## Tâches

- [ ] Exécuter `scripts/phase1_pipeline_california.py`.
- [ ] Vérifier les shapes train/validation/test.
- [ ] Confirmer que `StandardScaler.fit` est appelé uniquement sur `X_train`.
- [ ] Comparer rapidement la moyenne de `X_train` et `X_test` après scaling.
- [ ] Ajouter une note dans le README si une observation importante ressort.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase1_pipeline_california.py
```

## Critères d'acceptation

- Le script affiche les shapes attendues.
- Le commentaire de tête explique clairement pourquoi on split avant de scaler.
- Aucune donnée de validation/test n'est utilisée pour fitter le scaler.
