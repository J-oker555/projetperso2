## Objectif

Tester L2, Dropout et Early Stopping sur Pima Diabetes.

## Contexte

La phase 5 cherche à comprendre si la régularisation améliore réellement la généralisation. Il faut regarder l'accuracy, mais aussi precision/recall, surtout pour la classe positive diabétique.

## Tâches

- [ ] Exécuter `scripts/phase5_pima_regularisation.py`.
- [ ] Relever accuracy, precision et recall sur le test set.
- [ ] Relever le nombre d'epochs réellement exécutés.
- [ ] Comparer avec la baseline de phase 4.
- [ ] Documenter si L2/Dropout améliore ou non la généralisation.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase5_pima_regularisation.py
```

## Critères d'acceptation

- Early Stopping restaure les meilleurs poids.
- Les métriques test sont affichées.
- Le README indique si la régularisation a aidé, avec chiffres.
