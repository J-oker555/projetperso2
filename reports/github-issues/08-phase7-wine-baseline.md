## Objectif

Créer la baseline multiclassification Wine Quality en 3 classes.

## Contexte

Wine Quality est ordinal à l'origine, mais la phase demande une agrégation en trois classes `low`, `medium`, `high`, puis un modèle softmax avec `sparse_categorical_crossentropy`. Le déséquilibre de la classe `medium` doit être explicitement surveillé.

## Tâches

- [ ] Exécuter `scripts/phase7_wine_baseline.py`.
- [ ] Vérifier la distribution agrégée des classes.
- [ ] Vérifier que le split utilise `stratify=y`.
- [ ] Relever `val_accuracy` et `test_accuracy`.
- [ ] Vérifier si le modèle prédit autre chose que la classe majoritaire.
- [ ] Reporter les métriques dans le README.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase7_wine_baseline.py
```

## Critères d'acceptation

- La distribution `low/medium/high` est affichée.
- La loss est `sparse_categorical_crossentropy`.
- Le README mentionne la limite nominale vs ordinale.
