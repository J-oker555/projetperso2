## Objectif

Créer la baseline de classification binaire sur Pima Diabetes.

## Contexte

Pima est déséquilibré, avec environ 65% de classe majoritaire. Toute accuracy doit être comparée à ce baseline naïf, sinon le résultat est trompeur.

## Tâches

- [ ] Exécuter `scripts/phase4_pima_baseline.py`.
- [ ] Vérifier la distribution des classes.
- [ ] Relever le baseline majoritaire.
- [ ] Relever `val_accuracy` et `test_accuracy`.
- [ ] Comparer le modèle au baseline naïf.
- [ ] Reporter les métriques dans le README.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase4_pima_baseline.py
```

## Critères d'acceptation

- Le script affiche la distribution de classes.
- Le modèle bat clairement le baseline majoritaire ou l'écart est expliqué.
- Le split utilise `stratify=y`.
