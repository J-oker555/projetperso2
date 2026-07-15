## Objectif

Lancer une recherche d'hyperparamètres keras-tuner sur Pima Diabetes.

## Contexte

Le tuner doit explorer units, learning rate, L2, dropout et taille de couche. Le piège à éviter est un budget d'epochs trop court, qui sélectionne seulement les modèles qui démarrent vite.

## Tâches

- [ ] Exécuter `scripts/phase6_pima_kerastuner.py`.
- [ ] Vérifier que 15 trials sont terminés.
- [ ] Relever les 5 meilleurs essais.
- [ ] Relever le meilleur learning rate et la meilleure taille de couche.
- [ ] Comparer avec une exécution `max_trials=1` si le temps le permet.
- [ ] Reporter les résultats dans le README.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase6_pima_kerastuner.py
```

## Critères d'acceptation

- `tuner.results_summary(5)` affiche les meilleurs essais.
- Le meilleur modèle atteint une validation accuracy raisonnable pour Pima.
- Le README explique ce que le tuner a apporté par rapport au réglage manuel.
