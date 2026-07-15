## Objectif

Compléter la synthèse cross-dataset et exécuter le pipeline personnel.

## Contexte

La phase 9 transforme les runs en analyse : tableau comparatif, hypothèses, réponses de synthèse, post-mortem et pipeline sur un dataset personnel. Le script fourni utilise Breast Cancer comme dataset personnel par défaut.

## Tâches

- [ ] Exécuter `scripts/phase9_custom_pipeline.py`.
- [ ] Comparer Adam vs SGD.
- [ ] Observer la prédiction sur une entrée extrême hors distribution.
- [ ] Remplir tout le tableau comparatif du README avec des chiffres réels.
- [ ] Répondre aux trois questions de synthèse.
- [ ] Remplir le post-mortem.
- [ ] Nettoyer le repo avant rendu.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase9_custom_pipeline.py
git status --short
```

## Critères d'acceptation

- Le tableau comparatif ne contient plus `A remplir`.
- Les réponses de synthèse s'appuient sur les métriques réelles.
- Le post-mortem contient trois réponses concrètes.
- Le dépôt est propre ou les fichiers non commités sont justifiés.
