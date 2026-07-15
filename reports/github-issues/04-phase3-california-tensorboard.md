## Objectif

Brancher TensorBoard sur California Housing et comparer un run normalisé avec un run non normalisé.

## Contexte

TensorBoard doit devenir l'outil de diagnostic principal. Cette phase sert à lire les courbes `loss`, `mae`, `val_loss` et `val_mae`, puis à interpréter les écarts train/validation.

## Tâches

- [ ] Exécuter `scripts/phase3_tensorboard_california.py`.
- [ ] Vérifier la création de logs sous `logs/fit`.
- [ ] Lancer TensorBoard.
- [ ] Comparer les courbes du run `california_scaled` et `california_unscaled`.
- [ ] Ajouter dans le README une courte interprétation : sain, overfitting, ou suspicion de fuite.

## Commandes de validation

```powershell
.\.venv\Scripts\python.exe scripts\phase3_tensorboard_california.py
.\.venv\Scripts\tensorboard.exe --logdir logs/fit
```

## Critères d'acceptation

- Deux runs TensorBoard sont visibles.
- Le run normalisé converge mieux ou plus proprement que le run non normalisé.
- Une interprétation des courbes est documentée.
