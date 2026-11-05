# Détection de fraudes par carte de crédit avec un auto-encodeur (PyTorch)

Projet de cours : détecter des transactions frauduleuses avec un **auto-encodeur dense**
entraîné **uniquement sur des transactions normales**. Une fraude, que le modèle n'a jamais
apprise, est mal reconstruite : son **erreur de reconstruction** dépasse un seuil et elle est
signalée comme anomalie.

## Contenu

| Fichier | Description |
|---|---|
| `project_ua1_remis.ipynb` | Version telle que remise pour l'évaluation (avec ses résultats) |
| `project_ua1_v2.ipynb` | Version corrigée, après l'évaluation |

## Données

`credit_card_data.csv` : 1 000 000 de transactions, environ 8,7 % de fraudes, 7 variables + la cible `fraud`.
Le fichier **n'est pas inclus** (trop volumineux). Placez-le à côté des notebooks, ou changez `DATA_PATH`
au début du notebook.

| Variable | Type |
|---|---|
| `distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price` | continues |
| `repeat_retailer`, `used_chip`, `used_pin_number`, `online_order` | binaires |
| `fraud` | cible (jamais utilisée en entrée du modèle) |

## Méthode

1. Séparation 70 / 15 / 15 stratifiée ; entraînement et validation sur les **normales seulement**.
2. `StandardScaler` ajusté sur les normales d'entraînement uniquement.
3. Auto-encodeur dense (ReLU, sortie linéaire), perte MSE, early stopping.
4. Seuil de détection = 95e centile des erreurs de reconstruction des normales de validation.
5. Évaluation sur le test avec précision, rappel, F1 et matrice de confusion
   (l'accuracy est trompeuse avec 8,7 % de fraudes).

## Résultats de la version remise (jeu de test, 150 000 transactions)

Auto-encodeur classique, latente de 16 : précision 0,60 ; rappel 0,78 ; F1 0,68.
Matrice de confusion : 10 215 fraudes détectées, 2 895 manquées, 6 808 fausses alertes.

## Version 2 : ce qui a été corrigé

| Problème de la version remise | Correction |
|---|---|
| Latente de 16 pour 7 variables (pas de goulot) | `d_latent` testé de 1 à 7, choix par la courbe d'erreur |
| Validation contenant des fraudes | Validation sur les normales seulement |
| Seul le batch size testé | Recherche aléatoire (`lr` en échelle log, `weight_decay`, largeur) |
| Comparaison classique / débruitant faussée | Même validation propre, choix sur la validation (pas le test) |
| Seeds non fixées, deux `random_state` | Une seule graine |
| Sauvegarde des poids seuls | Poids + configuration + scaler + seuil, rechargement vérifié |
| Exemples = plus fortes erreurs (dont fausses alertes) | Vraies fraudes détectées, avec la variable responsable |

Les résultats de la version 2 sont à lire dans les sorties du notebook après exécution.

## Exécution

```bash
pip install torch numpy pandas scikit-learn matplotlib jupyter
jupyter notebook project_ua1_v2.ipynb
```

Sur Google Colab : activer un GPU (Exécution > Modifier le type d'exécution), téléverser
`credit_card_data.csv`, puis tout exécuter. Pour un essai rapide, régler `N_LIGNES = 100_000`
et réduire `EPOCHS_RECHERCHE` / `EPOCHS_FINAL` dans la première cellule.

## Limites et pistes

Une erreur élevée signifie « atypique », pas toujours « fraude ». Le seuil au 95e centile impose environ 5 % de
fausses alertes sans être optimisé selon le coût d'une fraude manquée. Pistes : `log1p` sur les variables
asymétriques, auto-encodeur épars, optimisation bayésienne (Optuna), courbe précision-rappel pour choisir le seuil.
