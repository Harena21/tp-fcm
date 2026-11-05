# TP Fuzzy C-means : segmentation floue d'une image

M2 Informatique VMI, Reconnaissance des formes avancée.

## Contenu

`fcm.py` contient l'algorithme : Fuzzy C-means écrit avec NumPy (initialisation, centres, degrés, fonction objectif), plus un k-means dur pour comparer.

`experiences.py` lance tous les essais du rapport et écrit `figures/` et `resultats.json`.

`rapport/rapport.tex` est le source du rapport.

## Lancer

```
pip install numpy matplotlib pillow
python experiences.py
```

Les images (`milky-way.jpg`, `milky-way-nvg.jpg`) sont cherchées par défaut dans le dossier `TP/` situé deux niveaux au-dessus de `tp_fcm/`. Pour un autre dossier, définir la variable d'environnement `FCM_DATA`. L'exécution complète prend une à deux minutes.

## Lire le code dans l'ordre

1. `init_U` : on tire U au hasard, chaque ligne somme à 1.
2. `update_centers` : chaque centre est la moyenne des pixels pondérée par u^m.
3. `update_U` : chaque pixel reçoit un degré pour chaque cluster selon ses distances relatives aux centres.
4. `fcm` : boucle 2 et 3 jusqu'à ce que U ne bouge presque plus (`eps`).

Un pixel est une donnée. En niveaux de gris, X a la taille (1048576, 1) ; en couleur, (1048576, 3). Pour afficher une heatmap, on remet une colonne de U au format 1024 x 1024.
