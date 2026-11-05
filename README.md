# TP Fuzzy C-means : segmentation floue d'une image

M2 Informatique VMI, Reconnaissance des formes avancée.

## Contenu

`fcm.py` contient l'algorithme : Fuzzy C-means écrit avec NumPy (initialisation, centres, degrés, fonction objectif), une variante accélérée sur l'histogramme pour les images en niveaux de gris, et un k-means dur pour comparer.

`experiences.py` lance tous les essais du rapport et écrit `figures/` et `resultats.json`.

`rapport/rapport.tex` est le source du rapport, `rapport/rapport.pdf` la version compilée.

## Lancer

1. Installer les dépendances :

   ```
   pip install numpy matplotlib pillow
   ```

2. Récupérer les deux images du sujet, `milky-way.jpg` et `milky-way-nvg.jpg` (elles ne sont pas dans le dépôt), et les mettre dans un dossier.

3. Indiquer ce dossier avec la variable d'environnement `FCM_DATA`, puis lancer le script :

   ```
   # Linux / macOS / Git Bash
   FCM_DATA=chemin/vers/images python experiences.py

   # PowerShell
   $env:FCM_DATA = "chemin\vers\images"; python experiences.py
   ```

   Sans `FCM_DATA`, le script cherche les images dans le dossier `TP/` situé deux niveaux au-dessus de ce dépôt.

L'exécution complète prend une à deux minutes, surtout à cause du cas couleur.

## Tester

```
python tests/test_fcm.py
```

Chaque ligne affiche PASS ou FAIL : formules de l'annexe, convergence, cas limites, k-means et version histogramme.

## Recompiler le rapport

```
cd rapport
pdflatex rapport.tex
pdflatex rapport.tex
```

Les figures sont lues dans `../figures/`.

## Lire le code dans l'ordre

1. `init_U` : on tire U au hasard, chaque ligne somme à 1.
2. `update_centers` : chaque centre est la moyenne des pixels pondérée par u^m.
3. `update_U` : chaque pixel reçoit un degré pour chaque cluster selon ses distances relatives aux centres.
4. `fcm` : boucle 2 et 3 jusqu'à ce que U ne bouge presque plus (`eps`).
5. `fcm_histogramme` : même algorithme sur les 256 niveaux de gris pondérés par leur nombre de pixels, environ 40 fois plus rapide pour le même résultat.

Un pixel est une donnée. En niveaux de gris, X a la taille (1048576, 1) ; en couleur, (1048576, 3). Pour afficher une heatmap, on remet une colonne de U au format 1024 x 1024.
