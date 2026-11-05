# TP Fuzzy C-means : segmentation floue d'une image

TP de reconnaissance des formes avancée, M2 Informatique VMI. Le code implémente Fuzzy C-means (FCM) avec NumPy et l'applique à l'image de la Voie lactée, en niveaux de gris et en couleur.

## Organisation du dépôt

Le fichier principal est le notebook `tp_fcm.ipynb`. Il déroule tout le TP dans l'ordre, avec une courte explication avant chaque étape, et il est enregistré avec ses sorties : on peut lire les résultats et voir les figures sur GitHub sans rien exécuter.

`fcm.py` contient l'algorithme : initialisation, calcul des centres, calcul des degrés et fonction objectif. On y trouve aussi une version plus rapide pour les images en niveaux de gris, qui travaille sur l'histogramme, et un k-means dur qui sert de point de comparaison. Le notebook importe ces fonctions.

`experiences.py` refait les mêmes essais en script. Il enregistre les figures dans `figures/` et les valeurs numériques dans `resultats.json`.

`tests/test_fcm.py` vérifie les formules et le comportement de l'algorithme. Les images du sujet sont dans `data/` et le rapport dans `rapport/rapport.pdf`.

## Lancer le notebook

Les dépendances s'installent avec pip :

```
pip install numpy matplotlib pillow jupyter
jupyter notebook tp_fcm.ipynb
```

Une exécution complète prend environ une minute, dont une demi-minute pour l'image couleur. Pour obtenir les figures sous forme de fichiers, on peut aussi lancer `python experiences.py`.

## Lancer les tests

```
python tests/test_fcm.py
```

Chaque test affiche PASS ou FAIL. Les tests comparent les mises à jour aux formules de l'annexe, contrôlent que la fonction objectif décroît, que les centres de données synthétiques sont retrouvés et qu'une distance nulle ne produit pas de NaN. Ils vérifient enfin que la version histogramme donne le même résultat que la version pixel par pixel.

## Lire le code

Le plus simple est de suivre l'ordre de l'annexe. `init_U` tire la matrice U au hasard puis normalise chaque ligne à 1. `update_centers` calcule chaque centre comme la moyenne des pixels pondérée par u^m. `update_U` donne à chaque pixel un degré par cluster, à partir de ses distances relatives aux centres. `fcm` répète ces deux mises à jour jusqu'à ce que U ne varie plus que de moins de `eps`.

Chaque pixel est une donnée : X a pour taille (1048576, 1) en niveaux de gris et (1048576, 3) en couleur. Pour afficher une heatmap, une colonne de U est remise au format 1024 x 1024.

`fcm_histogramme` applique la même boucle aux 256 niveaux de gris, chacun pondéré par son nombre de pixels. Sur l'image du TP, elle obtient les mêmes centres environ 100 fois plus vite.
