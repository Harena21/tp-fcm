"""
Implémentation de Fuzzy C-means (FCM), d'après l'annexe du TP
(Dunn 1973, amélioré par Bezdek 1981).

L'algorithme travaille sur n'importe quel tableau de données X de taille (N, d) :
N données décrites chacune par d caractéristiques. Pour une image en niveaux
de gris, un pixel = une donnée et d = 1 (son niveau de gris). Pour une image
couleur, d = 3 (R, G, B).

Notations reprises de l'annexe :
    U[i, j] = u_ij, degré d'appartenance de la donnée x_i au cluster j (entre 0 et 1)
    C[j]    = c_j, centre du cluster j
    m       = coefficient de flou (m > 1)

Les boucles portent sur les clusters (K vaut 2, 3 ou 4) et pas sur les pixels :
une boucle Python sur un million de pixels serait beaucoup trop lente. Pour un
cluster j fixé, NumPy calcule d'un coup la colonne j pour tous les pixels.
"""

import numpy as np


def init_U(N, K, rng):
    """Étape 1 : matrice U(0) tirée au hasard.

    Chaque ligne doit sommer à 1 (une donnée répartit 100 % de son
    appartenance entre les K clusters), donc on divise chaque ligne par sa somme.
    """
    U = rng.random((N, K))
    somme_lignes = U.sum(axis=1)
    for j in range(K):
        U[:, j] = U[:, j] / somme_lignes
    return U


def update_centers(X, U, m, w=None):
    """Étape 2 : c_j = somme_i u_ij^m x_i / somme_i u_ij^m.

    C'est une moyenne pondérée : un pixel qui appartient fortement au
    cluster j tire davantage le centre vers lui. Avec des u_ij valant 0 ou 1,
    on retrouve la moyenne classique de k-means.

    w (optionnel, taille N) : poids de chaque donnée. Sert pour l'histogramme,
    où une donnée = un niveau de gris qui compte pour w_i pixels.
    """
    N, d = X.shape
    K = U.shape[1]
    C = np.zeros((K, d))
    for j in range(K):
        poids = U[:, j] ** m                 # u_ij^m pour tous les pixels
        if w is not None:
            poids = poids * w
        numerateur = (poids[:, None] * X).sum(axis=0)   # somme_i u_ij^m x_i
        denominateur = poids.sum()                       # somme_i u_ij^m
        C[j] = numerateur / denominateur
    return C


def distances(X, C):
    """Distance euclidienne entre chaque donnée et chaque centre, taille (N, K)."""
    N = X.shape[0]
    K = C.shape[0]
    D = np.zeros((N, K))
    for j in range(K):
        ecart = X - C[j]                     # x_i - c_j pour tous les pixels
        D[:, j] = np.sqrt((ecart ** 2).sum(axis=1))
    # Si un pixel tombe exactement sur un centre, la division plus bas
    # exploserait. On remplace 0 par une valeur minuscule.
    D[D < 1e-10] = 1e-10
    return D


def update_U(X, C, m):
    """Étape 3 : u_ij = 1 / somme_k (d_ij / d_ik)^(2/(m-1)).

    Plus un pixel est proche du centre j par rapport aux autres centres,
    plus u_ij se rapproche de 1. C'est la loi en « carré inverse » dont le
    prof parlait en cours : pour m = 2, u_ij est proportionnel à 1/d_ij^2.
    La formule est écrite exactement comme dans l'annexe, avec deux boucles
    sur les clusters.
    """
    D = distances(X, C)
    N, K = D.shape
    p = 2.0 / (m - 1.0)
    U = np.zeros((N, K))
    for j in range(K):
        somme = np.zeros(N)
        for k in range(K):
            somme = somme + (D[:, j] / D[:, k]) ** p
        U[:, j] = 1.0 / somme
    return U


def objective(X, U, C, m, w=None):
    """Fonction objectif Jm = somme_i somme_j u_ij^m ||x_i - c_j||^2."""
    D = distances(X, C)
    J = 0.0
    for j in range(U.shape[1]):
        terme = (U[:, j] ** m) * D[:, j] ** 2
        if w is not None:
            terme = terme * w
        J += terme.sum()
    return float(J)


def fcm(X, K, m=2.0, eps=0.01, max_iter=300, seed=0, w=None):
    """Lance Fuzzy C-means.

    X        : données (N, d)
    K        : nombre de clusters
    m        : coefficient de flou, m > 1
    eps      : critère d'arrêt de l'annexe, on s'arrête quand
               max_ij |U(k+1) - U(k)| < eps
    max_iter : garde-fou si l'arrêt n'est jamais atteint
    seed     : graine du tirage aléatoire de U(0), pour pouvoir rejouer un essai
    w        : poids des données (optionnel), voir update_centers

    Retourne U (N, K), C (K, d), le nombre d'itérations et l'historique de Jm.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)           # un seul attribut : on passe en (N, 1)
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
    rng = np.random.default_rng(seed)

    U = init_U(X.shape[0], K, rng)      # étape 1
    history = []
    for k in range(1, max_iter + 1):
        C = update_centers(X, U, m, w)  # étape 2
        U_new = update_U(X, C, m)       # étape 3
        history.append(objective(X, U_new, C, m, w))
        delta = np.abs(U_new - U).max()
        U = U_new
        if delta < eps:                 # étape 4
            break
    return U, C, k, history


def fcm_histogramme(X, K, m=2.0, eps=0.01, max_iter=300, seed=0):
    """FCM accéléré pour une image en niveaux de gris (valeurs entières 0..255).

    Le degré u_ij ne dépend que de la valeur du pixel : deux pixels de même
    niveau de gris ont exactement les mêmes degrés. On fait donc tourner FCM
    sur les niveaux de gris présents dans l'image, chacun pondéré par son
    nombre de pixels (l'histogramme), puis on recopie le degré de chaque
    niveau sur ses pixels. On travaille sur au plus 256 données au lieu d'un
    million.

    Mêmes entrées et sorties que fcm (X de taille (N,) ou (N, 1)).
    """
    valeurs = np.asarray(X).ravel().astype(int)

    # Histogramme : effectifs[v] = nombre de pixels de niveau v
    # (np.bincount compte combien de fois chaque entier apparaît)
    effectifs = np.bincount(valeurs, minlength=256)

    # On garde seulement les niveaux qui apparaissent dans l'image
    niveaux = [v for v in range(256) if effectifs[v] > 0]
    poids = [effectifs[v] for v in niveaux]
    U_niveaux, C, k, history = fcm(np.array(niveaux), K, m=m, eps=eps,
                                   max_iter=max_iter, seed=seed, w=np.array(poids))

    # Table niveau -> degrés, puis lecture de la table pour chaque pixel
    table = np.zeros((256, K))
    for position, v in enumerate(niveaux):
        table[v] = U_niveaux[position]
    return table[valeurs], C, k, history


def kmeans(X, K, max_iter=300, seed=0):
    """K-means « dur », écrit à la main pour comparer avec FCM.

    Même boucle, mais chaque pixel est affecté à un seul cluster (le plus
    proche) au lieu d'être réparti entre tous.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)]
    for k in range(1, max_iter + 1):
        labels = distances(X, C).argmin(axis=1)
        C_new = C.copy()
        for j in range(K):
            pixels_j = X[labels == j]
            if len(pixels_j) > 0:            # cluster vide : on garde l'ancien centre
                C_new[j] = pixels_j.mean(axis=0)
        if np.allclose(C_new, C):
            break
        C = C_new
    return labels, C, k
