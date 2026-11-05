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
"""

import numpy as np


def init_U(N, K, rng):
    """Étape 1 : matrice U(0) tirée au hasard.

    Chaque ligne doit sommer à 1 (une donnée répartit 100 % de son
    appartenance entre les K clusters), donc on normalise ligne par ligne.
    """
    U = rng.random((N, K))
    return U / U.sum(axis=1, keepdims=True)


def update_centers(X, U, m, w=None):
    """Étape 2 : c_j = somme_i u_ij^m x_i / somme_i u_ij^m.

    C'est une moyenne pondérée : un pixel qui appartient fortement au
    cluster j tire davantage le centre vers lui. Avec des u_ij valant 0 ou 1,
    on retrouve la moyenne classique de k-means.

    w (optionnel, taille N) : poids de chaque donnée. Sert pour l'histogramme,
    où une donnée = un niveau de gris qui compte pour w_i pixels.
    """
    Um = U ** m                                   # (N, K)
    if w is not None:
        Um = Um * w[:, None]
    return (Um.T @ X) / Um.sum(axis=0)[:, None]   # (K, d)


def distances(X, C):
    """Distance euclidienne entre chaque donnée et chaque centre, taille (N, K)."""
    D = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    # Si un pixel tombe exactement sur un centre, la division plus bas
    # exploserait. On remplace 0 par une valeur minuscule.
    return np.fmax(D, 1e-10)


def update_U(X, C, m):
    """Étape 3 : u_ij = 1 / somme_k (d_ij / d_ik)^(2/(m-1)).

    Plus un pixel est proche du centre j par rapport aux autres centres,
    plus u_ij se rapproche de 1. C'est la loi en « carré inverse » dont le
    prof parlait en cours : pour m = 2, u_ij est proportionnel à 1/d_ij^2.
    """
    D = distances(X, C)
    p = 2.0 / (m - 1.0)
    # On écrit la formule sous la forme équivalente
    # u_ij = d_ij^(-p) / somme_k d_ik^(-p), plus simple à vectoriser.
    inv = D ** (-p)
    return inv / inv.sum(axis=1, keepdims=True)


def objective(X, U, C, m, w=None):
    """Fonction objectif Jm = somme_i somme_j u_ij^m ||x_i - c_j||^2."""
    D = distances(X, C)
    termes = (U ** m) * D ** 2
    if w is not None:
        termes = termes * w[:, None]
    return float(termes.sum())


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
        X = X[:, None]                 # un seul attribut : on passe en (N, 1)
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
    rng = np.random.default_rng(seed)

    U = init_U(X.shape[0], K, rng)     # étape 1
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
    """FCM accéléré pour une image en niveaux de gris.

    Une image 8 bits n'a que 256 valeurs possibles, et le degré u_ij ne dépend
    que de la valeur du pixel. On fait donc tourner FCM sur les valeurs
    distinctes, chacune pondérée par son nombre de pixels (l'histogramme),
    puis on recopie le degré de chaque valeur sur ses pixels.
    On travaille sur au plus 256 données au lieu d'un million.

    Mêmes entrées et sorties que fcm (X de taille (N,) ou (N, 1)).
    """
    valeurs, inverse, effectifs = np.unique(np.asarray(X).ravel(),
                                            return_inverse=True, return_counts=True)
    U, C, k, history = fcm(valeurs, K, m=m, eps=eps, max_iter=max_iter,
                           seed=seed, w=effectifs)
    return U[inverse], C, k, history


def kmeans(X, K, max_iter=300, seed=0):
    """K-means « dur », écrit à la main pour comparer avec FCM.

    Même boucle, mais chaque pixel est affecté à un seul cluster (le plus
    proche) au lieu d'être réparti entre tous.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)]
    for k in range(1, max_iter + 1):
        labels = distances(X, C).argmin(axis=1)
        C_new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                          for j in range(K)])
        if np.allclose(C_new, C):
            break
        C = C_new
    return labels, C, k
