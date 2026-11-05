"""Tests de fcm.py : python tests/test_fcm.py (chaque ligne affiche PASS ou FAIL)."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fcm import fcm, fcm_histogramme, kmeans, update_U, update_centers, init_U
rng = np.random.default_rng(0)
ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg)

# 1. init_U : lignes somment à 1
U = init_U(1000, 4, rng); ok(np.allclose(U.sum(1), 1) and (U >= 0).all(), "init_U lignes = 1")

# 2. update_U contre la formule de l'annexe écrite en boucle
X = rng.normal(size=(50, 2)); C = rng.normal(size=(3, 2)); m = 2.5
D = np.linalg.norm(X[:, None] - C[None], axis=2)
ref = np.array([[1 / sum((D[i, j] / D[i, k]) ** (2 / (m - 1)) for k in range(3)) for j in range(3)] for i in range(50)])
ok(np.allclose(update_U(X, C, m), ref), "update_U = formule annexe")

# 3. update_centers contre boucle
U = init_U(50, 3, rng)
refC = np.array([(U[:, j:j+1] ** m * X).sum(0) / (U[:, j] ** m).sum() for j in range(3)])
ok(np.allclose(update_centers(X, U, m), refC), "update_centers = formule annexe")

# 4. Données bien séparées : retrouve les centres
X = np.vstack([rng.normal(0, .3, (300, 2)), rng.normal(10, .3, (300, 2)), rng.normal([0, 10], .3, (300, 2))])
U, C, it, J = fcm(X, 3, seed=1)
Cr = C.round(); Cs = Cr[np.lexsort(Cr.T[::-1])]
ok(np.allclose(Cs, [[0, 0], [0, 10], [10, 10]], atol=0.2), f"centres retrouvés ({it} it.)")
ok(np.allclose(U.sum(1), 1), "U final lignes = 1")
ok(all(J[i+1] <= J[i] + 1e-6 for i in range(len(J) - 1)), "Jm décroissant")

# 5. Entrée 1D
U, C, it, _ = fcm(np.r_[np.zeros(100), np.ones(100) * 50] + rng.normal(0, 1, 200), 2)
ok(np.allclose(np.sort(C.ravel()), [0, 50], atol=1), "entrée 1D")

# 6. Pixel exactement sur un centre (distance nulle) : pas de NaN
X = np.array([[0.], [0.], [10.], [10.]])
U, C, it, _ = fcm(X, 2)
ok(np.isfinite(U).all() and np.isfinite(C).all(), "pas de NaN avec distances nulles")

# 7. k-means
X = np.vstack([rng.normal(0, .3, (200, 2)), rng.normal(10, .3, (200, 2))])
lab, C, it = kmeans(X, 2)
ok(len(set(lab[:200])) == 1 and len(set(lab[200:])) == 1 and lab[0] != lab[-1], "kmeans sépare 2 groupes")

# 8. Version histogramme : même résultat que la version pixel par pixel
X = np.r_[rng.integers(0, 60, 3000), rng.integers(150, 256, 2000)].astype(float)
U1, C1, _, _ = fcm(X, 2, seed=0, eps=1e-6)
U2, C2, _, _ = fcm_histogramme(X, 2, seed=0, eps=1e-6)
o1, o2 = np.argsort(C1.ravel()), np.argsort(C2.ravel())
ok(np.allclose(C1.ravel()[o1], C2.ravel()[o2], atol=1e-3) and np.allclose(U1[:, o1], U2[:, o2], atol=1e-4),
   "fcm_histogramme = fcm sur les pixels")


