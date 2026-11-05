"""
Expériences du TP : segmentation floue de l'image de la Voie lactée avec FCM.

Lancement : python experiences.py
Les figures sont écrites dans figures/ et les chiffres dans resultats.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from fcm import fcm, kmeans

import os

ICI = Path(__file__).resolve().parent
# Dossier des images fournies : variable d'environnement FCM_DATA si définie,
# sinon le dossier TP/ du cours (deux niveaux au-dessus de ce fichier).
DATA = Path(os.environ.get("FCM_DATA", ICI.parent.parent / "TP"))
OUT = ICI / "figures"
OUT.mkdir(exist_ok=True)
res = {}

gris = np.array(Image.open(DATA / "milky-way-nvg.jpg"), dtype=np.float64)
coul = np.array(Image.open(DATA / "milky-way.jpg"), dtype=np.float64)
H, W = gris.shape
Xg = gris.reshape(-1, 1)        # 1 pixel = 1 donnée, 1 attribut
Xc = coul.reshape(-1, 3)        # 1 pixel = 1 donnée, 3 attributs (R, G, B)


def trier(U, C):
    """Le tirage aléatoire de U(0) donne les clusters dans un ordre quelconque.
    On les range du plus sombre au plus clair pour que « cluster 0 » désigne
    toujours le fond, quel que soit l'essai."""
    ordre = np.argsort(C.sum(axis=1))
    return U[:, ordre], C[ordre]


# 1) Niveaux de gris, K = 2, m = 2 (cas conseillé par le sujet)
t = time.time()
U2, C2, it2, J2 = fcm(Xg, K=2, m=2.0, eps=0.01, seed=0)
U2, C2 = trier(U2, C2)
res["gris_K2"] = {"centres": C2.ravel().round(1).tolist(), "iterations": it2,
                  "temps_s": round(time.time() - t, 2),
                  "J_debut": round(J2[0]), "J_fin": round(J2[-1])}
umax = U2.max(axis=1)
res["gris_K2"]["part_pixels_ambigus_umax<0.7"] = round(float((umax < 0.7).mean()) * 100, 1)
res["gris_K2"]["part_pixels_galaxie_dur"] = round(float((U2[:, 1] > 0.5).mean()) * 100, 1)

# 2) Effet de l'initialisation : 5 graines différentes
res["graines"] = []
for s in range(5):
    U, C, it, _ = fcm(Xg, K=2, m=2.0, eps=0.01, seed=s)
    res["graines"].append({"seed": s, "iterations": it,
                           "centres": np.sort(C.ravel()).round(2).tolist()})

# 3) Effet du critère d'arrêt epsilon
res["epsilon"] = []
for e in [0.3, 0.1, 0.01, 0.001]:
    U, C, it, _ = fcm(Xg, K=2, m=2.0, eps=e, seed=0)
    res["epsilon"].append({"eps": e, "iterations": it,
                           "centres": np.sort(C.ravel()).round(2).tolist()})

# 4) Effet du coefficient de flou m
res["m"] = []
courbes = {}
niveaux = np.arange(256, dtype=np.float64)[:, None]
for m in [1.2, 1.5, 2.0, 3.0, 5.0]:
    U, C, it, _ = fcm(Xg, K=2, m=m, eps=0.01, seed=0)
    U, C = trier(U, C)
    amb = float((U.max(axis=1) < 0.7).mean()) * 100
    res["m"].append({"m": m, "iterations": it, "centres": C.ravel().round(1).tolist(),
                     "part_ambigus": round(amb, 1)})
    # Degré d'appartenance au cluster clair pour chaque niveau de gris 0..255
    D = np.fmax(np.abs(niveaux - C.T), 1e-10)
    inv = D ** (-2 / (m - 1))
    courbes[m] = (inv / inv.sum(axis=1, keepdims=True))[:, 1]

# 5) Comparaison avec k-means dur
lab, Ck, itk = kmeans(Xg, K=2, seed=0)
clair_km = lab == np.argmax(Ck.ravel())
clair_fcm = U2[:, 1] > 0.5
res["kmeans_K2"] = {"centres": np.sort(Ck.ravel()).round(1).tolist(), "iterations": itk,
                    "accord_avec_fcm_pct": round(float((clair_km == clair_fcm).mean()) * 100, 2)}

# 6) Niveaux de gris, K = 3
U3, C3, it3, _ = fcm(Xg, K=3, m=2.0, eps=0.01, seed=0)
U3, C3 = trier(U3, C3)
res["gris_K3"] = {"centres": C3.ravel().round(1).tolist(), "iterations": it3,
                  "part_ambigus": round(float((U3.max(axis=1) < 0.7).mean()) * 100, 1)}

# 7) Couleur RGB, K = 4
t = time.time()
Uc, Cc, itc, _ = fcm(Xc, K=4, m=2.0, eps=0.01, seed=0)
Uc, Cc = trier(Uc, Cc)
res["couleur_K4"] = {"centres_RGB": Cc.round(0).tolist(), "iterations": itc,
                     "temps_s": round(time.time() - t, 2)}

# 8) Alpha-coupes sur la région « galaxie » (cluster clair, K = 2)
res["alpha_coupes_pct_pixels"] = {a: round(float((U2[:, 1] >= a).mean()) * 100, 1)
                                  for a in [0.3, 0.5, 0.7, 0.9]}

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9})

# Figure 1 : K = 2 en niveaux de gris
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
ax[0].imshow(gris, cmap="gray"); ax[0].set_title("(a) image NVG")
im = ax[1].imshow(U2[:, 1].reshape(H, W), cmap="jet", vmin=0, vmax=1)
ax[1].set_title("(b) $u$ cluster clair (heatmap)")
fig.colorbar(im, ax=ax[1], fraction=0.046)
ax[2].imshow(U2.argmax(axis=1).reshape(H, W), cmap="gray")
ax[2].set_title("(c) segmentation dure (argmax)")
for m, c in courbes.items():
    ax[3].plot(np.arange(256), c, label=f"m={m}")
ax[3].axhline(0.5, color="k", lw=0.5, ls="--")
ax[3].set_xlabel("niveau de gris"); ax[3].set_ylabel("$u$ cluster clair")
ax[3].set_title("(d) effet de m"); ax[3].legend(fontsize=7)
for a in ax[:3]:
    a.axis("off")
fig.tight_layout(); fig.savefig(OUT / "fig1_K2.png", dpi=150); plt.close(fig)

# Figure 2 : K = 3, couleur K = 4, alpha-coupes
fig, ax = plt.subplots(1, 5, figsize=(14, 3.1))
noms = ["fond", "bras", "coeur"]
for j in range(3):
    ax[j].imshow(U3[:, j].reshape(H, W), cmap="jet", vmin=0, vmax=1)
    ax[j].set_title(f"({'abc'[j]}) K=3, $u$ {noms[j]}")
seg_coul = Cc[Uc.argmax(axis=1)].reshape(H, W, 3) / 255
ax[3].imshow(seg_coul); ax[3].set_title("(d) RGB, K=4 (argmax)")
coupes = np.zeros((H, W, 3))
coupes[..., 0] = (U2[:, 1] >= 0.5).reshape(H, W)
coupes[..., 1] = (U2[:, 1] >= 0.9).reshape(H, W)
coupes[..., 2] = (U2[:, 1] >= 0.9).reshape(H, W)
ax[4].imshow(coupes); ax[4].set_title(r"(e) $\alpha$=0.5 (rouge), 0.9 (blanc)")
for a in ax:
    a.axis("off")
fig.tight_layout(); fig.savefig(OUT / "fig2_K3_couleur.png", dpi=150); plt.close(fig)

# Convergence de Jm (gardée pour le README, pas dans le rapport)
plt.figure(figsize=(4, 3)); plt.plot(range(1, len(J2) + 1), J2, "o-")
plt.xlabel("itération"); plt.ylabel("$J_m$"); plt.tight_layout()
plt.savefig(OUT / "convergence_J.png", dpi=120); plt.close()

(ICI / "resultats.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
