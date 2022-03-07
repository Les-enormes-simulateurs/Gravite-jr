# -*- coding: utf-8 -*-
"""
Demonstrates use of GLScatterPlotItem with rapidly-updating plots.

"""

## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from Sauvegarde import objets_7 as obj
import pickle
import os


# ---- trucs de sauvegarde
# Si le fichier Sauvegarde.data est inexistant, il crée un fichier vide portant ce nom
if not os.path.isfile("Sauvegarde.data"):
    data = {}
    fw = open("Sauvegarde.data", 'wb')
    pickle.dump(data, fw)
    fw.close()
# Ouvre le fichier Sauvegarde.data et loade les sauvegardes existantes
fd = open("Sauvegarde.data", 'rb')
liste_objets = pickle.load(fd)
fd.close()
# Demande à l'utilisateur quelle sauvegarde il veut utiliser et l'importe
print(f"Sauvegardes disponibles: {liste_objets.keys()}")
nom_fichier = input(
    "Veuillez entrer le nom de la sauvegarde à utiliser ou enter pour utiliser le fichier déjà importé: ")
# nom_fichier = "objets_6"
if nom_fichier == '':
    objets = obj
else:
    objets = liste_objets[nom_fichier]
    print(liste_objets[nom_fichier])
# initialisation des principales variables
objets_total = objets  # les objets dans la sauvegarde, ainsi que les métadonnées qu'il pourrait éventuellement y avoir
objets = objets[0]  # Objets de la sauvegarde, sans métadonnées
temps = 0  # temps auquel la simulation est rendue
uns = np.array([[[1, 1, 1]], [[1, 1, 1]], [[1, 1, 1]]])  # hmmm, j'ai aucune idée ce que ça fait
delta_t = 10 # pas de temps
G = 6.67430 * 10 ** (-11) # constante gravitationnelle
# pos = []  # éventuellement une matrice de positions
v = []  # éventuellement une matrice de
# liste = []


n_objets = len(objets)
vecteur_masses = []
print(n_objets)
for i in range(n_objets):
    vecteur_masses.append(objets[i][0])
vecteur_masses = np.array(vecteur_masses).T
masse = []  # un vecteur avec les masses de chaque objet
masses = []  # une matrice carrée dont l'indice u_mn est (masse(m)) * (1 - delta_mn). Utile pour les calculs plus tard
# initialisation vecteur masse
for i in range(len(objets)):
    masse.append(objets[i][0])

# initialisation matrice masses
for i in range(len(objets)):
    masses.append(masse[0:i] + [0] + masse[i + 1:])
tailles = 0.35122595114987754 * np.power(np.array(masse), 0.333333333)  # vecteur de taille des objets
masses = G * np.array(masses)
pos = np.array([objets[0][2]]).T  # position du premier objet
# On concatène la position du premier objet avec toutes les autres positions pour obtenir une matrice.
for i in range(1, len(objets)):
    pos = np.concatenate((pos, np.array([objets[i][2]]).T), 1)
pos_init = np.array([pos], dtype='float64')  # positions initiales
pos = pos_init  # on initialise pos à la position initiale
v = np.array([objets[0][1]]).T  # vecteur vitesse du premier objet
# On concatène le vecteur vitesse du premier objet avec tous les autres pour obtenir une matrice.
for i in range(1, len(objets)):
    v = np.concatenate((v, np.array([objets[i][1]]).T), 1)
v_init = np.array([v], dtype='float64')  # matrice de vitesses initiale
v = v_init  # on initialise la matrice de vitesses à la vitesse initiale


def position():
    global pos
    global temps
    global v
    global n_objets
    global delta_t
    global liste
    temps += delta_t  # On incrémente le temps
    # On cherche à obtenir une expression de la forme a = GM/R^3 * \vec{r}

    # On concatène la matrice de position pour obtenir une matrice prismique m X 3 X m
    posQ = np.array([np.concatenate(pos, axis=0) for i in range(n_objets)])
    # On soustrait la matrice avec sa rotation de 90 degrés dans le but d'avoir une matrice 3D de toutes les différence
    #   de distance vectorielle entre les objets
    posR = posQ - np.rot90(posQ, k=1, axes=(2, 0))
    # On met tous les éléments au carré
    posR_carre = np.absolute(np.square(posR))
    # On additionne le carré de toutes les composantes pour avoir le R^2 associé à chaque paire d'objets sous forme de
    #   matrice 2D. On additionne aussi une matrice identité pour ne pas avoir de zéros sur la diagonale, car cela
    #   occasionnerait une division par 0
    uns_plus_posR = np.identity(n_objets) + posR_carre[:, 0, :] + posR_carre[:, 1, :] + posR_carre[:, 2, :]
    # On fait (R^2)^(-1.5) pour avoir le R^3 au dénominateur et on multiplie par GM, notre vecteur masses
    module = np.array([masses * np.power(uns_plus_posR, -1.5)])
    # On fait tourner notre matrice pour qu'elle soit dans le bon axe pour la multiplication matricielle
    module = np.rot90(module, axes=(1, 0))
    module = np.rot90(module, axes=(2, 1))
    # On multiplie la matrice de distance vectorielle par celle du module de GM/R^3 pour obtenir un vecteur
    #   d'accélération. En gros, ça revient à un produit scalaire, mais de matrices.
    acc = posR @ module
    # On fait tourner notre vecteur pour qu'il soit dans le bon axe pour la suite des calculs.
    acc = np.rot90(acc, k=3, axes=(2, 0))
    # On applique simplement l'algorithme d'Euler. On calcule le module de la vitesse.
    v += delta_t * acc
    # data.append((pos+delta_t * v)[0, :, :])
    # À partir de ce module, on calcule la nouvelle position. On rappelle au passage qu'il s'agit d'une matrice.
    pos += delta_t * v
    return pos


masse_totale = 0
# somme des masses pour calculer le barycentre du système
for i in objets:
    masse_totale += i[0]
print(f"pos = {pos}")
print(f"vecteur_masses = {vecteur_masses}")
# On trouve le barycentre à un instant t
barycentre_1 = pos[0, :, :] @ vecteur_masses / masse_totale
# On trouve la prochaine position pour pouvoir calculer le barycentre à l'instant t + delta_t
position()
barycentre_2 = pos[0, :, :] @ vecteur_masses / masse_totale
# On fait la différence de position des deux barycentres divisée par delta_t pour avoir la vitesse du centre
v_barycentre = np.array([((barycentre_2 - barycentre_1) / delta_t)]).T
# On concatène la vitesse pour avoir une matrice de vitesses
v_barycentre = np.array([np.concatenate(v_barycentre, axis=0) for i in range(n_objets)])
# On soustrait la vitesse de tous les corps par celle du barycentre pour que la vitesse résultante soit toujours nulle.
v = v_init - np.array([v_barycentre.T])
# On concatène la position du barycentre pour avoir une matrice de positions x, y z
barycentre_1 = np.array([np.array([np.concatenate(np.array([barycentre_1]).T, axis=0) for i in range(n_objets)]).T])
# On soustrait la position de tous les corps par celle du barycentre pour que le barycentre soit au centre de l'écran
pos = pos_init - barycentre_1
# position()

# J'ai sincèrement aucune idée de ce qui se passe à partir d'ici...
app = pg.mkQApp("GLScatterPlotItem Example")
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

# g = gl.GLGridItem()
# w.addItem(g)
pos3 = np.zeros((100, 100, 3))
pos3[:, :, :2] = np.mgrid[:100, :100].transpose(1, 2, 0) * [-0.1, 0.1]
pos3 = pos3.reshape(10000, 3)
print(pos3)
pos3 = pos[0, :, :]
pos3 = pos3.T
# ... jusqu'à ici. Ensuite, on change la couleur de nos points selon celle qui est spécifiée dans la sauvegarde importée
couleur = []
for i in range(n_objets):
    if len(objets[0]) == 4:
        couleur.append(objets[i][3])
    else:
        couleur = [1, 1, 1, .3]
couleur = np.array(couleur)
# plot les objets (points)
sp3 = gl.GLScatterPlotItem(pos=pos3, color=couleur, size=tailles, pxMode=False)
# On ajoute les points au graphique
w.addItem(sp3)

# Stack correspond à la matrice m X 3 X n (n = nombre d'itérations) des positions à travers le temps pour avoir les
#       trajectoires
stack = np.array([pos3])
stack = np.rot90(stack, axes=(0, 1))
liste = [] # liste des différentes trajectoires
if len(objets_total[1]) != 0:
    for i in objets_total[1]:
        w.addItem(i) # on ajoute les trajectoires en métadonnées, s'il y en a
for i in range(n_objets):  # On met les différentes trajectoires dans le graphique
    liste.append(gl.GLLinePlotItem(pos=stack[i, :, :],
                                   color=list(np.array(objets[n_objets - i - 1][3]) - np.array([0, 0, 0, 0.5])),
                                   width=1, antialias=True, mode='line_strip'))
    w.addItem(liste[i])
k = 0


def save(ecriture=False):  # Éventuellement pour sauvegarder. Fonctionne pas pour le moment
    global masse, v, pos, objets, obj, liste
    donnees = []
    for i in range(n_objets):
        donnees.append((masse[i], list(v[:, i]), list(pos[:, i]), objets[i][3]))
    obj[1].append(liste)
    if ecriture:
        nom = input("Veuillez entrer un nom: ")
        file = open("Sauvegarde.py", "a+")
        file.write(f"{nom} = [\n")
        for i in donnees:
            file.write(f"{i},\n")
        file.write("]\n")


# Pour avoir un référentiel en rotation (vraiment utile pour checker les points de Lagrange)
lagrange = False


def update():  # C'est ça qui update la position dans l'interface graphique
    # update surface positions and colors
    global sp3, pos3, pos, stack, liste, k
    position()  # On calcule la nouvelle position
    if lagrange:  # Si on veut que ça tourne avec le premier objet, on applique une matrice de rotation sur toutes
        #   les positions
        theta = np.arctan2(pos[0, 1, 1], pos[0, 0, 1])
        rotation = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0 , 0, 1]])
        pos3 = np.array([(rotation @ pos[0, :, :]).T])
    else:
        pos3 = np.array([pos[0, :, :].T])
    # On ajoute les positions à stack, la matrice 3D de toutes les positions servant à grapher les trajectoires.
    stack = np.concatenate((stack, np.rot90(pos3, axes=(0, 1))), axis=1)
    for i, j in enumerate(liste):  # on update les lignes de trajectoires dans la liste
        j.setData(pos=stack[i, :, :])
    sp3.setData(pos=pos3)  # On update les positions


# A rapport avec le graphique I guess
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(1)

# Boucle de simulation (j'ai aucune idée de ce qui se passe là-dedans)
if __name__ == '__main__':
    pg.mkQApp().exec_()
