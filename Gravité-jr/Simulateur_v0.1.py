# -*- coding: utf-8 -*-
"""
Demonstrates use of GLScatterPlotItem with rapidly-updating plots.

"""

## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from Sauvegarde import objets_5 as obj
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

objets_total = objets
objets = objets[0]
temps = 0
uns = np.array([[[1, 1, 1]], [[1, 1, 1]], [[1, 1, 1]]])
delta_t = 10
G = 6.67430 * 10 ** (-11)
pos = []
v = []
n_objets = 0
liste = []


n_objets = len(objets)
vecteur_masses = []
print(n_objets)
for i in range(n_objets):
    vecteur_masses.append(objets[i][0])
vecteur_masses = np.array(vecteur_masses).T
masse = []
masses = []
for i in range(len(objets)):
    masse.append(objets[i][0])
for i in range(len(objets)):
    masses.append(masse[0:i] + [0] + masse[i + 1:])
tailles = 1.0668117883456128e-18 * np.array(masse)
masses = G * np.array(masses)
pos = np.array([objets[0][2]]).T
for i in range(1, len(objets)):
    pos = np.concatenate((pos, np.array([objets[i][2]]).T), 1)
pos_init = np.array([pos], dtype='float64')
pos = pos_init
v = np.array([objets[0][1]]).T
for i in range(1, len(objets)):
    v = np.concatenate((v, np.array([objets[i][1]]).T), 1)
v_init = np.array([v], dtype='float64')
v = v_init


def position():
    global pos
    global temps
    global v
    global n_objets
    global delta_t
    global liste
    temps += delta_t
    posQ = np.array([np.concatenate(pos, axis=0) for i in range(n_objets)])
    posR = posQ - np.rot90(posQ, k=1, axes=(2, 0))
    posR_carre = np.absolute(np.square(posR))
    uns_plus_posR = np.identity(n_objets) + posR_carre[:, 0, :] + posR_carre[:, 1, :] + posR_carre[:, 2, :]
    module = np.array([masses * np.power(uns_plus_posR, -1.5)])
    module = np.rot90(module, axes=(1, 0))
    module = np.rot90(module, axes=(2, 1))
    acc = posR @ module
    acc = np.rot90(acc, k=3, axes=(2, 0))
    v += delta_t * acc
    # data.append((pos+delta_t * v)[0, :, :])
    pos += delta_t * v
    return pos

masse_totale = 0
for i in objets:
    masse_totale += i[0]
print(f"pos = {pos}")
print(f"vecteur_masses = {vecteur_masses}")
barycentre_1 = pos[0, :, :] @ vecteur_masses / masse_totale
position()
barycentre_2 = pos[0, :, :] @ vecteur_masses / masse_totale
v_barycentre = np.array([((barycentre_2 - barycentre_1) / delta_t)]).T
v_barycentre = np.array([np.concatenate(v_barycentre, axis=0) for i in range(n_objets)])
v = v_init - np.array([v_barycentre.T])
barycentre_1 = np.array([np.array([np.concatenate(np.array([barycentre_1]).T, axis=0) for i in range(n_objets)]).T])
pos = pos_init - barycentre_1
position()
app = pg.mkQApp("GLScatterPlotItem Example")
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

g = gl.GLGridItem()
w.addItem(g)
pos3 = np.zeros((100, 100, 3))
pos3[:, :, :2] = np.mgrid[:100, :100].transpose(1, 2, 0) * [-0.1, 0.1]
pos3 = pos3.reshape(10000, 3)
print(pos3)
pos3 = pos[0, :, :]
pos3 = pos3.T
couleur = []
for i in range(n_objets):
    if len(objets[0]) == 4:
        couleur.append(objets[i][3])
    else:
        couleur = [1, 1, 1, .3]
couleur = np.array(couleur)
sp3 = gl.GLScatterPlotItem(pos=pos3, color=couleur, size=tailles, pxMode=False)

w.addItem(sp3)
stack = np.array([pos3])
stack = np.rot90(stack, axes=(0, 1))
print(f"stack = {stack}")
liste = []
if len(objets_total[1]) != 0:
    for i in objets_total[1]:
        w.addItem(i)
for i in range(n_objets):
    liste.append(gl.GLLinePlotItem(pos=stack[i, :, :],
                                   color=list(np.array(objets[n_objets - i - 1][3]) - np.array([0, 0, 0, 0.5])),
                                   width=1, antialias=True, mode='line_strip'))
    w.addItem(liste[i])
k = 0


def save(ecriture=False):
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


def update():
    # update surface positions and colors
    global sp3, pos3, pos, stack, liste, k
    position()
    pos3 = pos[0, :, :]
    pos3 = pos3.T
    stackage = np.array([pos3])
    stack = np.concatenate((stack, np.rot90(stackage, axes=(0, 1))), axis=1)
    if k < 4:
        # print(stack)
        k += 1
    for i, j in enumerate(liste):
        j.setData(pos=stack[i, :, :])
    sp3.setData(pos=pos3)


t = QtCore.QTimer()
t.timeout.connect(update)
t.start(1)


if __name__ == '__main__':
    pg.mkQApp().exec_()
