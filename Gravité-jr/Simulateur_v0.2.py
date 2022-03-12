from vpython import *
import numpy as np
from Sauvegarde import dicto
import pickle
import os
from func_lagrange import l1, l2, l3, l4, l5

obj = dicto["objets_9"]
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
# nom_fichier = input(
#     "Veuillez entrer le nom de la sauvegarde à utiliser ou enter pour utiliser le fichier déjà importé: ")

nom_fichier = "objets_9"
lagrange = True  # Pour avoir un référentiel en rotation (vraiment utile pour checker les points de Lagrange)

if nom_fichier == '':
    objets = obj
else:
    objets = liste_objets[nom_fichier]

# --- initialisation des principales variables
objets_total = objets  # les objets dans la sauvegarde, ainsi que les métadonnées qu'il pourrait éventuellement y avoir
objets = objets[0]  # Objets de la sauvegarde, sans métadonnées
temps = 0  # temps auquel la simulation est rendue
# uns = np.array([[[1, 1, 1]], [[1, 1, 1]], [[1, 1, 1]]])  # hmmm, j'ai aucune idée ce que ça fait
delta_t = 10  # pas de temps
G = 6.67430 * 10 ** (-11)  # constante gravitationnelle kg m^3/s^2 * (1 km / 1000 m)^3
# pos = []  # éventuellement une matrice de positions
# v = []  # éventuellement une matrice de vitesses


n_objets = len(objets)
vecteur_masses = []
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
tailles = 0.035101372997968866 * np.power(np.array(masse), 0.333333333)  # vecteur de taille des objets
print(f"tailles = {tailles}")
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

posf = pos_init


def acc(posi):
    global n_objets
    # On cherche à obtenir une expression de la forme a = GM/R^3 * \vec{r}

    # On concatène la matrice de position pour obtenir une matrice prismique m X 3 X m
    posQ = np.array([np.concatenate(posi, axis=0) for i in range(n_objets)])
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
    # On multiplie la matrice de distance vectorielle par celle du module de GM/R^3 pour obtenir une matrice
    #   d'accélération. En gros, ça revient à un produit d'une matrice par un vecteur, mais avec une dimension de plus.
    # On fait ensuite tourner notre vecteur pour qu'il soit dans le bon axe pour la suite des calculs.
    return np.rot90(posR @ module, k=3, axes=(2, 0))


def position():
    global pos
    global temps
    global v
    global delta_t
    global lagrange
    global posf
    temps += delta_t  # On incrémente le temps
    # On applique un énorme RK4
    k_1 = delta_t * acc(pos)
    k_2 = delta_t * acc(pos + k_1 / 2)
    k_3i = delta_t * acc(pos + k_2 / 2)
    k_3ii = delta_t * acc(pos + k_3i)
    v += (k_1 + 2 * k_2 + 2 * k_3i + k_3ii) / 6
    # data.append((pos+delta_t * v)[0, :, :])
    # À partir de ce module, on calcule la nouvelle position. On rappelle au passage qu'il s'agit d'une matrice.
    pos += delta_t * v
    if lagrange:
        theta = np.arctan2(pos[0, 1, 1], pos[0, 0, 1])
        rotation = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0 , 0, 1]])
        posf = np.array([(rotation @ pos[0, :, :])])
    else:
        posf = pos
    return posf


masse_totale = 0
# somme des masses pour calculer le barycentre du système
for i in objets:
    masse_totale += i[0]
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

couleur = []
for i in range(n_objets):
    if len(objets[0]) == 4:
        couleur.append(vector(objets[i][3][0], objets[i][3][1], objets[i][3][2]))
    else:
        couleur = vector(1, 1, 1)
couleur = np.array(couleur)
# plot les objets (points)
liste_planetes = []


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


liste_courbes = []
liste_positions = []
for i in range(n_objets):
    liste_planetes.append(sphere(pos=vector(posf[0, 0, i], posf[0, 1, i], posf[0, 2, i]), radius=tailles[i],
                                 color=couleur[i]))
    liste_positions.append(vector(posf[0, 0, i], posf[0, 1, i], posf[0, 2, i]))
position()
for i in range(n_objets):
    liste_planetes[i].pos = vector(posf[0, 0, i], posf[0, 1, i], posf[0, 2, i])
    liste_courbes.append(curve(liste_positions[i], vector(posf[0, 0, i], posf[0, 1, i], posf[0, 2, i]), color=couleur[i],
                         retain=1000))
k = 0
if lagrange:
    M_1 = masse[0]
    M_2 = masse[1]
    print(f"M_1, M_2 = {M_1}, {M_2}")
    D = np.linalg.norm(pos[0, :, 0] - pos[0, :, 1])
    print(f"D = {D}")
    L_1 = l1(M_1, M_2, D)
    L_2 = l2(M_1, M_2, D)
    L_3 = l3(M_1, M_2, D)
    L_4 = l4(M_1, M_2, D)
    L_5 = l5(M_1, M_2, D)
    sphere(pos=vector(L_1[0], L_1[1], L_1[2]), radius=250000)
    sphere(pos=vector(L_2[0], L_2[1], L_2[2]), radius=250000)
    sphere(pos=vector(L_3[0], L_3[1], L_3[2]), radius=250000)
    sphere(pos=vector(L_4[0], L_4[1], L_4[2]), radius=250000)
    sphere(pos=vector(L_5[0], L_5[1], L_5[2]), radius=250000)
while True:
    position()
    for i in range(n_objets):
        situ = vector(posf[0, 0, i], posf[0, 1, i], posf[0, 2, i])
        liste_planetes[i].pos = situ
        if k > 5:
            liste_courbes[i].append(situ)
    k += 1
    if k > 6:
        k = 0


