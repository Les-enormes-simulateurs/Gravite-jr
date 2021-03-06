import sys
import numpy as np
from Sauvegarde import objets_5 as obj
import pickle
import os
from vispy import app, scene

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
    # print(liste_objets[nom_fichier])
# --- initialisation des principales variables
objets_total = objets  # les objets dans la sauvegarde, ainsi que les métadonnées qu'il pourrait éventuellement y avoir
objets = objets[0]  # Objets de la sauvegarde, sans métadonnées
temps = 0  # temps auquel la simulation est rendue
# uns = np.array([[[1, 1, 1]], [[1, 1, 1]], [[1, 1, 1]]])  # hmmm, j'ai aucune idée ce que ça fait
delta_t = 10  # pas de temps
G = 6.67430 * 10 ** (-11)  # constante gravitationnelle kg m^3/s^2 * (1 km / 1000 m)^3
# pos = []  # éventuellement une matrice de positions
# v = []  # éventuellement une matrice de vitesses
lagrange = True

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
tailles = 0.07020274599593773 * np.power(np.array(masse), 0.333333333)  # vecteur de taille des objets
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
pos_finale = pos_init


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
    global pos_finale
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
        pos_finale = np.array([(rotation @ pos[0, :, :])])
    else:
        pos_finale = pos
    return pos_finale



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
# position()

canvas = scene.SceneCanvas(keys='interactive', size=(1200, 1200), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera()
view.camera.scale_factor = 500



# g = gl.GLGridItem()
# w.addItem(g)
# pos3 = np.zeros((100, 100, 3))
# pos3[:, :, :2] = np.mgrid[:100, :100].transpose(1, 2, 0) * [-0.1, 0.1]
# pos3 = pos3.reshape(10000, 3)
pos3 = pos[0, :, :]
pos3 = pos3.T
# ... jusqu'à ici. Ensuite, on change la couleur de nos points selon celle qui est spécifiée dans la sauvegarde importée
couleur = []
for i in range(n_objets):
    if len(objets[0]) == 4:
        couleur.append(np.array([objets[i][3][:3]]))
    else:
        couleur = [1, 1, 1]
couleur = np.array(couleur)

# data = np.concatenate([data, [[0, 0, 0]]], axis=0)
# size = np.concatenate([tailles, [100]], axis=0)
colors = np.concatenate(couleur, axis=0)



# Create and show visual
vis = scene.visuals.Markers(
    pos=pos[0, :, :].T,
    size=tailles,
    antialias=0,
    face_color=colors,
    edge_color='white',
    edge_width=0,
    scaling=True,
    spherical=True,
)
vis.parent = view.scene
# plot les objets (points)
# On ajoute les points au graphique


# Stack correspond à la matrice m X 3 X n (n = nombre d'itérations) des positions à travers le temps pour avoir les
#       trajectoires
stack = np.array([pos3])
stack = np.rot90(stack, axes=(0, 1))
liste = []  # liste des différentes trajectoires
# if len(objets_total[1]) != 0:
#     for i in objets_total[1]:
#         w.addItem(i)  # on ajoute les trajectoires en métadonnées, s'il y en a
# for i in range(n_objets):  # On met les différentes trajectoires dans le graphique
#     liste.append(gl.GLLinePlotItem(pos=stack[i, :, :],
#                                    color=list(np.array(objets[n_objets - i - 1][3]) - np.array([0, 0, 0, 0.5])),
#                                    width=1, antialias=True, mode='line_strip'))
#     w.addItem(liste[i])
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


# def update():  # C'est ça qui update la position dans l'interface graphique
#     # update surface positions and colors
#     global sp3, pos3, pos, stack, liste, k
#     position()  # On calcule la nouvelle position
#     pos3 = np.array([pos_finale[0, :, :].T])
#     # On ajoute les positions à stack, la matrice 3D de toutes les positions servant à grapher les trajectoires.
#     stack = np.concatenate((stack, np.rot90(pos3, axes=(0, 1))), axis=1)
#     for i, j in enumerate(liste):  # on update les lignes de trajectoires dans la liste
#         j.setData(pos=stack[i, :, :])
#     sp3.setData(pos=pos3)  # On update les positions


def update(ev):
    global pos, vis
    position()
    vis._data = pos[0, :, :]


timer = app.Timer()
timer.connect(update)
timer.start(0)

# Boucle de simulation (j'ai aucune idée de ce qui se passe là-dedans)
if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()



