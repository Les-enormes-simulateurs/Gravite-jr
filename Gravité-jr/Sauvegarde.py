import numpy as np
import pickle
import os


M_1 = 50.972 * 10 ** 24
M_2 = 7.34767309 * 10 ** 23
G = 6.67430 * 10 ** (-11)
objets_1 = [
    [
        (5.972 * 10 ** 24, [0, 0, 0], [0, 0, 0]),
        (7.34767309 * 10 ** 22, [0, 1750, -600], [40000000, 0, 0]),
        (1000, [0, 7700, 2000], [7000000, 0, 0]),
        (1.0 * 10 ** 21, [0, 0, -6700], [-8000000, 0, 0])
    ],
    []
]
k = 2 * np.pi / 5
d = 200000000
v = np.sqrt(6.67430 * 10 ** (-11) * 6.0 * 10 ** 27 / d)
objets_2 = [
    [
        (6.0 * 10 ** 27, [0, 0, 0], [0, 0, 0]),
        (6.0 * 10 ** 24, [0, v, 0], [d, 0, 0]),
        (6.0 * 10 ** 24, [v * np.cos(k + np.pi / 2), v * np.sin(k + np.pi / 2), 0], [d * np.cos(k), d * np.sin(k), 0]),
        (6.0 * 10 ** 24, [v * np.cos(2 * k + np.pi / 2), v * np.sin(2 * k + np.pi / 2), 0],
         [d * np.cos(2 * k), d * np.sin(2 * k), 0]),
        (6.0 * 10 ** 24, [v * np.cos(3 * k + np.pi / 2), v * np.sin(3 * k + np.pi / 2), 0],
         [d * np.cos(3 * k), d * np.sin(3 * k), 0]),
        # (6.0 * 10 ** 24, [v*np.cos(4*k + np.pi/2), v*np.sin(4*k + np.pi/2), 0], [d * np.cos(4 * k), d * np.sin(4 * k), 0]),
    ],
    []
]
objets_3 = [
    [
        (5.972 * 10 ** 24, [0, 0, 0], [0, 0, 0]),
        (3.34767309 * 10 ** 23, [0, 900, -400], [80000000, 0, 0]),
        (3.34767309 * 10 ** 21, [0, 5700, -400], [81000000, 0, 0]),
        (1000, [0, 7440, 2000], [7000000, 0, 0]),
        (1.0 * 10 ** 22, [0, 0, 6700], [8000000, 0, 0])
    ],
    []
]

objets_4 = [
    [
        (5.972 * 10 ** 24, [0, 0, 0], [0, 0, 0]),
        (7.34767309 * 10 ** 23, [0, 1750, -600], [40000000, 0, 0]),
        (1000, [0, 7700, 2000], [7000000, 0, 0]),
        (1.0 * 10 ** 21, [0, 0, -6700], [-8000000, 0, 0]),
        (1.0 * 10 ** 23, [5000, 0, 5000], [0, 10000000, 0])
    ],
    []
]
objets_5 = [
    [
        (5.972 * 10 ** 24, [0, 0, 0], [0, 0, 0], [0, 0, 1, 1]),
        (7.34767309 * 10 ** 23, [0, 1750, -600], [40000000, 0, 0], [0.5, 0.5, 0.5, 1]),
        (1000, [0, 7700, 2000], [7000000, 0, 0], [0, 1, 0, 1]),
        (1.0 * 10 ** 21, [0, 0, -6700], [-8000000, 0, 0], [1, 0, 1, 1]),
        (1.0 * 10 ** 23, [5000, 0, 5000], [0, 10000000, 0], [0, 1, 1, 1]),
        (1.0 * 10 ** 21, [0, -7700, -2000], [-7000000, 0, 0], [1, 0, 1, 1]),
    ],
    []
]
D = 80000000
P = np.sqrt(4* np.pi**2*D**3/ G / (M_1 + M_2))
r = (1 - M_2 / (M_1 + M_2)) * D
R = D - r
v = 2 * np.pi * r / P
V = 2 * np.pi * R / P
r_cm = np.sqrt((D * (M_1 - M_2)/(M_1 + M_2) / 2 - R)**2 + (D * np.sqrt(3) / 2)**2)
v_particule = 2* np.pi* r_cm / P
theta = np.arctan((D * np.sqrt(3) / 2) / (D * (M_1 - M_2)/(M_1 + M_2) / 2))
objets_6 = [
    [
        (M_1, [0, -V, 0], [-R, 0, 0], [0, 0, 1, 1]),
        (M_2, [0, v, 0], [r, 0, 0], [0.5, 0.5, 0.5, 1]),
        (1, [-v_particule*np.sin(theta), v_particule*np.cos(theta), 0], [D * (M_1 - M_2)/(M_1 + M_2) / 2, D * np.sqrt(3) / 2, 100000], [0, 1, 0, 1]),
    ],
    []
]
os.remove("Sauvegarde.data")
dicto = {"objets_1": objets_1, "objets_2": objets_2, "objets_3": objets_3, "objets_4": objets_4, "objets_5": objets_5, "objets_6": objets_6}
fw = open("Sauvegarde.data", 'wb')
pickle.dump(dicto, fw)
fw.close()

print(np.sqrt((D * (M_1 - M_2)/(M_1 + M_2) / 2)**2 + (D * np.sqrt(3) / 2)**2))
print(f"v = {v}")
print(f"v_particule = {v_particule}")
k = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(np.sqrt(k))


