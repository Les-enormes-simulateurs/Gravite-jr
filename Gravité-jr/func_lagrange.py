import numpy as np


def l1(m_1, m_2, d):
    s_0, s_1 = -1, 1
    a = m_2 / (m_1 + m_2)
    b = m_1 / (m_1 + m_2)
    k = np.roots([1, (3 - a), (3 - 2 * a), (1 - s_1 - a * (1 + s_0 - s_1)), - 2 * a * s_0, - a * s_0])
    i = k[abs(k.imag) < 1e-5]
    return [d * (i[min(range(len(i.imag)), key=i.imag.__getitem__)].real + b), 0 , 0]


def l2(m_1, m_2, d):
    s_0, s_1 = 1, 1
    a = m_2 / (m_1 + m_2)
    b = m_1 / (m_1 + m_2)
    k = np.roots([1, (3 - a), (3 - 2 * a), (1 - s_1 - a * (1 + s_0 - s_1)), - 2 * a * s_0, - a * s_0])
    i = k[abs(k.imag) < 1e-5]
    return [d * (i[min(range(len(i.imag)), key=i.imag.__getitem__)].real + b), 0, 0]


def l3(m_1, m_2, d):
    s_0, s_1 = -1, -1
    a = m_2 / (m_1 + m_2)
    b = m_1 / (m_1 + m_2)
    k = np.roots([1, (3 - a), (3 - 2 * a), (1 - s_1 - a * (1 + s_0 - s_1)), - 2 * a * s_0, - a * s_0])
    i = k[abs(k.imag) < 1e-5]
    return [d * (i[min(range(len(i.imag)), key=i.imag.__getitem__)].real + b), 0, 0]


def l4(m_1, m_2, d):
    return [d * (0.5 - m_2 / m_1), d * np.sqrt(3) / 2, 0]


def l5(m_1, m_2, d):
    return [d * (0.5 - m_2 / m_1), d * np.sqrt(3) / 2, 0]


M_1 = 5000.972e24
M_2 = 7.34767309 * 10 ** 23
D = 80000000

L_1 = D - D * np.cbrt(M_2 / (3 * (M_1 + M_2)))
L_2 = D + D * np.cbrt(M_2 / (3 * (M_1 + M_2)))
L_3 = - D - D * 5 * M_2 / (12 * (M_1 + M_2))

# print(f"L_1 = {L_1}")
# print(f"L_2 = {L_2}")
# print(f"L_3 = {L_3}")
# print(f"L1 = {l1(M_1, M_2, D)}")
# print(f"L2 = {l2(M_1, M_2, D)}")
# print(f"L3 = {l3(M_1, M_2, D)}")
