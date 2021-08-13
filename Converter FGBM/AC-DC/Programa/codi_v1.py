import numpy as np

Pdc = 0.1
Rdc = 10
Xc = 0.5
U = 1.05
a = 0.8
k1 = 0.995 * 3 * np.sqrt(2) / np.pi
V0 = 0.0
G = 10
B = -10

Idc = np.sqrt(Pdc / Rdc)
Vdc = Idc * Rdc

Vre = Vdc / (k1 * a)

prof = 60

V = np.zeros(prof)
alpha = np.zeros(prof)
Ure = np.zeros(prof)
Uim = np.zeros(prof)
Vim = np.zeros(prof)


# Termes [0]

V[0] = np.sqrt(Vre * Vre + V0)
alpha[0] = (Vdc + 3 * Xc * Idc / np.pi) / (k1 * a * V[0])
Ure[0] = (a * k1 * Idc + G * Vre) / G
Vim[0] = (-B * Ure[0] + B * Vre) / (-G)
Uim[0] = np.sqrt(U * U + Ure[0] * Ure[0])  # mirar si agafar l'arrel positiva o negativa

# Termes [1]
V[1] = (Vim[0] * Vim[0] - V0) / (2 * V[0])
alpha[1] = (-V[1] * alpha[0]) / V[0]
Ure[1] = (B * Uim[0] - B * Vim[0]) / G
Vim[1] = (G * Uim[0] + B * Ure[1]) / G
Uim[1] = (-Ure[0] * Ure[1]) / Uim[0]

# Termes [c>=2]


def conv(a, b, j, lim, rest):
    suma = 0
    for k in range(j, lim + 1):
        suma += a[k] * b[lim - k - rest]
    return suma


for c in range(2, prof):
    V[c] = (-conv(V, V, 1, c-1, 0) + conv(Vim, Vim, 0, c-1, 1)) / (2 * V[0])
    alpha[c] = (-conv(alpha, V, 0, c-1, 0)) / V[0]
    Ure[c] = (B * Uim[c-1] - B * Vim[c-1]) / G
    Vim[c] = (G * Uim[c-1] + B * Ure[c]) / G
    Uim[c] = (-conv(Ure, Ure, 0, c, 0) - conv(Uim, Uim, 1, c-1, 0)) / (2 * Uim[0])

from Pade import pade

V_fi = pade(prof-1, V, 1)
Ure_fi = pade(prof-1, Ure, 1)
Uim_fi = pade(prof-1, Uim, 1)
alpha_fi = pade(prof-1, alpha, 1)
Vim_fi = pade(prof-1, Vim, 1)

print(V_fi)
print(Ure_fi)
print(Uim_fi)
print(alpha_fi)
print(Vim_fi)
print(Vre)
print((Vdc + 3 * Xc * Idc / np.pi) / (k1 * a))
print(Idc)
print(Vdc)

print(V)
print(alpha)