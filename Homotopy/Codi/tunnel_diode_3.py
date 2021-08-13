# ja he trobat les 9 solucions, ara veure com millorar-lo per tal que no marxi tant
# díodes tunnel que tenen 9 solucions, a partir del cas de la tesi de Delia Torres Muñoz

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# invento solucions
x10 = 5  # 3
x20 = 0.0  # 0.5
lam = 0.0
n_punts = 16000  # nombre de punts de la trajectòria, inventat, 16000
n_iter = 15  # vegades que vull que iteri per trobar la solució
r = 0.001  # radi arbitrari, ha de ser suficientment petit, 0.001

c1 = x10  # centre inicial de x1
c2 = x20  # centre inicial de x2
c3 = lam  # lambda inicial
x1 = x10  # valors de partida
x2 = x20  # valors de partida


def F1(x1, x2):  # funció de la primera eq.
    valor = 2.5 * x1 ** 3 - 10.5 * x1 ** 2 + 11.8 * x1 - 0.43 * x2 ** 3 + 2.69 * x2 ** 2 - 4.56 * x2
    return valor


def F2(x1, x2):  # funció de la segona eq.
    valor = -33.25 * x1 ** 3 + 139.65 * x1 ** 2 - 157.94 * x1 - x2 + 30
    return valor


def F3(x1, x2, lam, c1, c2, c3, r):
    valor = (x1 - c1) ** 2 + (x2 - c2) ** 2 + (lam - c3) ** 2 - r ** 2
    return valor


# funcions avaluades al punt de partida
F10 = F1(x10, x20)
F20 = F2(x10, x20)

J = np.zeros((3, 3), dtype=float)

vec_x1 = []
vec_x2 = []
vec_lam = []
vec_error = []

# increments passats
Apx1 = 0
Apx2 = 0
Aplam = 0

for c in range(n_punts):
    # guardo valors del punt inicial
    x1_i = x1
    x2_i = x2
    lam_i = lam

    # etapa predictiva, a millorar
    if c == 0:
        x1 -= r / 2  # triar la direcció, arbitrari
    else:
        x1 += Apx1
        x2 += Apx2
        lam += Aplam


    for k in range(n_iter):
        J[0, 0] = 7.5 * x1 ** 2 - 21 * x1 + 11.8
        J[0, 1] = -1.29 * x2 ** 2 + 5.38 * x2 - 4.56
        J[0, 2] = F10

        J[1, 0] = -99.75 * x1 ** 2 + 279.3 * x1 - 157.94
        J[1, 1] = -1
        J[1, 2] = F20

        J[2, 0] = 2 * (x1 - c1)
        J[2, 1] = 2 * (x2 - c2)
        J[2, 2] = 2 * (lam - c3)

        h1 = F1(x1, x2) - (1 - lam) * F10
        h2 = F2(x1, x2) - (1 - lam) * F20
        h3 = F3(x1, x2, lam, c1, c2, c3, r)
        f = np.block([h1, h2, h3])
        J1 = np.linalg.inv(J)
        Ax = - np.dot(J1, f)

        x1 += Ax[0]
        x2 += Ax[1]
        lam += Ax[2]

    Apx1 = x1 - x1_i
    Apx2 = x2 - x2_i
    Aplam = lam - lam_i
    c1 = x1
    c2 = x2
    c3 = lam
    vec_x1.append(x1)
    vec_x2.append(x2)
    vec_lam.append(lam)
    vec_error.append(max(abs(h1), abs(h2), abs(h3)))  # error màxim 
    print(lam)
    #print(vec_error[-1])

# solucions de les incògnites, miro el punt de tall amb lambda = 1
sol_x1 = []
sol_x2 = []
for i in range(len(vec_lam) - 1):
    if (vec_lam[i] > 1 and vec_lam[i + 1] < 1) or (vec_lam[i] < 1 and vec_lam[i + 1] > 1):
        lam1 = vec_lam[i]
        lam2 = vec_lam[i + 1]
        sol_x1.append((vec_x1[i + 1] - vec_x1[i] + lam2 * vec_x1[i] - lam1 * vec_x1[i + 1]) / (lam2 - lam1))
        sol_x2.append((vec_x2[i + 1] - vec_x2[i] + lam2 * vec_x2[i] - lam1 * vec_x2[i + 1]) / (lam2 - lam1))
        # sol_x1.append((vec_x1[i] + vec_x1[i + 1]) / 2)
        # sol_x2.append((vec_x2[i] + vec_x2[i + 1]) / 2)

# solucions
print(sol_x1)
print(sol_x2)

# errors
for i in range(len(sol_x1)):
    print(abs(F1(sol_x1[i], sol_x2[i])))
    print(abs(F2(sol_x1[i], sol_x2[i])))


sub2 = plt.subplot(2, 1, 1) 
sub12 = plt.subplot(2, 1, 2) 
sub2.plot(vec_lam, vec_x2)
sub12.plot(vec_x1, vec_x2)
plt.show() 

