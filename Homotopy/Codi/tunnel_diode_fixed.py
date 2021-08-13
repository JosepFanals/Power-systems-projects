# Homotopia fixed point del circuit de tunnel diodes de la tesi de Delia Torres Muñoz
# de moment no funciona


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# invento solucions
x10 = 5  # 5, voltatge gran
x20 = -3  # 0, voltatge petit
lam = 0.0
n_punts = 20000  # nombre de punts de la trajectòria, inventat, 16000
n_iter = 15  # vegades que vull que iteri per trobar la solució
r = 0.001  # radi arbitrari, ha de ser suficientment petit, 0.001

g1 = 1e-6
g2 = 1e-6
a1 = x10
a2 = x20

c1 = x10  # centre inicial de x1
c2 = x20  # centre inicial de x2
c3 = lam  # lambda inicial
x1 = x10  # valors de partida
x2 = x20  # valors de partida

R = 13.3
E = 30


def H1(x1, x2, lam):  # funció de la primera eq.
    valor = lam * (2.5 * x1 ** 3 - 10.5 * x1 ** 2 + x1 * (11.8 + 1 / R) + x2 * 1 / R - E / R) + (1 - lam) * g1 * (x1 - a1)
    return valor


def H2(x1, x2, lam):  # funció de la segona eq.
    valor = lam * (-2.5 * x1 ** 3 + 10.5 * x1 ** 2 - 11.8 * x1 + 0.43 * x2 ** 3 - 2.69 * x2 ** 2 + 4.56 * x2) + (1 - lam) * g2 * (x2 - a2)
    return valor


def H3(x1, x2, lam, c1, c2, c3, r):
    valor = (x1 - c1) ** 2 + (x2 - c2) ** 2 + (lam - c3) ** 2 - r ** 2
    return valor


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

    # etapa predictiva
    if c == 0:
        lam += r / 2  # és una bona inicialització, sabem que ha de créixer just al principi
    else:
        x1 += Apx1
        x2 += Apx2
        lam += Aplam


    for k in range(n_iter):
        J[0, 0] = lam * (7.5 * x1 ** 2 - 21 * x1 + 11.8 + 1 / R) + (1 - lam) * g1
        J[0, 1] = lam * 1 / R
        J[0, 2] = 2.5 * x1 ** 3 - 10.5 * x1 ** 2 + x1 * (11.8 + 1 / R) + x2 * 1 / R - E / R - g1 * (x1 - a1)

        J[1, 0] = lam * (-7.5 * x1 ** 2 + 21 * x1 - 11.8)
        J[1, 1] = lam * (1.29 * x2 ** 2 - 5.38 * x2 + 4.56) + (1 - lam) * g2
        J[1, 2] = -2.5 * x1 ** 3 + 10.5 * x1 ** 2 - 11.8 * x1 + 0.43 * x2 ** 3 - 2.69 * x2 ** 2 + 4.56 * x2 - g2 * (x2 - a2)

        J[2, 0] = 2 * (x1 - c1)
        J[2, 1] = 2 * (x2 - c2)
        J[2, 2] = 2 * (lam - c3)

        h1 = H1(x1, x2, lam)
        h2 = H2(x1, x2, lam)
        h3 = H3(x1, x2, lam, c1, c2, c3, r)
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
    print(vec_error[-1])

# solucions de les incògnites, miro el punt de tall amb lambda = 1
sol_x1 = []
sol_x2 = []
for i in range(len(vec_lam) - 1):
    if (vec_lam[i] > 1 and vec_lam[i + 1] < 1) or (vec_lam[i] < 1 and vec_lam[i + 1] > 1):
        lam1 = vec_lam[i]
        lam2 = vec_lam[i + 1]
        sol_x1.append((vec_x1[i + 1] - vec_x1[i] + lam2 * vec_x1[i] - lam1 * vec_x1[i + 1]) / (lam2 - lam1))
        sol_x2.append((vec_x2[i + 1] - vec_x2[i] + lam2 * vec_x2[i] - lam1 * vec_x2[i + 1]) / (lam2 - lam1))

# solucions
print(sol_x1)
print(sol_x2)

# errors
for i in range(len(sol_x1)):
    print(abs(H1(sol_x1[i], sol_x2[i], 1)))
    print(abs(H2(sol_x1[i], sol_x2[i], 1)))

sub1 = plt.subplot(1, 3, 1)
sub2 = plt.subplot(1, 3, 2)  # v2
sub12 = plt.subplot(1, 3, 3)   # v2(v1)
sub1.plot(vec_lam, vec_x1)
sub2.plot(vec_lam, vec_x2)
sub12.plot(vec_x1, vec_x2)
plt.show() 

