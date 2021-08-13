# Homotopia DBH de tunnel diodes de la tesi de Delia Torres Muñoz


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

lam = 0.5
x1 = 3
x2 = 1
c1 = x1
c2 = x2
c3 = lam

n_punts = 16  # nombre de punts de la trajectòria, inventat, 16000
n_iter = 5  # vegades que vull que iteri per trobar la solució
r = 0.001  # radi arbitrari, ha de ser suficientment petit, 0.001

E = 30
R = 13.3


def F1(x1, x2):
    valor = E - R * (2.5 * x1 ** 3 - 10.5 * x1 ** 2 + 11.8 * x1) - (x1 + x2)
    return valor


def F2(x1, x2):
    valor = 2.5 * x1 ** 3 - 10.5 * x1 ** 2 + 11.8 * x1 - 0.43 * x2 ** 3 + 2.69 * x2 ** 2 - 4.56 * x2
    return valor


def H1(x1, x2, lam):
    valor = 40 * lam * (lam - 1) + np.exp(lam * (lam - 1)) * np.log(0.01 * (F1(x1, x2)) ** 2 + 1)
    return valor


def H2(x1, x2, lam):
    valor = 40 * lam * (lam - 1) + np.exp(lam * (lam - 1)) * np.log(0.01 * (F2(x1, x2)) ** 2 + 1)
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
        # lam += r / 2  # és una bona inicialització, sabem que ha de créixer just al principi
        x1 += r / 2
    else:
        x1 += Apx1
        x2 += Apx2
        lam += Aplam

    for k in range(n_iter):
        print(lam)
        J[0, 0] = np.exp(lam * (lam - 1)) * 1 / (0.01 * F1(x1, x2) ** 2 + 1) * 0.02 * F1(x1, x2) * (-R * (7.5 * x1 ** 2 - 21 * x1 + 11.8) - 1)
        J[0, 1] = np.exp(lam * (lam - 1)) * 1 / (0.01 * F1(x1, x2) ** 2 + 1) * 0.02 * F1(x1, x2) * (-1)
        J[0, 2] = 40 * (2 * lam - 1) + np.log(0.01 * F1(x1, x2) ** 2 + 1) * np.exp(lam * (lam - 1)) * (2 * lam - 1)

        J[1, 0] = np.exp(lam * (lam - 1)) * 1 / (0.01 * F2(x1, x2) ** 2 + 1) * 0.02 * F2(x1, x2) * (7.5 * x1 ** 2 - 21 * x1 + 11.8)
        J[1, 1] = np.exp(lam * (lam - 1)) * 1 / (0.01 * F2(x1, x2) ** 2 + 1) * 0.02 * F2(x1, x2) * (-1.29 * x2 ** 2 + 5.38 * x2 - 4.56)
        J[1, 2] = 40 * (2 * lam - 1) + np.log(0.01 * F2(x1, x2) ** 2 + 1) * np.exp(lam * (lam - 1)) * (2 * lam - 1)

        J[2, 0] = 2 * (x1 - c1)
        J[2, 1] = 2 * (x2 - c2)
        J[2, 2] = 2 * (lam - c3)

        h1 = H1(x1, x2, lam)
        h2 = H2(x1, x2, lam)
        h3 = H3(x1, x2, lam, c1, c2, c3, r)
        f = np.block([h1, h2, h3])
        #print(J)
        J1 = np.linalg.inv(J)
        Ax = - np.dot(J1, f)

        print(max(abs(h1), abs(h2), abs(h3)))

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