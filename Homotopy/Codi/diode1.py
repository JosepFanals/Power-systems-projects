# homotopia i newton-raphson
# circuit simple amb díode i resistència

import numpy as np
import random

Is = 1e-10
n = 1 / 0.026  # la inversa de Vt
R = 10.0
Y = 1 / R
E = 5.0

# paràmetres inventats per a l'homotopia
a2 = 4.0
G2 = 1e-3

J = np.zeros((3, 3), dtype=float)
n_iter = 5  # número establert d'iteracions a cada etapa de l'homotopia
passos = 1000  # número de valors intermedis de lambda

# valors de l'etapa inicial de l'homotopia, lambda = 0
x1 = E
x2 = a2
x3 = - Is * np.exp(n * (x1 - x2)) + Is

# guardar els valors
vec_x1 = []
vec_x2 = []
vec_x3 = []

for lam in range(1, passos + 1, 1):
    lam = lam / passos

    for c in range(n_iter):
        J[0, 0] = 1
        J[0, 1] = 0
        J[0, 2] = 0

        J[1, 0] = - Is * np.exp(n * (x1 - x2)) * n
        J[1, 1] = Y - Is * np.exp(n * (x1 - x2)) * (-n) + (1 - lam) / lam * G2
        J[1, 2] = 0

        J[2, 0] = Is * np.exp(n * (x1 - x2)) * n
        J[2, 1] = Is * np.exp(n * (x1 - x2)) * (-n)
        J[2, 2] = 1

        f1 = x1 - E
        f2 = Y * x2 - Is * np.exp(n * (x1 - x2)) + Is + (x2 - a2) * (1 - lam) / lam * G2
        f3 = x3 + Is * np.exp(n * (x1 - x2)) - Is

        f = np.block([f1, f2, f3])
        J1 = np.linalg.inv(J)
        Ax = - np.dot(J1, f)
        
        x1 += Ax[0]
        x2 += Ax[1]
        x3 += Ax[2]

    vec_x1.append(x1)
    vec_x2.append(x2)
    vec_x3.append(x3)

    print(lam, max(abs(f1), abs(f2), abs(f3)))  # error màxim 
