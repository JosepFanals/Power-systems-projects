# homotopia i newton-raphson
# trigger schmitt com el del pdf
# ara soluciono H(x, lambda) = 0 amb la fixed point

import numpy as np
import random
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt

# dades d'entrada
Vcc = 10.0
R1 = 10e3
R2 = 5e3
R3 = 1.25e3
R4 = 1e6
Rc1 = 1.5e3
Rc2 = 1e3
Re = 100
af = 0.99
ar = 0.5
me = -1e-16 / af
mc = -1e-16 / ar
n = -38.78

# paràmetres inventats per a l'homotopia, podrien ser aleatoris
a1 = 0.5
a2 = 0.5
a3 = 0.5
a4 = 0.5
a5 = 0.5
a6 = 0.5
a7 = 0.5
G1 = 1e-3
G2 = 1e-3
G3 = 1e-3
G4 = 1e-3
G5 = 1e-3
G6 = 1e-3
G7 = 1e-3

# objectes per iterar
J = np.zeros((7, 7), dtype=float)
n_iter = 10  # número establert d'iteracions a cada etapa de l'homotopia
passos = 1000  # número de valors intermedis de lambda

# valors de l'etapa inicial de l'homotopia, lambda = 0.
x1 = a1
x2 = a2 
x3 = a3
x4 = a4
x5 = a5
x6 = Vcc
x7 = - (x6 - x2) / Rc1 - (x6 - x3) / Rc2 - (x6 - x5) / R2

# ara entrar al bucle, muntar jacobià, ja tinc les expressions a mà

start_time = time.time()

vec_x1 = []
vec_x2 = []
vec_x3 = []
vec_x4 = []
vec_x5 = []
vec_x6 = []
vec_x7 = []
vec_lam = []

for lam in range(1, passos + 1, 1):
    lam = lam / passos

    for c in range(n_iter):
        # jacobià
        J[0, 0] = 1 / Re + me * (np.exp(n * (x1 - x5))) * n + me * (np.exp(n * (x1 - x4))) * n + (1 - lam) / lam * G1
        J[0, 1] = - ar * mc * (np.exp(n * (x2 - x5))) * n
        J[0, 2] = - ar * mc * (np.exp(n * (x3 - x4))) * n
        J[0, 3] = me * (np.exp(n * (x1 - x4))) * (-n) - ar * mc * (np.exp(n * (x3 - x4))) * (-n)
        J[0, 4] = me * (np.exp(n * (x1 - x5))) * (-n) - ar * mc * (np.exp(n * (x2 - x5))) * (-n)
        J[0, 5] = 0
        J[0, 6] = 0

        J[1, 0] = - af * me * (np.exp(n * (x1 - x5))) * n
        J[1, 1] = 1 / R1 + 1 / Rc1 + mc * (np.exp(n * (x2 - x5))) * n + (1 - lam) / lam * G2
        J[1, 2] = 0
        J[1, 3] = -1 / R1
        J[1, 4] = - af * me * (np.exp(n * (x1 - x5))) * (-n) + mc * (np.exp(n * (x2 - x5))) * (-n)
        J[1, 5] = -1 / Rc1
        J[1, 6] = 0

        J[2, 0] = -af * me * (np.exp(n * (x1 - x4))) * n
        J[2, 1] = 0
        J[2, 2] = 1 / Rc2 + mc * (np.exp(n * (x3 - x4))) * n + (1 - lam) / lam * G3
        J[2, 3] = -af * me * (np.exp(n * (x1 - x4))) * (-n) + mc * (np.exp(n * (x3 - x4))) * (-n)
        J[2, 4] = 0
        J[2, 5] = -1 / Rc2
        J[2, 6] = 0

        J[3, 0] = -me * (np.exp(n * (x1 - x4))) * n + af * me * (np.exp(n * (x1 - x4))) * n
        J[3, 1] = -1 / R1
        J[3, 2] = ar * mc * (np.exp(n * (x3 - x4))) * n - mc * (np.exp(n * (x3 - x4))) * n
        J[3, 3] = 1 / R1 + 1 / R4 - me * (np.exp(n * (x1 - x4))) * (-n) + ar * mc * (np.exp(n * (x3 - x4))) * (-n) + af * me * (np.exp(n * (x1 - x4))) * (-n) - mc * (np.exp(n * (x3 - x4))) * (-n) + (1 - lam) / lam * G4
        J[3, 4] = 0
        J[3, 5] = 0
        J[3, 6] = 0

        J[4, 0] = af * me * (np.exp(n * (x1 - x5))) * n - mc * (np.exp(n * (x1 - x5))) * n
        J[4, 1] = -mc * (np.exp(n * (x2 - x5))) * n + ar * mc * (np.exp(n * (x2 - x5))) * n
        J[4, 2] = 0
        J[4, 3] = 0
        J[4, 4] = 1 / R2 + 1 / R3 + af * me * (np.exp(n * (x1 - x5))) * (-n) - mc * (np.exp(n * (x2 - x5))) * (-n) - mc * (np.exp(n * (x1 - x5))) * (-n) + ar * mc * (np.exp(n * (x2 - x5))) * (-n) + (1 - lam) / lam * G5
        J[4, 5] = -1 / R2
        J[4, 6] = 0

        J[5, 0] = 0
        J[5, 1] = -1 / Rc1
        J[5, 2] = -1 / Rc2
        J[5, 3] = 0
        J[5, 4] = -1 / R2
        J[5, 5] = 1 / Rc1 + 1 / Rc2 + 1 / R2
        J[5, 6] = 1

        J[6, 0] = 0
        J[6, 1] = 0
        J[6, 2] = 0
        J[6, 3] = 0
        J[6, 4] = 0
        J[6, 5] = 1
        J[6, 6] = 0

        # mismatches
        f1 = x1 / Re + me * (np.exp(n * (x1 - x5)) - 1) - ar * mc * (np.exp(n * (x2 - x5)) - 1) + me * (np.exp(n * (x1 - x4)) - 1) - ar * mc * (np.exp(n * (x3 - x4)) - 1) + (x1 - a1) * (1 - lam) / lam * G1
        f2 = (x2 - x4) / R1 + (x2 - x6) / Rc1 - af * me * (np.exp(n * (x1 - x5)) - 1) + mc * (np.exp(n * (x2 - x5)) - 1) + (x2 - a2) * (1 - lam) / lam * G2
        f3 = (x3 - x6) / Rc2 - af * me * (np.exp(n * (x1 - x4)) - 1) + mc * (np.exp(n * (x3 - x4)) - 1) + (x3 - a3) * (1 - lam) / lam * G3
        f4 = (x4 - x2) / R1 + x4 / R4 - me * (np.exp(n * (x1 - x4)) - 1) + ar * mc * (np.exp(n * (x3 - x4)) - 1) + af * me * (np.exp(n * (x1 - x4)) - 1) - mc * (np.exp(n * (x3 - x4)) - 1) + (x4 - a4) * (1 - lam) / lam * G4
        f5 = (x5 - x6) / R2 + x5 / R3 + af * me * (np.exp(n * (x1 - x5)) - 1) - mc * (np.exp(n * (x2 - x5)) - 1) - mc * (np.exp(n * (x1 - x5)) - 1) + ar * mc * (np.exp(n * (x2 - x5)) - 1) + (x5 - a5) * (1 - lam) / lam * G5
        f6 = (x6 - x2) / Rc1 + (x6 - x3) / Rc2 + (x6 - x5) / R2 + x7
        f7 = x6 - Vcc

        # actualitzar iteració
        f = np.block([f1, f2, f3, f4, f5, f6, f7])
        J1 = np.linalg.inv(J)
        Ax = - np.dot(J1, f)
        
        x1 += Ax[0]
        x2 += Ax[1]
        x3 += Ax[2]
        x4 += Ax[3]
        x5 += Ax[4]
        x6 += Ax[5]
        x7 += Ax[6]

    print(lam, max(abs(f1), abs(f2), abs(f3), abs(f4), abs(f5), abs(f6), abs(f7)))  # error màxim 
    vec_x1.append(x1)
    vec_x2.append(x2)
    vec_x3.append(x3)
    vec_x4.append(x4)
    vec_x5.append(x5)
    vec_x6.append(x6)
    vec_x7.append(x7)
    vec_lam.append(lam)


# resultats
end_time = time.time()
print('Temps: ', end_time - start_time, 'segons')
print('Solucions: ', x1, x2, x3, x4, x5, x6, x7)

sub1 = plt.subplot(3, 3, 1) 
sub2 = plt.subplot(3, 3, 2) 
sub3 = plt.subplot(3, 3, 3) 
sub4 = plt.subplot(3, 3, 4) 
sub5 = plt.subplot(3, 3, 5) 
sub6 = plt.subplot(3, 3, 6) 
sub7 = plt.subplot(3, 3, 7)

sub1.plot(vec_lam, vec_x1)
sub2.plot(vec_lam, vec_x2)
sub3.plot(vec_lam, vec_x3)
sub4.plot(vec_lam, vec_x4)
sub5.plot(vec_lam, vec_x5)
sub6.plot(vec_lam, vec_x6)
sub7.plot(vec_lam, vec_x7)
 
plt.show() 