# homotopia de newton i newton-raphson
# trigger schmitt com el del pdf

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

# objectes per iterar
J = np.zeros((8, 8), dtype=float)
n_iter = 10  # número establert d'iteracions a cada etapa de l'homotopia
n_punts = 40000  # número de valors intermedis de lambda
r = 0.001

# valors de l'etapa inicial de l'homotopia, lambda = 0.
x1 = 0.5
x2 = 0.5
x3 = 0.5
x4 = 0.5
x5 = 0.5
x6 = 0.5  # abans era Vcc
x7 = 0.5  # abans era -0.01
lam = 0

c1 = x1
c2 = x2
c3 = x3
c4 = x4
c5 = x5
c6 = x6
c7 = x7
c8 = lam

f10 = x1 / Re + me * (np.exp(n * (x1 - x5)) - 1) - ar * mc * (np.exp(n * (x2 - x5)) - 1) + me * (np.exp(n * (x1 - x4)) - 1) - ar * mc * (np.exp(n * (x3 - x4)) - 1)
f20 = (x2 - x4) / R1 + (x2 - x6) / Rc1 - af * me * (np.exp(n * (x1 - x5)) - 1) + mc * (np.exp(n * (x2 - x5)) - 1)
f30 = (x3 - x6) / Rc2 - af * me * (np.exp(n * (x1 - x4)) - 1) + mc * (np.exp(n * (x3 - x4)) - 1)
f40 = (x4 - x2) / R1 + x4 / R4 - me * (np.exp(n * (x1 - x4)) - 1) + ar * mc * (np.exp(n * (x3 - x4)) - 1) + af * me * (np.exp(n * (x1 - x4)) - 1) - mc * (np.exp(n * (x3 - x4)) - 1)
f50 = (x5 - x6) / R2 + x5 / R3 + af * me * (np.exp(n * (x1 - x5)) - 1) - mc * (np.exp(n * (x2 - x5)) - 1) - me * (np.exp(n * (x1 - x5)) - 1) + ar * mc * (np.exp(n * (x2 - x5)) - 1) 
f60 = (x6 - x2) / Rc1 + (x6 - x3) / Rc2 + (x6 - x5) / R2 + x7
f70 = x6 - Vcc

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

for c in range(n_punts):
    # guardo valors del punt inicial
    x1_i = x1
    x2_i = x2
    x3_i = x3
    x4_i = x4
    x5_i = x5
    x6_i = x6
    x7_i = x7
    lam_i = lam

    # etapa predictiva
    if c == 0:
        lam += r / 8  # abans era r / 2
    else:
        x1 += Apx1
        x2 += Apx2
        x3 += Apx3
        x4 += Apx4
        x5 += Apx5
        x6 += Apx6
        x7 += Apx7
        lam += Aplam

    for k in range(n_iter):
        # jacobià
        J[0, 0] = 1 / Re + me * (np.exp(n * (x1 - x5))) * n + me * (np.exp(n * (x1 - x4))) * n
        J[0, 1] = - ar * mc * (np.exp(n * (x2 - x5))) * n
        J[0, 2] = - ar * mc * (np.exp(n * (x3 - x4))) * n
        J[0, 3] = me * (np.exp(n * (x1 - x4))) * (-n) - ar * mc * (np.exp(n * (x3 - x4))) * (-n)
        J[0, 4] = me * (np.exp(n * (x1 - x5))) * (-n) - ar * mc * (np.exp(n * (x2 - x5))) * (-n)
        J[0, 5] = 0
        J[0, 6] = 0
        J[0, 7] = f10

        J[1, 0] = - af * me * (np.exp(n * (x1 - x5))) * n
        J[1, 1] = 1 / R1 + 1 / Rc1 + mc * (np.exp(n * (x2 - x5))) * n
        J[1, 2] = 0
        J[1, 3] = -1 / R1
        J[1, 4] = - af * me * (np.exp(n * (x1 - x5))) * (-n) + mc * (np.exp(n * (x2 - x5))) * (-n)
        J[1, 5] = -1 / Rc1
        J[1, 6] = 0
        J[1, 7] = f20

        J[2, 0] = -af * me * (np.exp(n * (x1 - x4))) * n
        J[2, 1] = 0
        J[2, 2] = 1 / Rc2 + mc * (np.exp(n * (x3 - x4))) * n
        J[2, 3] = -af * me * (np.exp(n * (x1 - x4))) * (-n) + mc * (np.exp(n * (x3 - x4))) * (-n)
        J[2, 4] = 0
        J[2, 5] = -1 / Rc2
        J[2, 6] = 0
        J[2, 7] = f30

        J[3, 0] = -me * (np.exp(n * (x1 - x4))) * n + af * me * (np.exp(n * (x1 - x4))) * n
        J[3, 1] = -1 / R1
        J[3, 2] = ar * mc * (np.exp(n * (x3 - x4))) * n - mc * (np.exp(n * (x3 - x4))) * n
        J[3, 3] = 1 / R1 + 1 / R4 - me * (np.exp(n * (x1 - x4))) * (-n) + ar * mc * (np.exp(n * (x3 - x4))) * (-n) + af * me * (np.exp(n * (x1 - x4))) * (-n) - mc * (np.exp(n * (x3 - x4))) * (-n)
        J[3, 4] = 0
        J[3, 5] = 0
        J[3, 6] = 0
        J[3, 7] = f40

        J[4, 0] = af * me * (np.exp(n * (x1 - x5))) * n - me * (np.exp(n * (x1 - x5))) * n
        J[4, 1] = -mc * (np.exp(n * (x2 - x5))) * n + ar * mc * (np.exp(n * (x2 - x5))) * n
        J[4, 2] = 0
        J[4, 3] = 0
        J[4, 4] = 1 / R2 + 1 / R3 + af * me * (np.exp(n * (x1 - x5))) * (-n) - mc * (np.exp(n * (x2 - x5))) * (-n) - me * (np.exp(n * (x1 - x5))) * (-n) + ar * mc * (np.exp(n * (x2 - x5))) * (-n)
        J[4, 5] = -1 / R2
        J[4, 6] = 0
        J[4, 7] = f50

        J[5, 0] = 0
        J[5, 1] = -1 / Rc1
        J[5, 2] = -1 / Rc2
        J[5, 3] = 0
        J[5, 4] = -1 / R2
        J[5, 5] = 1 / Rc1 + 1 / Rc2 + 1 / R2
        J[5, 6] = 1
        J[5, 7] = f60

        J[6, 0] = 0
        J[6, 1] = 0
        J[6, 2] = 0
        J[6, 3] = 0
        J[6, 4] = 0
        J[6, 5] = 1
        J[6, 6] = 0
        J[6, 7] = f70

        J[7, 0] = 2 * (x1 - c1)
        J[7, 1] = 2 * (x2 - c2)
        J[7, 2] = 2 * (x3 - c3)
        J[7, 3] = 2 * (x4 - c4)
        J[7, 4] = 2 * (x5 - c5)
        J[7, 5] = 2 * (x6 - c6)
        J[7, 6] = 2 * (x7 - c7)
        J[7, 7] = 2 * (lam - c8)
        

        # mismatches
        f1 = x1 / Re + me * (np.exp(n * (x1 - x5)) - 1) - ar * mc * (np.exp(n * (x2 - x5)) - 1) + me * (np.exp(n * (x1 - x4)) - 1) - ar * mc * (np.exp(n * (x3 - x4)) - 1) - (1 - lam) * f10
        f2 = (x2 - x4) / R1 + (x2 - x6) / Rc1 - af * me * (np.exp(n * (x1 - x5)) - 1) + mc * (np.exp(n * (x2 - x5)) - 1) - (1 - lam) * f20
        f3 = (x3 - x6) / Rc2 - af * me * (np.exp(n * (x1 - x4)) - 1) + mc * (np.exp(n * (x3 - x4)) - 1) - (1 - lam) * f30
        f4 = (x4 - x2) / R1 + x4 / R4 - me * (np.exp(n * (x1 - x4)) - 1) + ar * mc * (np.exp(n * (x3 - x4)) - 1) + af * me * (np.exp(n * (x1 - x4)) - 1) - mc * (np.exp(n * (x3 - x4)) - 1) - (1 - lam) * f40
        f5 = (x5 - x6) / R2 + x5 / R3 + af * me * (np.exp(n * (x1 - x5)) - 1) - mc * (np.exp(n * (x2 - x5)) - 1) - me * (np.exp(n * (x1 - x5)) - 1) + ar * mc * (np.exp(n * (x2 - x5)) - 1) - (1 - lam) * f50 
        f6 = (x6 - x2) / Rc1 + (x6 - x3) / Rc2 + (x6 - x5) / R2 + x7 - (1 - lam) * f60
        f7 = x6 - Vcc - (1 - lam) * f70
        f8 = (x1 - c1) ** 2 + (x2 - c2) ** 2 + (x3 - c3) ** 2 + (x4 - c4) ** 2 + (x5 - c5) ** 2 + (x6 - c6) ** 2 + (x7 - c7) ** 2 + (lam - c8) ** 2 - r ** 2

        # actualitzar iteració
        f = np.block([f1, f2, f3, f4, f5, f6, f7, f8])
        J1 = np.linalg.inv(J)
        Ax = - np.dot(J1, f)
        
        x1 += Ax[0]
        x2 += Ax[1]
        x3 += Ax[2]
        x4 += Ax[3]
        x5 += Ax[4]
        x6 += Ax[5]
        x7 += Ax[6]
        lam += Ax[7]

    Apx1 = x1 - x1_i
    Apx2 = x2 - x2_i
    Apx3 = x3 - x3_i
    Apx4 = x4 - x4_i
    Apx5 = x5 - x5_i
    Apx6 = x6 - x6_i
    Apx7 = x7 - x7_i
    Aplam = lam - lam_i
    c1 = x1
    c2 = x2
    c3 = x3
    c4 = x4
    c5 = x5
    c6 = x6
    c7 = x7
    c8 = lam

    print(lam)
    print(max(abs(f1), abs(f2), abs(f3), abs(f4), abs(f5), abs(f6), abs(f7)))  # error màxim 
    vec_x1.append(x1)
    vec_x2.append(x2)
    vec_x3.append(x3)
    vec_x4.append(x4)
    vec_x5.append(x5)
    vec_x6.append(x6)
    vec_x7.append(x7)
    vec_lam.append(lam)


def f1f(x1, x2, x3, x4, x5, x6, x7):
    return x1 / Re + me * (np.exp(n * (x1 - x5)) - 1) - ar * mc * (np.exp(n * (x2 - x5)) - 1) + me * (np.exp(n * (x1 - x4)) - 1) - ar * mc * (np.exp(n * (x3 - x4)) - 1)

def f2f(x1, x2, x3, x4, x5, x6, x7):
    return (x2 - x4) / R1 + (x2 - x6) / Rc1 - af * me * (np.exp(n * (x1 - x5)) - 1) + mc * (np.exp(n * (x2 - x5)) - 1)

def f3f(x1, x2, x3, x4, x5, x6, x7):
    return (x3 - x6) / Rc2 - af * me * (np.exp(n * (x1 - x4)) - 1) + mc * (np.exp(n * (x3 - x4)) - 1)

def f4f(x1, x2, x3, x4, x5, x6, x7):
    return (x4 - x2) / R1 + x4 / R4 - me * (np.exp(n * (x1 - x4)) - 1) + ar * mc * (np.exp(n * (x3 - x4)) - 1) + af * me * (np.exp(n * (x1 - x4)) - 1) - mc * (np.exp(n * (x3 - x4)) - 1)

def f5f(x1, x2, x3, x4, x5, x6, x7):
    return (x5 - x6) / R2 + x5 / R3 + af * me * (np.exp(n * (x1 - x5)) - 1) - mc * (np.exp(n * (x2 - x5)) - 1) - me * (np.exp(n * (x1 - x5)) - 1) + ar * mc * (np.exp(n * (x2 - x5)) - 1) 

def f6f(x1, x2, x3, x4, x5, x6, x7):
    return (x6 - x2) / Rc1 + (x6 - x3) / Rc2 + (x6 - x5) / R2 + x7

def f7f(x1, x2, x3, x4, x5, x6, x7):
    return x6 - Vcc


sol_x1 = []
sol_x2 = []
sol_x3 = []
sol_x4 = []
sol_x5 = []
sol_x6 = []
sol_x7 = []
vec_errors = []
for i in range(len(vec_lam) - 1):
    if (vec_lam[i] > 1 and vec_lam[i + 1] < 1) or (vec_lam[i] < 1 and vec_lam[i + 1] > 1):
        lam1 = vec_lam[i]
        lam2 = vec_lam[i + 1]
        sol_x1.append((vec_x1[i + 1] - vec_x1[i] + lam2 * vec_x1[i] - lam1 * vec_x1[i + 1]) / (lam2 - lam1))
        sol_x2.append((vec_x2[i + 1] - vec_x2[i] + lam2 * vec_x2[i] - lam1 * vec_x2[i + 1]) / (lam2 - lam1))
        sol_x3.append((vec_x3[i + 1] - vec_x3[i] + lam2 * vec_x3[i] - lam1 * vec_x3[i + 1]) / (lam2 - lam1))
        sol_x4.append((vec_x4[i + 1] - vec_x4[i] + lam2 * vec_x4[i] - lam1 * vec_x4[i + 1]) / (lam2 - lam1))
        sol_x5.append((vec_x5[i + 1] - vec_x5[i] + lam2 * vec_x5[i] - lam1 * vec_x5[i + 1]) / (lam2 - lam1))
        sol_x6.append((vec_x6[i + 1] - vec_x6[i] + lam2 * vec_x6[i] - lam1 * vec_x6[i + 1]) / (lam2 - lam1))
        sol_x7.append((vec_x7[i + 1] - vec_x7[i] + lam2 * vec_x7[i] - lam1 * vec_x7[i + 1]) / (lam2 - lam1))
        vec_errors.append(f1f(sol_x1[-1], sol_x2[-1], sol_x3[-1], sol_x4[-1], sol_x5[-1], sol_x6[-1], sol_x7[-1]))
        vec_errors.append(f2f(sol_x1[-1], sol_x2[-1], sol_x3[-1], sol_x4[-1], sol_x5[-1], sol_x6[-1], sol_x7[-1]))
        vec_errors.append(f3f(sol_x1[-1], sol_x2[-1], sol_x3[-1], sol_x4[-1], sol_x5[-1], sol_x6[-1], sol_x7[-1]))
        vec_errors.append(f4f(sol_x1[-1], sol_x2[-1], sol_x3[-1], sol_x4[-1], sol_x5[-1], sol_x6[-1], sol_x7[-1]))
        vec_errors.append(f5f(sol_x1[-1], sol_x2[-1], sol_x3[-1], sol_x4[-1], sol_x5[-1], sol_x6[-1], sol_x7[-1]))
        vec_errors.append(f6f(sol_x1[-1], sol_x2[-1], sol_x3[-1], sol_x4[-1], sol_x5[-1], sol_x6[-1], sol_x7[-1]))
        vec_errors.append(f7f(sol_x1[-1], sol_x2[-1], sol_x3[-1], sol_x4[-1], sol_x5[-1], sol_x6[-1], sol_x7[-1]))

print(sol_x1)
print(sol_x2)
print(sol_x3)
print(sol_x4)
print(sol_x5)
print(sol_x6)
print(sol_x7)
print(vec_errors)

# resultats
end_time = time.time()
print('Temps: ', end_time - start_time, 'segons')

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