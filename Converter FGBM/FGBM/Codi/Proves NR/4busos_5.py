# sense que el bus 4 sigui PQV
# prova de si les meves consideracions són correctes per un cas de 4 busos, amb NR
# jacobià 12 x 12, sense A i B. Veure per què és singular. Al principi no ho és com 
# a tal, però a l'invertir-la hi ha entrades molt grans, així que sí que ho és...

import numpy as np
import pandas as pd

V1 = 1.05
d1 = 0
V2 = 1.01
P4 = -0.1
Q4 = -0.03
Pf = 0.102  # era 0.12
r2 = 0.05
G0 = 0.02
r1 = 1  # realment és una admitància
x1 = -3  # és una admitància
bc1 = 0.02
r3 = 0.76  # és una admitància
x3 = -2.4  # és una admitància
bc3 = 0.04
m2 = 1  # ho tracto com una dada

# inicialitzo variables:
a21 = 0
a34 = 0
i21 = -0.01
i34 = 0.01
d2 = 0
d3 = 0
d4 = 0
V3 = 1
V4 = 1
m1 = 1
B2 = 0
th = 0.0  # theta_2


J = np.zeros((12, 12), dtype=float)
n_iter = 4

for cc in range(n_iter):

    J[0, 0] = -i21 * np.sin(a21)
    J[0, 1] = 0
    J[0, 2] = np.cos(a21)
    J[0, 3] = 0
    J[0, 4] = -V2 / r2 * np.sin(d2)
    J[0, 5] = V3 / r2 * np.sin(d3)
    J[0, 6] = 0
    J[0, 7] = -np.cos(d3) / r2
    # nuls

    J[1, 0] = i21 * np.cos(a21)
    J[1, 1] = 0
    J[1, 2] = np.sin(a21)
    J[1, 3] = 0
    J[1, 4] = V2 / r2 * np.cos(d2)
    J[1, 5] = -V3 / r2 * np.cos(d3)
    J[1, 6] = 0
    J[1, 7] = -np.sin(d3) / r2
    # nuls

    J[2, 0] = i21 * np.sin(a21)
    J[2, 1] = 0
    J[2, 2] = 2 * i21 * G0 * V2 * np.cos(d2) - np.cos(a21)
    J[2, 3] = 0
    J[2, 4] = - (G0 * i21 ** 2 + m1 ** 2 * r1) * V2 * np.sin(d2) - m1 ** 2 * (x1 + bc1 / 2) * V2 * np.cos(d2)
    J[2, 5] = 0
    J[2, 6] = 0
    J[2, 7] = 0
    J[2, 8] = 0
    J[2, 9] = 2 * m1 * r1 * V2 * np.cos(d2) - 2 * m1 * (x1 + bc1 / 2) * V2 * np.sin(d2) - r1 * V1 * np.cos(d1) + x1 * V1 * np.sin(d1)
    # nuls

    J[3, 0] = -i21 * np.cos(a21)
    J[3, 1] = 0
    J[3, 2] = 2 * i21 * G0 * V2 * np.sin(d2) - np.sin(a21)
    J[3, 3] = 0
    J[3, 4] = - m1 ** 2 * (x1 + bc1 / 2) * V2 * np.sin(d2) + (G0 * i21 ** 2 + m1 ** 2 * r1) * V2 * np.cos(d2)
    J[3, 5] = 0
    J[3, 6] = 0
    J[3, 7] = 0
    J[3, 8] = 0
    J[3, 9] = 2 * m1 * (x1 + bc1 / 2) * V2 * np.cos(d2) + 2 * m1 * r1 * V2 * np.sin(d2) - x1 * V1 * np.cos(d1) - r1 * V1 * np.sin(d1)
    # nuls

    J[4, 0] = 0
    J[4, 1] = i34 * np.sin(a34)
    J[4, 2] = 0
    J[4, 3] = 2 * i34 * G0 * V3 * np.cos(d3) - np.cos(a34)
    J[4, 4] = 0
    J[4, 5] = - (G0 * i34 ** 2 + m2 ** 2 * r3) * V3 * np.sin(d3) - m2 ** 2 * (x3 + bc3 / 2 + B2) * V3 * np.cos(d3)
    J[4, 6] = m2 * (r3 * np.cos(th) + x3 * np.sin(th)) * V4 * np.sin(d4) + m2 * (x3 * np.cos(th) - r3 * np.sin(th)) * V4 * np.cos(d4)
    J[4, 7] = (G0 * i34 ** 2 + m2 ** 2 * r3) * np.cos(d3) - m2 ** 2 * (x3 + bc3 / 2 + B2) * np.sin(d3)  # canviat
    J[4, 8] = - m2 * (r3 * np.cos(th) + x3 * np.sin(th)) * np.cos(d4) + m2 * (x3 * np.cos(th) - r3 * np.sin(th)) * np.sin(d4)
    J[4, 9] = 0
    J[4, 10] = - m2 ** 2 * V3 * np.sin(d3)
    J[4, 11] = - m2 * V4 * np.cos(d4) * (-r3 * np.sin(th) + x3 * np.cos(th)) + m2 * (-x3 * np.sin(th) - r3 * np.cos(th)) * V4 * np.sin(d4)
    # nuls

    J[5, 0] = 0
    J[5, 1] = - i34 * np.cos(a34)
    J[5, 2] = 0
    J[5, 3] = 2 * i34 * G0 * V3 * np.sin(d3) - np.sin(a34)
    J[5, 4] = 0
    J[5, 5] = (G0 * i34 ** 2 + m2 ** 2 * r3) * V3 * np.cos(d3) - m2 ** 2 * (x3 + bc3 / 2 + B2) * V3 * np.sin(d3)
    J[5, 6] = - m2 * (r3 * np.cos(th) + x3 * np.sin(th)) * V4 * np.cos(d4) + m2 * (x3 * np.cos(th) - r3 * np.sin(th)) * V4 * np.sin(d4)
    J[5, 7] = (G0 * i34 ** 2 + m2 ** 2 * r3) * np.sin(d3) + m2 ** 2 * (x3 + bc3 / 2 + B2) * np.cos(d3)
    J[5, 8] = - m2 * (r3 * np.cos(th) + x3 * np.sin(th)) * np.sin(d4) - m2 * (x3 * np.cos(th) - r3 * np.sin(th)) * np.cos(d4)
    J[5, 9] = 0
    J[5, 10] = m2 ** 2 * V3 * np.cos(d3)
    J[5, 11] = - m2 * V4 * np.sin(d4) * (-r3 * np.sin(th) + x3 * np.cos(th)) - m2 * V4 * np.cos(d4) * (-x3 * np.sin(th) - r3 * np.cos(th))
    # nuls

    J[6, 0] = 0
    J[6, 1] = -i34 * np.sin(a34)
    J[6, 2] = 0
    J[6, 3] = np.cos(a34)
    J[6, 4] = V2 / r2 * np.sin(d2)
    J[6, 5] = - V3 / r2 * np.sin(d3)
    J[6, 6] = 0
    J[6, 7] = np.cos(d3) / r2
    # nuls

    J[7, 0] = 0
    J[7, 1] = i34 * np.cos(a34)
    J[7, 2] = 0
    J[7, 3] = np.sin(a34)
    J[7, 4] = - V2 / r2 * np.cos(d2)
    J[7, 5] = V3 / r2 * np.cos(d3)
    J[7, 6] = 0
    J[7, 7] = np.sin(d3) / r2
    # nuls

    # F9, la canvio
    J[8, 0] = 0
    J[8, 1] = 0
    J[8, 2] = 0
    J[8, 3] = 0
    J[8, 4] = 0
    J[8, 5] = - m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) - m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th))
    J[8, 6] = - m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) + m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) - r3 * V4 * V4 * (- np.sin(d4) * np.cos(d4) - np.cos(d4) * np.sin(d4)) + (x3 + bc3 / 2) * V4 * V4 * (- np.sin(d4) * np.sin(d4) + np.cos(d4) * np.cos(d4)) + m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - (x3 + bc3 / 2) * V4 * V4 * (- np.sin(d4) * np.sin(d4) + np.cos(d4) * np.cos(d4)) - r3 * V4 * V4 * (np.cos(d4) * np.sin(d4) + np.sin(d4) * np.cos(d4))
    J[8, 7] = m2 * np.cos(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - m2 * np.sin(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * np.cos(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * np.sin(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th))
    J[8, 8] = m2 * V3 * np.cos(d3) * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - m2 * V3 * np.sin(d3) * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) - r3 * 2 * V4 * np.cos(d4) * np.cos(d4) + (x3 + bc3 / 2) * 2 * V4 * np.cos(d4) * np.sin(d4) + m2 * V3 * np.cos(d3) * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * V3 * np.sin(d3) * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - (x3 + bc3 / 2) * 2 * V4 * np.cos(d4) * np.sin(d4) - r3 * 2 * V4 * np.sin(d4) * np.sin(d4)
    J[8, 9] = 0
    J[8, 10] = 0
    J[8, 11] = m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (- r3 * np.sin(th) - x3 * np.cos(th)) - m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (- x3 * np.sin(th) + r3 * np.cos(th)) + m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (- x3 * np.sin(th) + r3 * np.cos(th)) + m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (- r3 * np.sin(th) - x3 * np.cos(th))

    # nuls

    # F10, la canvio
    J[9, 0] = 0
    J[9, 1] = 0
    J[9, 2] = 0
    J[9, 3] = 0
    J[9, 4] = 0
    J[9, 5] = - m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) + m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) + m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th))
    J[9, 6] = - m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) - m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - (x3 + bc3 / 2) * V4 * V4 * (- np.sin(d4) * np.cos(d4) - np.sin(d4) * np.cos(d4)) - r3 * V4 * V4 * (- np.sin(d4) * np.sin(d4) + np.cos(d4) * np.cos(d4)) - m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) + m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + r3 * V4 * V4 * ( - np.sin(d4) * np.sin(d4) + np.cos(d4) * np.cos(d4)) - (x3 + bc3 / 2) * V4 * V4 * (np.cos(d4) * np.sin(d4) + np.sin(d4) * np.cos(d4))

    J[9, 7] = m2 * np.cos(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * np.sin(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - m2 * np.cos(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) + m2 * np.sin(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th))
    J[9, 8] = m2 * V3 * np.cos(d3) * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * V3 * np.sin(d3) * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - (x3 + bc3 / 2) * 2 * V4 * np.cos(d4) * np.cos(d4) - r3 * 2 * V4 * np.cos(d4) * np.sin(d4) - m2 * V3 * np.cos(d3) * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) + m2 * V3 * np.sin(d3) * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + r3 * 2 * V4 * np.cos(d4) * np.sin(d4) - (x3 + bc3 / 2) * 2 * V4 * np.sin(d4) * np.sin(d4)
    J[9, 9] = 0
    J[9, 10] = 0
    J[9, 11] = m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (- x3 * np.sin(th) + r3 * np.cos(th)) + m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (- r3 * np.sin(th) - x3 * np.cos(th)) - m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (- r3 * np.sin(th) - x3 * np.cos(th)) + m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (- x3 * np.sin(th) + r3 * np.cos(th))
    # nuls

    J[10, 0] = 0
    J[10, 1] = - V3 * np.cos(d3) * i34 * np.sin(a34) + V3 * np.sin(d3) * i34 * np.cos(a34)
    J[10, 2] = 0
    J[10, 3] = V3 * np.cos(d3) * np.cos(a34) + V3 * np.sin(d3) * np.sin(a34)
    J[10, 4] = 0
    J[10, 5] = - V3 * np.sin(d3) * i34 * np.cos(a34) + V3 * np.cos(d3) * i34 * np.sin(a34)
    J[10, 6] = 0
    J[10, 7] = np.cos(d3) * i34 * np.cos(a34) + np.sin(d3) * i34 * np.sin(a34)
    # nuls

    J[11, 0] = 0
    J[11, 1] = - V3 * np.cos(d3) * i34 * np.cos(a34) - V3 * np.sin(d3) * i34 * np.sin(a34)
    J[11, 2] = 0
    J[11, 3] = - V3 * np.cos(d3) * np.sin(a34) + V3 * np.sin(d3) * np.cos(a34)
    J[11, 4] = 0
    J[11, 5] = V3 * np.sin(d3) * i34 * np.sin(a34) + V3 * np.cos(d3) * i34 * np.cos(a34)
    J[11, 6] = 0
    J[11, 7] = - np.cos(d3) * i34 * np.sin(a34) + np.sin(d3) * i34 * np.cos(a34)
    # nuls

    df = pd.DataFrame(J)
    print(df)

    # modificacions al jacobià per veure quina fila falla
    # J[9, 6] = J[9, 6] * 1.001
    print(cc)
    J1 = np.linalg.inv(J)
    #print(J1)

    F1 = i21 * np.cos(a21) + V2 / r2 * np.cos(d2) - 1 / r2 * V3 * np.cos(d3)
    F2 = i21 * np.sin(a21) + V2 / r2 * np.sin(d2) - 1 / r2 * V3 * np.sin(d3)
    F3 = (G0 * i21 ** 2 + m1 ** 2 * r1) * V2 * np.cos(d2) - m1 ** 2 * (x1 + bc1 / 2) * V2 * np.sin(d2) - m1 * r1 * V1 * np.cos(d1) + m1 * x1 * V1 * np.sin(d1) - i21 * np.cos(a21)
    F4 = m1 ** 2 * (x1 + bc1 / 2) * V2 * np.cos(d2) + (G0 * i21 ** 2 + m1 ** 2 * r1) * V2 * np.sin(d2) - m1 * x1 * V1 * np.cos(d1) - m1 * r1 * V1 * np.sin(d1) - i21 * np.sin(a21)
    F5 = (G0 * i34 ** 2 + m2 ** 2 * r3) * V3 * np.cos(d3) - m2 ** 2 * (x3 + bc3 / 2 + B2) * V3 * np.sin(d3) - m2 * ((r3 * np.cos(th) + x3 * np.sin(th)) * V4 * np.cos(d4) - (x3 * np.cos(th) - r3 * np.sin(th)) * V4 * np.sin(d4)) - i34 * np.cos(a34)
    F6 = (G0 * i34 ** 2 + m2 ** 2 * r3) * V3 * np.sin(d3) + m2 ** 2 * (x3 + bc3 / 2 + B2) * V3 * np.cos(d3) - m2 * ((r3 * np.cos(th) + x3 * np.sin(th)) * V4 * np.sin(d4) + (x3 * np.cos(th) - r3 * np.sin(th)) * V4 * np.cos(d4)) - i34 * np.sin(a34)
    F7 = i34 * np.cos(a34) + V3 / r2 * np.cos(d3) - V2 / r2 * np.cos(d2)
    F8 = i34 * np.sin(a34) + V3 / r2 * np.sin(d3) - V2 / r2 * np.sin(d2)
    F9 = P4 + m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) - r3 * V4 * V4 * np.cos(d4) * np.cos(d4) + (x3 + bc3 / 2) * V4 * V4 * np.cos(d4) * np.sin(d4) + m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - (x3 + bc3 / 2) * V4 * V4 * np.cos(d4) * np.sin(d4) - r3 * V4 * V4 * np.sin(d4) * np.sin(d4)
    F10 = - Q4 + m2 * V3 * np.cos(d3) * V4 * np.cos(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + m2 * V3 * np.sin(d3) * V4 * np.cos(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) - (x3 + bc3 / 2) * V4 * V4 * np.cos(d4) * np.cos(d4) - r3 * V4 * V4 * np.cos(d4) * np.sin(d4) - m2 * V3 * np.cos(d3) * V4 * np.sin(d4) * (r3 * np.cos(th) - x3 * np.sin(th)) + m2 * V3 * np.sin(d3) * V4 * np.sin(d4) * (x3 * np.cos(th) + r3 * np.sin(th)) + r3 * V4 * V4 * np.cos(d4) * np.sin(d4) - (x3 + bc3 / 2) * V4 * V4 * np.sin(d4) * np.sin(d4)
    F11 = V3 * np.cos(d3) * i34 * np.cos(a34) + V3 * np.sin(d3) * i34 * np.sin(a34) - Pf
    F12 = - V3 * np.cos(d3) * i34 * np.sin(a34) + V3 * np.sin(d3) * i34 * np.cos(a34)

    F = np.block([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12])
    Ax = np.dot(- J1, F)

    a21 += Ax[0]
    a34 += Ax[1]
    i21 += Ax[2]
    i34 += Ax[3]
    d2 += Ax[4]
    d3 += Ax[5]
    d4 += Ax[6]
    V3 += Ax[7]
    V4 += Ax[8]
    m1 += Ax[9]
    B2 += Ax[10]
    th += Ax[11]



   