# primera versió del sistema de 6 busos mallat seguint les indicacions dels de Durham

import numpy as np

# dades
V1 = 1.05
d1 = 0
V2 = 1.01
V4 = 0.97
P2 = -0.1
Q2 = -0.05
P3 = 0
Q3 = 0
P4 = 0
Q4 = 0
P5 = -0.2
Q5 = -0.1
P6 = -0.07
Q6 = -0.02
Pf = 0.1
# g12 = 4
# b12 = -15
# rs1 = 3.4  # és igualment una admitància, encara que l'hagi anomenat r
# xs1 = -16.3  # és igualment una admitància, encara que l'hagi anomenat x
# bc1 = -0.01
# g34 = 5.8
# b34 = -0
# rs2 = 3.7
# xs2 = -11.5
# bc2 = -0.03
# g25 = 6.7
# b25 = -21.4
# g56 = 2.8
# b56 = -17.2
g12 = 4.00060
b12 = -14.99984
rs1 = 3.39935
xs1 = -16.30080
bc1 = 0
g34 = 5.80013
b34 = -0
rs2 = 3.66962
xs2 = -11.50019
bc2 = 0
g25 = 6.69759
b25 = -21.41000
g56 = 2.79980
b56 = -17.19961

# inicialitzo incògnites
d2 = 0
d3 = 0
d4 = 0
d5 = 0
d6 = 0
V3 = 1
V5 = 1
V6 = 1
th = 0
Beq1 = 0
Beq2 = 0
m = 1

eps = 0.00000000000001


def f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return -P2 + V2 * np.cos(d2) * (V2 * np.cos(d2) * (g12 + g25 + rs1) + V2 * np.sin(d2) * (-b12 - b25 - xs1 - bc1 / 2) - V1 * np.cos(d1) * g12 + V1 * np.sin(d1) * b12 - V5 * np.cos(d5) * g25 + V5 * np.sin(d5) * b25 - V3 * np.cos(d3) * rs1 / m + V3 * np.sin(d3) * xs1 / m) + V2 * np.sin(d2) * (V2 * np.cos(d2) * (b12 + b25 + xs1 + bc1 / 2) + V2 * np.sin(d2) * (g12 + g25 + rs1) - V1 * np.cos(d1) * b12 - V1 * np.sin(d1) * g12 - V5 * np.cos(d5) * b25 - V5 * np.sin(d5) * g25 - V3 * np.cos(d3) * xs1 / m - V3 * np.sin(d3) * rs1 / m)


def f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return Q2 + V2 * np.cos(d2) * (V2 * np.cos(d2) * (b12 + b25 + xs1 + bc1 / 2) + V2 * np.sin(d2) * (g12 + g25 + rs1) - V1 * np.cos(d1) * b12 - V1 * np.sin(d1) * g12 - V5 * np.cos(d5) * b25 - V5 * np.sin(d5) * g25 - V3 * np.cos(d3) * xs1 / m - V3 * np.sin(d3) * rs1 / m) - V2 * np.sin(d2) * (V2 * np.cos(d2) * (g12 + g25 + rs1) + V2 * np.sin(d2) * (-b12 - b25 - xs1 - bc1 / 2) - V1 * np.cos(d1) * g12 + V1 * np.sin(d1) * b12 - V5 * np.cos(d5) * g25 + V5 * np.sin(d5) * b25 - V3 * np.cos(d3) * rs1 / m + V3 * np.sin(d3) * xs1 / m)


def f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return -P3 + V3 * np.cos(d3) * (V3 * np.cos(d3) * (g34 + rs1 / (m ** 2)) + V3 * np.sin(d3) * (-b34 + (-xs1 - bc1 / 2 - Beq1) / (m ** 2)) - V4 * np.cos(d4) * g34 + V4 * np.sin(d4) * b34 - V2 * np.cos(d2) * rs1 / m + V2 * np.sin(d2) * xs1 / m) + V3 * np.sin(d3) * (V3 * np.cos(d3) * (b34 + (xs1 + bc1 / 2 + Beq1) / (m ** 2)) + V3 * np.sin(d3) * (g34 + rs1 / (m ** 2)) - V4 * np.cos(d4) * b34 - V4 * np.sin(d4) * g34 - V2 * np.cos(d2) * xs1 / m - V2 * np.sin(d2) * rs1 / m)


def f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return Q3 + V3 * np.cos(d3) * (V3 * np.cos(d3) * (b34 + (xs1 + bc1 / 2 + Beq1) / (m ** 2)) + V3 * np.sin(d3) * (g34 + rs1 / (m ** 2)) - V4 * np.cos(d4) * b34 - V4 * np.sin(d4) * g34 - V2 * np.cos(d2) * xs1 / m - V2 * np.sin(d2)* rs1 / m) - V3 * np.sin(d3) * (V3 * np.cos(d3) * (g34 + rs1 / (m ** 2)) + V3 * np.sin(d3) * (-b34 + (-xs1 - bc1 / 2 - Beq1) / (m ** 2)) - V4 * np.cos(d4) * g34 + V4 * np.sin(d4) * b34 - V2 * np.cos(d2) * rs1 / m + V2 * np.sin(d2) * xs1 / m)


# def f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
#     return -P4 + V4 * np.cos(d4) * (V4 * np.cos(d4) * (g34 + rs2) + V4 * np.sin(d4) * (-b34 - xs2 - bc2 / 2 - Beq2) - V3 * np.cos(d3) * g34 + V3 * np.sin(d3) * b34 + V2 * np.cos(d2) * (-rs2 * np.cos(th) - xs2 * np.sin(th)) + V2 * np.sin(d2) * (-rs2 * np.sin(th) + xs2 * np.cos(th))) + V4 * np.sin(d4) * (V4 * np.cos(d4) * (b34 + xs2 + bc2 / 2 + Beq2) + V4 * np.sin(d4) * (g34 + rs2) - V3 * np.cos(d3) * b34 - V3 * np.sin(d3) * g34 + V2 * np.cos(d2) * (rs2 * np.sin(th) - xs2 * np.cos(th)) + V2 * np.sin(d2) * (-rs2 * np.cos(th) - xs2 * np.sin(th)))


# def f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
#     return Q4 + V4 * np.cos(d4) * (V4 * np.cos(d4) * (b34 + xs2 + bc2 / 2 + Beq2) + V4 * np.sin(d4) * (g34 + rs2) - V3 * np.cos(d3) * b34 - V3 * np.sin(d3) * g34 + V2 * np.cos(d2) * (rs2 * np.sin(th) - xs2 * np.cos(th)) + V2 * np.sin(d2) * (-rs2 * np.cos(th) - xs2 * np.sin(th))) - V4 * np.sin(d4) * (V4 * np.cos(d4) * (g34 + rs2) + V4 * np.sin(d4) * (-b34 - xs2 - bc2 / 2 - Beq2) - V3 * np.cos(d3) * g34 + V3 * np.sin(d3) * b34 + V2 * np.cos(d2) * (-rs2 * np.cos(th) - xs2 * np.sin(th)) + V2 * np.sin(d2) * (-rs2 * np.sin(th) + xs2 * np.cos(th)))

def f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return -P4 + V4 * np.cos(d4) * (V4 * np.cos(d4) * (g34 + rs2) + V4 * np.sin(d4) * (-b34 - xs2 - bc2 / 2 - Beq2) - V3 * np.cos(d3) * g34 + V3 * np.sin(d3) * b34 + V5 * np.cos(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th)) + V5 * np.sin(d5) * (-rs2 * np.sin(th) + xs2 * np.cos(th))) + V4 * np.sin(d4) * (V4 * np.cos(d4) * (b34 + xs2 + bc2 / 2 + Beq2) + V4 * np.sin(d4) * (g34 + rs2) - V3 * np.cos(d3) * b34 - V3 * np.sin(d3) * g34 + V5 * np.cos(d5) * (rs2 * np.sin(th) - xs2 * np.cos(th)) + V5 * np.sin(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th)))


def f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return Q4 + V4 * np.cos(d4) * (V4 * np.cos(d4) * (b34 + xs2 + bc2 / 2 + Beq2) + V4 * np.sin(d4) * (g34 + rs2) - V3 * np.cos(d3) * b34 - V3 * np.sin(d3) * g34 + V5 * np.cos(d5) * (rs2 * np.sin(th) - xs2 * np.cos(th)) + V5 * np.sin(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th))) - V4 * np.sin(d4) * (V4 * np.cos(d4) * (g34 + rs2) + V4 * np.sin(d4) * (-b34 - xs2 - bc2 / 2 - Beq2) - V3 * np.cos(d3) * g34 + V3 * np.sin(d3) * b34 + V5 * np.cos(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th)) + V5 * np.sin(d5) * (-rs2 * np.sin(th) + xs2 * np.cos(th)))


def f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return -P5 + V5 * np.cos(d5) * (V5 * np.cos(d5) * (g25 + g56 + rs2) + V5 * np.sin(d5) * (-b25 - b56 - xs2 - bc2 / 2) - V2 * np.cos(d2) * g25 + V2 * np.sin(d2) * b25 + V4 * np.cos(d4) * (-rs2 * np.cos(th) + xs2 * np.sin(th)) + V4 * np.sin(d4) * (xs2 * np.cos(th) + rs2 * np.sin(th)) - V6 * np.cos(d6) * g56 + V6 * np.sin(d6)  * b56) + V5 * np.sin(d5) * (V5 * np.cos(d5) * (b25 + b56 + xs2 + bc2 / 2) + V5 * np.sin(d5) * (g25 + g56 + rs2) - V2 * np.cos(d2) * b25 - V2 * np.sin(d2) * g25 + V4 * np.cos(d4) * (-xs2 * np.cos(th) - rs2 * np.sin(th)) + V4 * np.sin(d4) * (-rs2 * np.cos(th) + xs2 * np.sin(th)) - V6 * np.cos(d6) * b56 - V6 * np.sin(d6) * g56)


def f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return Q5 + V5 * np.cos(d5) * (V5 * np.cos(d5) * (b25 + b56 + xs2 + bc2 / 2) + V5 * np.sin(d5) * (g25 + g56 + rs2) - V2 * np.cos(d2) * b25 - V2 * np.sin(d2) * g25 + V4 * np.cos(d4) * (-xs2 * np.cos(th) - rs2 * np.sin(th)) + V4 * np.sin(d4) * (-rs2 * np.cos(th) + xs2 * np.sin(th)) - V6 * np.cos(d6) * b56 - V6 * np.sin(d6) * g56) - V5 * np.sin(d5) * (V5 * np.cos(d5) * (g25 + g56 + rs2) + V5 * np.sin(d5) * (-b25 - b56 - xs2 - bc2 / 2) - V2 * np.cos(d2) * g25 + V2 * np.sin(d2) * b25 + V4 * np.cos(d4) * (-rs2 * np.cos(th) + xs2 * np.sin(th)) + V4 * np.sin(d4) * (xs2 * np.cos(th) + rs2 * np.sin(th)) - V6 * np.cos(d6) * g56 + V6 * np.sin(d6) * b56)


def f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return -P6 + V6 * np.cos(d6) * (V6 * np.cos(d6) * g56 - V6 * np.sin(d6) * b56 - V5 * np.cos(d5) * g56 + V5 * np.sin(d5) * b56) + V6 * np.sin(d6) * (V6 * np.cos(d6) * b56 + V6 * np.sin(d6) * g56 - V5 * np.cos(d5) * b56 - V5 * np.sin(d5) * g56)


def f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):
    return Q6 + V6 * np.cos(d6) * (V6 * np.cos(d6) * b56 + V6 * np.sin(d6) * g56 - V5 * np.cos(d5) * b56 - V5 * np.sin(d5) * g56) - V6 * np.sin(d6) * (V6 * np.cos(d6) * g56 - V6 * np.sin(d6) * b56 - V5 * np.cos(d5) * g56 + V5 * np.sin(d5) * b56)


# def f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):  # amb I45
#     return -Pf + V4 * np.cos(d4) * (V4 * np.cos(d4) * rs2 + V4 * np.sin(d4) * (-xs2 - bc2 / 2 - Beq2) + V5 * np.cos(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th)) + V5 * np.sin(d5) * (-rs2 * np.sin(th) + xs2 * np.cos(th))) + V4 * np.sin(d4) * (V4 * np.cos(d4) * (xs2 + bc2 / 2 + Beq2) + V4 * np.sin(d4) * rs2 + V5 * np.cos(d5) * (rs2 * np.sin(th) - xs2 * np.cos(th)) + V5 * np.sin(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th)))

def f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):  # amb I43
    return Pf + V4 * np.cos(d4) * (V4 * np.cos(d4) * g34 - V4 * np.sin(d4) * b34 - V3 * np.cos(d3) * g34 + V3 * np.sin(d3) * b34) + V4 * np.sin(d4) * (V4 * np.cos(d4) * b34 + V4 * np.sin(d4) * g34 - V3 * np.cos(d3) * b34 - V3 * np.sin(d3) * g34)


# def f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):  # amb I45
#     return 0 + V4 * np.cos(d4) * (V4 * np.cos(d4) * (xs2 + bc2 / 2 + Beq2) + V4 * np.sin(d4) * rs2 + V5 * np.cos(d5) * (rs2 * np.sin(th) - xs2 * np.cos(th)) + V5 * np.sin(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th))) - V4 * np.sin(d4) * (V4 * np.cos(d4) * rs2 + V4 * np.sin(d4) * (-xs2 - bc2 / 2 - Beq2) + V5 * np.cos(d5) * (-rs2 * np.cos(th) - xs2 * np.sin(th)) + V5 * np.sin(d5) * (-rs2 * np.sin(th) + xs2 * np.cos(th)))

def f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m):  # amb I43
    return V4 * np.cos(d4) * (V4 * np.cos(d4) * b34 + V4 * np.sin(d4) * g34 - V3 * np.cos(d3) * b34 - V3 * np.sin(d3) * g34) - V4 * np.sin(d4) * (V4 * np.cos(d4) * g34 - V4 * np.sin(d4) * b34 - V3 * np.cos(d3) * g34 + V3 * np.sin(d3) * b34)


J = np.zeros((12, 12), dtype=float)
I = np.identity(12, dtype=float)
n_iter = 20

for k in range(n_iter):

    J[0, 0] = (f1(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 1] = (f1(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 2] = (f1(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 3] = (f1(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 4] = (f1(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 5] = (f1(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 6] = (f1(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 7] = (f1(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 8] = (f1(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 9] = (f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 10] = (f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[0, 11] = (f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[1, 0] = (f2(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 1] = (f2(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 2] = (f2(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 3] = (f2(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 4] = (f2(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 5] = (f2(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 6] = (f2(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 7] = (f2(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 8] = (f2(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 9] = (f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 10] = (f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[1, 11] = (f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[2, 0] = (f3(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 1] = (f3(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 2] = (f3(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 3] = (f3(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 4] = (f3(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 5] = (f3(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 6] = (f3(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 7] = (f3(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 8] = (f3(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 9] = (f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 10] = (f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[2, 11] = (f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[3, 0] = (f4(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 1] = (f4(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 2] = (f4(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 3] = (f4(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 4] = (f4(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 5] = (f4(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 6] = (f4(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 7] = (f4(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 8] = (f4(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 9] = (f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 10] = (f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[3, 11] = (f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[4, 0] = (f5(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 1] = (f5(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 2] = (f5(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 3] = (f5(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 4] = (f5(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 5] = (f5(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 6] = (f5(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 7] = (f5(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 8] = (f5(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 9] = (f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 10] = (f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[4, 11] = (f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[5, 0] = (f6(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 1] = (f6(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 2] = (f6(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 3] = (f6(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 4] = (f6(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 5] = (f6(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 6] = (f6(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 7] = (f6(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 8] = (f6(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 9] = (f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 10] = (f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[5, 11] = (f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[6, 0] = (f7(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 1] = (f7(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 2] = (f7(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 3] = (f7(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 4] = (f7(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 5] = (f7(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 6] = (f7(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 7] = (f7(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 8] = (f7(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 9] = (f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 10] = (f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[6, 11] = (f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[7, 0] = (f8(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 1] = (f8(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 2] = (f8(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 3] = (f8(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 4] = (f8(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 5] = (f8(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 6] = (f8(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 7] = (f8(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 8] = (f8(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 9] = (f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 10] = (f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[7, 11] = (f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[8, 0] = (f9(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 1] = (f9(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 2] = (f9(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 3] = (f9(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 4] = (f9(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 5] = (f9(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 6] = (f9(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 7] = (f9(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 8] = (f9(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 9] = (f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 10] = (f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[8, 11] = (f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[9, 0] = (f10(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 1] = (f10(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 2] = (f10(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 3] = (f10(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 4] = (f10(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 5] = (f10(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 6] = (f10(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 7] = (f10(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 8] = (f10(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 9] = (f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 10] = (f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[9, 11] = (f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[10, 0] = (f11(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 1] = (f11(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 2] = (f11(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 3] = (f11(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 4] = (f11(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 5] = (f11(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 6] = (f11(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 7] = (f11(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 8] = (f11(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 9] = (f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 10] = (f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[10, 11] = (f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J[11, 0] = (f12(d2 + eps, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 1] = (f12(d2, d3 + eps, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 2] = (f12(d2, d3, d4 + eps, d5, d6, V3, V5, V6, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 3] = (f12(d2, d3, d4, d5 + eps, d6, V3, V5, V6, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 4] = (f12(d2, d3, d4, d5, d6 + eps, V3, V5, V6, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 5] = (f12(d2, d3, d4, d5, d6, V3 + eps, V5, V6, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 6] = (f12(d2, d3, d4, d5, d6, V3, V5 + eps, V6, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 7] = (f12(d2, d3, d4, d5, d6, V3, V5, V6 + eps, th, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 8] = (f12(d2, d3, d4, d5, d6, V3, V5, V6, th + eps, Beq1, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 9] = (f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1 + eps, Beq2, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 10] = (f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2 + eps, m) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps
    J[11, 11] = (f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m + eps) - f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)) / eps

    J1 = np.linalg.inv(J)

    f1r = f1(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f2r = f2(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f3r = f3(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f4r = f4(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f5r = f5(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f6r = f6(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f7r = f7(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f8r = f8(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f9r = f9(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f10r = f10(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f11r = f11(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)
    f12r = f12(d2, d3, d4, d5, d6, V3, V5, V6, th, Beq1, Beq2, m)

    f = np.block([f1r, f2r, f3r, f4r, f5r, f6r, f7r, f8r, f9r, f10r, f11r, f12r])

    Ax = - np.dot(J1, f)

    d2 += Ax[0]
    d3 += Ax[1]
    d4 += Ax[2]
    d5 += Ax[3]
    d6 += Ax[4]
    V3 += Ax[5]
    V5 += Ax[6]
    V6 += Ax[7]
    th += Ax[8]
    Beq1 += Ax[9]
    Beq2 += Ax[10]
    m += Ax[11]

    print(max(f))

print(d2 * 180 / np.pi)
print(d3 * 180 / np.pi)
print(d4 * 180 / np.pi)
print(d5 * 180 / np.pi)
print(d6 * 180 / np.pi)
print(V1)
print(V2)
print(V3)
print(V4)
print(V5)
print(V6)
print(th * 180 / np.pi)
print(Beq1)
print(Beq2)
print(m)

V1 = V1 * np.cos(d1) + 1j * V1 * np.sin(d1)
V2 = V2 * np.cos(d2) + 1j * V2 * np.sin(d2)
V3 = V3 * np.cos(d3) + 1j * V3 * np.sin(d3)
V4 = V4 * np.cos(d4) + 1j * V4 * np.sin(d4)
V5 = V5 * np.cos(d5) + 1j * V5 * np.sin(d5)
V6 = V6 * np.cos(d6) + 1j * V6 * np.sin(d6)
Y12 = g12 + 1j * b12
Y25 = g25 + 1j * b25
Y34 = g34 + 1j * b34
Y56 = g56 + 1j * b56

Ytt1 = rs1 + 1j * xs1 + 1j * bc1 / 2
Ytf1 = - (rs1 + 1j * xs1) / m 
Yff1 = (rs1 + 1j * xs1 + 1j * bc1 / 2 + 1j * Beq1) / (m ** 2)
Yft1 = - (rs1 + 1j * xs1) / m

Yff2 = (rs2 + 1j * xs2 + 1j * bc2 / 2 + 1j * Beq2) / 1
Yft2 = - (rs2 + 1j * xs2) * (np.cos(th) - 1j * np.sin(th))
Ytt2 = rs2 + 1j * xs2 + 1j * bc2 / 2
Ytf2 = - (rs2 + 1j * xs2) * (np.cos(th) + 1j * np.sin(th))

sumai2 = (P2 - 1j * Q2) / np.conj(V2) - ((V2 - V1) * Y12 + (V2 - V5) * Y25 + V2 * Ytt1 + V3 * Ytf1)
sumai3 = (P3 - 1j * Q3) / np.conj(V3) - ((V3 - V4) * Y34 + V3 * Yff1 + V2 * Yft1)
sumai4 = (P4 - 1j * Q4) / np.conj(V4) - ((V4 - V3) * Y34 + V4 * Yff2 + V5 * Yft2)
sumai5 = (P5 - 1j * Q5) / np.conj(V5) - ((V5 - V6) * Y56 + (V5 - V2) * Y25 + V5 * Ytt2 + V4 * Ytf2)
sumai6 = (P6 - 1j * Q6) / np.conj(V6) - ((V6 - V5) * Y56)
print(abs(sumai2))
print(abs(sumai3))
print(sumai4)
print(sumai5)
print(abs(sumai6))

print(V4 * np.conj((V4 - V3) * Y34))