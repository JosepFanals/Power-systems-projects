import numpy as np
from Pade import pade

g = 5
b = -10
V1 = 1.05
Gx = 10
bc = 100
Pdc = -0.2  # positiva si genera, negativa si consumeix
W2 = 0.95  # el mòdul de V2 al quadrat


prof = 30

V2 = np.zeros(prof, dtype=complex)
V2_re = np.zeros(prof)
V2_im = np.zeros(prof)

If = np.zeros(prof, dtype=complex)
If_re = np.zeros(prof)
If_im = np.zeros(prof)

Beq = np.zeros(prof)


# Termes [0]

V2_re[0] = 1
V2_im[0] = 0
V2[0] = V2_re[0] + V2_im[0] * 1j

If_re[0] = 0
If_im[0] = 0
If[0] = 0

# Fi termes [0]

# Termes [1]

M = np.array([[1, 0, -g, b, 0], [0, 1, -b, -g, -V2_re[0]], [V2_re[0], 0, 0, 0, 0], [0, +V2_re[0], 0, 0, 0], [0, 0, 2 * V2_re[0], 0, 0]])
M_inv = np.linalg.inv(M)

rhs = np.array([[np.real(-g * (V1 - 1) + Gx * V2[0] * (If_re[0] ** 2 + If_im[0] ** 2) - b * (V1 - 1) * 1j + bc / 2 * V2[0] * 1j)], 
                [np.imag(-g * (V1 - 1) + Gx * V2[0] * (If_re[0] ** 2 + If_im[0] ** 2) - b * (V1 - 1) * 1j + bc / 2 * V2[0] * 1j)],
                [Pdc],
                [0],
                [W2 - 1]
               ])


lhs = np.real(np.dot(M_inv, rhs))

If_re[1] = lhs[0][0]
If_im[1] = lhs[1][0]
If[1] = If_re[1] + If_im[1] * 1j
V2_re[1] = lhs[2][0]
V2_im[1] = lhs[3][0]
V2[1] = V2_re[1] + V2_im[1] * 1j
Beq[0] = lhs[4][0]

# Fi termes [1]

# Termes [c>=2]


def conv2(a, b, inici, fi, retard):  # passar-li un retard de 0 de 1 com a molt
    suma = 0
    for k in range(inici, fi + 1):
        suma += a[k] * b[fi + 1 - k - retard]
    return suma


def conv3(a, b, c, inici, fi, retard):  # passar-li un retard de 0 a la conv de 3
    suma = 0
    for k in range(inici, fi + 1):
        suma += a[k] * conv2(b, c, 0, fi - k, 0)
    return suma


for cc in range(2, prof):
    rhs = np.array([[np.real(Gx * conv3(V2, If_re, If_re, 0, cc - 1, 0) + Gx * conv3(V2, If_im, If_im, 0, cc - 1, 0) + bc / 2 * V2[cc - 1] * 1j + conv2(V2, Beq, 1, cc - 1, 1) * 1j)], 
                [np.imag(Gx * conv3(V2, If_re, If_re, 0, cc - 1, 0) + Gx * conv3(V2, If_im, If_im, 0, cc - 1, 0) + bc / 2 * V2[cc - 1] * 1j + conv2(V2, Beq, 1, cc - 1, 1) * 1j)],
                [-conv2(V2_re, If_re, 1, cc - 1, 0) - conv2(V2_im, If_im, 1, cc - 1, 0)],
                [conv2(V2_im, If_re, 1, cc - 1, 0) - conv2(V2_re, If_im, 1, cc - 1, 0)],
                [-conv2(V2_re, V2_re, 1, cc - 1, 0) - conv2(V2_im, V2_im, 1, cc - 1, 0)]
               ])

    lhs = np.real(np.dot(M_inv, rhs))

    If_re[cc] = lhs[0][0]
    If_im[cc] = lhs[1][0]
    If[cc] = If_re[cc] + If_im[cc] * 1j
    V2_re[cc] = lhs[2][0]
    V2_im[cc] = lhs[3][0]
    V2[cc] = V2_re[cc] + V2_im[cc] * 1j
    Beq[cc - 1] = lhs[4][0]

V2_f = sum(V2)
If_f = sum(If)
Beq_f = sum(Beq)

ys = g + b * 1j

error1_f = abs(ys * (V2_f - V1) + bc / 2 * V2_f * 1j + Beq_f * V2_f * 1j + Gx * V2_f * abs(If_f) ** 2 - If_f)
error2_f = abs(V2_f * np.conj(If_f) - Pdc)
error3_f = abs(W2 - abs(V2_f) ** 2)

print(error1_f)
print(error2_f)
print(error3_f)

V2_p = pade(prof - 1, V2, 1)
If_p = pade(prof - 1, If, 1)
Beq_p = pade(prof - 1, Beq, 1)

error1_p = abs(ys * (V2_p - V1) + bc / 2 * V2_p * 1j + Beq_p * V2_p * 1j + Gx * V2_p * abs(If_p) ** 2 - If_p)
error2_p = abs(V2_p * np.conj(If_p) - Pdc)
error3_p = abs(W2 - abs(V2_p) ** 2)

print(error1_p)
print(error2_p)
print(error3_p)

print(Beq_f)