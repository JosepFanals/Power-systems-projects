# AQUESTA VERSIÓ FUNCIONA!! ARA SI DE CAS VEURE COM ACCELERAR LA CONVERGÈNCIA...
# PERÒ SI LES TENSIONS CONVERGEIXEN BÉ, LES ALTRES VARIABLES SÓN POC PREOCUPANTS
# IGUAL QUE v3 PERÒ ARA RETARDO E2-RE I NO E2-IM

import numpy as np
from Pade import pade
import pandas as pd

g12 = 5.7
b12 = -15
bc12 = 0.2
g23 = 8
g34 = 7.9
b34 = -13
bc34 = 0.15
G2 = 0.2
G3 = 0.1
Q1 = 0.1
Pf = 0.11
P4 = -0.1
Q4 = -0.1
W2 = 0.99
W4 = 0.96
V1 = 1.05

prof = 30

V2_re = np.zeros(prof)
V2_im = np.zeros(prof)
V2 = np.zeros(prof, dtype=complex)
V3_re = np.zeros(prof)
V3_im = np.zeros(prof)
V3 = np.zeros(prof, dtype=complex)
V4_re = np.zeros(prof)
V4_im = np.zeros(prof)
V4 = np.zeros(prof, dtype=complex)
X4 = np.zeros(prof, dtype=complex)
If3_re = np.zeros(prof)
If3_im = np.zeros(prof)
If3 = np.zeros(prof, dtype=complex)
If2_re = np.zeros(prof)
If2_im = np.zeros(prof)
If2 = np.zeros(prof, dtype=complex)
M1 = np.zeros(prof)
M2 = np.zeros(prof)
B1 = np.zeros(prof)
B2 = np.zeros(prof)
E_re = np.zeros(prof)
E_im = np.zeros(prof)
E = np.zeros(prof, dtype=complex)
V1_vec = np.zeros(prof)

V1_vec[0] = 1
V1_vec[1] = V1 - 1

V2_re[0] = 1
V2_im[0] = 0
V2[0] = 1
V3_re[0] = 1
V3_im[0] = 0
V3[0] = 1
V4_re[0] = 1
V4_im[0] = 0
V4[0] = 1
X4[0] = 1
If3_re[0] = 0
If3_im[0] = 0
If3[0] = 0
If2_re[0] = 0
If2_im[0] = 0
If2[0] = 0
M1[0] = 1
M2[0] = 1

E_re[0] = 1
E_im[0] = 0
E[0] = 1


mat = np.array([[b12, g12, 0, 0, 0, 0, 0, 0, 0, 0, b12, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-g23, 0, g23, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -g23, 0, g23, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [g23, 0, -g23, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, g23, 0, -g23, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -g34, b34, g34, -b34, 1, 0, 0, 0, 0, -g34, 0, g34, 0, 0],  # he tret la b34 d'aquí
                [-g12, b12, 0, 0, 0, 0, 0, 0, 1, 0, -g12, 0, 0, 0, 0, 0],
                [0, 0, -b34, -g34, b34, g34, 0, 1, 0, 0, 0, -b34, 0, b34, g34, 0],
                [-b12, -g12, 0, 0, 0, 0, 0, 0, 0, 1, -b12, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, -g34, b34, g34, -b34, 0, 0, 0, 0, 0, -g34, 0, -g34, b34, 0],
                [0, 0, -b34, -g34, b34, g34, 0, 0, 0, 0, 0, -b34, 0, -b34, g34, -1]
                ])


df = pd.DataFrame(mat)
print(df)
df.to_excel("matriu.xlsx")  
print(np.shape(mat))
mat_inv = np.linalg.inv(mat)




def conv1(a, b, k, lim1, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        suma += a[kk] * b[cc - kk - retard]
    return suma


def conv2(a, b, c, k, j, lim1, lim2, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        for jj in range(j, lim2 + 1 - kk):
            suma += a[kk] * b[jj] * c[cc - kk - jj - retard]
    return suma


def conv3(a, b, c, d, k, j, i, lim1, lim2, lim3, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        for jj in range(j, lim2 + 1 - kk):
            for ii in range(i, lim3 + 1 - kk - jj):
                suma += a[kk] * b[jj] * c[ii] * d[cc - kk - jj - ii - retard]
    return suma


# TERMES [1]
cc = 1

rhs = np.array([[Q1 + b12 * V1_vec[1] + bc12 / 2],
                [Pf],
                [W2 - 1],
                [W4 - 1],
                [0],
                [0],
                [0],
                [0],
                [0],
                [G3 * (conv2(If3_re, If3_re, V3_re, 0, 0, cc-1, cc-1, 1, cc) - conv2(If3_im, If3_im, V3_re, 0, 0, cc-1, cc-1, 1, cc) - 2 * conv2(If3_re, If3_im, V3_im, 0, 0, cc-1, cc-1, 1, cc)) + g34 * (conv1(M2, V3_re, 1, cc-1, 0, cc) + conv2(M2, M2, V3_re, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(M2, V3_im, 1, cc-1, 0, cc) + conv2(M2, M2, V3_im, 1, 0, cc-1, cc, 0, cc)) - bc34 / 2 * conv2(M2, M2, V3_im, 0, 0, cc-1, cc-1, 1, cc) - g34 * (conv1(V4_re, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_re, 1, 0, cc-1, cc, 0, cc)) - g34 * (conv1(V4_im, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_im, 1, 0, cc-1, cc, 0, cc)) + b34 * (conv1(V4_im, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_re, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(V4_re, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_im, 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B2, M2, M2, V3, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - b34 * E_im[cc - 1]],  # aquest últim terme per tal que la matriu tingui inversa
                [-g12 * M1[cc-1] * V1_vec[1] + g12 * conv2(M1, M1, V2_re, 0, 0, cc-1, cc-1, 0, cc) - b12 * conv2(M1, M1, V2_im, 0, 0, cc-1, cc-1, 0, cc) + np.real(G2 * conv2(If2, If2, V2, 0, 0, cc-1, cc-1, 1, cc) + bc12 / 2 * 1j * conv2(M1, M1, V2, 0, 0, cc-1, cc-1, 1, cc) + 1j * conv3(B1, M1, M1, V2, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))],
                [G3 * (conv2(If3_re, If3_re, V3_im, 0, 0, cc-1, cc-1, 1, cc) - conv2(If3_im, If3_im, V3_im, 0, 0, cc-1, cc-1, 1, cc) - 2 * conv2(If3_re, If3_im, V3_re, 0, 0, cc-1, cc-1, 1, cc)) + b34 * (conv1(M2, V3_re, 1, cc-1, 0, cc) + conv2(M2, M2, V3_re, 1, 0, cc-1, cc, 0, cc)) + g34 * (conv1(M2, V3_im, 1, cc-1, 0, cc) + conv2(M2, M2, V3_im, 1, 0, cc-1, cc, 0, cc)) + bc34 / 2 * conv2(M2, M2, V3_re, 0, 0, cc-1, cc-1, 1, cc) - g34 * (conv1(V4_im, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_re, 1, 0, cc-1, cc, 0, cc)) - g34 * (conv1(V4_re, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_im, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(V4_re, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_re, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(V4_im, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_im, 1, 0, cc-1, cc, 0, cc) + np.imag(1j * conv3(B2, M2, M2, V3, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)))],
                [-b12 * M1[cc-1] * V1_vec[1] + g12 * conv2(M1, M1, V2_im, 0, 0, cc-1, cc-1, 0, cc) + b12 * conv2(M1, M1, V2_re, 0, 0, cc-1, cc-1, 0, cc) + np.imag(G2 * conv2(If2, If2, V2, 0, 0, cc-1, cc-1, 1, cc) + bc12 / 2 * 1j * conv2(M1, M1, V2, 0, 0, cc-1, cc-1, 1, cc) + 1j * conv3(B1, M1, M1, V2, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))],
                [0],
                [bc34 / 2 * V4_im[cc-1] + g34 * conv1(V3_re, E_re, 1, cc-1, 0, cc) + g34 * conv2(M2, V3_re, E_re, 1, 0, cc-1, cc, 0, cc) - b34 * conv1(V3_re, E_im, 1, cc-1, 0, cc) - b34 * conv2(M2, V3_re, E_im, 1, 0, cc-1, cc, 0, cc) + g34 * conv1(V3_im, E_im, 1, cc-1, 0, cc) + g34 * conv2(M2, V3_im, E_im, 1, 0, cc-1, cc, 0, cc) - b34 * conv1(V3_im, E_re, 1, cc-1, 0, cc) - b34 * conv2(M2, V3_im, E_re, 1, 0, cc-1, cc, 0, cc) + np.real((P4 - Q4 * 1j) * X4[cc-1])],
                [-bc34 / 2 * V4_re[cc-1] - g34 * conv1(V3_re, E_im, 1, cc-1, 0, cc) - g34 * conv2(M2, V3_re, E_im, 1, 0, cc-1, cc, 0, cc) + b34 * conv1(V3_re, E_re, 1, cc-1, 0, cc) + b34 * conv2(M2, V3_re, E_re, 1, 0, cc-1, cc, 0, cc) + g34 * conv1(V3_im, E_re, 1, cc-1, 0, cc) + g34 * conv2(M2, V3_im, E_re, 1, 0, cc-1, cc, 0, cc) - b34 * conv1(V3_im, E_im, 1, cc-1, 0, cc) - b34 * conv2(M2, V3_im, E_im, 1, 0, cc-1, cc, 0, cc) + np.imag((P4 - Q4 * 1j) * X4[cc-1])]
                ])


lhs = np.real(np.dot(mat_inv, rhs))

V2_re[1] = lhs[0]
V2_im[1] = lhs[1]
V2[1] = V2_re[1] + 1j * V2_im[1]
V3_re[1] = lhs[2]
V3_im[1] = lhs[3]
V3[1] = V3_re[1] + 1j * V3_im[1]
V4_re[1] = lhs[4]
V4_im[1] = lhs[5]
V4[1] = V4_re[1] + 1j * V4_im[1]
If3_re[1] = lhs[6]
If3_im[1] = lhs[7]
If3[1] = If3_re[1] + 1j * If3_im[1]
If2_re[1] = lhs[8]
If2_im[1] = lhs[9]
If2[1] = If2_re[1] + 1j * If2_im[1]
M1[1] = lhs[10]
M2[1] = lhs[11]
B1[0] = lhs[12]
E_re[1] = lhs[13]
E_im[1] = lhs[14]
E[1] = E_re[1] + 1j * E_im[1]
B2[0] = lhs[15]

X4[1] = - conv1(X4, np.conj(V4), 0, cc-1, 0, cc)

for c in range(2, prof):
    cc = c

    rhs = np.array([[-g12 * conv1(V2_im, M1, 1, cc-1, 0, cc) - g12 * V1_vec[1] * conv1(V2_im, M1, 0, cc-1, 1, cc) - b12 * conv1(V2_re, M1, 1, cc-1, 0, cc) - b12 * V1_vec[1] * conv1(V2_re, M1, 0, cc-1, 1, cc) + b12 * conv1(V1_vec, V1_vec, 0, cc, 0, cc) + bc12 / 2 * conv1(V1_vec, V1_vec, 0, cc-1, 1, cc)],
                [-conv1(V3_re, If3_re, 1, cc-1, 0, cc) - conv1(V3_im, If3_im, 1, cc-1, 0, cc)],
                [-conv1(V2_re, V2_re, 1, cc-1, 0, cc) - conv1(V2_im, V2_im, 1, cc-1, 0, cc)],
                [-conv1(V4_re, V4_re, 1, cc-1, 0, cc) - conv1(V4_im, V4_im, 1, cc-1, 0, cc)],
                [0],
                [0],
                [0],
                [0],
                [-conv1(V3_re, If3_im, 1, cc-1, 0, cc) + conv1(V3_im, If3_re, 1, cc-1, 0, cc)],
                [G3 * (conv2(If3_re, If3_re, V3_re, 0, 0, cc-1, cc-1, 1, cc) - conv2(If3_im, If3_im, V3_re, 0, 0, cc-1, cc-1, 1, cc) - 2 * conv2(If3_re, If3_im, V3_im, 0, 0, cc-1, cc-1, 1, cc)) + g34 * (conv1(M2, V3_re, 1, cc-1, 0, cc) + conv2(M2, M2, V3_re, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(M2, V3_im, 1, cc-1, 0, cc) + conv2(M2, M2, V3_im, 1, 0, cc-1, cc, 0, cc)) - bc34 / 2 * conv2(M2, M2, V3_im, 0, 0, cc-1, cc-1, 1, cc) - g34 * (conv1(V4_re, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_re, 1, 0, cc-1, cc, 0, cc)) - g34 * (conv1(V4_im, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_im, 1, 0, cc-1, cc, 0, cc)) + b34 * (conv1(V4_im, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_re, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(V4_re, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_im, 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B2, M2, M2, V3, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))],
                [-g12 * M1[cc-1] * V1_vec[1] + g12 * conv2(M1, M1, V2_re, 0, 0, cc-1, cc-1, 0, cc) - b12 * conv2(M1, M1, V2_im, 0, 0, cc-1, cc-1, 0, cc) + np.real(G2 * conv2(If2, If2, V2, 0, 0, cc-1, cc-1, 1, cc) + bc12 / 2 * 1j * conv2(M1, M1, V2, 0, 0, cc-1, cc-1, 1, cc) + 1j * conv3(B1, M1, M1, V2, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - b34 * E_im[cc - 1]],  # aquest últim terme per tal que la matriu tingui inversa
                [G3 * (conv2(If3_re, If3_re, V3_im, 0, 0, cc-1, cc-1, 1, cc) - conv2(If3_im, If3_im, V3_im, 0, 0, cc-1, cc-1, 1, cc) - 2 * conv2(If3_re, If3_im, V3_re, 0, 0, cc-1, cc-1, 1, cc)) + b34 * (conv1(M2, V3_re, 1, cc-1, 0, cc) + conv2(M2, M2, V3_re, 1, 0, cc-1, cc, 0, cc)) + g34 * (conv1(M2, V3_im, 1, cc-1, 0, cc) + conv2(M2, M2, V3_im, 1, 0, cc-1, cc, 0, cc)) + bc34 / 2 * conv2(M2, M2, V3_re, 0, 0, cc-1, cc-1, 1, cc) - g34 * (conv1(V4_im, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_re, 1, 0, cc-1, cc, 0, cc)) - g34 * (conv1(V4_re, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_im, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(V4_re, E_re, 1, cc-1, 0, cc) + conv2(M2, V4_re, E_re, 1, 0, cc-1, cc, 0, cc)) - b34 * (conv1(V4_im, E_im, 1, cc-1, 0, cc) + conv2(M2, V4_im, E_im, 1, 0, cc-1, cc, 0, cc) + np.imag(1j * conv3(B2, M2, M2, V3, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)))],
                [-b12 * M1[cc-1] * V1_vec[1] + g12 * conv2(M1, M1, V2_im, 0, 0, cc-1, cc-1, 0, cc) + b12 * conv2(M1, M1, V2_re, 0, 0, cc-1, cc-1, 0, cc) + np.imag(G2 * conv2(If2, If2, V2, 0, 0, cc-1, cc-1, 1, cc) + bc12 / 2 * 1j * conv2(M1, M1, V2, 0, 0, cc-1, cc-1, 1, cc) + 1j * conv3(B1, M1, M1, V2, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))],
                [-conv1(E_re, E_re, 1, cc-1, 0, cc) - conv1(E_im, E_im, 1, cc-1, 0, cc)],
                [bc34 / 2 * V4_im[cc-1] + g34 * conv1(V3_re, E_re, 1, cc-1, 0, cc) + g34 * conv2(M2, V3_re, E_re, 1, 0, cc-1, cc, 0, cc) - b34 * conv1(V3_re, E_im, 1, cc-1, 0, cc) - b34 * conv2(M2, V3_re, E_im, 1, 0, cc-1, cc, 0, cc) + g34 * conv1(V3_im, E_im, 1, cc-1, 0, cc) + g34 * conv2(M2, V3_im, E_im, 1, 0, cc-1, cc, 0, cc) - b34 * conv1(V3_im, E_re, 1, cc-1, 0, cc) - b34 * conv2(M2, V3_im, E_re, 1, 0, cc-1, cc, 0, cc) + np.real((P4 - Q4 * 1j) * X4[cc-1])],  
                [-bc34 / 2 * V4_re[cc-1] - g34 * conv1(V3_re, E_im, 1, cc-1, 0, cc) - g34 * conv2(M2, V3_re, E_im, 1, 0, cc-1, cc, 0, cc) + b34 * conv1(V3_re, E_re, 1, cc-1, 0, cc) + b34 * conv2(M2, V3_re, E_re, 1, 0, cc-1, cc, 0, cc) + g34 * conv1(V3_im, E_re, 1, cc-1, 0, cc) + g34 * conv2(M2, V3_im, E_re, 1, 0, cc-1, cc, 0, cc) - b34 * conv1(V3_im, E_im, 1, cc-1, 0, cc) - b34 * conv2(M2, V3_im, E_im, 1, 0, cc-1, cc, 0, cc) + np.imag((P4 - Q4 * 1j) * X4[cc-1])]
                ])

    lhs = np.real(np.dot(mat_inv, rhs))

    V2_re[cc] = lhs[0]
    V2_im[cc] = lhs[1]
    V2[cc] = V2_re[cc] + 1j * V2_im[cc]
    V3_re[cc] = lhs[2]
    V3_im[cc] = lhs[3]
    V3[cc] = V3_re[cc] + 1j * V3_im[cc]
    V4_re[cc] = lhs[4]
    V4_im[cc] = lhs[5]
    V4[cc] = V4_re[cc] + 1j * V4_im[cc]
    If3_re[cc] = lhs[6]
    If3_im[cc] = lhs[7]
    If3[cc] = If3_re[cc] + 1j * If3_im[cc]
    If2_re[cc] = lhs[8]
    If2_im[cc] = lhs[9]
    If2[cc] = If2_re[cc] + 1j * If2_im[cc]
    M1[cc] = lhs[10]
    M2[cc] = lhs[11]
    B1[cc-1] = lhs[12]
    E_re[cc] = lhs[13]
    E_im[cc] = lhs[14]
    E[cc] = E_re[cc] + 1j * E_im[cc]
    B2[cc-1] = lhs[15]

    X4[cc] = - conv1(X4, np.conj(V4), 0, cc-1, 0, cc)

V2f = pade(prof - 1, V2, 1)
V3f = pade(prof - 1, V3, 1)
V4f = pade(prof - 1, V4, 1)
If2f = pade(prof - 1, If2, 1)
If3f = pade(prof - 1, If3, 1)
M1f = pade(prof - 1, M1, 1)
B1f = pade(prof - 2, B1, 1)
B2f = pade(prof - 2, B2, 1)
Ef = pade(prof - 1, E, 1)
M2f = pade(prof - 1, M2, 1)
X4f = pade(prof - 1, X4, 1)

print(V2f)
print(V3f)
print(V4f)
print(If2f)
print(If3f)
print(M1f)
print(B1f)
print(B2f)
print(Ef)
print(M2f)
print(X4f)

cos11 = g34 * np.real(V4f) - b34 * np.imag(V4f) - bc34 / 2 * np.imag(V4f) - M2f * (np.real(V3f) * g34 * np.real(Ef) - b34 * np.imag(Ef) * np.real(V3f) + g34 * np.imag(Ef) * np.imag(V3f) - b34 * np.real(Ef) * np.imag(V3f))
cos12 = np.real((P4 - Q4 * 1j) / np.conj(V4f))
print(cos11)
print(cos12)