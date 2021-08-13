import numpy as np
from Pade import pade

# he girat els signes a l'eq. de la reactiva de l'slack, s'hauria de girar i ja està

g = 5
b = -10
V1 = 1.05
Gx = 10
bc = 0.2
Pdc = -0.1  # positiva si genera, negativa si consumeix
W2 = 0.95  # el mòdul de V2 al quadrat
Q1 = 0.1

prof = 60

V2 = np.zeros(prof, dtype=complex)
V2_re = np.zeros(prof)
V2_im = np.zeros(prof)

If = np.zeros(prof, dtype=complex)
If_re = np.zeros(prof)
If_im = np.zeros(prof)

Beq = np.zeros(prof)
M = np.zeros(prof)

V2[0] = 1
V2_re[0] = 1
V2_im[0] = 0

If[0] = 0
If_re[0] = 0
If_im[0] = 0

M[0] = 1

mat = np.array([[b, g, 0, 0, b, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [2, 0, 0, 0, 0, 0], [-g, b, 1, 0, -g, 0], [-b, -g, 0, 1, -b, -1]])
mat_inv = np.linalg.inv(mat)


# [1]

def conv2(a, b, c, k, j, lim1, lim2, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        for jj in range(j, lim2 + 1 - kk):
            suma += a[kk] * b[jj] * c[cc - kk - jj - retard]
    return suma

# conv3 a partir de c>=2

cc = 1
rhs = np.array([[+Q1 + b * (V1 - 1) + bc / 2],  # canvio aquest signe, sí, ha de ser així
                [Pdc], 
                [0],
                [W2 - 1],
                [-g * M[cc-1] * (V1 - 1) + g * conv2(M, M, V2_re, 0, 0, cc-1, cc-1, 0, cc) - b * conv2(M, M, V2_im, 0, 0, cc-1, cc-1, 0, cc) + np.real(Gx * conv2(If, If, V2, 0, 0, cc-1, cc-1, 1, cc)) + 1j * bc / 2 * conv2(M, M, V2, 0, 0, cc-1, cc-1, 1, cc)],
                [-b * M[cc-1] * (V1 - 1) + g * conv2(M, M, V2_im, 0, 0, cc-1, cc-1, 0, cc) + b * conv2(M, M, V2_re, 0, 0, cc-1, cc-1, 0, cc) + np.imag(Gx * conv2(If, If, V2, 0, 0, cc-1, cc-1, 1, cc)) + 1j * bc / 2 * conv2(M, M, V2, 0, 0, cc-1, cc-1, 1, cc)]
                ])

lhs = np.real(np.dot(mat_inv, rhs))

V2_re[1] = lhs[0]
V2_im[1] = lhs[1]
If_re[1] = lhs[2]
If_im[1] = lhs[3]
M[1] = lhs[4]
Beq[0] = lhs[5]

V2[1] = V2_re[1] + V2_im[1] * 1j
If[1] = If_re[1] + If_im[1] * 1j


def conv1(a, b, k, lim1, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        suma += a[kk] * b[cc - kk - retard]
    return suma


def conv3(a, b, c, d, k, j, i, lim1, lim2, lim3, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        for jj in range(j, lim2 + 1 - kk):
            for ii in range(i, lim3 + 1 - kk - jj):
                suma += a[kk] * b[jj] * c[ii] * d[cc - kk - jj - ii - retard]
    return suma

V1_vec = np.zeros(prof)
V1_vec[0] = 1
V1_vec[1] = V1 - 1

for cc in range(2, prof):
    rhs = np.array([[-g * conv1(V2_im, M, 1, cc-1, 0, cc) - g * (V1 - 1) * conv1(V2_im, M, 0, cc-1, 1, cc) - b * conv1(V2_re, M, 1, cc-1, 0, cc) - b * (V1 - 1) * conv1(V2_re, M, 0, cc-1, 1, cc) + b * conv1(V1_vec, V1_vec, 0, cc, 0, cc) + bc / 2 * conv1(V1_vec, V1_vec, 0, cc-1, 1, cc)], 
                [-conv1(V2_re, If_re, 1, cc-1, 0, cc) - conv1(V2_im, If_im, 1, cc-1, 0, cc)], 
                [-conv1(V2_re, If_im, 1, cc-1, 0, cc) + conv1(V2_im, If_re, 1, cc-1, 0, cc)],
                [-conv1(V2_re, V2_re, 1, cc-1, 0, cc) - conv1(V2_im, V2_im, 1, cc-1, 0, cc)],
                [-g * M[cc-1] * (V1 - 1) + g * conv2(M, M, V2_re, 0, 0, cc-1, cc-1, 0, cc) - b * conv2(M, M, V2_im, 0, 0, cc-1, cc-1, 0, cc) + np.real(Gx * conv2(If, If, V2, 0, 0, cc-1, cc-1, 1, cc)) + 1j * bc / 2 * conv2(M, M, V2, 0, 0, cc-1, cc-1, 1, cc) + 1j * conv3(Beq, M, M, V2, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)],
                [-b * M[cc-1] * (V1 - 1) + g * conv2(M, M, V2_im, 0, 0, cc-1, cc-1, 0, cc) + b * conv2(M, M, V2_re, 0, 0, cc-1, cc-1, 0, cc) + np.imag(Gx * conv2(If, If, V2, 0, 0, cc-1, cc-1, 1, cc)) + 1j * bc / 2 * conv2(M, M, V2, 0, 0, cc-1, cc-1, 1, cc) + 1j * conv3(Beq, M, M, V2, 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)]
                ])

    lhs = np.real(np.dot(mat_inv, rhs))

    V2_re[cc] = lhs[0]
    V2_im[cc] = lhs[1]
    If_re[cc] = lhs[2]
    If_im[cc] = lhs[3]
    M[cc] = lhs[4]
    Beq[cc-1] = lhs[5]

    V2[cc] = V2_re[cc] + V2_im[cc] * 1j
    If[cc] = If_re[cc] + If_im[cc] * 1j


V2ff = pade(prof, V2, 1)
Beqff = pade(prof-1, Beq, 1)
Mff = pade(prof, M, 1)
Ifff = pade(prof, If, 1)

print(V2ff)
print(Ifff)
print(Beqff)
print(Mff)

It = - (g + b * 1j) * Mff * V2ff + (g + b * 1j + bc / 2 * 1j) * V1
print(V1 * np.conj(It))