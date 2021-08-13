# amb sistema matricial, a veure si convergeix així
import numpy as np

Pdc = 0.1
Rdc = 10
Xc = 0.5
U = 1.05
a = 5
k1 = 0.995 * 3 * np.sqrt(2) / np.pi
G = 5
B = -5

Idc = np.sqrt(Pdc / Rdc)
Vdc = Idc * Rdc

Vre = Vdc / (k1 * a)

prof = 100

V = np.zeros(prof)
alpha = np.zeros(prof)
ctheta = np.zeros(prof)
stheta = np.zeros(prof)
cdelta = np.zeros(prof)
sdelta = np.zeros(prof)


# Termes [0]

V[0] = Vdc / (k1 * a) + (3 * Xc * Idc) / (k1 * a * np.pi)
ctheta[0] = Vdc / (k1 * a * V[0])
stheta[0] = np.sqrt(1 - ctheta[0] * ctheta[0])

M2 = np.zeros((2, 2), dtype=float)

M2[0, 0] = G * U
M2[0, 1] = -B * U
M2[1, 0] = B * U
M2[1, 1] = G * U

rhs2 = np.zeros((2, 1), dtype=float)

rhs2[0, 0] = a * k1 * Idc + G * V[0] * ctheta[0] - B * V[0] * stheta[0]
rhs2[1, 0] = G * V[0] * stheta[0] + B * V[0] * ctheta [0]

lhs2 = np.dot(np.linalg.inv(M2), rhs2)

cdelta[0] = lhs2[0]
sdelta[0] = lhs2[1]


# Termes [1]

M = np.array([[ctheta[0], V[0], 0, 0, 0], 
              [-G * ctheta[0] + B * stheta[0], -G * V[0], B * V[0], G * U, - B * U],
              [-G * stheta[0] - B * ctheta[0], -B * V[0], -G * V[0], B * U, G * U],
              [0, 2 * ctheta[0], 2 * stheta[0], 0, 0],
              [0, 0, 0, 2 * cdelta[0], 2 * sdelta[0]]
              ])

rhs = np.array([[0], 
               [0], 
               [0], 
               [0], 
               [0]
               ])

Minv = np.linalg.inv(M)

lhs = np.dot(Minv, rhs)

print(rhs)
print(lhs)
print(Minv)

V[1] = lhs[0]
ctheta[1] = lhs[1]
stheta[1] = lhs[2]
cdelta[1] = lhs[3]
sdelta[1] = lhs[4]


# Termes [c>=2]

def conv(a, b, j, lim, rest):
    suma = 0
    for k in range(j, lim + 1):
        suma += a[k] * b[lim - k - rest]
    return suma


for c in range(2, prof):
    rhs = [- conv(V, ctheta, 1, c-1, 0),
           - conv(V, alpha, 1, c-1, 1),
           + G * conv(V, ctheta, 1, c-1, 0) - B * conv(V, stheta, 1, c-1, 0),
           + G * conv(V, stheta, 1, c-1, 0) + B * conv(V, ctheta, 1, c-1, 0),
           - conv(stheta, stheta, 1, c-1, 0) - conv(ctheta, ctheta, 1, c-1, 0),
           - conv(sdelta, sdelta, 1, c-1, 0) - conv(cdelta, cdelta, 1, c-1, 0)
          ]
   
    lhs = np.dot(Minv, rhs)
    V[c] = lhs[0]
    alpha[c-1] = lhs[1]
    ctheta[c] = lhs[2]
    stheta[c] = lhs[3]
    cdelta[c] = lhs[4]
    sdelta[c] = lhs[5]


from Pade import pade

print(V)

V_fi = pade(prof-1, V, 1)
alpha_fi = pade(prof-2, alpha, 1)
ctheta_fi = pade(prof-1, ctheta, 1)
stheta_fi = pade(prof-1, stheta, 1)
cdelta_fi = pade(prof-1, cdelta, 1)
sdelta_fi = pade(prof-1, sdelta, 1)
