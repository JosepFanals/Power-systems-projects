# 2 bus system, one slack and 1 PQ
# P and Q are parameters, v1 = e + jf are states
# we assume slack v = 1 + j0
# the parameters move between 1 and -1

import numpy as np
import matplotlib.pyplot as plt

# data
z1 = 0.01 + 0.05j
y1 = 1 / z1
g = np.real(y1)
b = np.imag(y1)

# init 
e = np.array([0.9, 0, 0, 0, 0, 0])
f = np.array([0.9, 0, 0, 0, 0, 0])
J = np.zeros((2,2))

# step 0, solve NR with some iterations
for k in range(10):
    Af1 = g * e[0] ** 2 - g * e[0] + g * f[0] ** 2 - b * f[0]
    Af2 = g * f[0] + b * e[0] ** 2 - b * e[0] + b * f[0] ** 2
    Af = np.array([Af1, Af2])

    J[0,0] = 2 * g * e[0] - g
    J[0,1] = 2 * g * f[0] - b
    J[1,0] = 2 * b * e[0] - b
    J[1,1] = 2 * b * f[0] + g

    Ax = - np.dot(np.linalg.inv(J), Af)
    e[0] += Ax[0]
    f[0] += Ax[1]

# step 1
Af1 = g * (2 * e[1] * e[0]) - g * e[1] + g * (2 * f[1] * f[0]) - b * f[1] - 1
Af2 = g * f[1] + b * (2 * e[1] * e[0]) - b * e[1] + b * (2 * f[1] * f[0]) + 0
Af = np.array([Af1, Af2])

J[0,0] = 2 * g * e[0] - g
J[0,1] = 2 * g * f[0] - b
J[1,0] = 2 * b * e[0] - b
J[1,1] = 2 * b * f[0] + g

Jinv = np.linalg.inv(J)

Ax = - np.dot(Jinv, Af)
e[1] += Ax[0]
f[1] += Ax[1]

# step 2
Af1 = g * (2 * e[2] * e[0] + e[1] * e[1]) - g * e[2] + g * (2 * f[2] * f[0] + f[1] * f[1]) - b * f[2] - 0
Af2 = g * f[2] + b * (2 * e[2] * e[0] + e[1] * e[1]) - b * e[2] + b * (2 * f[2] * f[0] + f[1] * f[1]) + 1
Af = np.array([Af1, Af2])

Ax = - np.dot(Jinv, Af)
e[2] += Ax[0]
f[2] += Ax[1]

# step 3
Af1 = g * (2 * e[3] * e[0] + e[1] * e[2] + e[2] * e[1]) - g * e[3] + g * (2 * f[3] * f[0] + f[1] * f[2] + f[2] * f[1]) - b * f[3] - 0
Af2 = g * f[3] + b * (2 * e[3] * e[0] + e[1] * e[2] + e[2] * e[1]) - b * e[3] + b * (2 * f[3] * f[0] + f[1] * f[2] + f[2] * f[1]) + 0
Af = np.array([Af1, Af2])

Ax = - np.dot(Jinv, Af)
e[3] += Ax[0]
f[3] += Ax[1]

# step 4
Af1 = g * (2 * e[4] * e[0] + e[3] * e[1] + e[2] * e[2] + e[3] * e[1]) - g * e[4] + g * (2 * f[4] * f[0] + f[3] * f[1] + f[2] * f[2] + f[3] * f[1]) - b * f[4] - 0
Af2 = g * f[4] + b * (2 * e[4] * e[0] + e[3] * e[1] + e[2] * e[2] + e[3] * e[1]) - b * e[4] + b * (2 * f[4] * f[0] + f[1] * f[3] + f[2] * f[2] + f[3] * f[1]) + 0
Af = np.array([Af1, Af2])

Ax = - np.dot(Jinv, Af)
e[4] += Ax[0]
f[4] += Ax[1]

# step 5
Af1 = g * (2 * e[5] * e[0] + e[1] * e[4] + e[2] * e[3] + e[3] * e[2] + e[4] * e[1]) - g * e[5] + g * (2 * f[5] * f[0] + f[1] * f[4] + f[2] * f[3] + f[3] * f[2] + f[4] * f[1]) - b * f[5] - 0
Af2 = g * f[5] + b * (2 * e[5] * e[0] + e[1] * e[4] + e[2] * e[3] + e[3] * e[2] + e[4] * e[1]) - b * e[5] + b * (2 * f[5] * f[0] + f[1] * f[4] + f[2] * f[3] + f[3] * f[2] + f[4] * f[1]) + 0
Af = np.array([Af1, Af2])

Ax = - np.dot(Jinv, Af)
e[5] += Ax[0]
f[5] += Ax[1]

print(e, f)

# calculate solution to particular cases and debug
p = 0.5
q = 0

phi = [1, p, q, p * q, 1 / 2 * (3 * p ** 2 - 1), 1 / 2 * (3 * q ** 2 - 1)]

e_fin = e[0] * phi[0] + e[1] * phi[1] + e[2] * phi[2] + e[3] * phi[3] + e[4] * phi[4] + e[5] * phi[5]
f_fin = f[0] * phi[0] + f[1] * phi[1] + f[2] * phi[2] + f[3] * phi[3] + f[4] * phi[4] + f[5] * phi[5]

v_fin = e_fin + 1j * f_fin
vv = abs(v_fin)

print(v_fin)


