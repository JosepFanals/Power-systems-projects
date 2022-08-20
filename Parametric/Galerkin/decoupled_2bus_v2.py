# 2 bus system, one slack and 1 PQ
# P and Q are parameters, v1 = e + jf are states
# we assume slack v = 1 + j0
# the parameters move between 1 and -1

import numpy as np
import matplotlib.pyplot as plt


def conv(vec, s):
    sum = 0
    for k in range(1, s):
        sum += vec[k] * vec[s - k]
    return sum


# data
z1 = 0.01 + 0.05j
y1 = 1 / z1
g = np.real(y1)
b = np.imag(y1)

# init 
n_steps = 10
e = np.zeros(n_steps)
f = np.zeros(n_steps)
coef_p = np.zeros(n_steps)
coef_q = np.zeros(n_steps)

e[0] = 0.9
f[0] = 0.1

coef_p[1] = 1
coef_q[2] = 1

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

    Jinv = np.linalg.inv(J)

    Ax = - np.dot(Jinv, Af)
    e[0] += Ax[0]
    f[0] += Ax[1]

for s in range(1, n_steps):
    # step 1
    Af1 = g * (2*e[s]*e[0] + conv(e, s)) - g*e[s] + g * (2*f[s]*f[0] + conv(f, s)) - b*f[s] - coef_p[s]
    Af2 = g*f[s] + b * (2*e[s]*e[0] + conv(e, s)) - b*e[s] + b * (2*f[s]*f[0] + conv(f, s)) + coef_q[s]

    Af = np.array([Af1, Af2])

    Ax = - np.dot(Jinv, Af)
    e[s] += Ax[0]
    f[s] += Ax[1]


# coeffs
print('Coefficients: ')
print('e: ', e)
print('f: ', f)


# calculate solution to particular cases and debug
pvec = np.arange(-0, 1, 0.02)
qvec = np.arange(-0, 1, 0.02)

err_vec = []
p_point = []
q_point = []

for p in pvec:
    for q in qvec:
        phi = [1, p, q, p * q, 1 / 2 * (3 * p ** 2 - 1), 1 / 2 * (3 * q ** 2 - 1), p * 1 / 2 * (3 * q ** 2 - 1), q * 1 / 2 * (3 * p ** 2 - 1), 1 / 2 * (5 * p ** 3 - 1), 1 / 2 * (5 * q ** 3 - 1)]

        e_fin = 0
        f_fin = 0
        for k in range(len(phi)):
            e_fin += e[k] * phi[k]
            f_fin += f[k] * phi[k]

        v_fin = e_fin + 1j * f_fin

        # error:
        iline = (v_fin - (1.0 + 0.0 * 1j)) / z1
        iload = np.conjugate((p + 1j * q) / v_fin)

        err = iline - iload
        err_vec.append(abs(err))
        p_point.append(p)
        q_point.append(q)


ax = plt.axes(projection='3d')
ax.plot3D(p_point, q_point, err_vec, 'white')
ax.scatter3D(p_point, q_point, err_vec, c=err_vec, cmap='Greens')
plt.show()



for i,v in enumerate(err_vec):
    print(v)