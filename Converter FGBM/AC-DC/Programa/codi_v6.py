# segueixo una altra incrustació, l'slack fa de referència
# al costat del codi_v5, corregeixo el retard de la 2.A que ho havia tingut en compte
# amb aquests valors es treu un error petit

import numpy as np

Pdc = 0.1
Rdc = 10
Xc = 0.05
U = 1.1
a = 2
k1 = 0.995 * 3 * np.sqrt(2) / np.pi
G = 1
B = -1

Idc = np.sqrt(Pdc / Rdc)
Vdc = Idc * Rdc

prof = 60

Vre = np.zeros(prof)
Vim = np.zeros(prof)
alpha = np.zeros(prof)
Vre2 = np.zeros(prof)
Vim2 = np.zeros(prof)


def V2(a, b, lim):
    suma = 0
    for k in range(0, lim + 1):
        suma += a[k] * b[lim - k]
    return suma


Vre[0] = 1
Vim[0] = 0
Vre2[0] = V2(Vre, Vre, 0)
Vim2[0] = V2(Vim, Vim, 0)
alpha[0] = np.sqrt((Vdc / (k1 * a) + (3 * Xc * Idc) / (np.pi * k1 * a)) ** 2)

M = np.array([[G - 2 * G, -B + 2 * B], [2, 0]])
Minv = np.linalg.inv(M)
rhs = np.array([[Pdc - G * Vre[0] * (U - 1)], 
                [(a * k1 * Idc) ** 2 / (G ** 2 + B ** 2) - 2 * (U - 1) + 0]  # he canviat això
               ])
lhs = np.dot(Minv, rhs)

Vre[1] = lhs[0]
Vim[1] = lhs[1]

Vre2[1] = V2(Vre, Vre, 1)
Vim2[1] = V2(Vim, Vim, 1)

alpha[1] = - Vre[1] * alpha[0] / Vre[0]


def conv2(a, b, j, lim):
    suma = 0
    for k in range(j, lim):
        suma += a[k] * b[lim - k]
    return suma


def conv3(a, b, c, lim):
    suma = 0
    for k in range(1, lim + 1):
        for kk in range(0, lim + 1 - k):
            suma += a[k] * b[kk] * c[lim - k - kk]
    return suma


rhs = np.array([[- G * Vre[1] * (U - 1) + B * Vim[1] * (U - 1) - 2 * B * conv2(Vre, Vim, 1, 2) + G * conv2(Vre, Vre, 1, 2) - G * conv2(Vim, Vim, 1, 2)], 
                [- (U - 1) * (U - 1) + 2 * 1 * Vre[1] + 2 * (U - 1) * Vre[0] - conv2(Vre, Vre, 1, 2) - conv2(Vim, Vim, 1, 2)]
               ])
lhs = np.dot(Minv, rhs)
Vre[2] = lhs[0]
Vim[2] = lhs[1]

Vre2[2] = V2(Vre, Vre, 2)
Vim2[2] = V2(Vim, Vim, 2)

alpha[2] = (- Vre2[0] * conv2(alpha, alpha, 1, 2) - conv3(Vre2, alpha, alpha, 2) - conv3(Vim2, alpha, alpha, 2)) / (2 * Vre2[0] * alpha[0])

for lk in range(3, prof):
    rhs = np.array([[- G * Vre[lk - 1] * (U - 1) + B * Vim[lk - 1] * (U - 1) - 2 * B * conv2(Vre, Vim, 1, lk) + G * conv2(Vre, Vre, 1, lk) - G * conv2(Vim, Vim, 1, lk)], 
                   [+ 2 * (U - 1) * Vre[lk - 2] + 2 * 1 * Vre[lk - 1] - conv2(Vre, Vre, 1, lk) - conv2(Vim, Vim, 1, lk)]
                   ])
    
    lhs = np.dot(Minv, rhs)
    Vre[lk] = lhs[0]
    Vim[lk] = lhs[1]

    Vre2[lk] = V2(Vre, Vre, lk)
    Vim2[lk] = V2(Vim, Vim, lk)

    alpha[lk] = (- Vre2[0] * conv2(alpha, alpha, 1, lk) - conv3(Vre2, alpha, alpha, lk) - conv3(Vim2, alpha, alpha, lk)) / (2 * Vre2[0] * alpha[0])


from Pade import pade

Vre_f = pade(prof-1, Vre, 1)
Vim_f = pade(prof-1, Vim, 1)
alpha_f = pade(prof-1, alpha, 1)
print(np.real(Vre_f))
print(np.real(Vim_f))
print(np.real(alpha_f))
print(Vdc)
print(Idc)

compr3 = (Vdc / (k1 * a) + (3 * Xc * Idc) / (np.pi * k1 * a)) ** 2 - Vre_f ** 2 * alpha_f ** 2 - Vim_f ** 2 * alpha_f ** 2
print(np.real(compr3))
compr2 = (a * k1 * Idc) ** 2 / (G ** 2 + B ** 2) - U * U - Vre_f ** 2 - Vim_f ** 2 + 2 * U * Vre_f
print(np.real(compr2))
compr1 = Pdc - Vre_f * U * G + Vim_f * U * B - 2 * Vre_f * Vim_f * B + Vre_f ** 2 * G - Vim_f ** 2 * G
print(np.real(compr1))
