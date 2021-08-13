# Santiago Peñate Vera and Josep Fanals i Batllori
# main reference: D. Borzacchiello, F. Chinesta, H. Malik, R. García-Blanco and P. Díez, "Unified formulation of a family of iterative solvers for power systems analysis," in Electric Power Systems Research, vol. 140, pp. 201–208, 2016.
# change gamma, psi and prof (number of iterations) accordingly

# 1.1, no funciona

import time
import pandas as pd
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

df_b = pd.read_excel('Code/data_IEEE14.xlsx', sheet_name="buses")
df_l = pd.read_excel('Code/data_IEEE14.xlsx', sheet_name="lines")

n_b = 0
n_pq = 0
n_pv = 0
pq = []
pv = []
pq0 = []  # store pq buses indices relative to its own
pv0 = []  # store pv buses indices relative to its own
d_pq = {}  # dict of pq
d_pv = {}  # dict of pv
for i in range(len(df_b)):
    if df_b.iloc[i, 4] == "slack":  # index 0 is reserved for the slack bus
        pass

    elif df_b.iloc[i, 4] == "PQ":
        pq0.append(n_pq)
        d_pq[df_b.iloc[i, 0]] = n_pq
        n_b += 1
        n_pq += 1
        pq.append(df_b.iloc[i, 0] - 1)
        
    elif df_b.iloc[i, 4] == "PV":
        pv0.append(n_pv)
        d_pv[df_b.iloc[i, 0]] = n_pv
        n_b += 1
        n_pv += 1
        pv.append(df_b.iloc[i, 0] - 1)

n_l = len(df_l)  # number of lines

V0 = df_b.iloc[0, 3]  # the slack is always positioned in the first row
I0_pq = np.zeros(n_pq, dtype=complex)
I0_pv = np.zeros(n_pv, dtype=complex)
Y = np.zeros((n_b, n_b), dtype=complex)  # I will build it with block matrices
Y11 = np.zeros((n_pq, n_pq), dtype=complex)  # pq pq
Y12 = np.zeros((n_pq, n_pv), dtype=complex)  # pq pv
Y21 = np.zeros((n_pv, n_pq), dtype=complex)  # pv pq
Y22 = np.zeros((n_pv, n_pv), dtype=complex)  # pv pv

for i in range(n_l):
    Ys = 1 / (df_l.iloc[i, 2] + 1j * df_l.iloc[i, 3])  # series element
    Ysh = df_l.iloc[i, 4] + 1j * df_l.iloc[i, 5]  # shunt element
    t = df_l.iloc[i, 6] * np.cos(df_l.iloc[i, 7]) + 1j * df_l.iloc[i, 6] * np.sin(df_l.iloc[i, 7])  # tap as a complex number

    a = df_l.iloc[i, 0]
    b = df_l.iloc[i, 1]

    if a == 0:
        if b - 1 in pq:
            I0_pq[d_pq[b]] += V0 * Ys / t
            Y11[d_pq[b], d_pq[b]] += Ys + Ysh
        if b - 1 in pv:
            I0_pv[d_pv[b]] += V0 * Ys / t
            Y22[d_pv[b], d_pv[b]] += Ys + Ysh

    elif b == 0:
        if a - 1 in pq:
            I0_pq[d_pq[a]] += V0 * Ys / np.conj(t)
            Y11[d_pq[a], d_pq[a]] += (Ys + Ysh) / (t * np.conj(t))
        if a - 1 in pv:
            I0_pv[d_pv[a]] += V0 * Ys / np.conj(t)
            Y22[d_pv[a], d_pv[a]] += (Ys + Ysh) / (t * np.conj(t))

    else:
        if a - 1 in pq and b - 1 in pq:
            Y11[d_pq[a], d_pq[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y11[d_pq[b], d_pq[b]] += Ys + Ysh
            Y11[d_pq[a], d_pq[b]] += - Ys / np.conj(t)
            Y11[d_pq[b], d_pq[a]] += - Ys / t
        
        if a - 1 in pq and b - 1 in pv:
            Y11[d_pq[a], d_pq[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y22[d_pv[b], d_pv[b]] += Ys + Ysh
            Y12[d_pq[a], d_pv[b]] += - Ys / np.conj(t)
            Y21[d_pv[b], d_pq[a]] += - Ys / t

        if a - 1 in pv and b - 1 in pq:
            Y22[d_pv[a], d_pv[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y11[d_pq[b], d_pq[b]] += Ys + Ysh
            Y21[d_pv[a], d_pq[b]] += - Ys / np.conj(t)
            Y12[d_pq[b], d_pv[a]] += - Ys / t

        if a - 1 in pv and b - 1 in pv:
            Y22[d_pv[a], d_pv[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y22[d_pv[b], d_pv[b]] += Ys + Ysh
            Y22[d_pv[a], d_pv[b]] += - Ys / np.conj(t)
            Y22[d_pv[b], d_pv[a]] += - Ys / t



for i in range(len(df_b)):  # add shunts connected directly to the bus
    a = df_b.iloc[i, 0]
    if a - 1 in pq:
        print(d_pq[a])
        Y11[d_pq[a], d_pq[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]
    elif a - 1 in pv:
        print(d_pv[a])
        Y22[d_pv[a], d_pv[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]


Y = np.block([[Y11, Y12], [Y21, Y22]])

V_mod = np.zeros(n_pv, dtype=float)
P_pq = np.zeros(n_pq, dtype=float)
P_pv = np.zeros(n_pv, dtype=float)
Q_pq = np.zeros(n_pq, dtype=float)
for i in range(len(df_b)):
    if df_b.iloc[i, 4] == "PV":
        V_mod[d_pv[df_b.iloc[i, 0]]] = df_b.iloc[i, 3]
        P_pv[d_pv[df_b.iloc[i, 0]]] = df_b.iloc[i, 1]
    elif df_b.iloc[i, 4] == "PQ":
        Q_pq[d_pq[df_b.iloc[i, 0]]] = df_b.iloc[i, 2]
        P_pq[d_pq[df_b.iloc[i, 0]]] = df_b.iloc[i, 1]

# scaling powers
lam = 1
P_pv = lam * P_pv
P_pq = lam * P_pq
Q_pq = lam * Q_pq


Vb = 1 + 0 * 1j  # could be another value and maybe different for each bus
alpha = np.zeros((n_b, n_b), dtype=complex)
alpha_pq = np.zeros(n_pq, dtype=complex)  # vector
alpha_pv = np.zeros(n_pv, dtype=complex)  # vector
beta = np.zeros((n_b, n_b), dtype=complex)

psi = 2.01
for i in range(n_pq):  # make sure the indexation is correct
    # alpha_pq[i] = psi * ((P_pq[i] - 1j * Q_pq[i]) / Vb ** 2)
    alpha_pq[i] = (P_pq[i] - 1j * Q_pq[i]) / Vb ** 2

for i in range(n_pv):
    # alpha_pv[i] = psi * ((P_pv[i]) / Vb ** 2)
    alpha_pv[i] = (P_pv[i]) / Vb ** 2

alpha_vec = np.block([alpha_pq, alpha_pv])
alpha = np.diag(alpha_vec)

for i in range(n_b):
    # beta[i, i] = psi * (Y[i, i] + alpha[i, i])
    beta[i, i] = Y[i, i] - alpha[i, i]

beta_pq = np.zeros(n_pq, dtype=complex)
beta_pv = np.zeros(n_pv, dtype=complex)
for i in range(n_pq):
    beta_pq[i] = beta[i, i]
for i in range(n_pv):
    beta_pv[i] = beta[i + n_pq, i + n_pq]
beta_vec = np.block([beta_pq, beta_pv])


Yalpha = Y - alpha
Yalpha_inv = np.linalg.inv(Yalpha)
Ybeta = Y - beta

I0 = np.block([I0_pq, I0_pv])

V1 = np.dot(Yalpha_inv, I0)  # V^(l)
V1_pq = V1[:n_pq]
V1_pv = V1[n_pq:n_b]
V2 = np.zeros(n_b, dtype=complex)  # V^(l + 1/2)
V2_pq = np.zeros(n_pq, dtype=complex)
V2_pv = np.zeros(n_pv, dtype=complex)
Q1 = np.zeros(n_pv, dtype=float)
Q2 = np.zeros(n_pv, dtype=float)
gamma = 0.25  # try different values

rhs_pq = np.zeros(n_pq, dtype=complex)
rhs_pv = np.zeros(n_pv, dtype=complex)
A = np.zeros(n_b, dtype=complex)
sigma = np.zeros(n_b, dtype=complex)
U = np.zeros(n_b, dtype=complex)


Y_kron = Y22 + np.dot(Y21, np.dot(np.linalg.inv(Y11), Y12))
Ypv = np.block([Y21, Y22])

start_time = time.time()

prof = 50  # number of iterations
for c in range(prof):

    # GLOBAL STEP
    for i in range(n_pq):
        if c == 0:
            rhs_pq[i] = (P_pq[i] - 1j * Q_pq[i]) / np.conj(V1_pq[i]) - alpha_pq[i] * V1_pq[i] + I0_pq[i]
        else:
            rhs_pq[i] = I1[i] - alpha_pq[i] * V1_pq[i] + I0_pq[i]

    for i in range(n_pv):
        if c == 0:
            rhs_pv[i] = (P_pv[i] - 1j * Q1[i]) / np.conj(V1_pv[i]) - alpha_pv[i] * V1_pv[i] + I0_pv[i]
        else:
            rhs_pv[i] = I1[i + n_pq] - alpha_pv[i] * V1_pv[i] + I0_pv[i]

    lhs = np.dot(Yalpha_inv, np.block([rhs_pq, rhs_pv]))
    V2[:] = lhs[:]
    V2_pq = V2[:n_pq]
    V2_pv = V2[n_pq:n_b]
    I2 = np.dot(Y, V2) - I0


    # PV BUS MODELLING ACORDING TO THE MAIN REFERENCE
    # err_V2_pv = np.zeros(n_pv, dtype=complex)
    # for i in range(n_pv):
    #     err_V2_pv[i] = (V_mod[i] - np.abs(V2_pv[i])) * V2_pv[i] / np.abs(V2_pv[i])

    # err_I2_pv = np.dot(Y_kron, err_V2_pv)
    # err_Q2_pv = np.zeros(n_pv, dtype=float)
    # for i in range(n_pv):
    #     err_Q2_pv[i] = np.imag(V2_pv[i] * np.conj(err_I2_pv[i]))
    # for i in range(n_pv):
    #     Q1[i] = Q1[i] + gamma * err_Q2_pv[i]


    # NEW PV BUS MODELLING
    for i in range(n_pv):
        V2_pv[i] = V2_pv[i] * V_mod[i] / np.abs(V2_pv[i])

    V2[n_pq:n_b] = V2_pv
    Q1_nou = np.imag(V2_pv * np.conj(np.dot(Ypv, V2) - I0_pv))
    for i in range(n_pv):
        Q1[i] = Q1[i] + gamma * (Q1_nou[i] - Q1[i])
    
    # print(Q1)

    # LOCAL STEP
    
    S_c = np.zeros(n_b, dtype=complex)

    for i in range(n_pq):
        S_c[i] = P_pq[i] - 1j * Q_pq[i]
    for i in range(n_pq, n_b):
        S_c[i] = P_pv[i - n_pq] - 1j * Q1[i - n_pq]

    YbetaV2 = np.dot(Ybeta, V2)
    I1 = np.zeros(n_b, dtype=complex)
    
    for i in range(n_b):
        V1[i] = S_c[i] / np.conj(I2[i])
        I1[i] = beta[i, i] * (V1[i] - V2[i]) + I2[i]
        # A[i] = (YbetaV2[i] - I0[i]) / beta[i, i]
        # sigma[i] = - S_c[i] / (A[i] * np.conj(A[i]) * beta[i, i])
        # U[i] = (- 1 - np.sqrt(1 - 4 * (np.imag(sigma[i]) * np.imag(sigma[i]) + np.real(sigma[i])))) / 2 + 1j * np.imag(sigma[i])  # negative root!!
        # V1[i] = U[i] * A[i]

    V1_pq = V1[:n_pq]
    V1_pv = V1[n_pq:n_b]

    vec_YV = np.dot(Y, V1)
    errors = []
    for i in range(n_b): 
        #print(V1[i])
        aa = vec_YV[i] - I0[i]
        bb = S_c[i] / np.conj(V1[i])
        print(abs(V1[i]))
        errors.append(abs(aa - bb))

    # print(c, np.max(errors))
    print(np.log10(np.max(errors)))



# RESULTS

end_time = time.time()

print('Temps: ', end_time - start_time, 'segons')

vec_YV = np.dot(Y, V1)
errors = []
for i in range(n_b): 
    print(V1[i])
    aa = vec_YV[i] - I0[i]
    bb = S_c[i] / np.conj(V1[i])
    errors.append(abs(aa - bb))

print('Màx error: ', np.max(errors))



