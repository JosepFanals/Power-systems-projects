# PGD code, generic. Less cases, more simple to debut it
import time
import pandas as pd
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

pd.options.display.precision = 2
pd.set_option('display.precision', 2)

df_b = pd.read_excel('data_5PQ_mesh_r.xlsx', sheet_name="buses")
df_l = pd.read_excel('data_5PQ_mesh_r.xlsx', sheet_name="lines")

# BEGIN INITIALIZATION OF DATA
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
        # print(d_pq[a])
        Y11[d_pq[a], d_pq[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]
    elif a - 1 in pv:
        # print(d_pv[a])
        Y22[d_pv[a], d_pv[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]


Y = np.block([[Y11, Y12], [Y21, Y22]])
Yinv = np.linalg.inv(Y)
Ydf = pd.DataFrame(Y)

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
# END INITIALIZATION OF DATA


# DECOMPOSITION OF APPARENT POWERS
SSk = []
SSp = []
SSq = []
n_buses = np.shape(Y)[0]  # number of buses
n_scale = 10  # number of discretized points, arbitrary. Before it was like 101
n_time = 8  # number of hours considered to solve the power flow (could be minutes...)

SKk0 = P_pq + Q_pq * 1j  # original load
SPp0 = np.ones(n_time)  # powers across time
SQq0 = np.ones(n_scale)  # discretization of powers

SSk.append(np.conj(SKk0))
SSp.append(np.conj(SPp0))
SSq.append(np.conj(SQq0))

SKk1 = np.array([0.5, 0.6, 0.7, 0.6, 0.5])
SPp1 = np.array([0.0, 0.2, 0.3, 0.7, 0.7, 0.3, 0.2, 0.0])
SQq1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

SSk.append(np.conj(SKk1))
SSp.append(np.conj(SPp1))
SSq.append(np.conj(SQq1))

print('SSk: ', SSk)
print('SSp: ', SSp)
print('SSq: ', SSq)
# ----------



# DECOMPOSITION OF VOLTAGES
Kkv = np.ones(n_buses, dtype=complex)  # amplitude vector
Ppv = np.ones(n_time)  # time vector
Qqv = np.ones(n_scale)  # scaling vector

VVk = []
VVp = []
VVq = []

VVk.append(np.conj(Kkv))
VVp.append(np.conj(Ppv))
VVq.append(np.conj(Qqv))


# DECOMPOSITION OF CURRENTS
IIk = []
IIp = []
IIq = []

# CREATION OF C (auxiliary variables). 
# THIS FUNCTION HAS TO BE CALLED EVERY TIME WE COMPUTE A NEW RESIDUE OF I, AND ALWAYS FOLLOWS THIS


def fun_C(SSk, SSp, SSq, VVk, VVp, VVq, IIk, IIp, IIq):
    Ns = len(SSk)
    Nv = len(VVk)
    n = len(IIk)
    Nc = Ns + Nv * n
    # CCk = SSk  # initialize with the S* decomposed variables
    # CCp = SSp
    # CCq = SSq
    CCk = SSk.copy()
    CCp = SSp.copy()
    CCq = SSq.copy() 
    for ii in range(Nv):
        for jj in range(n):
            CCk.append(- VVk[ii] * IIk[jj])
            CCp.append(- VVp[ii] * IIp[jj])
            CCq.append(- VVq[ii] * IIq[jj])
    return (CCk, CCp, CCq, Nc, Nv, n)


# DEFINITION OF NUMBER OF ITERATIONS, CAN CHANGE ARBITRARILY
n_gg = 10  # outer
n_mm = 10  # intermediate
n_kk = 10  # inner


for gg in range(n_gg):  # outer loop

    # add the blank initialization of C:
    CCk = []
    CCp = []
    CCq = []

    IIk = []
    IIp = []
    IIq = []

    for mm in range(n_mm):  # intermediate loop
        # define the new C
        [CCk, CCp, CCq, Nc, Nv, n] = fun_C(SSk, SSp, SSq, VVk, VVp, VVq, IIk, IIp, IIq)

        # initialize the residues we have to find
        IIk1 = (np.random.rand(n_buses) - np.random.rand(n_buses)) * 1  # could also try to set IIk1 = VVk1
        # IIk1 = VVk[0]
        # if gg == 0 and mm == 0:
        # if mm == 0:
            # IIk1 = np.ones(n_buses, dtype=complex)
        IIp1 = (np.random.rand(n_time) - np.random.rand(n_time)) * 1
        IIq1 = (np.random.rand(n_scale) - np.random.rand(n_scale)) * 1

        for kk in range(n_kk):  # inner loop
            # compute IIk1 (residues on Ik)

            prodRK = 0
            RHSk = np.zeros(n_buses, dtype=complex)
            for ii in range(Nc):
                prodRK = np.dot(IIp1, CCp[ii]) * np.dot(IIq1, CCq[ii])
                RHSk += prodRK * CCk[ii]

            prodLK = 0
            LHSk = np.zeros(n_buses, dtype=complex)
            for ii in range(Nv):
                prodLK = np.dot(IIp1, VVp[ii] * IIp1) * np.dot(IIq1, VVq[ii] * IIq1)
                LHSk += prodLK * VVk[ii]

            IIk1 = RHSk / LHSk


            # compute IIp1 (residues on Ip)
            prodRP = 0
            RHSp = np.zeros(n_time, dtype=complex)
            for ii in range(Nc):
                prodRP = np.dot(IIk1, CCk[ii]) * np.dot(IIq1, CCq[ii])
                RHSp += prodRP * CCp[ii]

            prodLP = 0
            LHSp = np.zeros(n_time, dtype=complex)
            for ii in range(Nv):
                prodLP = np.dot(IIk1, VVk[ii] * IIk1) * np.dot(IIq1, VVq[ii] * IIq1)
                LHSp += prodLP * VVp[ii]

            IIp1 = RHSp / LHSp


            # compute IIq1 (residues on Iq)
            prodRQ = 0
            RHSq = np.zeros(n_scale, dtype= complex)
            for ii in range(Nc):
                prodRQ = np.dot(IIk1, CCk[ii]) * np.dot(IIp1, CCp[ii])
                RHSq += prodRQ * CCq[ii]

            prodLQ = 0
            LHSq = np.zeros(n_scale, dtype=complex)
            for ii in range(Nv):
                prodLQ = np.dot(IIk1, VVk[ii] * IIk1) * np.dot(IIp1, VVp[ii] * IIp1)
                LHSq += prodLQ * VVq[ii]

            IIq1 = RHSq / LHSq

            # if gg == 0 and mm == 0 and kk >= 0:
            #     print('i')
            #     print(IIk1[0])
                # print(IIp1[:10])
                # print(IIq1[:10])


        # if gg == 0:
        #     print('m')
        #     print(IIk1[0])

        IIk.append(IIk1)
        IIp.append(IIp1)
        IIq.append(IIq1)

    VVk = []
    VVp = []
    VVq = []
    PP1 = np.ones(n_time)
    QQ1 = np.ones(n_scale)
    for ii in range(n_mm):
        # VVk.append(np.conj(np.dot(Yinv, IIk[ii] + I0_pq)))
        VVk.append(np.conj(np.dot(Yinv, IIk[ii])))
        VVp.append(IIp[ii])
        VVq.append(IIq[ii])
        # VVp = np.copy(IIp)
        # VVq = np.copy(IIq)

    # print(VVk[0][:10])
    # print(VVk[1][:10])

    # try to add I0 this way:
    VVk.append(np.conj(np.dot(Yinv, I0_pq)))
    VVp.append(PP1)
    VVq.append(QQ1)


# CHART OF VOLTAGES, but conjugated or not!?
# full_map = np.multiply.outer(VVk[0], np.multiply.outer(VVp[0], VVq[0]))  # initial tridimensional representation
V_map = np.multiply.outer(np.multiply.outer(VVp[0], VVk[0]), VVq[0])  # the tridimensional representation I am looking for
for i in range(1, len(VVk)):
    V_map += np.multiply.outer(np.multiply.outer(VVp[i], VVk[i]), VVq[i])  # the tridimensional representation I am looking for
writer = pd.ExcelWriter('Map_V.xlsx', engine='xlsxwriter')
for i in range(n_time):
    V_map_df = pd.DataFrame(V_map[:][i][:])
    V_map_df.to_excel(writer, sheet_name=str(i))  
writer.save()

# CHART OF CURRENTS
I_map = np.multiply.outer(np.multiply.outer(IIp[0], IIk[0]), IIq[0])
for i in range(1, len(IIk)):
    I_map += np.multiply.outer(np.multiply.outer(IIp[i], IIk[i]), IIq[i])
writer = pd.ExcelWriter('Map_I.xlsx', engine='xlsxwriter')
for i in range(n_time):
    I_map_df = pd.DataFrame(I_map[:][i][:])
    I_map_df.to_excel(writer, sheet_name=str(i))
writer.save()

# CHART OF POWERS
S_map = np.multiply.outer(np.multiply.outer(SSp[0], SSk[0]), SSq[0])
for i in range(1, len(SSk)):
    S_map += np.multiply.outer(np.multiply.outer(SSp[i], SSk[i]), SSq[i])
writer = pd.ExcelWriter('Map_S.xlsx', engine='xlsxwriter')
for i in range(n_time):
    S_map_df = pd.DataFrame(S_map[:][i][:])
    S_map_df.to_excel(writer, sheet_name=str(i))
writer.save()


# CHECKING
I_mis = []
for n_sheet in range(n_time):
    for n_col in range(n_scale):
        YV_prod = np.dot(Y, V_map[n_sheet,:,n_col])
        # YV_prod = np.dot(Y, np.conj(V_map[n_sheet,:,n_col]))
        I22 = YV_prod - I0_pq
        I11 = []
        for kk in range(n_buses):
            I11.append(np.conj(S_map[n_sheet,kk,n_col]) / np.conj(V_map[n_sheet,kk,n_col]))
            # I11.append(S_map[n_sheet,kk,n_col] / V_map[n_sheet,kk,n_col])
        I_mis.append(abs(max(np.abs(I11 - I22))))
        print(I11)
        print(I22)

for xx in range(len(I_mis)):
    print(I_mis[xx])