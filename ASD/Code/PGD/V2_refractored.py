# PGD code, merging Santiago's improvements

import time
import pandas as pd
import numpy as np
import sys
from math import *
# from mpmath import mp
# mp.dps = 100
np.set_printoptions(precision=4)
pd.options.display.precision = 2
pd.set_option('display.precision', 2)
import time


start = time.time()

def progress_bar(i, n, size):  # show the percentage
    percent = float(i) / float(n)
    sys.stdout.write("\r"
                     + str(int(i)).rjust(3, '0')
                     + "/"
                     + str(int(n)).rjust(3, '0')
                     + ' ['
                     + '='*ceil(percent*size)
                     + ' '*floor((1-percent)*size)
                     + ']')


def read_grid_data(fname):
    """
    Read the grid data
    :param fname: name of the excel file
    :return: n_buses, Qmax, Qmin, Y, Yinv, V_mod, P_pq, Q_pq, P_pv, I0_pq, n_pv, n_pq
    """
    df_b = pd.read_excel(fname, sheet_name="buses")
    df_l = pd.read_excel(fname, sheet_name="lines")

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

    # add shunts connected directly to the bus
    for i in range(len(df_b)):
        a = df_b.iloc[i, 0]
        if a - 1 in pq:
            Y11[d_pq[a], d_pq[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]

        elif a - 1 in pv:
            Y22[d_pv[a], d_pv[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]

    Y = np.block([[Y11, Y12], [Y21, Y22]])
    Yinv = np.linalg.inv(Y)

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

    n_buses = np.shape(Y)[0]  # number of buses

    return n_buses, Y, Yinv, V_mod, P_pq, Q_pq, P_pv, I0_pq, n_pv, n_pq


def time_scaling_Pd(t0, t1, x_step):
    t_vec = np.arange(t0, t1 + 1, x_step)
    Ssp_v = []
    for llm in range(len(t_vec)):
        x = t_vec[llm]
        Ssp_v.append((0.0363*(22.394708020782733 - 1.6354720294954328*x + 0.39434274588533585*x*x - 0.0035090092967717756*x*x*x - 0.029337034458808153*x*x*x*x + 0.007731225380636515*x*x*x*x*x - 0.0008557978395676469*x*x*x*x*x*x + 0.00004732882729208937 *x*x*x*x*x*x*x - 0.00000128996363643316*x*x*x*x*x*x*x*x + 1.381540655448e-8*x*x*x*x*x*x*x*x*x)) ** 1.5)

    return Ssp_v


def time_scaling_Pg(t0, t1, x_step):
    t_vec = np.arange(t0, t1 + 1, x_step)
    Ssp_g = []
    for llm in range(len(t_vec)):
        x = t_vec[llm]
        Ssp_g.append(5*1/(2*np.sqrt(2*np.pi))*np.exp(-((x-12)**2)/(2*2**2)))
    
    return Ssp_g


def init_apparent_powers_decomposition(n_buses, n_scale, n_time, P_pq, Q_pq):
    """

    :param n_buses:
    :param n_scale:
    :param n_time:
    :param P_pq:
    :param Q_pq:
    :return:
    """
    SSk = np.empty((n_buses, 2), dtype=complex)
    SSp = np.empty((n_time, 2), dtype=complex)
    SSq = np.empty((n_scale, 2), dtype=complex)

    SKk0 = P_pq + Q_pq * 1j  # original load
    # SPp0 = np.ones(n_time)  # the original loads do not change with time
    SPp0 = time_scaling_Pd(0, 23, 1)  # scale load with  time
    SQq0 = np.ones(n_scale)  # the original loads are not scaled
    # SQq0 = np.linspace(0.9, 1.1, num=n_scale)

    SSk[:, 0] = np.conj(SKk0)
    SSp[:, 0] = np.conj(SPp0)
    SSq[:, 0] = np.conj(SQq0)


    # SKk1 = - 0.02 * np.random.rand(n_buses)  # load/generator of random active power, change this to try different cases
    # SKk1 = [0, 0.1, 0, 0, 0, 0, 0, 0.0327, 0.0295, 0, 0.0448, 0]
    # SKk1 = [0, 0.2, 0, 0, 0, 0, 0, 0.0654, 0.059, 0, 0.0896, 0]
    SKk1 = [0, 0.3, 0, 0, 0, 0, 0, 0.0654 * 1.5, 0.059 * 1.5, 0, 0.0896 * 1.5, 0]
    SPp1 = time_scaling_Pg(0, 23, 1)  # scale PV with time
    # SPp1 = np.random.rand(n_time)
    # SQq1 = np.random.rand(n_scale)
    SQq1 = np.linspace(0, 1, num=n_scale)

    # SSk[:, 0] = np.conj(SKk1)
    # SSp[:, 0] = np.conj(SPp1)
    # SSq[:, 0] = np.conj(SQq1)

    SSk[:, 1] = np.conj(SKk1)
    SSp[:, 1] = np.conj(SPp1)
    SSq[:, 1] = np.conj(SQq1)

    return SSk, SSp, SSq


def init_voltages_decomposition(n_mm, n_buses, n_scale, n_time):
    """

    :param n_buses:
    :param n_scale:
    :param n_time:
    :return:
    """
    # DECOMPOSITION OF VOLTAGES
    VVk = np.zeros((n_buses, n_mm + 1), dtype=complex)
    VVp = np.zeros((n_time, n_mm + 1), dtype=complex)
    VVq = np.zeros((n_scale, n_mm + 1), dtype=complex)

    Kkv = np.ones(n_buses, dtype=complex)  # amplitude vector
    Ppv = np.ones(n_time)  # position vector
    Qqv = np.ones(n_scale)  # scaling vector

    VVk[:, 0] = np.conj(Kkv)
    VVp[:, 0] = np.conj(Ppv)
    VVq[:, 0] = np.conj(Qqv)

    return VVk, VVp, VVq


def init_currents_decomposition(n_gg, n_mm, n_buses, n_scale, n_time):
    """

    :return:
    """

    IIk = np.zeros((n_buses, n_mm + 1), dtype=complex)
    IIp = np.zeros((n_time, n_mm + 1), dtype=complex)
    IIq = np.zeros((n_scale, n_mm + 1), dtype=complex)
    return IIk, IIp, IIq


def fun_C(SSk, SSp, SSq, VVk, VVp, VVq, IIk, IIp, IIq, n_i_coeff, n_v_coeff, n_bus, n_scale, n_time):
    """

    :param SSk:
    :param SSp:
    :param SSq:
    :param VVk:
    :param VVp:
    :param VVq:
    :param IIk:
    :param IIp:
    :param IIq:
    :param n: number of coefficients fo far
    :return:
    """

    Nc = n_v_coeff * n_i_coeff + 2

    CCk = np.empty((n_bus, Nc), dtype=complex)
    CCp = np.empty((n_time, Nc), dtype=complex)
    CCq = np.empty((n_scale, Nc), dtype=complex)

    CCk[:, :2] = SSk
    CCp[:, :2] = SSp
    CCq[:, :2] = SSq
    idx = 2
    for ii in range(n_v_coeff):
        for jj in range(n_i_coeff):
            CCk[:, idx] = - VVk[:, ii] * IIk[:, jj]
            CCp[:, idx] = - VVp[:, ii] * IIp[:, jj]
            CCq[:, idx] = - VVq[:, ii] * IIq[:, jj]
            idx += 1

    return CCk, CCp, CCq, Nc, n_v_coeff, n_i_coeff


def build_map(MMk, MMp, MMq, n_mm):
    """
    Build 3-D mapping from decomposed variables
    :param MMk:
    :param MMp:
    :param MMq:
    :return:
    """
    MM_map = np.multiply.outer(np.multiply.outer(MMp[:, 0], MMk[:, 0]), MMq[:, 0])
    n = len(MMk)
    for i in range(1, n_mm + 1):
        print(i)
        # the tri-dimensional representation I am looking for
        MM_map += np.multiply.outer(np.multiply.outer(MMp[:, i], MMk[:, i]), MMq[:, i])
        progress_bar(i+1, n, 50)

    return MM_map


def save_mapV(MM_map, fname, n_time):
    """

    :param MM_map: 3D tensor for voltages
    :param fname:
    :return:
    """
    writer = pd.ExcelWriter(fname)
    for i in range(n_time):
        V_map_df = pd.DataFrame(np.conj(MM_map[:][i][:]))
        V_map_df.to_excel(writer, sheet_name=str(i))
    writer.save()


def save_mapI(MM_map, fname, n_time):
    """

    :param MM_map: 3D tensor for currents
    :param fname:
    :return:
    """
    writer = pd.ExcelWriter(fname)
    for i in range(n_time):
        V_map_df = pd.DataFrame(MM_map[:][i][:])
        V_map_df.to_excel(writer, sheet_name=str(i))
    writer.save()


def save_mapS(MM_map, fname, n_time):
    """

    :param MM_map: 3D tensor for powers
    :param fname:
    :return:
    """
    writer = pd.ExcelWriter(fname)
    for i in range(n_time):
        V_map_df = pd.DataFrame(np.conj(MM_map[:][i][:]))
        V_map_df.to_excel(writer, sheet_name=str(i))
    writer.save()



def pgd(fname, n_gg=20, n_mm=20, n_kk=20, n_scale=12, n_time=8):
    """

    :param fname: data file name
    :param n_gg: outer iterations
    :param n_mm: intermediate iterations
    :param n_kk: inner iterations
    :param n_scale: number of discretized points, arbitrary
    :param n_time: number of discretized time periods, arbitrary
    :return:
    """


    n_buses, Y, Yinv, V_mod, P_pq, Q_pq, P_pv, I0_pq, n_pv, n_pq = read_grid_data(fname)

    SSk, SSp, SSq = init_apparent_powers_decomposition(n_buses, n_scale, n_time, P_pq, Q_pq)

    n_max = n_gg * n_mm * n_kk
    iter_count = 0
    idx_i = 0
    idx_v = 1
    for gg in range(n_gg):  # outer loop: iterate on γ to solve the power flow as such
        VVk, VVp, VVq = init_voltages_decomposition(n_mm, n_buses, n_scale, n_time)
        IIk, IIp, IIq = init_currents_decomposition(n_gg, n_mm, n_buses, n_scale, n_time)

        idx_i = 0
        idx_v = 1

        for mm in range(n_mm):  # intermediate loop: iterate on i to find the superposition of terms of the I tensor.
            # define the new C
            CCk, CCp, CCq, Nc, Nv, n = fun_C(SSk, SSp, SSq, VVk, VVp, VVq, IIk, IIp, IIq, idx_i, idx_v, n_buses, n_scale, n_time)

            # initialize the residues we have to find
            # IIk1 = (np.random.rand(n_buses) - np.random.rand(n_buses)) * 1  # could also try to set IIk1 = VVk1
            IIk1 = (np.random.rand(n_buses) - np.random.rand(n_buses)) * (n_mm - mm) ** 2 / n_mm ** 2
            IIp1 = (np.random.rand(n_time) - np.random.rand(n_time)) * 1 
            IIq1 = (np.random.rand(n_scale) - np.random.rand(n_scale)) * 1

            for kk in range(n_kk):  # inner loop: iterate on Γ to find the residues.

                # compute IIk1 (residues on Ik)
                RHSk = np.zeros(n_buses, dtype=complex)
                for ii in range(Nc):
                    prodRK = np.dot(IIp1, CCp[:, ii]) * np.dot(IIq1, CCq[:, ii])
                    RHSk += prodRK * CCk[:, ii]

                LHSk = np.zeros(n_buses, dtype=complex)
                for ii in range(Nv):
                    prodLK = np.dot(IIp1, VVp[:, ii] * IIp1) * np.dot(IIq1, VVq[:, ii] * IIq1)
                    LHSk += prodLK * VVk[:, ii]

                # IIk1 = RHSk / LHSk
                IIk1 = RHSk / (LHSk + 1e-8)

                # compute IIp1 (residues on Ip)
                RHSp = np.zeros(n_time, dtype=complex)
                for ii in range(Nc):
                    prodRP = np.dot(IIk1, CCk[:, ii]) * np.dot(IIq1, CCq[:, ii])
                    RHSp += prodRP * CCp[:, ii]

                LHSp = np.zeros(n_time, dtype=complex)
                for ii in range(Nv):
                    prodLP = np.dot(IIk1, VVk[:, ii] * IIk1) * np.dot(IIq1, VVq[:, ii] * IIq1)
                    LHSp += prodLP * VVp[:, ii]

                # IIp1 = RHSp / LHSp
                IIp1 = RHSp / (LHSp + 1e-8)

                # compute IIq1 (residues on Iq)
                RHSq = np.zeros(n_scale, dtype=complex)
                for ii in range(Nc):
                    prodRQ = np.dot(IIk1, CCk[:, ii]) * np.dot(IIp1, CCp[:, ii])
                    RHSq += prodRQ * CCq[:, ii]

                LHSq = np.zeros(n_scale, dtype=complex)
                for ii in range(Nv):
                    prodLQ = np.dot(IIk1, VVk[:, ii] * IIk1) * np.dot(IIp1, VVp[:, ii] * IIp1)
                    LHSq += prodLQ * VVq[:, ii]

                # IIq1 = RHSq / LHSq
                IIq1 = RHSq / (LHSq + 1e-8)

                progress_bar(iter_count, n_max, 50)  # display the inner operations
                iter_count += 1

            IIk[:, idx_i] = IIk1
            IIp[:, idx_i] = IIp1
            IIq[:, idx_i] = IIq1
            idx_i += 1

        for ii in range(n_mm):
            VVk[:, ii] = np.conj(np.dot(Yinv, IIk[:, ii]))
            VVp[:, ii] = np.conj(IIp[:, ii])
            VVq[:, ii] = np.conj(IIq[:, ii])

        # try to add I0 this way:
        VVk[:, n_mm] = np.conj(np.dot(Yinv, I0_pq))
        VVp[:, n_mm] = np.ones(n_time)
        VVq[:, n_mm] = np.ones(n_scale)
        idx_v = n_mm + 1


    # VVk: size (n_mm + 1, nbus)
    # VVp: size (n_mm + 1, nbus)
    # VVq: size (n_mm + 1, n_scale)
    v_map = build_map(VVk, VVp, VVq, n_mm)

    # SSk: size (2, nbus)
    # SSp: size (2, nbus)
    # SSq: size (2, n_scale)
    s_map = build_map(SSk, SSp, SSq, 1)
    # s_map = build_map(SSk, SSp, SSq, n_mm)

    # IIk: size (n_gg * n_mm, nbus)
    # IIp: size (n_gg * n_mm, nbus)
    # IIq: size (n_gg * n_mm, n_scale)
    i_map = build_map(IIk, IIp, IIq, n_mm)

    # the size of the maps is nbus, ntime, n_scale

    vec_error = checking(Y, v_map, s_map, I0_pq, n_buses, n_time, n_scale)

    return v_map, s_map, i_map, vec_error


def checking(Y, V_map, S_map, I0_pq, n_buses, n_time, n_scale):
    """

    :param Y: data file name
    :param V_map: outer iterations
    :param S_map: intermediate iterations
    :param I0_pq: inner iterations
    :param n_buses: number of buses
    :param n_scale: number of discretized points, arbitrary
    :param n_time: number of discretized time periods, arbitrary
    :return: maximum current error
    """

    print(Y)

    I_mis = []
    for n_sheet in range(n_time):
        for n_col in range(n_scale):
            YV_prod = np.dot(Y, np.conj(V_map[n_sheet,:,n_col]))
            # YV_prod = np.dot(Y, V_map[n_sheet,:,n_col])
            I22 = YV_prod - I0_pq
            I11 = []
            for kk in range(n_buses):
                I11.append(S_map[n_sheet,kk,n_col] / V_map[n_sheet,kk,n_col])
                # I11.append(np.conj(S_map[n_sheet,kk,n_col]) / np.conj(V_map[n_sheet,kk,n_col]))
            I_mis.append(abs(max(np.abs(I11 - I22))))
    return I_mis


# v_map_, s_map_, i_map_, vec_error_ = pgd('data_10PQ_mesh_r.xlsx', n_gg=25, n_mm=25, n_kk=25, n_scale=10, n_time=80)
v_map_, s_map_, i_map_, vec_error_ = pgd('data_13_Portugal.xlsx', n_gg=10, n_mm=10, n_kk=10, n_scale=20, n_time=24)

# for i in range(len(vec_error_)):
    # print(vec_error_[i])

print(max(vec_error_))

n_scale = 20
n_time = 24
n_buses = 12
save_mapV(v_map_, 'Map_V2.xlsx', n_time)
save_mapI(i_map_, 'Map_I2.xlsx', n_time)
save_mapS(s_map_, 'Map_S2.xlsx', n_time)


# print(s_map_[0][2][3])
print('$$$$$$$$$$$$$$$$$')

def S_eval(v_map_, s_map_, n_time, n_scale, n_buses):
    for ll in range(n_time):
        eff_scale = []
        vec_Plosses = []
        vec_Ptotal = []
        vec_Pbuses = []
        for kk in range(n_scale):
            P_buses = 0
            for mm in range(n_buses):
                P_buses += np.real(np.conj(s_map_[ll, mm, kk]))

            P_loss_line = 0
            
            par1 = [1, 2, 0.015, 0.035]
            par2 = [1, 3, 0.02, 0.04]
            par3 = [1, 4, 0.03, 0.05]
            par4 = [2, 5, 0.01, 0.02]
            par5 = [2, 6, 0.01, 0.02]
            par6 = [3, 7, 0.06, 0.08]
            par7 = [3, 11, 0.023, 0.047]
            par8 = [4, 7, 0.012, 0.028]
            par9 = [5, 8, 0.01, 0.025]
            par10 = [8, 9, 0.03, 0.05]
            par11 = [9, 10, 0.07, 0.11]
            par12 = [10, 11, 0.06, 0.09]
            par13 = [11, 12, 0.02, 0.04]
            par14 = [12, 13, 0.024, 0.051]
            par_tot = [par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14]
            for xl in range(14):
                # print(par_tot[xl][0] - 1)
                if xl < 3:
                    iix = (1 - np.conj(v_map_[ll, par_tot[xl][1] - 2, kk])) / (par_tot[xl][2] + 1j * par_tot[xl][3])
                    ppx = abs(iix)**2 * par_tot[xl][2]
                else:
                    iix = (np.conj(v_map_[ll, par_tot[xl][0] - 2, kk]) - np.conj(v_map_[ll, par_tot[xl][1] - 2, kk])) / (par_tot[xl][2] + 1j * par_tot[xl][3])
                    ppx = abs(iix)**2 * par_tot[xl][2]

                P_loss_line += ppx

            P_tot = abs(P_buses) + abs(P_loss_line)
            eff_x = abs(P_buses) / abs(P_tot)
            eff_scale.append(eff_x)
            vec_Plosses.append(abs(P_loss_line))
            vec_Ptotal.append(abs(P_buses) + abs(P_loss_line))
            vec_Pbuses.append(abs(P_buses))
            # print('P_losses: ', abs(P_loss_line))
            # print('P_buses: ', abs(P_buses))
            # print('P_total: ', abs(P_tot))
            # print('Efficiency: ', abs(P_buses) / abs(P_tot))
        # print('Vector efficiencies: ', eff_scale)
        ind_max = np.argmax(eff_scale) 
        # print('Scaling for max efficiency: ', (ind_max - 1) / n_scale)
        # print('Ptotal: ', vec_Ptotal[ind_max])
        # print('Ploss: ', vec_Plosses[ind_max])
        Skmax = [0, 0.3, 0, 0, 0, 0, 0, 0.0654 * 1.5, 0.059 * 1.5, 0, 0.0896 * 1.5, 0]
        SPp1 = time_scaling_Pg(0, 23, 1)  # scale PV with time
        f_time = SPp1[ll]
        P_Pv = (0.3 + 0.0654 * 1.5 + 0.059 * 1.5 + 0.0896 * 1.5) * f_time

        P_base =  [0.3, 0.0654 * 1.5, 0.059 * 1.5, 0.0896 * 1.5]
        # P_ff = f_time * (ind_max + 1) / n_scale * 100
        # print(P_base[0] * P_ff, P_base[1] * P_ff, P_base[2] * P_ff, P_base[3] * P_ff)

        print(P_base[0] * f_time, P_base[1] * f_time, P_base[2] * f_time, P_base[3] * f_time)

        # print(eff_scale)
        # print('PPV: ', P_Pv)
        # print('Self gen fraction: ', P_Pv / vec_Ptotal[ind_max])

        # print('Full line: ')
        # print(str(ll) + ' & ' + str(round(P_Pv / vec_Ptotal[ind_max]),4) + ' & ' + str((ind_max - 1) / n_scale) + ' & ' + str(round(vec_Plosses[ind_max]), 4) + ' & ' + str(round(P_Pv / vec_Ptotal[ind_max]), 4))

        # print(str(ll) + ' & ' + str((P_Pv * (ind_max + 1) / n_scale) / vec_Ptotal[ind_max] * 100) + ' & ' + str(vec_Pbuses[ind_max] / vec_Ptotal[ind_max] * 100) + ' & ' + str(vec_Plosses[ind_max] * 100e3) + ' & ' + str((ind_max + 1) / n_scale))

        # print(str(ll) + ' & ' + str(round(P_Pv / vec_Ptotal[ind_max], 6)) + ' & ' + str((ind_max - 1) / n_scale) + ' & ' + str(round(vec_Plosses[ind_max],6)) + ' & ' + str(round(P_Pv / vec_Ptotal[ind_max], 6)))

        qq_f = ind_max / n_scale
            
    return P_buses

aaa = S_eval(v_map_, s_map_, n_time, n_scale, n_buses)
# print(aaa)
# print(v_map_[0][2][3])


end = time.time()
print('Time: ', end - start)