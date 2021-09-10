# Parametric analysis of power systems
# Addition: normalize parameters [0, 1]

# Packages
import numpy as np
import GridCal.Engine as gc
import random
import math
import itertools
import time
from smt.sampling_methods import LHS
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
np.set_printoptions(precision=10)


def test_grid():

    grid = gc.MultiCircuit()

    # Buses
    bus1 = gc.Bus('Bus 1', vnom=20)
    bus1.is_slack = True
    grid.add_bus(bus1)
    gen1 = gc.Generator('Slack Generator', voltage_module=1.0)
    grid.add_generator(bus1, gen1)

    bus2 = gc.Bus('Bus 2', vnom=20)
    grid.add_bus(bus2)
    grid.add_load(bus2, gc.Load('load 2', P=0, Q=0))

    bus3 = gc.Bus('Bus 3', vnom=20)
    grid.add_bus(bus3)
    grid.add_load(bus3, gc.Load('load 3', P=0, Q=0))

    bus4 = gc.Bus('Bus 4', vnom=20)
    grid.add_bus(bus4)
    grid.add_load(bus4, gc.Load('load 4', P=0, Q=0))

    bus5 = gc.Bus('Bus 5', vnom=20)
    grid.add_bus(bus5)
    grid.add_load(bus5, gc.Load('load 5', P=0, Q=0))

    # more buses added
    bus6 = gc.Bus('Bus 6', vnom=20)
    grid.add_bus(bus6)
    grid.add_load(bus6, gc.Load('load 6', P=0, Q=0))

    bus7 = gc.Bus('Bus 7', vnom=20)
    grid.add_bus(bus7)
    grid.add_load(bus7, gc.Load('load 7', P=0, Q=0))

    bus8 = gc.Bus('Bus 8', vnom=20)
    grid.add_bus(bus8)
    grid.add_load(bus8, gc.Load('load 8', P=0, Q=0))

    bus9 = gc.Bus('Bus 9', vnom=20)
    grid.add_bus(bus9)
    grid.add_load(bus9, gc.Load('load 9', P=0, Q=0))

    bus10 = gc.Bus('Bus 10', vnom=20)
    grid.add_bus(bus10)
    grid.add_load(bus10, gc.Load('load 10', P=0, Q=0))

    # Lines
    grid.add_line(gc.Line(bus1, bus2, 'line 1-2', r=0.05, x=0.11, b=0.0))
    grid.add_line(gc.Line(bus1, bus3, 'line 1-3', r=0.05, x=0.11, b=0.0))
    grid.add_line(gc.Line(bus1, bus5, 'line 1-5', r=0.03, x=0.08, b=0.0))
    grid.add_line(gc.Line(bus2, bus3, 'line 2-3', r=0.04, x=0.09, b=0.0))
    grid.add_line(gc.Line(bus2, bus5, 'line 2-5', r=0.04, x=0.09, b=0.0))
    grid.add_line(gc.Line(bus3, bus4, 'line 3-4', r=0.06, x=0.13, b=0.0))
    grid.add_line(gc.Line(bus4, bus5, 'line 4-5', r=0.04, x=0.09, b=0.0))

    # more lines added
    grid.add_line(gc.Line(bus1, bus6, 'line 1-6', r=0.03, x=0.10, b=0.0))
    grid.add_line(gc.Line(bus4, bus6, 'line 4-6', r=0.04, x=0.08, b=0.0))
    grid.add_line(gc.Line(bus5, bus7, 'line 5-7', r=0.04, x=0.11, b=0.0))
    grid.add_line(gc.Line(bus3, bus8, 'line 3-8', r=0.03, x=0.09, b=0.0))
    grid.add_line(gc.Line(bus6, bus9, 'line 6-9', r=0.03, x=0.08, b=0.0))
    grid.add_line(gc.Line(bus7, bus10, 'line 7-10', r=0.04, x=0.12, b=0.0))

    return grid


def power_flow(snapshot: gc.SnapshotData, S: np.ndarray, V0: np.ndarray):

    options = gc.PowerFlowOptions()

    res = gc.single_island_pf(circuit=snapshot,
                              Vbus=V0,
                              Sbus=S,
                              Ibus=snapshot.Ibus,
                              branch_rates=snapshot.Rates,
                              options=options,
                              logger=gc.Logger())
    return abs(res.voltage)


def samples_calc(snapshot: gc.SnapshotData, M, n_param, param_lower_bnd, param_upper_bnd):

    """
    Calculate the gradients, build the hx vector, the covariance C matrix and store the parameters

    :param M: number of samples
    :param n_param: number of parameters
    :param param_lower_bnd: array of lower bounds for the parameters
    :param param_upper_bnd: array of upper bounds for the parameters
    :return: hx, C and the stored parameters

    """

    nP = int(n_param / 2)
    nQ = int(n_param / 2)

    pq_vec = snapshot.pq
    nV = len(pq_vec)
    Sg = snapshot.generator_data.get_injections_per_bus()[:, 0]

    hx = np.zeros((M, nV), dtype=float)  # x solutions at each sample, for all nV buses
    C = np.zeros((nV, n_param, n_param), dtype=float)  # nV covariance matrices

    # create samples with Latin Hypercube
    xlimits = np.zeros((n_param, 2), dtype=float)  # 2 columns: [lower_bound, upper_bound]
    for ll in range(n_param):
        xlimits[ll, 0] = param_lower_bnd[ll]
        xlimits[ll, 1] = param_upper_bnd[ll]

    sampling_lh = LHS(xlimits=xlimits)
    param_store = sampling_lh(M)  # matrix with all samples

    for ll in range(M):

        # compose the power injections from the sample
        P = param_store[ll, :nP]
        Q = param_store[ll, nP:nP+nQ]
        Sl = snapshot.load_data.C_bus_load * ((P + 1j * Q))  # already normalized P and Q
        S = Sg - Sl

        # x solution for each sample
        v = power_flow(snapshot=snapshot, S=S, V0=snapshot.Vbus)
        hx[ll, :] = v[pq_vec]

        # calculate gradients and form C matrix
        Ag = np.zeros((n_param, nV), dtype=float)
        for kk in range(n_param):
            params_delta = np.copy(param_store[ll, :])
            params_delta[kk] += delta  # increase a parameter by delta

            # compose the power injections from the sample delta
            P = params_delta[:nP]
            Q = params_delta[nP:nP + nQ]
            Sl = snapshot.load_data.C_bus_load * ((P + 1j * Q))
            S = Sg - Sl

            # run the power flow with the increments in power
            v2 = power_flow(snapshot=snapshot, S=S, V0=v)

            # compute the delta
            # Ag[kk, :] = (v2[pq_vec] - hx[ll, :]) / delta  # compute gradient as [x(p + delta) - x(p)] / delta
            Ag[kk, :] = (v2[pq_vec] - hx[ll, :]) / (10 * delta)  # go from [0, 0.2] -> [0, 1] 

        for ii in range(nV):  # build all nV covariance matrices
            Ag_prod = np.outer(Ag[:, ii], Ag[:, ii])
            C[ii] = 1 / M * Ag_prod

    return hx, C, param_store, nV, pq_vec


def orthogonal_decomposition(C, tr_error, l_exp, nV):

    """
    Orthogonal decomposition of the covariance matrix to determine the meaningful directions

    :param C: covariance matrix
    :param tr_error: allowed truncation error
    :param l_exp: expansion order
    :param nV: number of PQ buses
    :return: transformation matrix Wy, number of terms N_t and meaningful directions k, for each PQ bus

    """

    Wy_mat = []
    k_vec = []
    N_t_vec = []

    for ll in range(nV):
        # eigenvalues and eigenvectors
        v, w = np.linalg.eig(C[ll])

        v_sum = np.sum(v)
        err_v = 1
        k = 0  # meaningful directions
        while err_v > tr_error:
            err_v = 1 - v[k] / v_sum
            k += 1

        N_t = int(math.factorial(l_exp + k) / (math.factorial(k) * math.factorial(l_exp)))  # number of terms
        Wy = w[:,:k]  # and for now, do not define Wz

        Wy_mat.append(Wy)
        k_vec.append(k)
        N_t_vec.append(N_t)

    # return Wy, N_t, k
    return Wy_mat, N_t_vec, k_vec


def permutate(k, l_exp, nV):

    """
    Generate the permutations for all exponents of y

    :param k: vector with number of meaningful directions
    :param l: expansion order
    :param nV: number of PQ buses
    :return perms: array of all permutations
    """

    perms_vec = []

    for nn in range(nV):
        Nt = int(math.factorial(l_exp + k[nn]) / (math.factorial(l_exp) * math.factorial(k[nn])))

        lst = [ll for ll in range(l_exp + 1)] * k[nn]
        perms_all = set(itertools.permutations(lst, k[nn]))
        perms = []
        for per in perms_all:
            if sum(per) <= l_exp:
                perms.append(per)

        perms_vec.append(perms)

    return perms_vec


def polynomial_coeff(M, N_t, Wy, param_store, hx, perms, k, nV):

    """
    Calculate the coefficients c

    :param M: number of samples
    :param N_t: number of terms
    :param Wy: transformation matrix to go from p to y
    :param param_store: stored values of the parameters for each sample
    :param hx: solutions of the state for each sample
    :param perms: permutations of exponents
    :param k: number of meaningful directions
    :param nV: number of PQ buses
    :return: array with c coefficients
    """

    param_store = param_store * 10  # from [0, 0.2] -> [0, 1]

    c_vec_all = []

    for hh in range(nV):
        Q = np.zeros((M, N_t[hh]), dtype=float)  # store the values of the basis function

        for ll in range(M):
            yy = np.dot(Wy[hh].T, param_store[ll, :])  # go from p to y
            for nn in range(N_t[hh]):  # fill a row
                res = 1.0
                for kk in range(k[hh]):  # basis function, with all permutations
                    res = res * yy[kk] ** perms[hh][nn][kk]
                Q[ll, nn] = np.real(res)  # to avoid error of casting a complex number

        c_vec = np.dot(np.linalg.solve(np.dot(Q.T, Q), Q.T), hx[:, hh])
        c_vec_all.append(c_vec)

    return c_vec_all


def stretch_translation(param_up, param_low, n_param):
    """
    Stretches and translates from [a, b] to [0, 1]
    :param param_up: array with the upper limit of the parameters
    :param param_low: array with the lower limit of the parameters
    :param n_param: number of parameters
    :return: arrays with m and n, y = 1 / (b - a) * x - a / (b - a), of the form y = m * x + n
    """

    m_norm = np.zeros(n_param, dtype=float)  # stretch
    n_norm = np.zeros(n_param, dtype=float)  # translation

    for ll in range(n_param):
        m_norm[ll] = 1 / (param_up[ll] - param_low[ll])
        n_norm[ll] = - param_low[ll] / (param_up[ll] - param_low[ll])

    return m_norm, n_norm


def normalize(m_norm, n_norm, x, n_param):
    """
    Goes from x to y, where y = m * x + n, that is, it normalizes the input
    :param m_norm: normalized slopes
    :param n_norm: normalized origin
    :param x: list with n_param parameters where the normalization has to be applied
    :param n_param: number of parameters
    :return: list with normalized parameters
    """

    y = np.zeros(n_param, dtype=float)  # normalized array
    for ll in range(n_param):
        y[ll] = m_norm[ll] * x[ll] + n_norm[ll]

    return y


if __name__ == '__main__':

    # start time
    start_time = time.time()

    # load grid
    grid = test_grid()
    snapshot = gc.compile_snapshot_circuit(grid)

    # Input values
    n_param = 18  # number of parameters
    l_exp = 3  # expansion order
    k_est = 0.2  # proportion of expected meaningful directions
    factor_MNt = 2.5  # M = factor_MNt * Nterms, should be around 1.5 and 3
    param_lower_bnd = [0.0] * n_param  # lower limits for all parameters
    param_upper_bnd = [0.1] * n_param  # upper limits for all parameters
    delta = 1e-5  # small increment to calculate gradients
    tr_error = 0.1  # truncation error allowed
    print('Running...')

    # O. Obtain m and n to normalize inputs
    m_norm, n_norm = stretch_translation(param_upper_bnd, param_lower_bnd, n_param)

    # 1. Initial calculations
    n_k_est = int(k_est * n_param)  # estimated meaningful directions
    N_terms_est = int(math.factorial(l_exp + n_k_est) / (math.factorial(n_k_est) * math.factorial(l_exp)))  # estimated number of terms
    M = int(factor_MNt * N_terms_est)  # estimated number of samples

    # 2. Compute gradients and covariance matrix
    hx, C, param_store, nV, pq_vec = samples_calc(snapshot, M, n_param, param_lower_bnd, param_upper_bnd)

    # 3. Perform orthogonal decomposition
    Wy, N_t, k = orthogonal_decomposition(C, tr_error, l_exp, nV)

    # 4. Generate permutations
    perms = permutate(k, l_exp, nV)

    # 5. Find polynomial coefficients
    c_vec = polynomial_coeff(M, N_t, Wy, param_store, hx, perms, k, nV)
    print('Array of coefficients:     ', c_vec)

    # 6. Test
    pp = np.array([random.uniform(param_lower_bnd[kk], param_upper_bnd[kk]) for kk in range(n_param)])  # random parameters

    nP = int(n_param / 2)
    nQ = int(n_param / 2)

    P = np.array(pp[:nP])
    Q = np.array(pp[nP:nP+nQ])
    Sl = snapshot.load_data.C_bus_load * ((P + 1j * Q))  # already normalized P and Q
    Sg = snapshot.generator_data.get_injections_per_bus()[:, 0]
    S = Sg - Sl

    # calculate the real state
    v = power_flow(snapshot=snapshot, S=S, V0=snapshot.Vbus)
    x_real_vec = v[pq_vec]

    # calculate the estimated state
    x_est_vec = np.zeros(nV, dtype=float)
    pp = normalize(m_norm, n_norm, pp, n_param)

    for hh in range(nV):
        y_red = np.dot(Wy[hh].T, np.array(pp))
        x_est = 0

        for nn in range(N_t[hh]):
            res = 1
            for kk in range(k[hh]):
                res = res * y_red[kk] ** perms[hh][nn][kk]
            x_est += c_vec[hh][nn] * res

        x_est_vec[hh] = np.real(x_est)  # discard complex 0j

    stop_time = time.time()
    # ----------------------------------------------------------------

    print('Actual state:               ', x_real_vec)
    print('Estimated state:            ', x_est_vec)
    print('Error:                      ', abs(x_real_vec - x_est_vec))
    print('Number of power flow calls: ', M * (n_param + 1))
    print('Original calls n^m = M^m:   ', M ** n_param)
    print('Time elapsed:               ', stop_time - start_time, 's')

