# Parametric analysis of power systems

# Packages
import sys
import numpy as np
import GridCal.Engine as gc
import numba as nb
import pandas as pd
import math
import itertools
import time

import scipy as sp
from smt.sampling_methods import LHS
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
np.set_printoptions(precision=10)


def loadingBar(count, total, size):
    percent = float(count)/float(total)*100
    sys.stdout.write("\r" + str(int(count)).rjust(3, '0') + "/" + str(int(total)).rjust(3, '0') +
                     ' [' + '=' * int(percent / 10) * size + ' ' * (10 - int(percent / 10)) * size + ']')


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

    options = gc.PowerFlowOptions(tolerance=1e-3)

    res = gc.single_island_pf(circuit=snapshot,
                              Vbus=V0,
                              Sbus=S,
                              Ibus=snapshot.Ibus,
                              branch_rates=snapshot.Rates,
                              options=options,
                              logger=gc.Logger())
    return res.voltage


def params2power(x, snapshot: gc.SnapshotData, fix_generation=True):
    """
    Convert the parameters vector into the power injections
    :param x: array of parameter [Pgen | Pload | Qload]
    :param snapshot: GridCal snapshot of a circuit
    :param fix_generation: make the generation match the load active power (this is a condition for power flow convergence)
    :return: Power injections in complex form ready for the power flow
    """
    ng = snapshot.ngen
    nl = snapshot.nload
    a = ng
    b = a + nl
    c = b + nl

    Pgen = x[0:a]
    Pload = x[a:b]
    Qload = x[b:c]

    # make Pgen match Pload
    if fix_generation:
        Pgen = (Pgen / Pgen.sum()) * Pload.sum()

    Sl = snapshot.load_data.C_bus_load * (Pload + 1j * Qload)  # already normalized P and Q
    Sg = snapshot.generator_data.C_bus_gen * Pgen
    S = Sg - Sl
    return S


def estimate_voltage_sensitivity(J, Ybus, V, S, pvpq, pq, npvpq, nbus):
    # evaluate F(x0)
    Scalc = V * np.conj(Ybus * V)
    dS = Scalc - S  # compute the mismatch
    f = np.r_[dS[pvpq].real, dS[pq].imag]

    # compute update step
    dx = sp.sparse.linalg.spsolve(J, f)

    # reassign the solution vector
    dVa = np.zeros(nbus)
    dVm = np.zeros(nbus)
    dVa[pvpq] = dx[:npvpq]
    dVm[pq] = dx[npvpq:]

    return dVa, dVm


def samples_calc(snapshot: gc.SnapshotData, M, n_param, param_lower_bnd, param_upper_bnd, m_norm, delta):

    """
    Calculate the gradients, build the hx vector, the covariance C matrix and store the parameters

    :param M: number of samples
    :param n_param: number of parameters
    :param param_lower_bnd: array of lower bounds for the parameters
    :param param_upper_bnd: array of upper bounds for the parameters
    :param m_norm: factor by which the delta has to be stretched
    :return: hx, C and the stored parameters

    """

    # nP = int(n_param / 2)
    # nQ = int(n_param / 2)
    #
    # pq_vec = snapshot.pq
    # nV = len(pq_vec)
    # Sg = snapshot.generator_data.get_injections_per_bus()[:, 0]
    nV = snapshot.nbus  # number of variables
    hx = np.zeros((M, nV), dtype=float)  # x solutions at each sample, for all nV buses
    C = np.zeros((nV, n_param, n_param), dtype=float)  # nV covariance matrices

    # create samples with Latin Hypercube
    xlimits = np.zeros((n_param, 2), dtype=float)  # 2 columns: [lower_bound, upper_bound]
    for ll in range(n_param):
        xlimits[ll, 0] = param_lower_bnd[ll]
        xlimits[ll, 1] = param_upper_bnd[ll]

    sampling_lh = LHS(xlimits=xlimits)
    param_store = sampling_lh(M)  # matrix with all samples

    pq = snapshot.pq
    pv = snapshot.pv
    pvpq = np.r_[pv, pq]
    npvpq = len(pvpq)
    npv = len(pv)
    npq = len(pq)
    nbus = snapshot.nbus
    # generate lookup pvpq -> index pvpq (used in createJ)
    pvpq_lookup = np.zeros(np.max(snapshot.Ybus.indices) + 1, dtype=int)
    pvpq_lookup[pvpq] = np.arange(npvpq)

    for ll in range(M):

        # compose the power injections from the sample
        S = params2power(x=param_store[ll, :], snapshot=snapshot)

        # x solution for each sample
        v = power_flow(snapshot=snapshot, S=S, V0=snapshot.Vbus)
        hx[ll, :] = np.abs(v)

        J = gc.AC_jacobian(snapshot.Ybus, v, pvpq, pq, pvpq_lookup, npv, npq)

        # calculate gradients and form C matrix
        Ag = np.zeros((n_param, nV), dtype=float)
        for kk in range(n_param):
            params_delta = np.copy(param_store[ll, :])
            params_delta[kk] += delta  # increase a parameter by delta

            # compose the power injections from the sample delta
            S = params2power(x=params_delta, snapshot=snapshot)

            # compute the voltage sensitivity with the Jacobian matrix
            dVa, dVm = estimate_voltage_sensitivity(J, snapshot.Ybus, v, S, pvpq, pq, npvpq, nbus)

            Ag[kk, :] = dVm

        for ii in range(nV):  # build all nV covariance matrices
            Ag_prod = np.outer(Ag[:, ii], Ag[:, ii])
            C[ii] = 1 / M * Ag_prod

        loadingBar(ll, M, 2)

    return hx, C, param_store, nV


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
    k_vec = np.empty(nV, dtype=int)
    N_t_vec = np.empty(nV, dtype=int)

    for ll in range(nV):
        # eigenvalues and eigenvectors
        v, w = np.linalg.eig(C[ll])

        v_sum = np.sum(v)

        if v_sum > 0:
            err_v = 1
            k = 0  # meaningful directions
            while err_v > tr_error:
                err_v = 1.0 - v[k] / v_sum
                k += 1
        else:
            k = 1

        N_t = int(math.factorial(l_exp + k) / (math.factorial(k) * math.factorial(l_exp)))  # number of terms

        # TODO: Al final, k siempre es 2...no podríamos saber esto con antelación para declarar Wy como una matriz?
        Wy = w[:, :k]  # and for now, do not define Wz

        Wy_mat.append(np.real(Wy))
        k_vec[ll] = k
        N_t_vec[ll] = N_t

    # return Wy, N_t, k
    return Wy_mat, N_t_vec, k_vec


def permutate(k, l_exp, nV):
    """
    Generate the permutations for all exponents of y

    :param k: vector with number of meaningful directions
    :param l_exp: expansion order
    :param nV: number of PQ buses
    :return perms: array of all permutations
    """

    # TODO: Me da la impresión de que debe existir una forma más eficiente de conseguir el resultado final

    perms_vec = []

    for nn in range(nV):
        # Nt = int(math.factorial(l_exp + k[nn]) / (math.factorial(l_exp) * math.factorial(k[nn])))

        lst = [ll for ll in range(l_exp + 1)] * k[nn]
        perms_all = set(itertools.permutations(lst, int(k[nn])))  # TODO: porqué usamos "set", no se descartan abajo con el "if sum(per)..." ?
        perms = []
        for per in perms_all:
            if sum(per) <= l_exp:
                perms.append(per)

        perms_vec.append(perms)

    return perms_vec


def polynomial_coeff(M, N_t, Wy, param_store, hx, perms, k, nV, m_norm, n_norm, n_param):

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
    :param m_norm: factor by which the parameters are stretched
    :param n_norm: factor by which the parameters are translated
    :param n_param: number of parameters
    :return: array with c coefficients
    """

    # param_store = param_store * 10  # from [0, 0.2] -> [0, 1]
    for ll in range(M):
        param_store[ll, :] = normalize(m_norm, n_norm, param_store[ll, :], n_param)

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


@nb.njit()
def stretch_translation(param_up, param_low, n_param):
    """
    Stretches and translates from [a, b] to [0, 1]
    :param param_up: array with the upper limit of the parameters
    :param param_low: array with the lower limit of the parameters
    :param n_param: number of parameters
    :return: arrays with m and n, y = 1 / (b - a) * x - a / (b - a), of the form y = m * x + n
    """

    m_norm = np.empty(n_param, dtype=nb.float64)  # stretch
    n_norm = np.empty(n_param, dtype=nb.float64)  # translation

    for ll in range(n_param):
        m_norm[ll] = 1 / (param_up[ll] - param_low[ll])
        n_norm[ll] = - param_low[ll] / (param_up[ll] - param_low[ll])

    return m_norm, n_norm


@nb.njit()
def normalize(m_norm, n_norm, x, n_param):
    """
    Goes from x to y, where y = m * x + n, that is, it normalizes the input
    :param m_norm: normalized slopes
    :param n_norm: normalized origin
    :param x: list with n_param parameters where the normalization has to be applied
    :param n_param: number of parameters
    :return: list with normalized parameters
    """

    y = np.empty(n_param, dtype=nb.float64)  # normalized array
    for ll in range(n_param):
        y[ll] = m_norm[ll] * x[ll] + n_norm[ll]

    return y


# @nb.njit()
def solve_parametric(pp, Wy, nV, N_t, k, perms, c_vec):  # parallelize!
    """
    Compute the solution with the polynomial coefficients
    :param pp: array with the normalized parameters
    :param Wy: matrix to go from p to y
    :param nV: number of PQ buses
    :param k: number of meaningful directions
    :param perms: exponents to use at each permutation
    :param c_vec: vector of coefficients
    :return: array with the voltages
    """

    x_est_vec = np.empty(nV, dtype=float)

    for hh in range(nV):
        y_red = np.dot(Wy[hh].transpose(), pp)
        x_est = 0

        for nn in range(N_t[hh]):
            res = 1.0
            for kk in range(k[hh]):
                res *= y_red[kk] ** perms[hh][nn][kk]
            x_est += c_vec[hh][nn] * res

        x_est_vec[hh] = x_est

    return x_est_vec


def get_bounds(snapshot: gc.SnapshotData, max_gen, min_gen, max_load, min_load):

    nl = snapshot.nload
    ng = snapshot.ngen
    n_param = ng + nl + nl
    upper = np.r_[max_gen * np.ones(ng), max_load * np.ones(nl), max_load * np.ones(nl)] / snapshot.Sbase
    lower = np.r_[min_gen * np.ones(ng), min_load * np.ones(nl), min_load * np.ones(nl)] / snapshot.Sbase
    return lower, upper, n_param


def PCA_training(snapshot, k_est, l_exp, factor_MNt, tr_error, param_lower_bnd, param_upper_bnd, n_param, delta):

    print('Running...')

    # 0. Obtain m and n to normalize inputs
    m_norm, n_norm = stretch_translation(param_upper_bnd, param_lower_bnd, n_param)

    # 1. Initial calculations
    n_k_est = int(k_est * n_param)  # estimated meaningful directions
    N_terms_est = int(math.factorial(l_exp + n_k_est) / (
                math.factorial(n_k_est) * math.factorial(l_exp)))  # estimated number of terms
    M = int(factor_MNt * N_terms_est)  # estimated number of samples

    # 2. Compute gradients and covariance matrix
    hx, C, param_store, nV = samples_calc(snapshot, M, n_param, param_lower_bnd, param_upper_bnd, m_norm, delta)

    # 3. Perform orthogonal decomposition
    Wy, N_t, k = orthogonal_decomposition(C, tr_error, l_exp, nV)

    # 4. Generate permutations
    perms = permutate(k, l_exp, nV)

    # 5. Find polynomial coefficients
    c_vec = polynomial_coeff(M, N_t, Wy, param_store, hx, perms, k, nV, m_norm, n_norm, n_param)

    return m_norm, n_norm, c_vec, Wy, nV, N_t, k, perms, M

def find_sampling_points(n_tests, n_param, param_lower_bnd, param_upper_bnd):

    # 6. Test
    pp_list = np.empty((n_tests, n_param))
    for ntt in range(n_tests):
        pp_list[ntt, :] = np.random.uniform(param_lower_bnd, param_upper_bnd)


    return pp_list

def sample_test_values(snapshot: gc.SnapshotData, n_tests, pp_list):


    # calculate the real state
    x_real_store = []
    for ntt in range(n_tests):
        S = params2power(x=pp_list[ntt, :], snapshot=snapshot)
        v = power_flow(snapshot=snapshot, S=S, V0=snapshot.Vbus)
        x_real_vec = np.abs(v)
        x_real_store.append(x_real_vec)

    return x_real_store


def calculate_estimation_at_the_sampling_points(n_tests, m_norm, n_norm, pp_list, n_param, Wy, nV, N_t, k, perms, c_vec):
    # calculate the estimated state
    x_est_store = []
    for ntt in range(n_tests):
        pp = normalize(m_norm, n_norm, pp_list[ntt], n_param)
        x_est_vec = solve_parametric(pp, Wy, nV, N_t, k, perms, c_vec)
        x_est_store.append(x_est_vec)
    return x_est_store


def main_comparison(grid: gc.MultiCircuit, min_p=0, max_p=20, n_tests = 2000):

    snapshot = gc.compile_snapshot_circuit(grid)
    l_exp = 3  # expansion order
    k_est = 0.2  # proportion of expected meaningful directions
    factor_MNt = 2.5  # M = factor_MNt * Nterms, should be around 1.5 and 3
    delta = 1e-5  # small increment to calculate gradients
    tr_error = 0.1  # truncation error allowed
     # number of tests to perform, totally arbitrary

    # generate bounds
    param_lower_bnd, param_upper_bnd, n_param = get_bounds(snapshot=snapshot,
                                                           max_gen=max_p,
                                                           min_gen=min_p,
                                                           max_load=max_p,
                                                           min_load=min_p)

    # start time
    start_time = time.time()
    m_norm, n_norm, c_vec, Wy, nV, N_t, k, perms, M = PCA_training(snapshot=snapshot,
                                                                   k_est=k_est,
                                                                   l_exp=l_exp,
                                                                   factor_MNt=factor_MNt,
                                                                   tr_error=tr_error,
                                                                   param_lower_bnd=param_lower_bnd,
                                                                   param_upper_bnd=param_upper_bnd,
                                                                   n_param=n_param,
                                                                   delta=delta)
    stop_time1 = time.time()  # time to find the coefficients

    # Input values
    pp_list = find_sampling_points(n_tests=n_tests,
                                   n_param=n_param,
                                   param_lower_bnd=param_lower_bnd,
                                   param_upper_bnd=param_upper_bnd)
    stop_time2 = time.time()  # time up to generate the random parameters

    x_real_store = sample_test_values(snapshot=snapshot,
                                      n_tests=n_tests,
                                      pp_list=pp_list)
    stop_time3 = time.time()  # time up to solve the traditional

    x_est_store = calculate_estimation_at_the_sampling_points(n_tests=n_tests,
                                                              m_norm=m_norm,
                                                              n_norm=n_norm,
                                                              pp_list=pp_list,
                                                              n_param=n_param,
                                                              Wy=Wy,
                                                              nV=nV,
                                                              N_t=N_t,
                                                              k=k,
                                                              perms=perms,
                                                              c_vec=c_vec)
    stop_time4 = time.time()  # time up to solve the parametric

    # error calculation
    err_abs = np.zeros(nV)  # calculate the average of all errors, for each bus
    for ntt in range(n_tests):
        err_abs += 1 / n_tests * (abs(x_real_store[ntt] - x_est_store[ntt]))

    # ----------------------------------------------------------------
    print()
    data = {'Names': snapshot.bus_data.bus_names,
            'Actual state': x_real_store[-1],
            'Estimated state': x_est_store[-1],
            'Mean error': err_abs}
    df = pd.DataFrame(data=data)
    print(df)
    print('Number of power flow calls: ', M * (n_param + 1))
    print('Original calls n^m = M^m:   ', M ** n_param)
    print('Time to find polynomial:    ', stop_time1 - start_time, 's')
    print('Time with traditional:      ', stop_time3 - stop_time2, 's')
    print('Time with parametric:       ', stop_time4 - stop_time3, 's')


if __name__ == '__main__':
    # load grid
    # grid = test_grid()
    fname = '/home/santi/Documentos/Git/GitHub/GridCal/Grids_and_profiles/grids/IEEE39.gridcal'
    # fname = '/home/santi/Documentos/Git/GitHub/GridCal/Grids_and_profiles/grids/Illinois 200 Bus.gridcal'
    grid = gc.FileOpen(fname).open()
    main_comparison(grid=grid, min_p=1, max_p=20, n_tests=100)




