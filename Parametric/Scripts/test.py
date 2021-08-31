# python file for testing small scripts

# Packages
import numpy as np
import GridCal.Engine as gc
import random
import math
import itertools
from smt.sampling_methods import LHS
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
np.set_printoptions(precision=10)


def dSbus_dV(Ybus, V):
    """
    Computes partial derivatives of power injection w.r.t. voltage.
    """

    Ibus = Ybus * V
    ib = range(len(V))
    diagV = csr_matrix((V, (ib, ib)))
    diagIbus = csr_matrix((Ibus, (ib, ib)))
    diagVnorm = csr_matrix((V / np.abs(V), (ib, ib)))
    dS_dVm = diagV * np.conj(Ybus * diagVnorm) + np.conj(diagIbus) * diagVnorm
    dS_dVa = 1j * diagV * np.conj(diagIbus - Ybus * diagV)

    return dS_dVm, dS_dVa

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
    return abs(res.voltage), np.angle(res.voltage), res.voltage


# load grid
grid = test_grid()
snapshot = gc.compile_snapshot_circuit(grid)

Ybus = snapshot.Ybus
V0 = snapshot.Vbus
n_bus = 10
n_no_slack = n_bus - 1
Pp = 0.2 * np.random.rand(1, n_bus)
Qq = 0.2 * np.random.rand(1, n_bus)
Sg = snapshot.generator_data.get_injections_per_bus()[:,0]
Sl = Pp + 1j * Qq
S = Sg - Sl[0]

mVbus, aVbus, Vbus = power_flow(snapshot=snapshot, S=S, V0=snapshot.Vbus)
print(Vbus)
print(mVbus)
print(aVbus)

# compute derivatives for later
# dS_dVm, dS_dVa = dSbus_dV(Ybus=snapshot.Ybus, V=V0)
# dS_dVm_red = dS_dVm[np.ix_(snapshot.pqpv, snapshot.pqpv)]
# dSdVm_array = dS_dVm.toarray()
# print(dSdVm_array)
# # print(dS_dVm_red)

# derivatives my way: introduce a Ai in the bus where there is a AS
indx = 4  # for instance, to look at a voltage
delta = 1e-5
Pp[0,indx] += delta  # varying P
# Qq[0,indx] += delta  # varying Q, one at a time
Sl_before = Sl
Sl_after = Pp + 1j * Qq
ASl = Sl_after - Sl_before
AS = - ASl[0]
Ivec = np.array(np.conj(Sl_after / Vbus))
print(Ivec[0])

# print(AI)
Zbus = np.linalg.pinv(Ybus.toarray())
# Vvec = np.linalg.solve(Ybus_ar, Ivec[0])
Vvec = np.dot(Zbus, Ivec[0])
AV = (Vvec - Vbus) / delta
print('Vvec')
# print(Vvec)
# print(abs(AV))






# AI = Y AV

# Ybus = Ybus.toarray()
# Zbus = np.linalg.inv(Ybus)
# Ppinc = delta
# Qqinc = 0  # comment
# Sinc = Ppinc + 1j * Qqinc
# Iinc = np.conj(Sinc / Vbus[indx])
# print(Iinc)



# test with the manual method (v_new - v_old) / delta
delta = 1e-5
Pp[0,indx] += delta  # varying P
# Qq[0,indx] += delta  # varying Q, one at a time
Sl = Pp + 1j * Qq
S = Sg - Sl[0]

mVbus_n, aVbus_n, Vbus_n = power_flow(snapshot=snapshot, S=S, V0=snapshot.Vbus)
print((mVbus_n - mVbus) / delta)


