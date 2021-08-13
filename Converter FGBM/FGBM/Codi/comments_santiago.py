####################################################################################################################
# Compile
####################################################################################################################
nc = compile_snapshot_circuit(grid)

V = nc.Vbus
S0 = nc.Sbus
pqpv = nc.pqpv
pq = nc.pq
pv = nc.pv
Ybus, Yf, Yt = compile_y(circuit=nc,
                         tr_tap_mod=nc.tr_tap_mod,
                         tr_tap_ang=nc.tr_tap_ang,
                         vsc_m=nc.vsc_m,
                         vsc_theta=nc.vsc_theta,
                         vsc_Beq=nc.vsc_Beq,
                         vsc_If=nc.vsc_Inom)

print(Ybus.toarray())

S_calc = V * np.conj(Ybus * V)

# equation (6)
gs = S_calc - S0

If = Yf * V
It = Yt * V
Vf = nc.C_branch_bus_f * V
Vt = nc.C_branch_bus_t * V
Sf = Vf * np.conj(If)  # eq. (8)
St = Vt * np.conj(It)  # eq. (9)

gp = gs.real[pqpv]  # eq. (12)
gq = gs.imag[pq]  # eq. (13)

"""
Control modes:

in the paper's scheme:
from -> DC
to   -> AC

|   Mode    |   const.1 |   const.2 |   type    |
-------------------------------------------------
|   1       |   theta   |   Vac     |   I       |   
|   2       |   Pf      |   Qac     |   I       |   
|   3       |   Pf      |   Vac     |   I       |   
-------------------------------------------------
|   4       |   Vdc     |   Qac     |   II      | 
|   5       |   Vdc     |   Vac     |   II      |
-------------------------------------------------
|   6       | Vdc droop |   Qac     |   III     |
|   7       | Vdc droop |   Vac     |   III     |
-------------------------------------------------
"""

# controls that the specified power flow is met
# Applicable to:
# - VSC devices where the power flow is set, this is types 2 and 3
# - Transformer devices where the power flow is set (is it?, apparently it only depends on the value of theta)
gsh = Sf.real[idx_sh] - Pf_set  # eq. (14)

# controls that 'Beq' absorbs the reactive power. So, Beq is always variable for all VSC?
# Applicable to:
# - All the VSC converters (is it?)
gqz = Sf.imag[idx_qz]  # eq. (15)

# Controls that 'ma' modulates to control the "voltage from" module.
# Applicable to:
# - Transformers that control the "from" voltage side
# - VSC that control the "from" voltage side, this is types 4 and 5
gvf = gs.imag[idx_vf]  # eq. (16)

# Controls that 'ma' modulates to control the "voltage to" module.
# Applicable to:
# - Transformers that control the "to" voltage side
# - VSC that control the "from" voltage side, this is types 1, 3, 5 and 7
gvt = gs.imag[idx_vt]  # eq. (17)

# controls that the specified reactive power flow is met, this is Qf=0
# Applicable to:
# - All VSC converters (is it?)
gqt = St.imag[idx_qt] - Qt_set  # eq. (18)


### JOSEP:
# controls that the specified power flow is met
# Applicable to:
# - VSC devices where the power flow is set, this is type 2 and 3 because Pf is the set power that the converter sends and phase shifting (theta_sh) mainly causes changes in the power flow
# - Transformer devices where the power flow is set (is it?, yes by the same last reason and by definition of the phase shifting transformer)
gsh = Sf.real[idx_sh] - Pf_set  # eq. (14)

# controls that 'Beq' absorbs the reactive power. So, Beq is always variable for all VSC? From what I have understood, yes, Beq is always an unknown. I found that if you set Beq to 0, the reactive 
# power may not be 0 in this given bus. Beq is meant to absorb all reactive power so that there is none in the DC grid. That is what I understand for Zero constraint.
# Applicable to:
# - All the VSC converters (is it? I am almost completely certain that yes, the Beq has to be found for each converter.)
gqz = Sf.imag[idx_qz]  # eq. (15)

# Controls that 'Beq' modulates to control the "voltage from" module. ma only controls AC variables.
# Applicable to:
# - VSC that control the "from" voltage side, this is types 4 and 5. Exactly, and there has to be one of them in each DC grid.
gvf = gs.imag[idx_vf]  # eq. (16)

# Controls that 'ma' modulates to control the "voltage to" module.
# Applicable to:
# - Transformers that control the "to" voltage side
# - VSC that control the "to" voltage side, this is types 1, 3, 5 and 7
gvt = gs.imag[idx_vt]  # eq. (17)

# controls that the specified reactive power flow in the AC side is met, this is Qt. ma is the variable responsible for that
# Applicable to:
# - All VSC converters that connect to an AC bus where the reactive power is controlled, this is types 2, 4 and 6.
gqt = St.imag[idx_qt] - Qt_set  # eq. (18)