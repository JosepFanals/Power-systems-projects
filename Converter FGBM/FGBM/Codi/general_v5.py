# debuggar, veure per què no dóna bé però no tinc errors

import pandas as pd
import numpy as np

df_busos_AC = pd.read_excel('dades_v1.xlsx', sheet_name="busos_AC")
df_busos_DC = pd.read_excel('dades_v1.xlsx', sheet_name="busos_DC")
df_linies_AC = pd.read_excel('dades_v1.xlsx', sheet_name="linies_AC")
df_linies_DC = pd.read_excel('dades_v1.xlsx', sheet_name="linies_DC")
df_convertidors = pd.read_excel('dades_v1.xlsx', sheet_name="convertidors")

print(df_busos_AC)
print(df_busos_DC)
print(df_linies_AC)
print(df_linies_DC)
print(df_convertidors)

# ----------- MATRIUS AC -----------

n_busos_AC = len(df_busos_AC)
n_linies_AC = len(df_linies_AC)

X_linies_AC = np.zeros((n_linies_AC, n_linies_AC), dtype=complex)
A_linies_AC = np.zeros((n_busos_AC, n_linies_AC), dtype=float)
Ysh = np.zeros(n_busos_AC, dtype=complex)

for i in range(n_linies_AC):
    X_linies_AC[i, i] = 1 / (df_linies_AC.iloc[i, 2] + 1j * df_linies_AC.iloc[i, 3])
    A_linies_AC[df_linies_AC.iloc[i, 0], i] = 1
    A_linies_AC[df_linies_AC.iloc[i, 1], i] = -1
    Ysh[df_linies_AC.iloc[i, 0]] += df_linies_AC.iloc[i, 4] + 1j * df_linies_AC.iloc[i, 5]
    Ysh[df_linies_AC.iloc[i, 1]] += df_linies_AC.iloc[i, 4] + 1j * df_linies_AC.iloc[i, 5]

Yadm_AC = np.dot(A_linies_AC, np.dot(X_linies_AC, np.transpose(A_linies_AC)))

for i in range(len(df_convertidors)):
    Yadm_AC[df_convertidors.iloc[i, 1], df_convertidors.iloc[i, 1]] += 1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])

Gadm_AC = np.real(Yadm_AC)
Badm_AC = np.imag(Yadm_AC)

# ----------- MATRIUS DC -----------

n_busos_DC = len(df_busos_DC)
n_linies_DC = len(df_linies_DC)

X_linies_DC = np.zeros((n_linies_DC, n_linies_DC), dtype=complex)
A_linies_DC = np.zeros((n_busos_DC, n_linies_DC), dtype=float)

for i in range(n_linies_DC):
    X_linies_DC[i, i] = 1 / (df_linies_DC.iloc[i, 2])
    A_linies_DC[df_linies_DC.iloc[i, 0], i] = 1
    A_linies_DC[df_linies_DC.iloc[i, 1], i] = -1

Yadm_DC = np.dot(A_linies_DC, np.dot(X_linies_DC, np.transpose(A_linies_DC)))
Gdc = Yadm_DC


# ----------- ÍNDEXS AC -----------
sl = []
pq = []
pv = []
pqv = []

pq0 = []  # basat en l'índex 0
pv0 = []
pqv0 = []
nsl_compt = 0
vsl_compt = []

for i in range(n_busos_AC):
    if df_busos_AC.iloc[i, 4] == "slack":
        sl.append(i)
        nsl_compt += 1
        vsl_compt.append(nsl_compt)

    elif df_busos_AC.iloc[i, 4] == "PQ":
        pq.append(i)
        pq0.append(i - nsl_compt)
        vsl_compt.append(nsl_compt)

    elif df_busos_AC.iloc[i, 4] == "PV":
        pv.append(i)
        pv0.append(i - nsl_compt)
        vsl_compt.append(nsl_compt)

    elif df_busos_AC.iloc[i, 4] == "PQV":
        pqv.append(i)
        pqv0.append(i - nsl_compt)
        vsl_compt.append(nsl_compt)

npq = len(pq)
npv = len(pv)
npqv = len(pqv)

n_no_sl = npq + npv + npqv
Ysl = np.zeros(n_no_sl, dtype=complex)

for i in range(n_no_sl):
    Ysl[i] = Yadm_AC[i + vsl_compt[i], sl]

Gac = np.delete(Gadm_AC, sl[0], 0)  # considero que només hi hauria d'haver un slack
Gac = np.delete(Gac, sl[0], 1)

Bac = np.delete(Badm_AC, sl[0], 0)  # considero que només hi hauria d'haver un slack
Bac = np.delete(Bac, sl[0], 1)


# ----------- MATRIUS CONVERTIDORS -----------
n_convertidors = len(df_convertidors)
ind_pf = []
ind_pf2 = []  # del convertidor, no del bus DC
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 4]):
        ind_pf.append(df_convertidors.iloc[i, 2])
        ind_pf2.append(df_convertidors.iloc[i, 0])

npf = len(ind_pf)
mat_1pf = np.zeros((npf, n_convertidors), dtype=float)
for i in range(npf):
    mat_1pf[i, ind_pf[i]] = 1

G0 = np.zeros(n_convertidors, dtype=float)

Y_conv_dc = np.zeros((n_convertidors, n_busos_DC), dtype=complex)
Y_conv_ac = np.zeros((n_convertidors, n_no_sl), dtype=complex)
Y_conv_conv = np.zeros((n_convertidors, n_convertidors), dtype=complex)
Y_conv_pf = np.zeros((n_convertidors, npf), dtype=complex)

for i in range(n_convertidors):
    X_links_conv = 1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])
    Y_conv_dc[i, df_convertidors.iloc[i, 2]] = X_links_conv
    Y_conv_ac[i, df_convertidors.iloc[i, 1] - vsl_compt[df_convertidors.iloc[i, 1]]] = X_links_conv
    Y_conv_conv[i, i] = X_links_conv
    G0[i] = df_convertidors.iloc[i, 5]

compt_pf = 0
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 4]):
        X_links_conv = 1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])
        Y_conv_pf[i, compt_pf] = X_links_conv

print(Y_conv_pf)

print(Y_conv_dc)
print(Y_conv_ac)

# ----------- MATRIU GLOBAL -----------

mat_1pv = np.zeros((n_no_sl, npv), dtype=float)
trobats = 0
for i in range(n_no_sl):
    if i in pv0:
        mat_1pv[i, trobats] = 1
        trobats += 1

mat_2w = np.zeros((npv + npqv, n_no_sl), dtype=float)
trobats = 0
for j in range(n_no_sl):
    if j in pqv0 or j in pv0:
        mat_2w[trobats, j] = 2
        trobats += 1

ind_vdc = []
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 3]):
        ind_vdc.append(df_convertidors.iloc[i, 2])

nvdc = len(ind_vdc)
mat_2vdc = np.zeros((nvdc, n_busos_DC), dtype=float)
for i in range(nvdc):
    mat_2vdc[i, ind_vdc[i]] = 2

m02 = np.zeros((n_no_sl, npv))
m03 = np.zeros((n_no_sl, n_busos_DC))
m04 = np.zeros((n_no_sl, n_busos_DC))
m05 = np.zeros((n_no_sl, n_convertidors))
m06 = np.zeros((n_no_sl, n_convertidors))
m07 = np.zeros((n_no_sl, n_convertidors))
m08 = np.zeros((n_no_sl, nvdc))
m09 = np.zeros((n_no_sl, npf))
m0_10 = np.zeros((n_no_sl, npf))

m13 = np.zeros((n_no_sl, n_busos_DC))
m14 = np.zeros((n_no_sl, n_busos_DC))
m15 = np.zeros((n_no_sl, n_convertidors))
m16 = np.zeros((n_no_sl, n_convertidors))
m17 = np.zeros((n_no_sl, n_convertidors))
m18 = np.zeros((n_no_sl, nvdc))
m19 = np.zeros((n_no_sl, npf))
m1_10 = np.zeros((n_no_sl, npf))

m21 = np.zeros((npv + npqv, n_no_sl))
m22 = np.zeros((npv + npqv, npv))
m23 = np.zeros((npv + npqv, n_busos_DC))
m24 = np.zeros((npv + npqv, n_busos_DC))
m25 = np.zeros((npv + npqv, n_convertidors))
m26 = np.zeros((npv + npqv, n_convertidors))
m27 = np.zeros((npv + npqv, n_convertidors))
m28 = np.zeros((npv + npqv, nvdc))
m29 = np.zeros((npv + npqv, npf))
m2_10 = np.zeros((npv + npqv, npf))

m30 = np.zeros((n_busos_DC, n_no_sl))
m31 = np.zeros((n_busos_DC, n_no_sl))
m32 = np.zeros((n_busos_DC, npv))
m34 = np.zeros((n_busos_DC, n_busos_DC))
m36 = np.zeros((n_busos_DC, n_convertidors))
m37 = np.zeros((n_busos_DC, n_convertidors))
m38 = np.zeros((n_busos_DC, nvdc))
m39 = np.zeros((n_busos_DC, npf))
m3_10 = np.zeros((n_busos_DC, npf))

m40 = np.zeros((n_busos_DC, n_no_sl))
m41 = np.zeros((n_busos_DC, n_no_sl))
m42 = np.zeros((n_busos_DC, npv))
m43 = np.zeros((n_busos_DC, n_busos_DC))
m45 = np.zeros((n_busos_DC, n_convertidors))
m47 = np.zeros((n_busos_DC, n_convertidors))
m48 = np.zeros((n_busos_DC, nvdc))
m49 = np.zeros((n_busos_DC, npf))
m4_10 = np.zeros((n_busos_DC, npf))

mat_1ifre = np.zeros((n_busos_DC, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifre[df_convertidors.iloc[i, 2], i] = 1

mat_1ifim = np.zeros((n_busos_DC, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifim[df_convertidors.iloc[i, 2], i] = 1

mat_1qz = np.zeros((nvdc, n_convertidors), dtype=float)
trobats = 0
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 3]):
        mat_1qz[trobats, df_convertidors.iloc[i, 0]] = 1

m50 = np.zeros((npf, n_no_sl))
m51 = np.zeros((npf, n_no_sl))
m52 = np.zeros((npf, npv))
m53 = np.zeros((npf, n_busos_DC))
m54 = np.zeros((npf, n_busos_DC))
m56 = np.zeros((npf, n_convertidors))
m57 = np.zeros((npf, n_convertidors))
m58 = np.zeros((npf, nvdc))
m59 = np.zeros((npf, npf))
m5_10 = np.zeros((npf, npf))

m60 = np.zeros((nvdc, n_no_sl))
m61 = np.zeros((nvdc, n_no_sl))
m62 = np.zeros((nvdc, npv))
m63 = np.zeros((nvdc, n_busos_DC))
m64 = np.zeros((nvdc, n_busos_DC))
m65 = np.zeros((nvdc, n_convertidors))
m67 = np.zeros((nvdc, n_convertidors))
m68 = np.zeros((nvdc, nvdc))
m69 = np.zeros((nvdc, npf))
m6_10 = np.zeros((nvdc, npf))

m70 = np.zeros((nvdc, n_no_sl))
m71 = np.zeros((nvdc, n_no_sl))
m72 = np.zeros((nvdc, npv))
m74 = np.zeros((nvdc, n_busos_DC))
m75 = np.zeros((nvdc, n_convertidors))
m76 = np.zeros((nvdc, n_convertidors))
m77 = np.zeros((nvdc, n_convertidors))
m78 = np.zeros((nvdc, nvdc))
m79 = np.zeros((nvdc, npf))
m7_10 = np.zeros((nvdc, npf))

mat_2E = np.zeros((npf, npf), dtype=float)
for i in range(npf):
    mat_2E[i, i] = 2

m80 = np.zeros((npf, n_no_sl))
m81 = np.zeros((npf, n_no_sl))
m82 = np.zeros((npf, npv))
m83 = np.zeros((npf, n_busos_DC))
m84 = np.zeros((npf, n_busos_DC))
m85 = np.zeros((npf, n_convertidors))
m86 = np.zeros((npf, n_convertidors))
m87 = np.zeros((npf, n_convertidors))
m88 = np.zeros((npf, nvdc))
m8_10 = np.zeros((npf, npf))

gij_ac = np.real(Y_conv_ac)
bij_ac = np.imag(Y_conv_ac)
gij_dc = np.real(Y_conv_dc)
bij_dc = np.imag(Y_conv_dc)

gij_conv = np.real(Y_conv_conv)
bij_conv = np.imag(Y_conv_conv)
gij_pf = np.real(Y_conv_pf)
bij_pf = np.imag(Y_conv_pf)

m92 = np.zeros((n_convertidors, npv))
m96 = np.zeros((n_convertidors, n_convertidors))
m98 = np.zeros((n_convertidors, nvdc))
m9_10 = np.zeros((n_convertidors, npf))

m10_2 = np.zeros((n_convertidors, npv))
m10_5 = np.zeros((n_convertidors, n_convertidors))
m10_9 = np.zeros((n_convertidors, npf))

mat_1ifre2 = np.zeros((n_convertidors, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifre2[i, i] = 1

mat_1ifim2 = np.zeros((n_convertidors, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifim2[i, i] = 1

mat_1B = np.zeros((n_convertidors, nvdc), dtype=float)
trobats = 0
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 3]):
        mat_1B[i, trobats] = -1
        trobats += 1

mat_gbij_1ac = np.zeros((n_no_sl, n_busos_DC), dtype = complex)
for i in range(n_convertidors):
    print(df_convertidors.iloc[i, 2])
    mat_gbij_1ac[df_convertidors.iloc[i, 1] - vsl_compt[df_convertidors.iloc[i, 1]], df_convertidors.iloc[i, 2]] = 1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])

mat_gbij_1M = np.zeros((n_no_sl, n_convertidors), dtype = complex)
for i in range(n_convertidors):
    mat_gbij_1M[df_convertidors.iloc[i, 1] - vsl_compt[df_convertidors.iloc[i, 1]], df_convertidors.iloc[i, 0]] = 1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])

mat_gbij_1E = np.zeros((n_no_sl, npf), dtype = complex)
trobats = 0
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 4]):
        mat_gbij_1E[df_convertidors.iloc[i, 1] - vsl_compt[df_convertidors.iloc[i, 1]], trobats] = 1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])
        trobats += 1

gij_1ac = np.real(mat_gbij_1ac)
bij_1ac = np.imag(mat_gbij_1ac)

gij_1M = np.real(mat_gbij_1M)
bij_1M = np.imag(mat_gbij_1M)

gij_1E = np.real(mat_gbij_1E)
bij_1E = np.imag(mat_gbij_1E)

mat_IE = np.zeros((n_convertidors, npf), dtype = complex)
trobats = 0
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 4]):
        mat_IE[i, trobats] = 1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])
        trobats += 1

gij_IE = np.real(mat_IE)
bij_IE = np.imag(mat_IE)

# anar provant si trec depenent de quins blocs matricials té inversa
# mat = np.block([[Gac, -Bac, m02, -gij_1ac, bij_1ac, m05, m06, -gij_1M, m08, -gij_1E, bij_1E], 
#                 [Bac, Gac, mat_1pv, -bij_1ac, -gij_1ac, m15, m16, -bij_1M, m18, -bij_1E, -gij_1E],
#                 [mat_2w, m21, m22, m23, m24, m25, m26, m27, m28, m29, m2_10],
#                 [m30, m31, m32, Gdc, m34, mat_1ifre, m36, m37, m38, m39, m3_10],
#                 [m40, m41, m42, m43, Gdc, m45, mat_1ifim, m47, m48, m49, m4_10],
#                 [m50, m51, m52, m53, m54, mat_1pf, m56, m57, m58, m59, m5_10],
#                 [m60, m61, m62, m63, m64, m65, mat_1qz, m67, m68, m69, m6_10],
#                 [m70, m71, m72, mat_2vdc, m74, m75, m76, m77, m78, m79, m7_10],
#                 [m80, m81, m82, m83, m84, m85, m86, m87, m88, mat_2E, m8_10],
#                 [gij_ac, -bij_ac, m92, -gij_dc, bij_dc, mat_1ifre2, m96, -gij_conv, m98, gij_IE, bij_IE],  # potser retardar aquesta bij_IE per tal de tenir inversa
#                 [bij_ac, gij_ac, m10_2, -bij_dc, -gij_dc, m10_5, mat_1ifim2, -bij_conv, mat_1B, bij_IE, -gij_IE]
#                 ])

mat = np.block([[Gac, -Bac, m02, -gij_1ac, bij_1ac, m05, m06, -gij_1M, m08, -gij_1E, bij_1E], 
                [Bac, Gac, mat_1pv, -bij_1ac, -gij_1ac, m15, m16, -bij_1M, m18, -bij_1E, -gij_1E],
                [mat_2w, m21, m22, m23, m24, m25, m26, m27, m28, m29, m2_10],
                [m30, m31, m32, Gdc, m34, mat_1ifre, m36, m37, m38, m39, m3_10],
                [m40, m41, m42, m43, Gdc, m45, mat_1ifim, m47, m48, m49, m4_10],
                [m50, m51, m52, m53, m54, mat_1pf, m56, m57, m58, m59, m5_10],
                [m60, m61, m62, m63, m64, m65, mat_1qz, m67, m68, m69, m6_10],
                [m70, m71, m72, mat_2vdc, m74, m75, m76, m77, m78, m79, m7_10],
                [m80, m81, m82, m83, m84, m85, m86, m87, m88, mat_2E, m8_10],
                [gij_ac, -bij_ac, m92, -gij_dc, bij_dc, mat_1ifre2, m96, -gij_conv, m98, gij_IE, 0 * bij_IE],  # aquesta última que sigui bij_IE i no m9_10??
                [bij_ac, 0 * gij_ac, m10_2, -bij_dc, -gij_dc, m10_5, mat_1ifim2, -bij_conv, mat_1B, bij_IE, -gij_IE]
                ])

# imprescindible fer aquestes multiplicacions per 0, així la inversa surt molt millor

mat = np.array(np.real(mat))

df = pd.DataFrame(mat)
print(df)
df.to_excel("matriu.xlsx")  

mat_inv = np.linalg.inv(mat)

for i in range(len(mat_inv)):
    for j in range(len(mat_inv)):
        if mat_inv[i, j] < 1e-15 and mat_inv[i, j] > -1e-15:
            mat_inv[i, j] = 0

df = pd.DataFrame(np.real(mat_inv))
#print(df)
df.to_excel("matriu_inv.xlsx") 

print(np.linalg.det(mat))


# ARA MUNTO EL RHS, PAS PER PAS
# primer inicialitzo tots els subvectors de RHS. Després els vaig emplenant, no em preocupa si he d'entrar en diversos for loops

rhs0 = np.zeros(n_no_sl, dtype=float)
rhs1 = np.zeros(n_no_sl, dtype=float)
rhs2 = np.zeros(npv + npqv, dtype=float)
rhs3 = np.zeros(n_busos_DC, dtype=float)
rhs4 = np.zeros(n_busos_DC, dtype=float)
rhs5 = np.zeros(npf, dtype=float)
rhs6 = np.zeros(nvdc, dtype=float)
rhs7 = np.zeros(nvdc, dtype=float)
rhs8 = np.zeros(npf, dtype=float)
rhs9 = np.zeros(n_convertidors, dtype=float)
rhs10 = np.zeros(n_convertidors, dtype=float)

prof = 60

# incògnites
Vac_re = np.zeros((prof, n_no_sl), dtype=float)
Vac_im = np.zeros((prof, n_no_sl), dtype=float)
Vac = np.zeros((prof, n_no_sl), dtype=complex)

Qac_i = np.zeros((prof, npv), dtype=float)

Vdc_re = np.zeros((prof, n_busos_DC), dtype=float)
Vdc_im = np.zeros((prof, n_busos_DC), dtype=float)
Vdc = np.zeros((prof, n_busos_DC), dtype=complex)

If_re = np.zeros((prof, n_convertidors), dtype=float)
If_im = np.zeros((prof, n_convertidors), dtype=float)
If = np.zeros((prof, n_convertidors), dtype=complex)

M = np.zeros((prof, n_convertidors), dtype=float)
B = np.zeros((prof, nvdc), dtype=float)  # no estic segur de si ha de ser nvdc sempre
E_re = np.zeros((prof, npf), dtype=float)
E_im = np.zeros((prof, npf), dtype=float)

Xac = np.zeros((prof, n_no_sl), dtype=complex)
Xdc = np.zeros((prof, n_busos_DC), dtype=complex)
V0 = np.zeros(prof, dtype=complex)  # vector de les tensions de l'slack. Assumeixo que només hi ha un slack!
# falta inicialitzar-les

Vac_re[0, :] = 1
Vac[0, :] = 1
Xac[0, :] = 1
Xdc[0, :] = 1

Vdc_re[0, :] = 1
Vdc[0, :] = 1

M[0, :] = 1
E_re[0, :] = 1


def conv1x(a, b, i1, i2, k, lim, cc, retard):
    suma = 0
    for kk in range(k, lim + 1):
        suma += a[kk, i1] * b[cc - kk - retard, i2]
    return suma


# dades:
Pac = np.zeros(n_no_sl, dtype=float)
Qac = np.zeros(n_no_sl, dtype=float)
Wac = np.zeros(n_no_sl, dtype=float)

Pdc = np.zeros(n_no_sl, dtype=float)

trobats = 0
for i in range(len(df_busos_AC)):
    if df_busos_AC.iloc[i, 4] == "PQ":
        Pac[trobats] = df_busos_AC.iloc[i, 1]
        Qac[trobats] = df_busos_AC.iloc[i, 2]
        trobats += 1
    elif df_busos_AC.iloc[i, 4] == "PV":
        Pac[trobats] = df_busos_AC.iloc[i, 1]
        Wac[trobats] = df_busos_AC.iloc[i, 3] ** 2  # al quadrat
        trobats += 1
    elif df_busos_AC.iloc[i, 4] == "PQV":
        Pac[trobats] = df_busos_AC.iloc[i, 1]
        Qac[trobats] = df_busos_AC.iloc[i, 2]
        Wac[trobats] = df_busos_AC.iloc[i, 3] ** 2  # al quadrat
        trobats += 1

for i in range(len(df_busos_DC)):
    Pdc[df_busos_DC.iloc[i, 0]] = df_busos_DC.iloc[i, 1]

for i in range(len(df_busos_AC)):
    if df_busos_AC.iloc[i, 4] == "slack":
        V0[0] = 1
        V0[1] = df_busos_AC.iloc[i, 3] - 1

for i in range(len(df_convertidors)):  # afegeixo els shunts de les connexions amb els convertidors
    Ysh[df_convertidors.iloc[i, 1]] += 1j * df_convertidors.iloc[i, 8]

Y_sh = np.zeros(n_no_sl, dtype=complex)  # és el vector de shunts retallat, sense l'slack
trobats = 0
for i in range(n_busos_AC):
    if df_busos_AC.iloc[i, 4] != "slack":
        Y_sh[trobats] = Ysh[df_busos_AC.iloc[i, 0]]
        trobats += 1



cc = 1

trobats_pv = 0
for i in range(len(rhs0)):
    if i in pq0:
        rhs0[i] = np.real((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
    elif i in pv0:
        rhs0[i] = np.real(Pac[i] * Xac[cc - 1, i] - 1j * conv1x(Xac, Qac_i, i, trobats_pv, 1, cc-1, cc, 0)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
        trobats_pv += 1
    elif i in pqv0:
        rhs0[i] = np.real((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])

trobats_pv = 0
for i in range(len(rhs0)):
    if i in pq0:
        rhs1[i] = np.imag((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
    elif i in pv0:
        rhs1[i] = np.imag(Pac[i] * Xac[cc - 1, i] - 1j * conv1x(Xac, Qac_i, i, trobats_pv, 1, cc-1, cc, 0)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
        trobats_pv += 1
    elif i in pqv0:
        rhs1[i] = np.imag((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])

# ara construir rhs2 i així anar fent. Després generalitzar per cc >= 2. O potser generalitzar per c >= 3 perquè he de retardar aaquells termes que desapareixen a la matriu

trobats_pqv_pv = 0
for i in range(len(df_busos_AC)):
    if df_busos_AC.iloc[i, 4] == "PV" or df_busos_AC.iloc[i, 4] == "PQV":
        rhs2[trobats_pqv_pv] = df_busos_AC.iloc[i, 3] ** 2 - 1  # em faltava aquest -1!
        trobats_pqv_pv += 1

for i in range(len(df_busos_DC)):
    rhs3[df_busos_DC.iloc[i, 0]] = np.real(Pdc[i] * Xdc[cc - 1, i])

for i in range(len(df_busos_DC)):
    rhs4[df_busos_DC.iloc[i, 0]] = np.imag(Pdc[i] * Xdc[cc - 1, i])

trobats = 0
for i in range(len(df_convertidors)):
    if not np.isnan(df_convertidors.iloc[i, 4]):
        rhs5[trobats] = df_convertidors.iloc[i, 4]
        trobats += 1

for i in range(len(rhs6)):
    rhs6[i] = 0  # és la de la Qz, quan c > 1 hi haurà convolucions

trobats = 0
for i in range(len(df_convertidors)):
    if not np.isnan(df_convertidors.iloc[i, 3]):
        rhs7[trobats] = df_convertidors.iloc[i, 3] ** 2 - 1  # em faltava aquest -1
        trobats += 1

trobats = 0
for i in range(len(rhs8)):
    rhs8[i] = 0  # és l'eq. del mòdul de E, hi haurà convolucions per c > 1

# emplenar rhs9 i rhs10 mirant els de 4busosv7.py

def conv1(a, b, k, lim1, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        suma += a[kk] * b[cc - kk - retard]
    return suma


def conv2(a, b, c, k, j, lim1, lim2, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        for jj in range(j, lim2 + 1 - kk):
            suma += a[kk] * b[jj] * c[cc - kk - jj - retard]
    return suma


def conv3(a, b, c, d, k, j, i, lim1, lim2, lim3, retard, cc):
    suma = 0
    for kk in range(k, lim1 + 1):
        for jj in range(j, lim2 + 1 - kk):
            for ii in range(i, lim3 + 1 - kk - jj):
                suma += a[kk] * b[jj] * c[ii] * d[cc - kk - jj - ii - retard]
    return suma

b_dc_c = []
b_ac_c = []
g_conv = []
b_conv = []
bc_shunt = []

for i in range(len(df_convertidors)):
    b_ac_c.append(df_convertidors.iloc[i, 1] - vsl_compt[df_convertidors.iloc[i, 1]])
    b_dc_c.append(df_convertidors.iloc[i, 2])
    g_conv.append(np.real(1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])))
    b_conv.append(np.imag(1 / (df_convertidors.iloc[i, 6] + 1j * df_convertidors.iloc[i, 7])))
    bc_shunt.append(df_convertidors.iloc[i, 8])  # agafo la shunt però sense part imaginària

print(b_dc_c)
print(b_ac_c)
print(G0)
print(bc_shunt)

tr_E = 0  # trobats E
tr_B = 0  # trobats B

for i in range(len(rhs9)):

    if not np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i amb E
        # rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))

        rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - b_conv[i] * E_im[cc - 1, tr_E]

        tr_E += 1
        tr_B += 1

    elif not np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i no amb E
        rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) + b_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))

        tr_B += 1

    elif np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb E i no amb B
        # rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc))

        rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * E_im[cc - 1, tr_E]

        tr_E += 1

    elif np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # no incloure la convolució ni amb E ni amb B 
        rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) + b_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc)



tr_E = 0  # trobats E
tr_B = 0  # trobats B

for i in range(len(rhs10)):

    if not np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i amb E

        # rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))

        rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]

        tr_E += 1
        tr_B += 1

    elif not np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i no amb E
        # rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) - b_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))

        rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) - b_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]

        tr_B += 1

    elif np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb E i no amb B
        # rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc))

        rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]

        tr_E += 1

    elif np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # no incloure la convolució ni amb E ni amb B 
        # rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) - b_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc)

        rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) - b_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]


# ----------

rhs_tot = np.block([rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, rhs9, rhs10])
lhs = np.dot(mat_inv, rhs_tot)

print('rhs:')
for i in range(len(rhs_tot)):
    print(rhs_tot[i])

ind_ara = 0
Vac_re[cc, :n_no_sl] = lhs[0: n_no_sl]
ind_ara += n_no_sl
Vac_im[cc, :n_no_sl] = lhs[ind_ara:ind_ara + n_no_sl]
ind_ara += n_no_sl
Qac_i[cc, :npv] = lhs[ind_ara:ind_ara + npv]
ind_ara += npv
Vdc_re[cc, :n_busos_DC] = lhs[ind_ara:ind_ara + n_busos_DC]
ind_ara += n_busos_DC
Vdc_im[cc, :n_busos_DC] = lhs[ind_ara:ind_ara + n_busos_DC]
ind_ara += n_busos_DC
If_re[cc, :n_convertidors] = lhs[ind_ara:ind_ara + n_convertidors]
ind_ara += n_convertidors
If_im[cc, :n_convertidors] = lhs[ind_ara:ind_ara + n_convertidors]
ind_ara += n_convertidors
M[cc, :n_convertidors] = lhs[ind_ara:ind_ara + n_convertidors]
ind_ara += n_convertidors
B[cc-1, :nvdc] = lhs[ind_ara:ind_ara + nvdc]
ind_ara += nvdc
E_re[cc, :npf] = lhs[ind_ara:ind_ara + npf]
ind_ara += npf
E_im[cc, :npf] = lhs[ind_ara:ind_ara + npf]
ind_ara = 0

Vac[cc, :n_no_sl] = Vac_re[cc, :n_no_sl] + 1j * Vac_im[cc, :n_no_sl]
Vdc[cc, :n_busos_DC] = Vdc_re[cc, :n_busos_DC] + 1j * Vdc_im[cc, :n_busos_DC]
If[cc, :n_convertidors] = If_re[cc, :n_convertidors] + 1j * If_im[cc, :n_convertidors]

for i in range(n_no_sl):
    Xac[cc, i] = - conv1(Xac[:, i], np.conj(Vac[:, i]), 0, cc-1, 0, cc)
    #Xac[cc, i] = - np.conj(Vac[cc, i]) - conv1(Xac[:, i], np.conj(Vac[:, i]), 1, cc-1, 0, cc)

for i in range(n_busos_DC):
    Xdc[cc, i] = - conv1(Xdc[:, i], np.conj(Vdc[:, i]), 0, cc-1, 0, cc)


# ----------


bac_pf = []
ind_convE = []
bdc_pf = []
for i in range(len(df_convertidors)):
    if not np.isnan(df_convertidors.iloc[i, 4]):
        bac_pf.append(df_convertidors.iloc[i, 1] - vsl_compt[df_convertidors.iloc[i, 1]])
        bdc_pf.append(df_convertidors.iloc[i, 2])
        ind_convE.append(df_convertidors.iloc[i, 0])


# --------------


for cc in range(2, prof):

    trobats_pv = 0
    tr_convE = 0  # número de convertidors trobats on es controla la E

    tr = 0
    for i in range(len(rhs0)):  # partir en 2 opcions, si connecta amb conv. on es controla la Pf o no. S'hauran d'afegir convolucions

        if i in bac_pf:

            if i in pq0:
                rhs0[i] = g_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) - g_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) + np.real((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])

            elif i in pv0:
                # rhs0[i] = np.real(Pac[i] * Xac[cc - 1, i] - 1j * conv1x(Xac, Qac, i, trobats_pv, 1, cc-1, cc, 0)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
                rhs0[i] = g_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) - g_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) + np.real(Pac[i] * Xac[cc - 1, i] - 1j * conv1(Xac[:, i], Qac_i[:, trobats_pv], 1, cc-1, 0, cc)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
                trobats_pv += 1
            elif i in pqv0:
                rhs0[i] = g_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) - g_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) + np.real((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
            
            tr += 1

        else:
            if i in pq0:
                rhs0[i] = np.real((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
            elif i in pv0:
                rhs0[i] = np.real(Pac[i] * Xac[cc - 1, i] - 1j * conv1(Xac[:, i], Qac_i[:, trobats_pv], 1, cc-1, 0, cc)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
                trobats_pv += 1
            elif i in pqv0:
                rhs0[i] = np.real((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])

    trobats_pv = 0
    tr_convE = 0  # número de convertidors trobats on es controla la E

    tr = 0
    for i in range(len(rhs1)):  # partir en 2 opcions, si connecta amb conv. on es controla la Pf o no. S'hauran d'afegir convolucions

        if i in bac_pf:

            if i in pq0:
                rhs1[i] = g_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) + b_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) + g_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) + np.imag((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])

            elif i in pv0:
                # rhs0[i] = np.real(Pac[i] * Xac[cc - 1, i] - 1j * conv1x(Xac, Qac, i, trobats_pv, 1, cc-1, cc, 0)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
                rhs1[i] = g_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) + b_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) + g_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) + np.imag(Pac[i] * Xac[cc - 1, i] - 1j * conv1(Xac[:, i], Qac_i[:, trobats_pv], 1, cc-1, 0, cc)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
                trobats_pv += 1
            elif i in pqv0:
                rhs1[i] = g_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) + b_conv[ind_convE[tr]] * conv1(Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_re[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) + g_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, cc-1, 0, cc) + g_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_re[:, tr], 1, 0, cc-1, cc, 0, cc) - b_conv[ind_convE[tr]] * conv1(Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, cc-1, 0, cc) - b_conv[ind_convE[tr]] * conv2(M[:, ind_convE[tr]], Vdc_im[:, bdc_pf[tr]], E_im[:, tr], 1, 0, cc-1, cc, 0, cc) + np.imag((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
            
            tr += 1

        else:
            if i in pq0:
                rhs1[i] = np.imag((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
            elif i in pv0:
                rhs1[i] = np.imag(Pac[i] * Xac[cc - 1, i] - 1j * conv1(Xac[:, i], Qac_i[:, trobats_pv], 1, cc-1, 0, cc)  - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])
                trobats_pv += 1
            elif i in pqv0:
                rhs1[i] = np.imag((Pac[i] - 1j * Qac[i]) * Xac[cc - 1, i] - Ysl[i] * V0[cc] - Y_sh[i] * Vac[cc - 1, i])


    
    trobats_pqv_pv = 0
    for i in range(len(df_busos_AC)):  # |Vac|
        if df_busos_AC.iloc[i, 4] == "PV" or df_busos_AC.iloc[i, 4] == "PQV":  # hi haurà convolució
            rhs2[trobats_pqv_pv] = - conv1(Vac_re[:, df_busos_AC.iloc[i, 0] - vsl_compt[df_busos_AC.iloc[i, 0]]], Vac_re[:, df_busos_AC.iloc[i, 0] - vsl_compt[df_busos_AC.iloc[i, 0]]], 1, cc-1, 0, cc) - conv1(Vac_im[:, df_busos_AC.iloc[i, 0] - vsl_compt[df_busos_AC.iloc[i, 0]]], Vac_im[:, df_busos_AC.iloc[i, 0] - vsl_compt[df_busos_AC.iloc[i, 0]]], 1, cc-1, 0, cc)
            trobats_pqv_pv += 1

    for i in range(len(df_busos_DC)):  # sumatori Idc_re = 0
        rhs3[df_busos_DC.iloc[i, 0]] = np.real(Pdc[i] * Xdc[cc - 1, i])

    for i in range(len(df_busos_DC)):  # sumatori Idc_im = 0
        rhs4[df_busos_DC.iloc[i, 0]] = np.imag(Pdc[i] * Xdc[cc - 1, i])

    trobats = 0
    for i in range(len(df_convertidors)):  # Pf
        if not np.isnan(df_convertidors.iloc[i, 4]):
            rhs5[trobats] = -conv1(Vdc_re[:, b_dc_c[i]], If_re[:, i], 1, cc-1, 0, cc) - conv1(Vdc_im[:, b_dc_c[i]], If_im[:, i], 1, cc-1, 0, cc)
            trobats += 1

    trobats = 0
    for i in range(len(df_convertidors)):  # Qz
        if not np.isnan(df_convertidors.iloc[i, 3]):
            rhs6[trobats] = - conv1(Vdc_re[:, b_dc_c[i]], If_im[:, i], 1, cc-1, 0, cc) + conv1(Vdc_im[:, b_dc_c[i]], If_re[:, i], 1, cc-1, 0, cc)
            trobats += 1

    trobats = 0
    for i in range(len(df_convertidors)):  # |Vdc|
        if not np.isnan(df_convertidors.iloc[i, 3]):
            rhs7[trobats] = - conv1(Vdc_re[:, b_dc_c[i]], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) - conv1(Vdc_im[:, b_dc_c[i]], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc)
            trobats += 1

    trobats = 0
    for i in range(len(df_convertidors)):  # |E|
        if not np.isnan(df_convertidors.iloc[i, 4]):
            rhs8[trobats] = - conv1(E_re[:, trobats], E_re[:, trobats], 1, cc-1, 0, cc) - conv1(E_im[:, trobats], E_im[:, trobats], 1, cc-1, 0, cc)
            trobats += 1

    
    tr_E = 0  # trobats E
    tr_B = 0  # trobats B

    for i in range(len(rhs9)):

        if not np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i amb E

            rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - b_conv[i] * E_im[cc - 1, tr_E]

            tr_E += 1
            tr_B += 1

        elif not np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i no amb E
            rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) + b_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc))

            tr_B += 1

        elif np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb E i no amb B

            rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * E_im[cc - 1, tr_E]

            tr_E += 1

        elif np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # no incloure la convolució ni amb E ni amb B 
            rhs9[i] = G0[i] * np.real(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) - bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) + b_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc)



    tr_E = 0  # trobats E
    tr_B = 0  # trobats B

    for i in range(len(rhs10)):

        if not np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i amb E

            rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]

            tr_E += 1
            tr_B += 1

        elif not np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb B i no amb E

            rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) - b_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) + np.real(1j * conv3(B[:, tr_B], M[:, i], M[:, i], Vdc[:, b_dc_c[i]], 0, 0, 0, cc-2, cc-1, cc-1, 1, cc)) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]

            tr_B += 1

        elif np.isnan(df_convertidors.iloc[i, 3]) and not np.isnan(df_convertidors.iloc[i, 4]):  # incloure la convolució amb E i no amb B

            rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_re[:, b_ac_c[i]], E_re[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - b_conv[i] * (conv1(Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, cc-1, 0, cc) + conv2(M[:, i], Vac_im[:, b_ac_c[i]], E_im[:, tr_E], 1, 0, cc-1, cc, 0, cc)) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]

            tr_E += 1

        elif np.isnan(df_convertidors.iloc[i, 3]) and np.isnan(df_convertidors.iloc[i, 4]):  # no incloure la convolució ni amb E ni amb B 

            rhs10[i] = G0[i] * np.imag(conv2(If[:, i], If[:, i], Vdc[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc)) + b_conv[i] * (conv1(M[:, i], Vdc_re[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + g_conv[i] * (conv1(M[:, i], Vdc_im[:, b_dc_c[i]], 1, cc-1, 0, cc) + conv2(M[:, i], M[:, i], Vdc_im[:, b_dc_c[i]], 1, 0, cc-1, cc, 0, cc)) + bc_shunt[i] / 2 * conv2(M[:, i], M[:, i], Vdc_re[:, b_dc_c[i]], 0, 0, cc-1, cc-1, 1, cc) - g_conv[i] * conv1(M[:, i], Vac_im[:, b_ac_c[i]], 1, cc-1, 0, cc) - b_conv[i] * conv1(M[:, i], Vac_re[:, b_ac_c[i]], 1, cc-1, 0, cc) - g_conv[i] * Vac_im[cc-1, b_ac_c[i]]

    
    rhs_tot = np.block([rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, rhs9, rhs10])
    lhs = np.dot(mat_inv, rhs_tot)

    ind_ara = 0
    Vac_re[cc, :n_no_sl] = lhs[0: n_no_sl]
    ind_ara += n_no_sl
    Vac_im[cc, :n_no_sl] = lhs[ind_ara:ind_ara + n_no_sl]
    ind_ara += n_no_sl
    Qac_i[cc, :npv] = lhs[ind_ara:ind_ara + npv]
    ind_ara += npv
    Vdc_re[cc, :n_busos_DC] = lhs[ind_ara:ind_ara + n_busos_DC]
    ind_ara += n_busos_DC
    Vdc_im[cc, :n_busos_DC] = lhs[ind_ara:ind_ara + n_busos_DC]
    ind_ara += n_busos_DC
    If_re[cc, :n_convertidors] = lhs[ind_ara:ind_ara + n_convertidors]
    ind_ara += n_convertidors
    If_im[cc, :n_convertidors] = lhs[ind_ara:ind_ara + n_convertidors]
    ind_ara += n_convertidors
    M[cc, :n_convertidors] = lhs[ind_ara:ind_ara + n_convertidors]
    ind_ara += n_convertidors
    B[cc-1, :nvdc] = lhs[ind_ara:ind_ara + nvdc]
    ind_ara += nvdc
    E_re[cc, :npf] = lhs[ind_ara:ind_ara + npf]
    ind_ara += npf
    E_im[cc, :npf] = lhs[ind_ara:ind_ara + npf]
    ind_ara = 0

    Vac[cc, :n_no_sl] = Vac_re[cc, :n_no_sl] + 1j * Vac_im[cc, :n_no_sl]
    Vdc[cc, :n_busos_DC] = Vdc_re[cc, :n_busos_DC] + 1j * Vdc_im[cc, :n_busos_DC]
    If[cc, :n_convertidors] = If_re[cc, :n_convertidors] + 1j * If_im[cc, :n_convertidors]

    for i in range(n_no_sl):
        Xac[cc, i] = - conv1(Xac[:, i], np.conj(Vac[:, i]), 0, cc-1, 0, cc)

    for i in range(n_busos_DC):
        Xdc[cc, i] = - conv1(Xdc[:, i], np.conj(Vdc[:, i]), 0, cc-1, 0, cc)


# RESULTATS
from Pade import pade


# print('Vac_re', Vac_re[:, 0])
# print('Vac_im', Vac_im[:, 0])
# print('Vdc_re', Vdc_re[:, 0])
# print('Vdc_im', Vdc_im[:, 0])
# print('If_re', If_re[:, 0])
# print('If_im', If_im[:, 0])
# print('M', M[:, 0])
# print('B', B[:, 0])
# print('E_re', E_re[:, 0])
# print('E_im', E_im[:, 0])


# print('Vac_re', pade(prof-1, Vac_re[:, 0], 1))
# print('Vac_im', pade(prof-1, Vac_im[:, 0], 1))
# print('Vac', pade(prof-1, Vac[:, 0], 1))
# print('Vdc_re', pade(prof-1, Vdc_re[:, 0], 1))
# print('Vdc_im', pade(prof-1, Vdc_im[:, 0], 1))
# print('If_re', pade(prof-1, If_re[:, 1], 1))
# print('If_im', pade(prof-1, If_im[:, 1], 1))
# print('M', pade(prof-1, M[:, 0], 1))
# print('B', pade(prof-2, B[:, 0], 1))
# print('E_re', pade(prof-1, E_re[:, 0], 1))
# print('E_im', pade(prof-1, E_im[:, 0], 1))


# print('Vac_re', Vac_re[:, 0])
# print('Vac_im', Vac_im[:, 0])
# print('Vac', Vac[:, 0])


print(Y_sh)
print(Ysh)