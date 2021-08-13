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

m02 = np.zeros((n_no_sl, npv))
m03 = np.zeros((n_no_sl, n_busos_DC))
m04 = np.zeros((n_no_sl, n_busos_DC))
m05 = np.zeros((n_no_sl, n_convertidors))
m06 = np.zeros((n_no_sl, n_convertidors))
m07 = np.zeros((n_no_sl, n_convertidors))
m08 = np.zeros((n_no_sl, n_convertidors))
m09 = np.zeros((n_no_sl, npf))
m0_10 = np.zeros((n_no_sl, npf))

m13 = np.zeros((n_no_sl, n_busos_DC))
m14 = np.zeros((n_no_sl, n_busos_DC))
m15 = np.zeros((n_no_sl, n_convertidors))
m16 = np.zeros((n_no_sl, n_convertidors))
m17 = np.zeros((n_no_sl, n_convertidors))
m18 = np.zeros((n_no_sl, n_convertidors))
m19 = np.zeros((n_no_sl, npf))
m1_10 = np.zeros((n_no_sl, npf))

m21 = np.zeros((npv + npqv, n_no_sl))
m22 = np.zeros((npv + npqv, npv))
m23 = np.zeros((npv + npqv, n_busos_DC))
m24 = np.zeros((npv + npqv, n_busos_DC))
m25 = np.zeros((npv + npqv, n_convertidors))
m26 = np.zeros((npv + npqv, n_convertidors))
m27 = np.zeros((npv + npqv, n_convertidors))
m28 = np.zeros((npv + npqv, n_convertidors))
m29 = np.zeros((npv + npqv, npf))
m2_10 = np.zeros((npv + npqv, npf))

m30 = np.zeros((n_busos_DC, n_no_sl))
m31 = np.zeros((n_busos_DC, n_no_sl))
m32 = np.zeros((n_busos_DC, npv))
m34 = np.zeros((n_busos_DC, n_busos_DC))
m36 = np.zeros((n_busos_DC, n_convertidors))
m37 = np.zeros((n_busos_DC, n_convertidors))
m38 = np.zeros((n_busos_DC, n_convertidors))
m39 = np.zeros((n_busos_DC, npf))
m3_10 = np.zeros((n_busos_DC, npf))

m40 = np.zeros((n_busos_DC, n_no_sl))
m41 = np.zeros((n_busos_DC, n_no_sl))
m42 = np.zeros((n_busos_DC, npv))
m43 = np.zeros((n_busos_DC, n_busos_DC))
m45 = np.zeros((n_busos_DC, n_convertidors))
m47 = np.zeros((n_busos_DC, n_convertidors))
m48 = np.zeros((n_busos_DC, n_convertidors))
m49 = np.zeros((n_busos_DC, npf))
m4_10 = np.zeros((n_busos_DC, npf))

mat_1ifre = np.zeros((n_busos_DC, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifre[df_convertidors.iloc[i, 2], i] = 1

mat_1ifim = np.zeros((n_busos_DC, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifim[df_convertidors.iloc[i, 2], i] = 1

mat_1qz = np.zeros((n_convertidors, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1qz[i, i] = 1

m50 = np.zeros((npf, n_no_sl))
m51 = np.zeros((npf, n_no_sl))
m52 = np.zeros((npf, npv))
m53 = np.zeros((npf, n_busos_DC))
m54 = np.zeros((npf, n_busos_DC))
m56 = np.zeros((npf, n_convertidors))
m57 = np.zeros((npf, n_convertidors))
m58 = np.zeros((npf, n_convertidors))
m59 = np.zeros((npf, npf))
m5_10 = np.zeros((npf, npf))

m60 = np.zeros((n_convertidors, n_no_sl))
m61 = np.zeros((n_convertidors, n_no_sl))
m62 = np.zeros((n_convertidors, npv))
m63 = np.zeros((n_convertidors, n_busos_DC))
m64 = np.zeros((n_convertidors, n_busos_DC))
m65 = np.zeros((n_convertidors, n_convertidors))
m67 = np.zeros((n_convertidors, n_convertidors))
m68 = np.zeros((n_convertidors, n_convertidors))
m69 = np.zeros((n_convertidors, npf))
m6_10 = np.zeros((n_convertidors, npf))

ind_vdc = []
for i in range(n_convertidors):
    if not np.isnan(df_convertidors.iloc[i, 3]):
        ind_vdc.append(df_convertidors.iloc[i, 2])

nvdc = len(ind_vdc)
mat_2vdc = np.zeros((nvdc, n_busos_DC), dtype=float)
for i in range(nvdc):
    mat_2vdc[i, ind_vdc[i]] = 2

m70 = np.zeros((nvdc, n_no_sl))
m71 = np.zeros((nvdc, n_no_sl))
m72 = np.zeros((nvdc, npv))
m74 = np.zeros((nvdc, n_busos_DC))
m75 = np.zeros((nvdc, n_convertidors))
m76 = np.zeros((nvdc, n_convertidors))
m77 = np.zeros((nvdc, n_convertidors))
m78 = np.zeros((nvdc, n_convertidors))
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
m88 = np.zeros((npf, n_convertidors))
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
m98 = np.zeros((n_convertidors, n_convertidors))
m9_10 = np.zeros((n_convertidors, npf))

m10_2 = np.zeros((n_convertidors, npv))
m10_5 = np.zeros((n_convertidors, n_convertidors))

mat_1ifre2 = np.zeros((n_convertidors, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifre2[i, i] = 1

mat_1ifim2 = np.zeros((n_convertidors, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1ifim2[i, i] = 1

mat_1B = np.zeros((n_convertidors, n_convertidors), dtype=float)
for i in range(n_convertidors):
    mat_1B[i, i] = -1

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
                [gij_ac, -bij_ac, m92, -gij_dc, bij_dc, mat_1ifre2, m96, -gij_conv, m98, gij_IE, bij_IE],  # potser retardar aquesta bij_IE per tal de tenir inversa
                [bij_ac, gij_ac, m10_2, -bij_dc, -gij_dc, m10_5, mat_1ifim2, -bij_conv, mat_1B, bij_IE, -gij_IE]
                ])

print(mat.shape)
#print('det', np.linalg.det(mat))
df = pd.DataFrame(mat)
print(df)
df.to_excel("matriu.xlsx")  
mat_inv = np.linalg.inv(mat)

