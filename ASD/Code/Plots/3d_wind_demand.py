import numpy as np

# P_wind = np.arange(5, 16, 1)
# P_wind_3x = [0, 0, 0.25**3, 0.5**3, 0.75**3, 1**3, 1.25**3, 1.5**3, 1.75**3, 2**3, 2.25**3, 2.5**3, 2.75**3, 3**3, 3**3, 0]
P_wind_3x_2 = []
for kk in range(32):
    if kk < 4:
        P_wind_3x_2.append(0)
    elif kk < 24:
        P_wind_3x_2.append((kk / 2) ** 3)
    else:
        P_wind_3x_2.append((23 / 2) ** 3)

for ii in range(len(P_wind_3x_2)):
    P_wind_3x_2[ii] = P_wind_3x_2[ii] * 0.012

print(P_wind_3x_2)

# for ii in range(len(P_wind_3x)):
    # P_wind_3x[ii] = P_wind_3x[ii] * 0.7
P_demand = [22.33, 21.38, 20.19, 19.70, 19.75, 19.31, 19.88, 19.83, 20.19, 22.56, 24.79, 25.32, 26.02, 25.63, 25.37, 23.80, 22.87, 22.47, 22.55, 23.35, 25.20, 27.88, 26.90, 24.79]
print(P_demand)

P_mat = np.zeros((len(P_wind_3x_2), len(P_demand)), dtype=float)
for ii in range(len(P_wind_3x_2)):
    for nn in range(len(P_demand)):
        P_mat[ii, nn] = P_demand[nn] - P_wind_3x_2[ii]


for ii in range(len(P_wind_3x_2)):
    for nn in range(len(P_demand)):
        print(P_wind_3x_2[ii])
        # print(P_demand[nn])

print(P_wind_3x_2)

# for ii in range(len(P_demand)):
    # print(str(ii) + ', ' + str(P_demand[ii]))

print('0-00000000000000')

# for ii in range(len(P_wind_3x_2)):
    # print(str(ii / 2) + ', ' + str(P_wind_3x_2[ii]))


# http://polynomialregression.drque.net/online.php

# Pwind: f( x ) = 1.6349058426679717 - 10.719343177393107x + 11.204527200243698x2 - 2.55136190066593x3 + 0.17642202760515993x4

# Pdemand: f( x ) = 22.394708020782733 - 1.6354720294954328x + 0.39434274588533585x2 - 0.0035090092967717756x3 - 0.029337034458808153x4 + 0.007731225380636515x5 - 0.0008557978395676469x6 + 0.00004732882729208937x7 - 0.00000128996363643316x8 + 1.381540655448e-8x9

# Pwind new: f(x) = -0.04848479278074868 + 0.3828985204520174x - 0.2608981245129764x2 + 0.061004598841255586x3 - 0.002658684188611232x4