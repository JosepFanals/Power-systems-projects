import numpy as np
from GridCal.Engine import *
import random
from functions_PQ import *
# from GridCal.Engine.CalculationEngine import *

np.set_printoptions(precision=12)

# EDITS
x_bus = 4
indx_Vbus = x_bus - 1
ssz = 0.1


P2 = np.arange(0, 40, ssz)
Q2 = np.arange(0, 30, ssz)
P3 = np.arange(0, 30, ssz)
Q3 = np.arange(0, 45, ssz)
P4 = np.arange(0, 25, ssz)
Q4 = np.arange(0, 20, ssz)
P5 = np.arange(0, 30, ssz)
Q5 = np.arange(0, 20, ssz)


P2m = translation_stretch_vec(P2)
Q2m = translation_stretch_vec(Q2)
P3m = translation_stretch_vec(P3)
Q3m = translation_stretch_vec(Q3)
P4m = translation_stretch_vec(P4)
Q4m = translation_stretch_vec(Q4)
P5m = translation_stretch_vec(P5)
Q5m = translation_stretch_vec(Q5)


n_param = 8  # the 8 powers
Mm = 10  # number of modes, arbitrary value, >1.5Nterms
delta = 1e-5
mat_mp = np.zeros((Mm, n_param), dtype=float)

for ll in range(Mm):  # gathering samples of parameters
	for kk in range(n_param):
		# with the normalized parameters
		# mat_mp[ll, 0] = random.sample(P2m, 1)[0]
		# mat_mp[ll, 1] = random.sample(Q2m, 1)[0]
		# mat_mp[ll, 2] = random.sample(P3m, 1)[0]
		# mat_mp[ll, 3] = random.sample(Q3m, 1)[0]
		# mat_mp[ll, 4] = random.sample(P4m, 1)[0]
		# mat_mp[ll, 5] = random.sample(Q4m, 1)[0]
		# mat_mp[ll, 6] = random.sample(P5m, 1)[0]
		# mat_mp[ll, 7] = random.sample(Q5m, 1)[0]

		# non-normalized parameters
		mat_mp[ll, 0] = random.sample(list(P2), 1)[0]
		mat_mp[ll, 1] = random.sample(list(Q2), 1)[0]
		mat_mp[ll, 2] = random.sample(list(P3), 1)[0]
		mat_mp[ll, 3] = random.sample(list(Q3), 1)[0]
		mat_mp[ll, 4] = random.sample(list(P4), 1)[0]
		mat_mp[ll, 5] = random.sample(list(Q4), 1)[0]
		mat_mp[ll, 6] = random.sample(list(P5), 1)[0]
		mat_mp[ll, 7] = random.sample(list(Q5), 1)[0]

print(mat_mp)

# vv52 = V5(1, 2, 3, 4, 5, 6, 7, 8, indx_Vbus)
# print(vv52)

C = np.zeros((n_param, n_param), dtype=float)
Ag_store = []

for ll in range(Mm):
	pp0 = mat_mp[ll, 0]
	pp1 = mat_mp[ll, 1]
	pp2 = mat_mp[ll, 2]
	pp3 = mat_mp[ll, 3]
	pp4 = mat_mp[ll, 4]
	pp5 = mat_mp[ll, 5]
	pp6 = mat_mp[ll, 6]
	pp7 = mat_mp[ll, 7]

	v5_sol = V5(pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7, indx_Vbus)

	v5_sol_pp0 = V5(pp0 + delta, pp1, pp2, pp3, pp4, pp5, pp6, pp7, indx_Vbus)
	v5_sol_pp1 = V5(pp0, pp1 + delta, pp2, pp3, pp4, pp5, pp6, pp7, indx_Vbus)
	v5_sol_pp2 = V5(pp0, pp1, pp2 + delta, pp3, pp4, pp5, pp6, pp7, indx_Vbus)
	v5_sol_pp3 = V5(pp0, pp1, pp2, pp3 + delta, pp4, pp5, pp6, pp7, indx_Vbus)
	v5_sol_pp4 = V5(pp0, pp1, pp2, pp3, pp4 + delta, pp5, pp6, pp7, indx_Vbus)
	v5_sol_pp5 = V5(pp0, pp1, pp2, pp3, pp4, pp5 + delta, pp6, pp7, indx_Vbus)
	v5_sol_pp6 = V5(pp0, pp1, pp2, pp3, pp4, pp5, pp6 + delta, pp7, indx_Vbus)
	v5_sol_pp7 = V5(pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7 + delta, indx_Vbus)

	Ag = np.array([[(v5_sol_pp0 - v5_sol) / delta],
	 [(v5_sol_pp1 - v5_sol) / delta], 
	 [(v5_sol_pp2 - v5_sol) / delta], 
	 [(v5_sol_pp3 - v5_sol) / delta], 
	 [(v5_sol_pp4 - v5_sol) / delta], 
	 [(v5_sol_pp5 - v5_sol) / delta], 
	 [(v5_sol_pp6 - v5_sol) / delta], 
	 [(v5_sol_pp7 - v5_sol) / delta]])

	# print(Ag)
	print(Ag.T)
	Ag_store.append(Ag.T)

	Ag_prod = np.dot(Ag, Ag.T)
	# print(Ag_prod)
	C += 1 / Mm * Ag_prod
	# print(C)

print(C)

w, v = np.linalg.eig(C)
print(w)  # eigenvalues
print(v)  # eigenvectors

# length(y) = 1, 1 dominant dimension is good enough here
# build A matrix
nterms = 4

matA = np.zeros((Mm, nterms), dtype=float)
d1 = 1
Wy = v[:,d1 - 1]  # only first column
Wz = v[:,d1:]
print(Wy)
print(Wz)
# zz = np.zeros(n_param - d1, dtype=float)

yy_vec = []

for ll in range(Mm):
	yy = np.dot(Wy.T, mat_mp[ll, :])
	yy_vec.append(yy)
	for nn in range(nterms):
		matA[ll, nn] = yy ** nn  # original

	zz = np.dot(Wz.T, mat_mp[ll, :])	

zz_mean = 1 / Mm * zz
print(matA)

# compute h(y)
h_vec = []
yy_vec = np.array(yy_vec)
for ll in range(Mm):
	y_in = np.dot(Wy, yy_vec[ll]) + np.dot(Wz, zz_mean)
	print(y_in)
	vv5 = V5(y_in[0], y_in[1], y_in[2], y_in[3], y_in[4], y_in[5], y_in[6], y_in[7], indx_Vbus)
	h_vec.append(vv5)

h_vec = np.array(h_vec)
print(h_vec)

# finally, compute c vector
c_vec = np.dot(np.dot(np.linalg.inv(np.dot(matA.T, matA)), matA.T), h_vec)
print(c_vec)


# everything done, now compute one solution and check
ppp = [48, 39, 38, 53, 33, 29, 38, 27]
vv5s = V5(ppp[0], ppp[1], ppp[2], ppp[3], ppp[4], ppp[5], ppp[6], ppp[7], indx_Vbus)  # traditional way


# P2 = np.arange(0, 40, ssz)
# Q2 = np.arange(0, 30, ssz)
# P3 = np.arange(0, 30, ssz)
# Q3 = np.arange(0, 45, ssz)
# P4 = np.arange(0, 25, ssz)
# Q4 = np.arange(0, 20, ssz)
# P5 = np.arange(0, 30, ssz)
# Q5 = np.arange(0, 20, ssz)




y_val = np.dot(Wy.T, np.array(ppp))
# y_inn = np.dot(Wy, vec_y) + np.dot(Wz, zz_mean)
vv5p = c_vec[0] * y_val ** 0 + c_vec[1] * y_val ** 1 + c_vec[2] * y_val ** 2 + c_vec[3] * y_val ** 3  # new way
# vv5p = c_vec[0] * y_val ** 0 + c_vec[1] * y_val ** 1 + c_vec[2] * y_val ** 2 + c_vec[3] * y_val ** 3 + c_vec[4] * y_val ** 4 # new way
# vv5p = c_vec[0] * y_val ** 0 + c_vec[1] * y_val ** 1 + c_vec[2] * y_val ** 2  # new way

# print results, very close!!
print(vv5p)
print(vv5s)
print(abs(vv5p-vv5s))

print(';;;;;;;;;;;;;;;;;;;;;;;;;')

# calculated from basic linealization:
for ll in range(Mm):
	App = - mat_mp[ll,:] + ppp
	Vff = h_vec[ll] + np.dot(Ag_store[ll], App)[0]
	# print(Vff)
	print(abs(Vff - vv5s))

