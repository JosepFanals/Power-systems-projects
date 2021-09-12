# python file for testing small scripts

# Packages
import numpy as np
import time
import itertools

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



k = [3] * 3
l_exp = 3
nV = 3


start_time = time.time()

permss = permutate(k, l_exp, nV)
for ll in range(len(permss)):
    print(permss[ll])

stop_time = time.time()

print(stop_time - start_time)



# new method
def f(r, n, t, acc=[]):
    if t == 0:
        if n >= 0:
            yield acc
        return
    for x in r:
        if x > n:  # <---- do not recurse if sum is larger than `n`
            break
        for lst in f(r, n-x, t-1, acc + [x]):
            yield lst



# full function
def permut(k, l_exp, nV):
    permm_all = []
    for nn in range(nV):
        xs_all = []
        for xs in f(range(l_exp+1), l_exp, k[nn]):
            xs_all.append(xs)
        permm_all.append(xs_all)

    return permm_all


start_time = time.time()

abc = permut(k, l_exp, nV)
# print(abc[0])

stop_time = time.time()

print(stop_time - start_time)


