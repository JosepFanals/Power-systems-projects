# python file for testing small scripts

import numpy as np
import math
import itertools

k = 2
l = 3
Nt = int(math.factorial(l + k) / (math.factorial(l) * math.factorial(k)))
print(Nt)

lst = [ll for ll in range(l + 1)] * k
perms = set(itertools.permutations(lst, k))
print(len(perms))
perms_good = []
for per in perms:
    if sum(per) <= l:
        perms_good.append(per)

print(perms_good)
print(len(perms_good))
print(perms_good[1][0])




# working! now merge with the parametric code

yyy = [2, 2, 3]
exp = [0, 1, 2]
# res = [yyy[kk] ** exp[kk] for kk in range(len(yyy))]
res = 1
for kk in range(len(yyy)):  # range(k)
    res = res * yyy[kk] ** exp[kk]
print(res)

Wy = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
paramm = np.array([1, 2, 3, 4, 5, 6])
yy = np.array(np.dot(Wy, paramm))
print(yy)
