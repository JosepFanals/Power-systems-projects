# python file for testing small scripts

import numpy as np
import math
import itertools

k = 3
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

# working! now merge with the parametric code
