# python file for testing small scripts

# Packages
import numpy as np

Cc = np.zeros((5, 3, 4), dtype=float)
print(Cc[1, 2, 1])

hx = np.zeros((10, 3), dtype=float)
v = np.ones(3, dtype=float)
hx[0, :] = v[:]
print(hx)
