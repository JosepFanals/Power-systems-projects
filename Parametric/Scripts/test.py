# python file for testing small scripts

# Packages
import numpy as np
import time

start_time = time.time()


x = 100
res = 1
for kk in range(x):
    res = res * kk ** (kk / 3) * np.sqrt(kk)


print(time.time() - start_time)
