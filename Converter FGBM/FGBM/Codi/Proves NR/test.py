import numpy as np

x = 285041.4326994716
enter = int(x / (2 * np.pi))
xmod = x - enter * 2 * np.pi
print(np.sin(xmod))
print(np.sin(x))

print(np.sin(285041.4326994716))

