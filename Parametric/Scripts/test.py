# python file for testing small scripts

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

xlimits = np.array([[0.0, 50]] * 8)
sampling = LHS(xlimits=xlimits)

print(xlimits)
num = 50
x = sampling(num)

print(x.shape)

print(x)
# plt.plot(x[:, 0], x[:, 1], "o")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
