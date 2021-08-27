# python file for testing small scripts

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

low = [1, 2, 3, 4, 5, 6, 7, 8]
high = [10, 20, 30, 40, 50, 60, 70, 80]
n_param = 8

xlimits = np.zeros((n_param, 2), dtype=float)  # 2 columns [lower, upper]
for ll in range(len(low)):
    xlimits[ll, 0] = low[ll]
    xlimits[ll, 1] = high[ll]

print(xlimits)

sampling_lh = LHS(xlimits=xlimits)

num = 50
x = sampling_lh(num)


print(x)
print(x[0,1])
# plt.plot(x[:, 0], x[:, 1], "o")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
