# comprovo si les equacions del trigger schmitt donen

import numpy as np

Vcc = 10.0
R1 = 10e3
R2 = 5e3
R3 = 1.25e3
R4 = 1e6
Rc1 = 1.5e3
Rc2 = 1e3
Re = 100
af = 0.99
ar = 0.5
me = -1e-16 / af
mc = -1e-16 / ar
n = -38.78

# solució 1
x1 = 1.1763
x2 = 5.4897
x3 = 1.2689
x4 = 2.0055
x5 = 1.9734
x6 = 10.0
x7 = -0.0133

ie1 = me * (np.exp(n * (x1 - x5)) - 1) - ar * mc * (np.exp(n * (x2 - x5)) - 1)
ic1 = -af * me * (np.exp(n * (x1 - x5)) - 1) + mc * (np.exp(n * (x2 - x5)) - 1)

ie2 = me * (np.exp(n * (x1 - x4)) - 1) - ar * mc * (np.exp(n * (x3 - x4)) - 1)
ic2 = -af * me * (np.exp(n * (x1 - x4)) - 1) + mc * (np.exp(n * (x3 - x4)) - 1)

f1 = x1 / Re + ie1 + ie2
f2 = (x2 - x4) / R1 + (x2 - x6) / Rc1 + ic1
f3 = (x3 - x6) / Rc2 + ic2
f4 = (x4 - x2) / R1 + x4 / R4 - ie2 - ic2
f5 = (x5 - x6) / R2 + x5 / R3 - ic1 - ie1
f6 = (x6 - x2) / Rc1 + (x6 - x3) / Rc2 + (x6 - x5) / R2 + x7
f7 = x6 - Vcc

print(abs(f1))
print(abs(f2))
print(abs(f3))
print(abs(f4))
print(abs(f5))
print(abs(f6))
print(abs(f7))