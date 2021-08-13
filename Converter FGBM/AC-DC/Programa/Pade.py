import numpy as np


def pade(order, coeff_mat, s):

    nbus = coeff_mat.ndim
    voltages = np.zeros(nbus, dtype=complex)
    if order % 2 != 0:
        nn = int(order / 2)
        L = nn
        M = nn
        for d in range(nbus):
            if nbus > 1:
                rhs = coeff_mat[L + 1:L + M + 1, d]
            else:
                rhs = coeff_mat[L + 1:L + M + 1]
            C = np.zeros((M, M), dtype=complex)
            for i in range(M):
                k = i + 1
                if nbus > 1:
                    C[i, :] = coeff_mat[L - M + k:L + k, d]
                else:
                    C[i, :] = coeff_mat[L - M + k:L + k]
            b = np.zeros(rhs.shape[0] + 1, dtype=complex)
            x = np.linalg.solve(C, -rhs)  # bn to b1
            b[0] = 1
            b[1:] = x[::-1]
            a = np.zeros(L + 1, dtype=complex)
            if nbus > 1:
                a[0] = coeff_mat[0, d]
            else:
                a[0] = coeff_mat[0]
            for i in range(L):
                val = complex(0)
                k = i + 1
                for j in range(k + 1):
                    if nbus > 1:
                        val += coeff_mat[k - j, d] * b[j]
                    else:
                        val += coeff_mat[k - j] * b[j]
                a[i + 1] = val
            p = complex(0)
            q = complex(0)

            for i in range(len(a)):
                p += a[i] * s ** i
            for i in range(len(b)):
                q += b[i] * s ** i
            voltages[d] = p / q
    else:
        nn = int(order / 2)
        L = nn
        M = nn - 1
        for d in range(nbus):
            if nbus > 1:
                rhs = coeff_mat[M + 2: 2 * M + 2, d]
            else:
                rhs = coeff_mat[M + 2: 2 * M + 2]
            C = np.zeros((M, M), dtype=complex)
            for i in range(M):
                k = i + 1
                if nbus > 1:
                    C[i, :] = coeff_mat[L - M + k:L + k, d]
                else:
                    C[i, :] = coeff_mat[L - M + k:L + k]
            b = np.zeros(rhs.shape[0] + 1, dtype=complex)
            x = np.linalg.solve(C, -rhs)  # de bn a b1, en aquest ordre
            b[0] = 1
            b[1:] = x[::-1]
            a = np.zeros(L + 1, dtype=complex)
            if nbus > 1:
                a[0] = coeff_mat[0, d]
            else:
                a[0] = coeff_mat[0]
            for i in range(1, L):
                val = complex(0)
                for j in range(i + 1):
                    if nbus > 1:
                        val += coeff_mat[i - j, d] * b[j]
                    else:
                        val += coeff_mat[i - j] * b[j]
                a[i] = val
            val = complex(0)
            for j in range(L):
                if nbus > 1:
                    val += coeff_mat[M - j + 1, d] * b[j]
                else:
                    val += coeff_mat[M - j + 1] * b[j]
            a[L] = val
            p = complex(0)
            q = complex(0)

            for i in range(len(a)):
                p += a[i] * s ** i
            for i in range(len(b)):
                q += b[i] * s ** i
            voltages[d] = p / q
    return voltages