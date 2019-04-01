import numpy as np
from profilehooks import profile
import matplotlib.pyplot

def rosenbrook_start(n):
    if n % 2 != 0:
        print("n must be a multiple of 2")
        return []
    else:
        x_0 = np.array([0.0 for i in range(n)])
        for i in range(n // 2):
            i1 = 2 * i
            i2 = 2 * i + 1
            x_0[i1] = -1.2
            x_0[i2] = 1
        return x_0

def powell_start(n):
    if n % 4 != 0:
        print("n must be a multiple of 4")
        return []
    else:
        x_0 = np.array([0.0 for i in range(n)])
        for i in range(n // 4):

            i1 = 4 * i
            i2 = 4 * i + 1
            i3 = 4 * i + 2
            i4 = 4 * i + 3
            x_0[i1] = 3
            x_0[i2] = -1
            x_0[i3] = 0
            x_0[i4] = 1
        return x_0

def rosenbrook_obj(x):
    n = len(x)
    if n % 2 != 0:
        print("n must be a multiple of 2")
        return []
    else:
        f_x = 0

        for i in range(n // 2):
            i1 = 2 * i
            i2 = 2 * i + 1
            f_x += (
                100 * (x[i1] ** 4)
                + (x[i1] ** 2)
                - 200 * x[i2] * (x[i1] ** 2)
                - 2 * x[i1] +
                + 100 * (x[i2] ** 2)
                + 1
                )
        return f_x

def rosenbrook_grad(x):
    n = len(x)
    if n % 2 != 0:
        print("n must be a multiple of 2")
        return []
    else:
        g = np.array([0.0 for i in range(n)])

        for i in range(n // 2):
            i1 = 2 * i
            i2 = 2 * i + 1

            g[i1] = (
                400 * (x[i1] ** 3)
                + 2 * x[i1]
                - 400 * x[i2] * x[i1]
                - 2
            )
            g[i2] = (
                - 200 * (x[i1] ** 2)
                + 200 * x[i2]
            )
        return g

def rosenbrook_hess(x):
    n = len(x)
    if n % 2 != 0:
        print("n must be a multiple of 2")
        return []
    else:
        H = np.array([[0.0 for i in range(n)] for j in range(n)])

        for i in range(n // 2):
            i1 = 2 * i
            i2 = 2 * i + 1
            H[i1][i1] = (
                800 * (x[i1] ** 2)
                - 400 * x[i2]
                + 400 * (x[i1] ** 2)
                + 2
            )
            H[i1][i2] = - 400 * x[i1]
            H[i2][i1] = - 400 * x[i1]
            H[i2][i2] = 200
        return H

def powell_obj(x):
    n = len(x)
    if n % 4 != 0:
        print("n must be a multiple of 4")
        return []
    else:
        f_x = 0
        for i in range(n // 4):
            i1 = 4 * i
            i2 = 4 * i + 1
            i3 = 4 * i + 2
            i4 = 4 * i + 3
            f_x += (
                10 * (x[i1]**4)
                - 40 * x[i4] * (x[i1]**3)
                + 60 * (x[i4]**2) * (x[i1]**2)
                + (x[i1]**2)
                - 40 * (x[i4]**3) * x[i1]
                + 20 * x[i2] * x[i1]
                + (x[i2]**4)
                + 16 * (x[i3]**4)
                + 10 * (x[i4]**4)
                - 32 * x[i2] * (x[i3]**3)
                + 100 * (x[i2]**2)
                + 24 * (x[i2]**2) * (x[i3]**2)
                + 5 * (x[i3]**2)
                + 5 * (x[i4]**2)
                - 8 * (x[i2]**3) * x[i3]
                - 10 * x[i3] * x[i4]
            )

        return f_x


def powell_grad(x):
    n = len(x)
    if n % 4 != 0:
        print("n must be a multiple of 4")
        return []
    else:
        g = np.array([0.0 for i in range(n)])
        for i in range(n // 4):
            i1 = 4 * i
            i2 = 4 * i + 1
            i3 = 4 * i + 2
            i4 = 4 * i + 3
            g[i1] = (
                40 * (x[i1]**3)
                - 120 * x[i4] * (x[i1]**2)
                + 120 * (x[i4]**2) * x[i1]
                + 2 * x[i1]
                - 40 * (x[i4]**3)
                + 20 * x[i2]
            )
            g[i2] = (
                4 * (x[i2]**3)
                - 24 * x[i3] * (x[i2]**2)
                + 48 * (x[i3]**2) * x[i2]
                + 200 * x[i2]
                - 32 * (x[i3]**3)
                + 20 * x[i1]
            )
            g[i3] = (
                -8 * (x[i2]**3)
                + 48 * x[i3] * (x[i2]**2)
                - 96 * (x[i3]**2) * x[i2]
                + 64 * (x[i3]**3)
                + 10 * x[i3]
                - 10 * x[i4]
            )
            g[i4] = (
                - 40 * (x[i1]**3)
                + 120 * x[i4] * (x[i1]**2)
                - 120 * (x[i4]**2) * x[i1]
                + 40 * (x[i4]**3)
                - 10 * x[i3]
                + 10 * x[i4]
            )
        return g


def powell_hess(x):
    n = len(x)
    if n % 4 != 0:
        print("n must be a multiple of 4")
        return []
    else:
        f_x = 0
        g = np.array([0.0 for i in range(n)])
        H = np.array([[0.0 for i in range(n)] for j in range(n)])
        for i in range(n // 4):
            i1 = 4 * i
            i2 = 4 * i + 1
            i3 = 4 * i + 2
            i4 = 4 * i + 3
            H[i1][i1] = (
                120 * (x[i1]**2)
                - 240 * x[i4] * x[i1]
                + 120 * (x[i4]**2)
                + 2
            )
            H[i2][i2] = (
                12 * (x[i2]**2)
                - 48 * x[i3] * x[i2]
                + 48 * (x[i3]**2)
                + 200
            )
            H[i3][i3] = (
                48 * (x[i2]**2)
                - 192 * x[i3] * x[i2]
                + 192 * (x[i3]**2)
                + 10
            )
            H[i4][i4] = (
                120 * (x[i1]**2)
                - 240 * x[i4] * x[i1]
                + 120 * (x[i4]**2)
                + 10
            )
            H[i3][i2] = (
                - 24 * (x[i2]**2)
                + 96 * x[i3] * x[i2]
                - 96 * (x[i3]**2)
            )
            H[i4][i1] = (
                - 120 * (x[i1]**2)
                + 240 * x[i4] * x[i1]
                - 120 * (x[i4]**2)
            )
            H[i2][i1] = 20
            H[i3][i1] = 0
            H[i4][i2] = 0
            H[i4][i3] = -10
            i_arr = [i1, i2, i3, i4]
            for a in range(4):
                for b in range(4):
                    if b > a:
                        H[i_arr[a]][ i_arr[b]] = H[i_arr[b]][i_arr[a]]
        return H
