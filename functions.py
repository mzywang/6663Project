import numpy as np
from profilehooks import profile
import matplotlib.pyplot
import scipy as sp
import scipy.optimize
import math

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
        f_x = 0.0
        for i in range(n // 2):
            i1 = 2 * i
            i2 = 2 * i + 1
            f_x += (10 * (x[i2] - (x[i1] ** 2))) ** 2
            f_x += (1 - x[i1]) ** 2
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

def powell_obj(x):
    n = len(x)
    if n % 4 != 0:
        print("n must be a multiple of 4")
        return []
    else:
        f_x = 0.0
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
            g[i1] = 40 * (x[i1] - x[i4])**3 + 2 *(x[i1] + 10 * x[i2])
            g[i2] = 4 * (x[i2] - 2*x[i3])**3 + 20 *(x[i1] + 10*x[i2])
            g[i3] = 10*(x[i3] - x[i4]) - 8*(x[i2] - 2 * x[i3])**3
            g[i4] = -40*(x[i1] - x[i4])**3 - 10*(x[i3] - x[i4])
        return g
