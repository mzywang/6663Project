from functions import *
from algorithms import *
import numpy as np
import time
import matplotlib.pyplot

import scipy as sp
import scipy.optimize

x_0 = rosenbrook_start(2)

x, f_x = line_search(
    meth = pr_cg,
    x_0 = x_0,
    f = rosenbrook_obj,
    fprime = rosenbrook_grad,
    term = gradient_progress,
    epsilon = 10 ** -6,
    DEBUG = True
)

print("Found Solution x = {} after {} iterations with f(x) = {}.".format(x[-1], len(x) - 1, f_x[-1]))
