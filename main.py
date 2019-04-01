from functions import *
from algorithms import *
import numpy as np
import time
import matplotlib.pyplot

import scipy as sp
import scipy.optimize

import warnings
warnings.filterwarnings("ignore")


function_names = ["Extended Rosenbrook", "Extended Powell"]
starts = [rosenbrook_start, powell_start]
objs = [rosenbrook_obj, powell_obj]
gradients = [rosenbrook_grad, powell_grad]
method_names = ["Steepest Descent", "Fletcher-Reeves", "Polak-Ribiere", "BFGS", "DFP", "L-BFGS"]
methods = [sd, fr_cg, pr_cg, bfgs, dfp, l_bfgs]
term_names = ["Objective Progress", "Gradient Progress"]
terms = [obj_progress, gradient_progress]

for i in range(len(function_names)):
    function_name = function_names[i]
    start = starts[i]
    f = objs[i]
    fprime = gradients[i]
    for j in range(len(method_names)):
        method_name = method_names[j]
        meth = methods[j]
        for k in range(len(term_names)):
            term_name = term_names[k]
            term = terms[k]
            print("Running the [{}] method on the [{}] function with terminating condition [{}].".format(method_name, function_name, term_name))
            x_0 = start(4)
            x, grad, f_x = line_search(
                meth=meth,
                x_0=x_0,
                f=f,
                fprime=fprime,
                term=term,
                epsilon=10**-6,
                DEBUG=False
            )
            print("Found Solution x = {} after {} iterations with f(x) = {}.".format(x[-1], len(x) - 1, f_x[-1]))
            print("Objective Progress: {}".format(obj_progress(f_x, grad)))
            print("Gradient Progress: {}".format(gradient_progress(f_x, grad)))
            print('---------------------------------------------------------------------------------------------------------------------------')
