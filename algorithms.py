import numpy as np
from profilehooks import timecall
import matplotlib.pyplot as plt
import math
import scipy as sp
import scipy.optimize

def primal_gap_progess(f_x, grad):
    return (f_x[-1] / f_x[0])

def gradient_progress(f_x, grad):
    return np.linalg.norm(grad[-1]) / (np.linalg.norm(grad[0]) + 1)

def get_step_size(f, fprime, x, d):
    ret_vals = sp.optimize.line_search(f = f, myfprime = fprime, xk = x, pk = d, amax=1)
    alpha = ret_vals[0]
    return alpha

def sd(d, grad, grad_next):
    if d is None or grad_next is None:
        return - grad[0]
    return - grad_next

def fr_cg(d, grad, grad_next):
    beta = (np.inner(grad_next, grad_next)) / (np.inner(grad[-1], grad[-1]))
    return - grad_next + beta * d[-1]

def pr_cg(d, grad, grad_next):
    beta = (np.inner(grad_next, grad_next - grad[-1])) / (np.linalg.norm(grad[-1]) ** 2)
    return - grad_next + beta * d[-1]

def print_progress(x, d, f_x, grad, term):
    k = len(x) - 1
    progress = term(f_x, grad)
    print("Iteration {}".format(k))
    print("x = {}".format(x[-1]))
    print("gradient = {}".format(grad[-1]))
    print("d = {}".format(d[-1]))
    print("f(x) = {}".format(f_x[-1]))
    print("Progress = {}".format(progress))
    print()

@timecall
def line_search(meth, x_0, f, fprime, term, epsilon, DEBUG):
    errors = []
    alphas = []

    f_x_0 = f(x_0)
    grad_0 = fprime(x_0)

    x = [x_0]
    f_x = [f_x_0]
    grad = [grad_0]
    d = [-grad_0]
    print_progress(x, d, f_x, grad, term)

    while (term(f_x, grad) > epsilon):
        alpha = get_step_size(f = f, fprime = fprime, x = x[-1], d = d[-1])
        x_next = x[-1] + alpha * d[-1]
        f_x_next = f(x_next)
        grad_next = fprime(x_next)
        d_next = meth(d, grad, grad_next)
        x.append(x_next)
        f_x.append(f_x_next)
        grad.append(grad_next)
        d.append(d_next)
        errors.append(term(f_x, grad))
        print_progress(x, d, f_x, grad, term)

    return (x, f_x)
