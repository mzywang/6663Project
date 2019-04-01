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
    ret_vals = sp.optimize.line_search(f = f, myfprime = fprime, xk = x, pk = d)
    alpha = ret_vals[0]
    return alpha

def sd(d, grad, grad_next, H, H_next):
    if d is None or grad_next is None:
        return - grad[0]
    return -grad_next

def fr_cg(d, grad, grad_next, H, H_next):
    if d is None or grad_next is None:
        return - grad[0]
    beta = (np.linalg.norm(grad_next) ** 2) / (np.linalg.norm(grad[-1]) ** 2)
    return -grad_next + beta * d[-1]

def pr_cg(d, grad, grad_next, H, H_next):
    if d is None or grad_next is None:
        return - grad[0]
    beta = (np.inner(grad_next, grad_next - grad[-1])) / (np.linalg.norm(grad[-1]) ** 2)
    return -grad_next + beta * d[-1]

def bfgs(d, grad, grad_next, H, H_next):
    if H_next is None:
        return - np.matmul(H[0], grad[0])
    else:
        return - np.matmul(H_next, grad_next)

def dfp(d, grad, grad_next, H, H_next):
    if H_next is None:
        return - np.matmul(H[0], grad[0])
    else:
        return - np.matmul(H_next, grad_next)

def print_progress(x, f_x, grad, term):
    k = len(x) - 1
    progress = term(f_x, grad)
    print("Iteration {}".format(k))
    print("x = {}".format(x[-1]))
    print("f(x) = {}".format(f_x[-1]))
    print("Progress = {}".format(progress))

@timecall
def line_search(meth, x_0, f, fprime, term, epsilon, DEBUG):
    errors = []
    alphas = []

    f_x_0 = f(x_0)
    grad_0 = fprime(x_0)

    x = [x_0]
    f_x = [f_x_0]
    grad = [grad_0]
    H = [np.identity(len(x_0))] if meth == bfgs or meth == dfp else [None]
    d = [meth(d = None, grad = grad, grad_next = None, H = H, H_next = None)]
    p_arr = []
    q_arr = []

    while (term(f_x, grad) > epsilon):
        alpha = get_step_size(f = f, fprime = fprime, x = x[-1], d = d[-1])
        x_next = x[-1] + alpha * d[-1]
        f_x_next = f(x_next)
        grad_next = fprime(x_next)
        H_next = H[-1]
        if meth == bfgs or meth == dfp:
            p = np.transpose(np.array([x_next - x[-1]]))
            p_t = np.transpose(p)
            q = np.transpose(np.array([grad_next - grad[-1]]))
            q_t = np.transpose(q)
            if meth == bfgs:
                A = np.matmul(q_t, np.matmul(H[-1], q))
                B = np.matmul(q_t, p)
                C = np.matmul(p, p_t)
                D = np.matmul(p_t, q)
                E = (np.matmul(p, np.matmul(p_t, H[-1])))
                F = np.matmul(H[-1], np.matmul(q, p_t))
                G = np.matmul(q_t, p)
                H_next = H_next + (1 + A / B) * C / D - (E + F) / G
            if meth == dfp:
                H_next = H_next + np.matmul(p, p_t) / np.matmul(p_t, q) - np.matmul(np.matmul(H[-1], q), np.matmul(q_t, H[-1])) / np.matmul(q_t, np.matmul(H[-1], q))
            p_arr.append(p)
            q_arr.append(q)

        d_next = meth(d, grad, grad_next, H, H_next)
        x.append(x_next)
        f_x.append(f_x_next)
        grad.append(grad_next)
        H.append(H_next)
        d.append(d_next)
        errors.append(term(f_x, grad))

        k = k + 1

        print_progress(x, f_x, grad, term)
        print("alpha: {}".format(alpha))

    return (x, f_x)
