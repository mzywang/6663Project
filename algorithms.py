import numpy as np
from profilehooks import timecall
import matplotlib.pyplot as plt
import math
import scipy as sp
import scipy.optimize
from copy import deepcopy

def obj_progress(f_x, grad):
    return (f_x[-1] / f_x[0])

def gradient_progress(f_x, grad):
    return np.linalg.norm(grad[-1]) / (np.linalg.norm(grad[0]) + 1)

def get_step_size(f, fprime, x, d):
    ret_vals = sp.optimize.line_search(f = f, myfprime = fprime, xk = x, pk = d)
    alpha = ret_vals[0]
    if alpha is None:
        alpha = 10**-9
    return alpha

def sd(d, grad, grad_next, H, H_next, p, q):
    return - grad_next

def fr_cg(d, grad, grad_next, H, H_next, p, q):
    beta = (np.inner(grad_next, grad_next)) / (np.inner(grad[-1], grad[-1]))
    return - grad_next + beta * d[-1]

def pr_cg(d, grad, grad_next, H, H_next, p, q):
    beta = (np.inner(grad_next, grad_next - grad[-1])) / (np.linalg.norm(grad[-1]) ** 2)
    return - grad_next + beta * d[-1]

def bfgs(d, grad, grad_next, H, H_next, p, q):
    return - np.matmul(H_next, grad_next)

def dfp(d, grad, grad_next, H, H_next, p, q):
    return - np.matmul(H_next, grad_next)

def l_bfgs(d, grad, grad_next, H, H_next, p, q):
    u = np.transpose(np.array([grad_next]))
    a = {}
    for i in reversed(list(range(len(p) - min(5, len(p)), len(p)))):
        a[i] = (np.matmul(np.transpose(p[i]), u) / np.matmul(np.transpose(p[i]), q[i]))[0][0]
        u = u - a[i] * q[i]
    r = np.matmul(H[0], u)
    for i in list(range(len(p) - min(5, len(p)), len(p))):
        r = r + (a[i] - np.matmul(np.transpose(q[i]), r) / np.matmul(np.transpose(p[i]), q[i])) * p[i]
    return -np.transpose(r)[0]

def print_progress(x, d, f_x, grad, term):
    print("Iteration {}".format(len(x) - 1))
    print("x = {}".format(x[-1]))
    print("gradient = {}".format(grad[-1]))
    print("d = {}".format(d[-1]))
    print("f(x) = {}".format(f_x[-1]))
    print("Progress = {}".format(term(f_x, grad)))
    print()

@timecall
def line_search(meth, x_0, f, fprime, term, epsilon, DEBUG):
    f_x_0 = f(x_0)
    grad_0 = fprime(x_0)

    x = [x_0]
    f_x = [f_x_0]
    grad = [grad_0]
    d = [-grad_0]

    I = np.identity(len(x_0))
    H = [I]
    p = []
    q = []

    if DEBUG:
        print_progress(x, d, f_x, grad, term)

    while (term(f_x, grad) > epsilon):
        alpha = get_step_size(f = f, fprime = fprime, x = x[-1], d = d[-1])
        x_next = x[-1] + alpha * d[-1]
        f_x_next = f(x_next)
        grad_next = fprime(x_next)

        H_next = deepcopy(H[-1])
        p_k = np.transpose(np.array([x_next - x[-1]]))
        q_k = np.transpose(np.array([grad_next - grad[-1]]))
        p.append(p_k)
        q.append(q_k)

        if meth == bfgs:
            H_next = np.matmul((I - np.matmul(p_k, np.transpose(q_k)) / np.matmul(np.transpose(q_k), p_k)), np.matmul(H_next, (I - np.matmul(q_k, np.transpose(p_k)) / np.matmul(np.transpose(q_k), p_k)))) + (np.matmul(p_k, np.transpose(p_k)) / np.matmul(np.transpose(q_k), p_k))
        if meth == dfp:
            H_next = H_next + np.matmul(p_k, np.transpose(p_k)) / np.matmul(np.transpose(p_k), q_k) - np.matmul(np.matmul(H_next, q_k), np.matmul(np.transpose(q_k), H_next)) / np.matmul(np.transpose(q_k), np.matmul(H_next, q_k))

        d_next = meth(d, grad, grad_next, H, H_next, p, q)

        x.append(x_next)
        f_x.append(f_x_next)
        grad.append(grad_next)
        d.append(d_next)

        H.append(H_next)

        if DEBUG and (len(x) - 1) % 1 == 0 :
            print_progress(x, d, f_x, grad, term)

    return (x, grad, f_x)
