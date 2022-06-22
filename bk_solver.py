import numpy as np
import csv
import pandas as pd
from multiprocessing import Pool
import time
import numba
from numba import jit
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer
from scipy import LowLevelCallable as llc
import scipy.interpolate as interpolate
from scipy.interpolate import UnivariateSpline
from scipy.integrate import nquad
from scipy.integrate import dblquad
import solver as so
import subprocess

import warnings
warnings.filterwarnings('ignore')

# variables
n = 399     # number of r points to be evaluated at each evolution step in Y
r1 = 1.e-6  # limits of r
r2 = 1.e2

xr1 = np.log(r1)
xr2 = np.log(r2)

hr = (xr2 - xr1) / n

hy = 0.2
ymax = 16.
y = np.arange(0.0, ymax, hy)

# Arrays for N and r in N(r), evaluated at some rapidity Y (including next step N(r,Y) in the evolution
xlr_ = [xr1 + i * hr for i in range(n + 1)]
r_ = np.exp(xlr_)
n_ = []

# parameters
nc = 3        # number of colors
nf = 3        # number of active flavors
lamb = 0.241  # lambda QCD (default)

beta = (11 * nc - 2. * nf)/(12 * np.pi)
afr = 0.7     # frozen coupling constant (default)

c2, gamma, qs02, ec = 0. , 0., 0., 0.   # fitting parameters
e  = np.exp(1)

# initial condition
def mv(r):
    xlog = np.log(1/(lamb * r) + ec * e)
    xexp = np.power(qs02 * r * r, gamma) * xlog/4.0
    return 1 - np.exp(-xexp)

def intg(xx):
    index = xlr_.index(xx)
    nr0 = n_[index]

    so.set_vars(xx, nr0, xlr_, n_)
    func = llc.from_cython(so, 'f', signature='double (int, double *)')
    return dblquad(func, xr1, xr2, 1.e-6, 0.5 * np.pi, epsabs=0.0, epsrel=1.e-4)[0]

# return type: array
def evolve(order):

    # Euler's method
    so.set_k(xlr_, [0 for i in range(n)])
    with Pool(processes=5) as pool:
        k1 = np.array(pool.map(intg, xlr_, chunksize=80))

    if order=='RK1':
        return hy * k1

    # RK2
    list_k1 = list(k1 * hy * 0.5)
    so.set_k(xlr_, list_k1)
    with Pool(processes=5) as pool:
        k2 = np.array(pool.map(intg, xlr_, chunksize=80))

    if order=='RK2':
        return hy * k2

    # RK3
    list_k2 = list(k2 * hy * 0.5)
    so.set_k(xlr_, list_k2)
    with Pool(processes=5) as pool:
        k3 = np.array(pool.map(intg, xlr_, chunksize=80))

    # RK4
    list_k3 = list(k3 * hy)
    so.set_k(xlr_, list_k3)
    with Pool(processes=5) as pool:
        k4 = np.array(pool.map(intg, xlr_, chunksize=80))

    if order=='RK4':
        return (1/6) * hy * (k1 + 2 * k2 + 2 * k3 + k4)

# pass fitting variables q_, c_, g_ to set variables in master.py
def master(q_, c2_, g_, ec_, filename='', order='RK4'):
    global n_, qs02, c2, gamma, ec

    # variables
    qs02  = q_
    c2    = c2_
    gamma = g_
    ec    = ec_

    so.set_params(c2, gamma, qs02) 

    l = ['n   ', 'r1  ', 'r2  ', 'y   ', 'hy  ', 'ec  ', 'qs02 ', 'c2  ', 'g ', 'order']
    v = [n, r1, r2, ymax, hy, ec, qs02, c2, gamma, order]
    bk_arr = []
    t1 = time.time()

    # initial condition----------------------------------------------------------
    n_ = [mv(r_[i]) for i in range(len(r_))]
    #----------------------------------------------------------------------------
    # begin evolution
    for i in range(len(y)):
        y0 = y[i]
        # print("y = " + str(y0))

        for j in range(len(r_)):
            # print('r = ' + str(r_[j]) + ', N(r,Y) = ' + str(n_[j]))
            bk_arr.append([y0, r_[j], n_[j]])

        # calculate correction and update N(r,Y) to next step in rapidity

        xk = evolve(order)
        n_ = [n_[j] + xk[j] for j in range(len(n_))]

        # remove nan values from solution
        xx = np.array(xlr_)
        nn = np.array(n_)
        idx_finite = np.isfinite(nn)
        f_finite = interpolate.interp1d(xx[idx_finite], nn[idx_finite])
        nn = f_finite(xx)
        n_ = nn.tolist()

        # solutions should not be greater than one or less than zero
        for j in range(len(n_)):
            if n_[j] < 0.:
                n_[j] = np.round(0.0, 2)
            if n_[j] > 0.9999:
                n_[j] = np.round(1.0, 2)

    t2 = time.time()
    print('bk run time: ' + str((t2 - t1)/3600) + ' hours')

    if filename != '':
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(l)
            writer.writerow(v)
            for j in range(len(bk_arr)):
                writer.writerow(bk_arr[j])

    return pd.DataFrame(bk_arr, columns=['y', 'r', 'N(r,Y)'])

if __name__ == "__main__":
    # qsq2, c^2, g, ec, filename
    p = []

    with open('params.csv', 'r') as foo:
        reader = csv.reader(foo, delimiter='\t')
        header = next(reader)
        p      = next(reader)

    bk = master(float(p[0]), float(p[1]), float(p[2]), float(p[3]), p[4], p[5])
