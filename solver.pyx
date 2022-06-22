cimport cython
from ctypes import c_int, c_double
import numpy as np
cimport numpy as cnp
cimport libc.stdlib as lib
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport exp, log, sqrt, cos, M_PI, isnan
cnp.import_array()
import matplotlib.pyplot as plt

# load interpolation routines from spline_c.c
cdef extern from "spline_c.c":
    void spline(double *x, double *y, double *b, double *c, double *d, int n)
    double ispline(double u, double *x, double *y, double *b, double *c, double *d, int n)

# variables
cdef int n = 399                   # number of r points evaluated at each evolution step in Y
cdef double r1 = 1.e-6             # lower limit of r
cdef double r2 = 1.e2              # upper limit of r

cdef double xr1 = log(r1)          # convert lower r limit to logspace
cdef double xr2 = log(r2)          # convert upper r limit to logspace

cdef double hr = (xr2 - xr1) / n

# parameters
cdef int nc = 3                     # number of colors
cdef int nf = 3                     # number of active flavors
cdef double lamb = 0.241

cdef double beta = (11 * nc - 2. * nf)/(12 * M_PI)
cdef double afr = 0.7               # frozen coupling constant (default)

cdef double c2, gamma, qsq2            # fitting parameters
cdef double xr0, r0, n0, rfr2

# allocating memory space for arrays
# xlr_, n_ describe the most current BK solution
# coeff1, coeff2, coeff3 are arrays for interpolation coefficients
cdef double *xlr_ = <double*>malloc(n * sizeof(double))
cdef double *n_ = <double*>malloc(n * sizeof(double))
cdef double *coeff1 = <double*>malloc(n * sizeof(double))
cdef double *coeff2 = <double*>malloc(n * sizeof(double))
cdef double *coeff3 = <double*>malloc(n * sizeof(double))

cdef double *k_ = <double*>malloc(n * sizeof(double))
cdef double *kcoeff1 = <double*>malloc(n * sizeof(double))
cdef double *kcoeff2 = <double*>malloc(n * sizeof(double))
cdef double *kcoeff3 = <double*>malloc(n * sizeof(double))
cdef double *xx_     = <double*>malloc(n * sizeof(double))

cpdef void set_params(double c_, double gamma_, double qsq_):
    global c2, gamma, qsq2, rfr2

    c2 = c_
    gamma = gamma_
    qsq2 = qsq_

    rfr2 = 4 * c2/(lamb * lamb * exp(1/(beta * afr)))
    print('c2 = ' + str(c2) + ', g = ' + str(gamma) + ', qsq2 = ' + str(qsq2))

# called to set coefficients at the beginning of each step of the evolution
cpdef void set_vars(double x, double n0_, list xlr_arr, list n_arr):
    global xr0, r0, n0, xlr_, n_, coeff1, coeff2, coeff3

    xr0 = x
    r0 = exp(x)
    n0 = n0_

    # clearing coefficient array
    memset(coeff1, 0, n * sizeof(double))
    memset(coeff2, 0, n * sizeof(double))
    memset(coeff3, 0, n * sizeof(double))

    # make arrays xlr_, n_ compatible with C
    convert_to_c(xlr_arr, xlr_)
    convert_to_c(n_arr, n_)

    # fill coefficient array
    spline(xlr_, n_, coeff1, coeff2, coeff3, n)

cpdef void set_k(list xlr_arr, list k_arr):
    global k_, kcoeff1, kcoeff2, kcoeff3, xx_

    memset(kcoeff1, 0, n * sizeof(double))
    memset(kcoeff2, 0, n * sizeof(double))
    memset(kcoeff3, 0, n * sizeof(double))

    convert_to_c(xlr_arr, xx_)
    convert_to_c(k_arr, k_)

    spline(xx_, k_, kcoeff1, kcoeff2, kcoeff3, n)

cdef void convert_to_c(list l1, double *arr):
    cdef int i
    for i in range(len(l1)):
        arr[i] = l1[i]


cdef convert_to_python(double *ptr, int n):
    cdef int i
    lst = []
    for i in range(n):
        lst.append(ptr[i])
    return lst


cdef double nfunc(double qlr):
    cdef double x = 0.0
    if qlr < xr1:
        x = n_[0] * exp(2 * qlr)/(r1 * r1)
    elif qlr >= xr2:
        x = 1.
    else:
        x = ispline(qlr, xlr_, n_, coeff1, coeff2, coeff3, n)

    if x < 0.: return 0.0
    if x > 1.: return 1.0

    return x


cdef double alphaS(double rsq):
    cdef double xlog
    if rsq > rfr2:
        return afr
    else:
        xlog = log((4 * c2)/(rsq * lamb * lamb))
        return 1/(beta * xlog)
       
cdef double find_r1(double r, double z, double thet):
    cdef double r12 = (0.25 * r * r) + (z * z) - (r * z * cos(thet))
    return sqrt(r12)


cdef double find_r2(double r, double z, double thet):
    cdef double r22 = (0.25 * r * r) + (z * z) + (r * z * cos(thet))
    return sqrt(r22)


# kernel
cdef double k(double r, double r1_, double r2_):
    cdef double rr, r12, r22
    cdef double t1, t2, t3
    cdef double prefac

    if (r1_ < 1e-20) or (r2_ < 1e-20):
        return 0
    else:
        rr = r * r
        r12 = r1_ * r1_
        r22 = r2_ * r2_

        t1 = rr/(r12 * r22)
        t2 = (1/r12) * (alphaS(r12)/alphaS(r22) - 1)
        t3 = (1/r22) * (alphaS(r22)/alphaS(r12) - 1)

        prefac = (nc * alphaS(rr))/(2 * M_PI * M_PI)
        return prefac * (t1 + t2 + t3)

# combined integrand
cdef double f(int num, double *xx):
    cdef double z, r1_, r2_
    cdef double xlr1, xlr2, kr0, kr1, kr2
    cdef double nr1, nr2

    z = exp(xx[1])
    r1_ = find_r1(r0, z, xx[0])
    r2_ = find_r2(r0, z, xx[0])

    xlr1 = log(r1_)
    xlr2 = log(r2_)

    kr0 = ispline(xr0 , xlr_, k_, kcoeff1, kcoeff2, kcoeff3, n)
    kr1 = ispline(xlr1, xlr_, k_, kcoeff1, kcoeff2, kcoeff3, n)
    kr2 = ispline(xlr2, xlr_, k_, kcoeff1, kcoeff2, kcoeff3, n)

    nr0 = n0 + kr0
    nr1 = nfunc(xlr1) + kr1
    nr2 = nfunc(xlr2) + kr2

    return 4 * z * z * k(r0, r1_, r2_) * (nr1 + nr2 - nr0 - nr1 * nr2)
