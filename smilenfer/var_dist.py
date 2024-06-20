import numpy as np
import scipy.special as sp
import scipy.optimize as opt

# Compute density of variance contributions for 
# 1-D stabilizing selection
def f_stab_1D(zz, eta):
    return np.exp(-2*eta*(zz+1)) / ( sp.exp1(2*eta) * (zz + 1) )

# Compute density of variance contributions for 
# high-D stabilizing selectio
def f_stab_hD(zz, eta):
    return np.exp(-2*np.sqrt(eta*(zz+1))) / ( 2*sp.exp1(2*np.sqrt(eta)) * (zz + 1) )

# Compute CDF of variance contributions for
# 1-D stabilizing selection
def F_stab_1D(zz, eta):
    return 1 - sp.exp1(2*eta*(zz+1)) / sp.exp1(2*eta)

# Compute CDF of variance contributions for
# high-D stabilizing selection
def F_stab_hD(zz, eta):
    return 1 - sp.exp1(2*np.sqrt(eta*(zz+1))) / sp.exp1(2*np.sqrt(eta))

# 1-D stabilizing selection
# Use the CDF to solve for the inverse CDF given quantile Q
# by solving for the root of F(z) - Q = 0
def inv_F_stab_1D(Q, eta):
    # Check if Q is a numpy array, if yes find the root for each element
    if isinstance(Q, np.ndarray):
        return np.array([opt.root_scalar(lambda z: F_stab_1D(z, eta) - q, bracket=[0, 1e10]).root for q in Q])
    else:
        return opt.root_scalar(lambda z: F_stab_1D(z, eta) - Q, bracket=[0, 1e10]).root

# high-D stabilizing selection
# Use the CDF to solve for the inverse CDF given quantile Q
# by solving for the root of F(z) - Q = 0
def inv_F_stab_hD(Q, eta):
    # Check if Q is a numpy array, if yes find the root for each element
    if isinstance(Q, np.ndarray):
        return np.array([opt.root_scalar(lambda z: F_stab_hD(z, eta) - q, bracket=[0, 1e10]).root for q in Q])
    else:
        return opt.root_scalar(lambda z: F_stab_hD(z, eta) - Q, bracket=[0, 1e10]).root

# Compute pointwise log-likelihood of variance contributions for 
# 1-D stabilizing selection
def ll_stab_1D(zz, eta):
    return -2*eta * (zz + 1) - np.log(sp.exp1(2*eta)) - np.log(zz + 1)

# Compute pointwise log-likelihood of variance contributions for 
# high-D stabilizing selection
def ll_stab_hD(zz, eta):
    return -2*np.sqrt(eta * (zz + 1)) - np.log(2*sp.exp1(2*np.sqrt(eta))) - np.log(zz + 1)

# Fit the variance distribution for 1-D stabilizing selection
def fit_stab_1D(zz):
    neg_ll_data = lambda eta: -np.sum(ll_stab_1D(zz, eta))
    res = opt.minimize_scalar(neg_ll_data, bounds=(1e-320, 500), method="bounded")
    return res.x

# Fit the variance distribution for 1-D stabilizing selection
def fit_stab_hD(zz):
    neg_ll_data = lambda eta: -np.sum(ll_stab_hD(zz, eta))
    res = opt.minimize_scalar(neg_ll_data, bounds=(1e-320, 700), method="bounded")
    return res.x
