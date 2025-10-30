import dadi
import math
import numpy as np
import scipy.stats as stats
import scipy.misc as misc
import scipy.special as special

import scipy.special as spy

from copy import deepcopy

def _zero_nan(arr):
    arr[np.where(np.isnan(arr))] = 0

def erf(x):
    """Error function"""
    return 2*stats.norm.cdf(x*np.sqrt(2)) - 1

def pi_dir_db(SS):
    SS = np.asarray(SS, dtype=np.float64)
    result = np.empty_like(SS, dtype=np.float64)

    mask0 = (SS == 0)
    result[mask0] = 0.5

    mask_neg = (SS < 0)
    if np.any(mask_neg):
        result[mask_neg] = np.expm1(2 * SS[mask_neg]) / np.expm1(4 * SS[mask_neg])

    mask_pos = (SS > 0)
    if np.any(mask_pos):
        result[mask_pos] = np.exp(-2 * SS[mask_pos]) * np.expm1(-2 * SS[mask_pos]) / np.expm1(-4 * SS[mask_pos])

    return result

def pi_full_db_hack(SS_dir, SS_stab):
    delta_SS = 2*SS_dir - SS_stab
    decr_ratio = (delta_SS + 2*SS_stab)/delta_SS * (1-np.exp(delta_SS))/(np.exp(-delta_SS-2*SS_stab)-1)
    decr_ratio_0 = 2*SS_stab/(1-np.exp(-2*SS_stab))
    return np.where(SS_dir==0,
                    0.5,
                    np.where(delta_SS==0,
                             1/(1+decr_ratio_0),
                             1/(1+decr_ratio)))

def hypergeo(a, b, c, z, ee=1e-3):
    n_max = 20
    result = np.exp(spy.gammaln(a+1) + spy.gammaln(b+1+ee) + spy.gammaln(c) -
                    spy.gammaln(a) -  spy.gammaln(b+ee) - spy.gammaln(c+1) + np.log(z))
    for nn in range(2, n_max+1):
        result += np.exp(spy.gammaln(a+nn) + spy.gammaln(b+nn+ee) + spy.gammaln(c) -
                         spy.gammaln(a) -  spy.gammaln(b+ee) - spy.gammaln(c+nn) + np.log(z) - spy.gammaln(nn+1))
    return result

def levy_density(ss, cc):
    return np.sqrt(cc/(2*np.pi))*np.exp(-cc/(2*ss))/(ss**(3/2))

def levy_cdf(ss, cc):
    return spy.erfc(np.sqrt(cc/(2*ss)))

def nplei_density(ss, beta, I2, nn):
    result_log = (0.5*np.log(2*np.pi) - 0.5*np.log(beta**2*I2) +
                  spy.gammaln((nn-1)/2) - spy.gammaln(nn/2) -
                  1.5*np.log(ss) + (nn-2)/2*np.log(1-beta**2*I2/(2*ss)))
    return np.where(ss < beta**2*I2/2, np.zeros_like(result_log), np.exp(result_log))

def nplei_cdf(ss, beta, I2, nn):
    log_part = spy.gammaln(nn/2) - spy.gammaln((nn-1)/2) + 0.5*np.log(2*beta**2*I2) - 0.5*np.log(np.pi*ss)
    result = 1 - np.exp(log_part)*hypergeo(0.5, (3-nn)/2, 1.5, beta**2*I2/(2*ss))
    return np.where(ss < beta**2*I2/2, 0*result, result)

def sfs_del_params(xx, theta, ss, Ne):
    """Calculate the intensity of the site frequency spectrum under purifying selection."""
    ss = ss + 1e-8/Ne
    xx = xx*np.array([1.])
    result_neg = (2*theta*(np.exp(4*Ne*ss)-np.exp(xx*4*Ne*ss))/
                  (xx*(1-xx)*(np.exp(4*Ne*ss)-1)))
    result_pos = (2*theta*(1-np.exp((xx-1)*4*Ne*ss))/
                  (xx*(1-xx)*(1-np.exp(-4*Ne*ss))))
    return np.where(np.full_like(result_neg, ss) < 0, result_neg, result_pos)

def sfs_dir_params(xx, theta, S_dir):
    """Calculate the intensity of the site frequency spectrum under directional selection."""
    S_dir = S_dir + 1e-8
    result_neg = (2*theta*(np.exp(2*S_dir)-np.exp(xx*2*S_dir))/
                  (xx*(1-xx)*(np.exp(2*S_dir)-1)))
    result_pos = (2*theta*(1-np.exp((xx-1)*2*S_dir))/
                  (xx*(1-xx)*(1-np.exp(-2*S_dir))))
    return np.where(np.full_like(result_neg, S_dir) < 0, result_neg, result_pos)

def sfs_dir_params_multi_S(xx, theta, S_dir):
    """Calculate the intensity of the site frequency spectrum under directional selection."""
    S_dir = S_dir + 1e-8
    result_neg = (2*theta*(np.exp(2*S_dir)-np.exp(xx*2*S_dir))/
                  (xx*(1-xx)*(np.exp(2*S_dir)-1)))
    result_pos = (2*theta*(1-np.exp((xx-1)*2*S_dir))/
                  (xx*(1-xx)*(1-np.exp(-2*S_dir))))
    return np.where(S_dir < 0, result_neg, result_pos)

def sfs_del_params_1d(x_set, theta, s_set, Ne):
    """Calculate the intensity of the site frequency spectrum under purifying selection."""
    result_neg = (2*theta*(np.exp(4*Ne*s_set)-np.exp(x_set*4*Ne*s_set))/
                  (x_set*(1-x_set)*(np.exp(4*Ne*s_set)-1)))
    result_pos = (2*theta*(1-np.exp((x_set-1)*4*Ne*s_set))/
                  (x_set*(1-x_set)*(1-np.exp(-4*Ne*s_set))))
    return np.where(np.ones_like(x_set)*s_set < 0, result_neg, result_pos)

def sfs_del_params_2d(x_set, theta, s_set, Ne):
    """Calculate the intensity of the site frequency spectrum under purifying selection."""
    s_tmp = s_set.reshape((-1,1))
    result_neg = (2*theta*(np.exp(4*Ne*s_tmp)-np.exp(x_set*4*Ne*s_tmp))/
                  (x_set*(1-x_set)*(np.exp(4*Ne*s_tmp)-1)))
    result_pos = (2*theta*(1-np.exp((x_set-1)*4*Ne*s_tmp))/
                  (x_set*(1-x_set)*(1-np.exp(-4*Ne*s_tmp))))
    return np.where(np.ones_like(x_set)*s_tmp < 0, result_neg, result_pos)

def sfs_ud_params(xx, theta, ss, Ne):
    """Calculate the intensity of the site frequency spectrum under underdominant selection."""
    ss = np.abs(ss) + 1e-8/Ne
    return (theta*np.exp(-2*ss*Ne*xx*(1-xx))/(xx*(1-xx))*
                 (1 + spy.erf(np.sqrt(2*ss*Ne)*(0.5-xx))/
                  spy.erf(np.sqrt(2*ss*Ne)/2)))

def sfs_ud_params_sigma(xx, theta, S_ud):
    """Calculate the intensity of the site frequency spectrum under underdominant selection."""
    S_ud = np.abs(S_ud) + 1e-8
    return (theta*np.exp(-S_ud*xx*(1-xx))/(xx*(1-xx))*
                 (1 + spy.erf(np.sqrt(S_ud)*(0.5-xx))/
                  spy.erf(np.sqrt(S_ud)/2)))

def sfs_ud_params_log(xx, theta, S_ud):
    """Calculate the log intensity of the site frequency spectrum under underdominant selection."""
    S_ud = np.abs(S_ud) + 1e-8
    A = spy.log_ndtr(np.sqrt(2*S_ud)*(0.5-xx))
    B = spy.log_ndtr(-np.sqrt(S_ud/2))
    non_erf_term = np.log(theta) - np.log(xx*(1-xx)) - S_ud*xx*(1-xx)
    erf_term_low = np.log1p(spy.erf(np.sqrt(S_ud)*(0.5-xx))/spy.erf(np.sqrt(S_ud)/2))
    erf_term_high = np.log(2) - np.log(spy.erf(np.sqrt(S_ud)/2)) + A + np.log(1-np.exp(B-A))
    return np.where(xx < 0.5, non_erf_term + erf_term_low, non_erf_term + erf_term_high)

###
def sfs_full_WF_grid(S_dir_set, S_ud_set, sfs_pile, x_set=None):
    s_dir_comp = S_dir_set/(2*sfs_pile["N_traj"][0])
    s_dir_comp[s_dir_comp >= np.max(sfs_pile["s_set"])] = np.max(sfs_pile["s_set"]) - 1/(2*sfs_pile["N_traj"][0]*1000)
    s_dir_comp[s_dir_comp <= np.min(sfs_pile["s_set"])] = np.min(sfs_pile["s_set"]) + 1/(2*sfs_pile["N_traj"][0]*1000)

    s_ud_comp = np.abs(S_ud_set)/(2*sfs_pile["N_traj"][0])# if sfs_pile is not None:

    s_ud_wf = np.abs(sfs_pile["s_ud_set"])
    s_ud_wf_max = np.max(s_ud_wf)
    s_ud_comp[s_ud_comp >= s_ud_wf_max] = s_ud_wf_max - 1/(2*sfs_pile["N_traj"][0]*1000)

    s_ii_upper = np.argmax(sfs_pile["s_set"][:,np.newaxis] > s_dir_comp, axis=0)
    s_ii_lower = s_ii_upper - 1

    s_ud_ii_upper = np.argmax(s_ud_wf[:,np.newaxis] > s_ud_comp, axis=0)
    s_ud_ii_lower = s_ud_ii_upper - 1

    s_upper = sfs_pile["s_set"][s_ii_upper]
    s_lower = sfs_pile["s_set"][s_ii_lower]

    s_ud_upper = s_ud_wf[s_ud_ii_upper]
    s_ud_lower = s_ud_wf[s_ud_ii_lower]

    sfs_11 = sfs_pile["sfs_grid"][s_ii_lower, s_ud_ii_lower]
    sfs_21 = sfs_pile["sfs_grid"][s_ii_upper, s_ud_ii_lower]
    sfs_12 = sfs_pile["sfs_grid"][s_ii_lower, s_ud_ii_upper]
    sfs_22 = sfs_pile["sfs_grid"][s_ii_upper, s_ud_ii_upper]

    _zero_nan(sfs_11)
    _zero_nan(sfs_21)
    _zero_nan(sfs_12)
    _zero_nan(sfs_22)

    sfs = (s_upper - s_dir_comp)*(s_ud_upper - s_ud_comp)*sfs_11.T
    sfs += (s_dir_comp - s_lower)*(s_ud_upper - s_ud_comp)*sfs_21.T
    sfs += (s_upper - s_dir_comp)*(s_ud_comp - s_ud_lower)*sfs_12.T
    sfs += (s_dir_comp - s_lower)*(s_ud_comp - s_ud_lower)*sfs_22.T
    sfs /= (s_upper - s_lower)*(s_ud_upper - s_ud_lower)

    _zero_nan(sfs)

    if x_set is not None:
        result = np.zeros((len(s_ii_upper), len(x_set)))
        for ii in range(len(s_ii_upper)):
            result[ii] = np.interp(x_set, sfs_pile["interp_x"], sfs[:,ii])
        return result.T
    return sfs

def sfs_dir_WF_single(S_dir, sfs_pile, x_set=None):
    S_ud_0_ii = np.where(sfs_pile["s_ud_set"]==0)[0][0]
    sfs_grid = sfs_pile["sfs_grid"][:,S_ud_0_ii]
    _zero_nan(sfs_grid)
    s_dir_comp = S_dir/(2*sfs_pile["N_traj"][0])
    if s_dir_comp >= np.max(sfs_pile["s_set"]):
        s_dir_comp = np.max(sfs_pile["s_set"]) - 1/(2*sfs_pile["N_traj"][0]*1000)
    elif s_dir_comp <= np.min(sfs_pile["s_set"]):
        s_dir_comp = np.min(sfs_pile["s_set"]) + 1/(2*sfs_pile["N_traj"][0]*1000)
    s_ii_upper = np.argmax(sfs_pile["s_set"] > s_dir_comp)
    s_ii_lower = s_ii_upper - 1
    w_lower = (sfs_pile["s_set"][s_ii_upper] - s_dir_comp)/(sfs_pile["s_set"][s_ii_upper] -
                                                           sfs_pile["s_set"][s_ii_lower])
    w_upper = 1 - w_lower

    sfs = (w_upper*sfs_grid[s_ii_upper] +
           w_lower*sfs_grid[s_ii_lower])
    if x_set is not None:
        return np.interp(x_set, sfs_pile["interp_x"], sfs)
    return sfs

def sfs_dir_WF_grid(S_dir_set, sfs_pile, x_set=None):
    S_ud_0_ii = np.where(sfs_pile["s_ud_set"]==0)[0][0]
    sfs_grid = sfs_pile["sfs_grid"][:,S_ud_0_ii]
    _zero_nan(sfs_grid)
    s_dir_comp = S_dir_set/(2*sfs_pile["N_traj"][0])
    s_dir_comp[s_dir_comp >= np.max(sfs_pile["s_set"])] = np.max(sfs_pile["s_set"]) - 1/(2*sfs_pile["N_traj"][0]*100)
    s_dir_comp[s_dir_comp <= np.min(sfs_pile["s_set"])] = np.min(sfs_pile["s_set"]) + 1/(2*sfs_pile["N_traj"][0]*100)
    s_ii_upper = np.argmax(sfs_pile["s_set"][:,np.newaxis] > s_dir_comp, axis=0)
    s_ii_lower = s_ii_upper - 1
    w_lower = (sfs_pile["s_set"][s_ii_upper] - s_dir_comp)/(sfs_pile["s_set"][s_ii_upper] -
                                                           sfs_pile["s_set"][s_ii_lower])
    w_upper = 1 - w_lower
    sfs = (w_upper[:,np.newaxis]*sfs_grid[s_ii_upper] +
           w_lower[:,np.newaxis]*sfs_grid[s_ii_lower])

    ## Interpolate x_set for all s_ud_wf if given
    if x_set is not None:
        result = np.zeros((len(s_ii_upper), len(x_set)))
        for ii in range(len(s_ii_upper)):
            result[ii] = np.interp(x_set, sfs_pile["interp_x"], sfs[ii])
        return np.transpose(result)
    return np.transpose(sfs)

def sfs_ud_WF_single(S_ud, sfs_pile, x_set=None):
    S_ud = np.abs(S_ud)
    ## Get UD grid and selection values
    sfs_grid = sfs_pile["sfs_grid"][np.where(sfs_pile["s_set"] == 0)[0][0]]
    _zero_nan(sfs_grid)
    s_ud_comp = S_ud/(2*sfs_pile["N_traj"][0])
    s_ud_wf = np.abs(sfs_pile["s_ud_set"])
    s_ud_ii_upper = np.searchsorted(s_ud_wf, s_ud_comp)
    s_ud_ii_upper = min(len(s_ud_wf) - 1, s_ud_ii_upper)
    s_ud_ii_lower = s_ud_ii_upper - 1
    w_lower = (s_ud_wf[s_ud_ii_upper] - s_ud_comp) / (s_ud_wf[s_ud_ii_upper] - s_ud_wf[s_ud_ii_lower])
    w_upper = 1 - max(w_lower, 0)

    sfs = (w_upper*sfs_grid[s_ud_ii_upper] +
           w_lower*sfs_grid[s_ud_ii_lower])
    if x_set is not None:
        return np.interp(x_set, sfs_pile["interp_x"], sfs)
    return sfs
    
def sfs_ud_WF_grid(S_ud_set, sfs_pile, x_set=None):
    S_ud_set = np.abs(S_ud_set)
    S_0_ii = np.where(sfs_pile["s_set"] == 0)[0][0]
    sfs_grid = sfs_pile["sfs_grid"][S_0_ii]
    _zero_nan(sfs_grid)
    s_ud_comp = S_ud_set/(2*sfs_pile["N_traj"][0])
    s_ud_wf = np.abs(sfs_pile["s_ud_set"])
    s_ud_wf_max = np.max(s_ud_wf)
    s_ud_comp[s_ud_comp >= s_ud_wf_max] = s_ud_wf_max - 1/(2*sfs_pile["N_traj"][0]*1000)
    s_ud_ii_upper = np.argmax(s_ud_wf[:,np.newaxis] > s_ud_comp, axis=0)
    s_ud_ii_lower = s_ud_ii_upper - 1

    w_lower = (s_ud_wf[s_ud_ii_upper] - s_ud_comp) / (s_ud_wf[s_ud_ii_upper] - s_ud_wf[s_ud_ii_lower])
    w_upper = 1 - w_lower

    sfs = (w_upper[:,np.newaxis]*sfs_grid[s_ud_ii_upper] +
           w_lower[:,np.newaxis]*sfs_grid[s_ud_ii_lower])

    ## Interpolate x_set for all s_ud_wf if given
    if x_set is not None:
        result = np.zeros((len(s_ud_ii_upper), len(x_set)))
        for ii in range(len(s_ud_ii_upper)):
            result[ii] = np.interp(x_set, sfs_pile["interp_x"], sfs[ii])
        return np.transpose(result)
    return np.transpose(sfs)

def ud_interp(x_set, interp_x, sfs, s_len):
    result = np.zeros((s_len, len(x_set)))
    for ii in range(s_len):
        result[ii] = np.interp(x_set, interp_x, sfs[ii])
    return np.transpose(result)

def sfs_full_params(xx, theta, ss_dir, ss_ud, Ne):
    """Calculate the intensity of the site frequency spectrum under
       directional and stabilizing selection."""
    sigma_1 = 2*Ne*ss_dir
    sigma_2 = 2*Ne*ss_ud
    part_A = 2*theta*np.exp(2*sigma_1*xx + sigma_2*xx*(1-xx))/(xx*(1-xx))
    part_B = spy.erf((-2*sigma_1+sigma_2*(2*xx-1))/(2*np.sqrt(np.abs(sigma_2))))
    part_C = spy.erf((-2*sigma_1+sigma_2)/(2*np.sqrt(np.abs(sigma_2))))
    part_D = spy.erf((-2*sigma_1-sigma_2)/(2*np.sqrt(np.abs(sigma_2))))
    return np.where(np.isinf(part_A) & (part_B-part_C==0),
                    np.zeros_like(part_A),
                    part_A*np.true_divide((part_B-part_C), (part_D-part_C)))

def sfs_full_params_alt(xx, theta, sigma_1, sigma_2):
    """Calculate the intensity of the site frequency spectrum under
       directional and stabilizing selection."""
    part_A = 2*theta*np.exp(2*sigma_1*xx + sigma_2*xx*(1-xx))/(xx*(1-xx))
    part_B = spy.erf((-2*sigma_1+sigma_2*(2*xx-1))/(2*np.sqrt(np.abs(sigma_2))))
    part_C = spy.erf((-2*sigma_1+sigma_2)/(2*np.sqrt(np.abs(sigma_2))))
    part_D = spy.erf((-2*sigma_1-sigma_2)/(2*np.sqrt(np.abs(sigma_2))))
    return np.where(np.isinf(part_A) & (part_B-part_C==0),
                    np.zeros_like(part_A),
                    part_A*np.true_divide((part_B-part_C), (part_D-part_C)))

def HH(X, kk=4):
    result = X**-1.
    for nn in np.arange(1., kk+1):
        if (2*nn - 1) % 2: #if odd
            result += ((-1)**nn * spy.gamma((2*nn-1)/2+1)*2**nn/np.sqrt(np.pi)/2**nn *
                       X**-(2*nn-1))
        else:
            result += (-1)**nn * 2**(nn/2) * spy.gamma(nn+1) * X**-(2*nn-1)
    return result

def HH_alt(X, kk=4):
    result = X**-1.
    for nn in np.arange(1., kk+1):
        if (2*nn - 1) % 2: #if odd
            result += ((-1)**nn * special.gamma((2*nn-1)/2+1)*2**nn/np.sqrt(np.pi)/2**nn *
                       X**-(2*nn-1))
        else:
            result += (-1)**nn * 2**(nn/2) * special.gamma(nn+1) * X**-(2*nn-1)
    return result

def sfs_full_negdir_approx(xx, theta, ss_dir, ss_ud, Ne):
    sigma_1 = 2*Ne*ss_dir
    sigma_2 = 2*Ne*ss_ud
    eta = np.sqrt(np.abs(sigma_2))
    AA = -sigma_1/eta - eta*(xx-0.5)
    BB = -sigma_1/eta - 0.5*eta
    CC = -sigma_1/eta + 0.5*eta
    return 2*theta*((HH(BB)*np.exp(2*sigma_1*xx-eta**2*xx*(1-xx)) - HH(AA)*np.exp(2*sigma_1))/
                    (xx*(1-xx)*(HH(BB)-HH(CC)*np.exp(2*sigma_1))))

def sfs_full_negdir_approx_alt(xx, theta, AA, BB, CC, eta, sigma_1, sigma_2):
    return 2*theta*((HH(BB)*np.exp(2*sigma_1*xx-eta**2*xx*(1-xx)) - HH(AA)*np.exp(2*sigma_1))/
                    (xx*(1-xx)*(HH(BB)-HH(CC)*np.exp(2*sigma_1))))

def sfs_full_posdir_approx(xx, theta, ss_dir, ss_ud, Ne):
    sigma_1 = 2*Ne*ss_dir
    sigma_2 = 2*Ne*ss_ud
    eta = np.sqrt(np.abs(sigma_2))
    AA = -sigma_1/eta - eta*(xx-0.5)
    BB = -sigma_1/eta - 0.5*eta
    CC = -sigma_1/eta + 0.5*eta
    return 2*theta*((HH(BB)*np.exp(-2*sigma_1*(1-xx)-eta**2.*xx*(1-xx))-HH(AA))/
                    (xx*(1-xx)*(HH(BB)*np.exp(-2*sigma_1)-HH(CC))))

def sfs_full_posdir_approx_alt(xx, theta, AA, BB, CC, eta,  sigma_1, sigma_2):
    return 2*theta*((HH(BB)*np.exp(-2*sigma_1*(1-xx)-eta**2.*xx*(1-xx))-HH(AA))/
                    (xx*(1-xx)*(HH(BB)*np.exp(-2*sigma_1)-HH(CC))))

def sfs_pos_stab(xx, theta, ss_dir, ss_ud, Ne):
    sigma_1 = 2*Ne*ss_dir
    sigma_2 = 2*Ne*ss_ud
    eta = np.sqrt(np.abs(sigma_2))
    AA = -sigma_1/eta - eta*(xx-0.5)
    BB = -sigma_1/eta - 0.5*eta
    CC = -sigma_1/eta + 0.5*eta
    num_1 = np.exp(-AA**2. + 2.*sigma_1*xx + sigma_2*xx*(1-xx))*HH(np.abs(AA))
    num_2 = np.exp(-BB**2. + 2.*sigma_1*xx + sigma_2*xx*(1-xx))*HH(np.abs(BB))
    denom = (spy.erf(CC) - spy.erf(BB))*xx*(1-xx)
    return 2*theta*(num_1-num_2)/denom

def sfs_pos_stab_alt(xx, theta, AA, BB, CC, sigma_1, sigma_2):
    num_1 = np.exp(-AA**2. + 2.*sigma_1*xx + sigma_2*xx*(1-xx))*HH(np.abs(AA))
    num_2 = np.exp(-BB**2. + 2.*sigma_1*xx + sigma_2*xx*(1-xx))*HH(np.abs(BB))
    denom = (spy.erf(CC) - spy.erf(BB))*xx*(1-xx)
    return 2*theta*(num_1-num_2)/denom

def sfs_full_params_stable_vec(xx, theta, sigma_1, sigma_2, UU=4):
    sigma_2 = -np.abs(sigma_2) - 1e-8
    eta = np.sqrt(np.abs(sigma_2))
    AA = -sigma_1/eta - eta*(xx-0.5)
    BB = -sigma_1/eta - 0.5*eta
    CC = -sigma_1/eta + 0.5*eta

    use_stable = (np.sign(BB) == np.sign(CC)) & ((np.abs(BB) > UU) & (np.abs(CC) > UU))
    pos_stab = ((AA < -UU) & (BB < -UU)) & (sigma_1 > 0)

    negdir_approx = sfs_full_negdir_approx_alt(xx, theta, AA, BB, CC, eta, sigma_1, sigma_2)
    posdir_approx = sfs_full_posdir_approx_alt(xx, theta, AA, BB, CC, eta, sigma_1, sigma_2)
    positive_stabilizing = sfs_pos_stab_alt(xx, theta, AA, BB, CC, sigma_1, sigma_2)
    full_params = sfs_full_params_alt(xx, theta, sigma_1, sigma_2)

    result = np.where(use_stable,
                      np.where(sigma_1<0, negdir_approx, posdir_approx),
                      np.where(pos_stab, positive_stabilizing, full_params))
    return result

def sfs_full_params_stable(xx, theta, ss_dir, ss_ud, Ne, UU=4):
    ss_dir = ss_dir + 1e-8/Ne
    ss_ud = ss_ud + 1e-8/Ne
    xx = np.array([1.])*xx
    sigma_1 = 2*Ne*ss_dir
    sigma_2 = 2*Ne*ss_ud
    eta = np.sqrt(np.abs(sigma_2))
    AA = -sigma_1/eta - eta*(xx-0.5)
    BB = -sigma_1/eta - 0.5*eta
    CC = -sigma_1/eta + 0.5*eta

    use_stable = (np.sign(BB) == np.sign(CC)) & ((np.abs(BB) > UU) & (np.abs(CC) > UU))
    pos_stab = ((AA < -UU) & (BB < -UU)) & (sigma_1 > 0)

    result = xx

    if use_stable:
        if ss_dir < 0:
            result = sfs_full_negdir_approx_alt(xx, theta, AA, BB, CC, eta, sigma_1, sigma_2)
        else:
            result =  sfs_full_posdir_approx_alt(xx, theta, AA, BB, CC, eta, sigma_1, sigma_2)
    else:
        result = np.where(pos_stab, sfs_pos_stab_alt(xx, theta, AA, BB, CC, sigma_1, sigma_2),
                          sfs_full_params_alt(xx, theta, sigma_1, sigma_2))
    return result

def sfs_full_params_stable_sig(xx, theta, sigma_1, sigma_2, UU=4):
    sigma_1 += 1e-8
    sigma_2 = -np.abs(sigma_2) - 1e-8
    xx = np.array([1.])*xx
    eta = np.sqrt(np.abs(sigma_2))
    AA = -sigma_1/eta - eta*(xx-0.5)
    BB = -sigma_1/eta - 0.5*eta
    CC = -sigma_1/eta + 0.5*eta

    use_stable = (np.sign(BB) == np.sign(CC)) & ((np.abs(BB) > UU) & (np.abs(CC) > UU))

    pos_stab = ((AA < -UU) & (BB < -UU)) & (sigma_1 > 0)

    result = xx

    if use_stable:
        if sigma_1 < 0:
            result = sfs_full_negdir_approx_alt(xx, theta, AA, BB, CC, eta, sigma_1, sigma_2)
        else:
            result =  sfs_full_posdir_approx_alt(xx, theta, AA, BB, CC, eta, sigma_1, sigma_2)
    else:
        result = np.where(pos_stab, sfs_pos_stab_alt(xx, theta, AA, BB, CC, sigma_1, sigma_2),
                          sfs_full_params_alt(xx, theta, sigma_1, sigma_2))

    return result

def sfs_neut_params(xx, theta):
    """Calculate the intensity of the site frequency spectrum under the standard neutral model."""
    return 2*theta/xx

def ds_sfs_del_params(xx, theta, ss, Ne):
    if type(ss) is not np.ndarray:
        ss = np.array([ss])
    result_neg = ((2*Ne*(np.exp(2*Ne*ss) +
                        (xx-1)*np.exp(2*Ne*ss*(1+xx)) -
                        xx*np.exp(2*Ne*ss*xx)))/
                  (xx*(xx-1)*(np.exp(2*Ne*ss)-1)**2))
    result_pos = ((-2*Ne*np.exp(-2*Ne*ss)*(1-np.exp(2*Ne*ss*(xx-1)))/
                   ((1-np.exp(-2*Ne*ss))**2*(1-xx)*xx)) -
                  2*Ne*(xx-1)*np.exp(2*Ne*ss*(xx-1))/
                  (xx*(1-xx)*(1-np.exp(-2*Ne*ss))))

    isnan_neg = np.isnan(result_neg)
    result_neg[ss > 0] = result_pos[ss > 0]
    result_neg[isnan_neg] = result_pos[isnan_neg]
    return result_neg

def ds_sfs_ud_params(xx, theta, ss, Ne):
    ss = np.abs(ss)
    mult_factor = np.exp(-2*Ne*ss*(1-xx)*xx)
    A1 = (np.exp(-2*Ne*ss*(0.5-xx)**2)*np.sqrt(Ne)*(1-2*xx)/
          (np.sqrt(2*np.pi*ss)*spy.erf(np.sqrt(Ne*ss/2))))
    A2 = (np.sqrt(Ne)*np.exp(-Ne*ss/2)*spy.erf(np.sqrt(Ne*ss*2)*(0.5-xx))/
          (np.sqrt(2*np.pi*ss)*spy.erf(np.sqrt(Ne*ss/2))**2))
    A3 = spy.erf(np.sqrt(2*Ne*ss)*(0.5-xx))/spy.erf(np.sqrt(Ne*ss/2))
    return mult_factor*(A1-A2)/(xx*(1-xx)) - mult_factor*Ne*(1 + A3)

def tennessen_model(kk=1):
    N0 = math.ceil(7310*kk)
    N_old_growth = math.ceil(14474*kk)
    N_ooa_bn = math.ceil(1861*kk)
    N_ooa_bn_2 = math.ceil(1032*kk)
    N_growth_1 = math.ceil(9300*kk)
    N_growth_2 = math.ceil(512000*kk)

    t_old_growth = math.ceil(3880*kk)
    t_ooa_bn = math.ceil(1120*kk)
    t_growth_1 = math.ceil(715*kk)
    t_growth_2 = math.ceil(205*kk)

    r_growth_1 = (N_growth_1/N_ooa_bn_2)**(1/(t_growth_1-1))
    r_growth_2 = (N_growth_2/N_growth_1)**(1/(t_growth_2))

    N_set = np.array([N0] + [N_old_growth]*t_old_growth)
    N_set = np.append(N_set, [N_ooa_bn]*t_ooa_bn)
    N_set = np.append(N_set, N_ooa_bn_2*r_growth_1**np.arange(t_growth_1))
    N_set = np.append(N_set, N_growth_1*r_growth_2**np.arange(1, t_growth_2+1))
    return N_set.astype(np.int)

def gutenkunst_model_chb(kk=1):
    N0 = math.ceil(7300*kk)
    N_old_growth = math.ceil(12300*kk)
    N_ooa_bn = math.ceil(2100*kk)
    N_ooa_split = math.ceil(510)

    t_old_growth = math.ceil(3200*kk)
    t_ooa_bn = math.ceil(4752*kk)
    t_growth = math.ceil(848*kk)

    growth_rate_per_gen = 0.0055

    N_set = np.array([N0] + [N_old_growth]*t_old_growth)
    N_set = np.append(N_set, [N_ooa_bn]*t_ooa_bn)
    N_set = np.append(N_set, N_ooa_split*np.array([np.exp(growth_rate_per_gen * tt) 
                                                   for tt in range(t_growth)]))
    return N_set.astype(np.int)

def jouganous_model_jpt():
    N0 = 11293
    N_old_growth = 23721
    N_ooa_bn = 2831
    N_chb_bn = 1019
    N_jpt_bn = 4384

    t_old_growth = math.ceil((357000 - 119000) / 29)
    t_ooa_bn = math.ceil((119000 - 46000) / 29)
    t_chb_bn = math.ceil((46000 - 9000) / 29)
    t_jpt_bn = math.ceil(9000 / 29)

    r_chb_bn = 1.0026
    r_jpt_bn = 1.0129

    N_set = np.array([N0] + [N_old_growth]*t_old_growth)                 # Old growth period
    N_set = np.append(N_set, [N_ooa_bn]*t_ooa_bn)                        # OOA bottleneck period
    N_set = np.append(N_set, N_chb_bn * r_chb_bn ** np.arange(t_chb_bn)) # CHB bottleneck period
    N_set = np.append(N_set, N_jpt_bn * r_jpt_bn ** np.arange(t_jpt_bn)) # JPT bottleneck period
    
    return N_set.astype(int)

def generate_tennessen_dadi_func():
    """
    Generate a function that interpolates the Tennessen et al. (2012) model
    of human population growth in units of 2N0 generations where N0 is the
    initial effective population size (7310).
    """
    tenn_N = tennessen_model()
    tenn_nu = tenn_N / tenn_N[0]
    tenn_T = np.arange(len(tenn_N)) / (2 * tenn_N[0])
    tenn_nu_func = lambda t: np.interp(t, tenn_T, tenn_nu)
    return tenn_nu_func, tenn_T[-1]

def generate_gutenkunst_dadi_func():
    """
    Generate a function that interpolates the Gutenkunst et al. (2009) model
    of human population growth in units of 2N0 generations where N0 is the
    initial effective population size (7310).
    """
    gutenkunst_N = gutenkunst_model_chb()
    gutenkunst_nu = gutenkunst_N / gutenkunst_N[0]
    gutenkunst_T = np.arange(len(gutenkunst_N)) / (2 * gutenkunst_N[0])
    gutenkunst_nu_func = lambda t: np.interp(t, gutenkunst_T, gutenkunst_nu)
    return gutenkunst_nu_func, gutenkunst_T[-1]

def generate_jouganous_dadi_func():
    """
    Generate a function that interpolates the Jouganous et al. (2017) model
    of human population growth in units of 2N0 generations where N0 is the
    initial effective population size (7310).
    """
    jouganous_N = jouganous_model_jpt()
    jouganous_nu = jouganous_N / jouganous_N[0]
    jouganous_T = np.arange(len(jouganous_N)) / (2 * jouganous_N[0])
    jouganous_nu_func = lambda t: np.interp(t, jouganous_T, jouganous_nu)
    return jouganous_nu_func, jouganous_T[-1]

def sm_to_dadi_params(S_ud, S_dir, gamma_fudge=-0.0001):
    gamma_ud = -np.abs(S_ud) / 2
    gamma_dir = S_dir + gamma_fudge
    hh = (gamma_dir + gamma_ud) / (2 * gamma_dir)
    return gamma_dir, hh

def run_tennessen_dadi(S_dir, S_ud, pts=10000, xx=None):
    if xx is None:
        xx = dadi.Numerics.default_grid(pts)
    gamma, hh = sm_to_dadi_params(S_ud, S_dir)
    tenn_nu_func, tenn_T_end = generate_tennessen_dadi_func()
    phi = dadi.PhiManip.phi_1D(xx, gamma=gamma, h=hh)
    phi = dadi.Integration.one_pop(phi, xx, T=tenn_T_end, nu=tenn_nu_func, gamma=gamma, h=hh)
    return xx, phi

def run_gutenkunst_dadi(S_dir, S_ud, pts=10000, xx=None):
    if xx is None:
        xx = dadi.Numerics.default_grid(pts)
    gamma, hh = sm_to_dadi_params(S_ud, S_dir)
    gutenkunst_nu_func, gutenkunst_T_end = generate_gutenkunst_dadi_func()
    phi = dadi.PhiManip.phi_1D(xx, gamma=gamma, h=hh)
    phi = dadi.Integration.one_pop(phi, xx, T=gutenkunst_T_end, nu=gutenkunst_nu_func, gamma=gamma, h=hh)
    return xx, phi

def run_jouganous_dadi(S_dir, S_ud, pts=10000, xx=None):
    if xx is None:
        xx = dadi.Numerics.default_grid(pts)
    gamma, hh = sm_to_dadi_params(S_ud, S_dir)
    jouganous_nu_func, jouganous_T_end = generate_jouganous_dadi_func()
    phi = dadi.PhiManip.phi_1D(xx, gamma=gamma, h=hh)
    phi = dadi.Integration.one_pop(phi, xx, T=jouganous_T_end, nu=jouganous_nu_func, gamma=gamma, h=hh)
    return xx, phi

def poly_ratio(neut_sfs, sel_sfs, xx):
    """
    Calculate the ratio of the area under the selected sfs to the area under the neutral sfs

    Parameters
    ----------
    neut_sfs : array
        The neutral sfs
    sel_sfs : array
        The selected sfs
    xx : array
        The set of derived allele frequencies for sfs grids
    """
    neut_area = np.trapz(neut_sfs, xx)
    sel_area = np.trapz(sel_sfs, xx)
    return sel_area / neut_area

def truncate_pile(sfs_pile, factor=1e-8):
    """
    Truncate the selected sfs so that there is never 1/factor more selected sfs than neutral sfs
    above the derived allele frequency where the ratio is first greater than that factor.
    Purpose of this is to ignore regions where the sfs is flat some region in the selected sfs
    but it is incredibly unlikely to actually see mutations in that frequency range.

    Assuming ~15 million SNPs with frequency > 1% in humans, the factor of 1e-8 corresponds to
    less than 1 selected SNP in the entire genome.

    Parameters
    ----------
    sfs_pile : dict
        The sfs pile dictionary
    factor : float
        The ratio of the selected sfs to the neutral sfs
    """

    # Get the neutral sfs
    xx = sfs_pile["interp_x"]
    s_zero = np.where(sfs_pile["s_set"]==0)[0][0]
    s_ud_zer = np.where(sfs_pile["s_ud_set"]==0)[0][0]
    sfs_neutral = sfs_pile["sfs_grid"][s_zero, s_ud_zer, :]
    # make a copy of the entire pile dict
    new_pile = deepcopy(sfs_pile)

    for ii, ss in enumerate(sfs_pile["s_set"]):
        for jj, s_ud in enumerate(sfs_pile["s_ud_set"]):
            sfs_sel = sfs_pile["sfs_grid"][ii, jj, :]
            new_sfs = sfs_sel.copy()
            kk = 0
            while poly_ratio(sfs_neutral[kk:], sfs_sel[kk:], xx[kk:]) > factor:
                kk += 1
                if kk == len(xx)-1:
                    break
            if kk < len(xx)-1:
                new_sfs[kk:] = 0
            new_pile["sfs_grid"][ii, jj, :] = new_sfs
    return new_pile