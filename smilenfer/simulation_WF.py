import numpy as np
import scipy.stats as stats
import scipy.special as spy
import math

from copy import deepcopy

from numba import jit, prange

from . import simulation as sim

def approx_eq_sfs_del(NN, ss, theta):
    x_set = np.arange(1, 2*NN, 1)/(2*NN)
    sfs = sim.sfs_del_params(x_set, theta, ss, NN)/(2*NN)
    return sfs.astype(np.float32)

def approx_eq_sfs_full(NN, ss, ss_ud, theta):
    x_set = np.arange(1, 2*NN, 1)/(2*NN)
    sfs = sim.sfs_full_params_stable(x_set, theta, ss, ss_ud, NN)/(2*NN)
    return sfs.astype(np.float32)

def make_transition_mat_del(N1, N2, ss):
    x_set = np.arange(1, 2*N1, 1)/(2*N1)
    w_bar = (1+2*ss)*x_set**2 + (1+ss)*2*x_set*(1-x_set) + (1-x_set)**2
    psi = ((1+2*ss)*x_set**2 + (1+ss)*x_set*(1-x_set))/w_bar
    return stats.binom.pmf(np.arange(1, 2*N2, 1), 2*N2, psi[:,np.newaxis]).astype(np.float32)

def make_transition_mat(N1, N2, w11, w12, w22):
    x_set = np.arange(1, 2*N1, 1)/(2*N1)
    w_bar = w22*x_set**2 + w12*2*x_set*(1-x_set) + w11*(1-x_set)**2
    psi = (w22*x_set**2 + w12*x_set*(1-x_set))/w_bar
    return stats.binom.pmf(np.arange(1, 2*N2, 1), 2*N2, psi[:,np.newaxis]).astype(np.float32)

@jit(nopython=True, parallel=True)
def transition_loop_del(N1, N2, ss, sfs_prev):
    x_set = np.arange(1, 2*N1, 1)/(2*N1)
    w_bar = (1+2*ss)*x_set**2 + (1+ss)*2*x_set*(1-x_set) + (1-x_set)**2
    psi_set = ((1+2*ss)*x_set**2 + (1+ss)*x_set*(1-x_set))/w_bar
    new_counts = np.arange(1, 2*N2, 1).astype(np.float64)
    result = np.zeros_like(new_counts)
    ## Loop through counts in previous generation and add up how
    ## much each contributes to counts in the next generation on average
    for ii in prange(psi_set.size):
        result += sfs_prev[ii]*np.exp(spy.gammaln(2*N2+1) - (spy.gammaln(new_counts+1) +
                                                            spy.gammaln(2*N2-new_counts+1)) +
                                      new_counts*np.log(psi_set[ii]) +
                                      (2*N2-new_counts)*np.log1p(-psi_set[ii]))
    return result

@jit(nopython=True, parallel=True)
def transition_loop(N1, N2, w11, w12, w22, sfs_prev):
    x_set = np.arange(1, 2*N1, 1)/(2*N1)
    w_bar = w22*x_set**2 + w12*2*x_set*(1-x_set) + w11*(1-x_set)**2
    psi_set = (w22*x_set**2 + w12*x_set*(1-x_set))/w_bar
    new_counts = np.arange(1, 2*N2, 1).astype(np.float64)
    result = np.zeros_like(new_counts)
    bin_coeff = spy.gammaln(2*N2+1) - (spy.gammaln(new_counts+1) +
                                      spy.lgamma(2*N2-new_counts+1))
    ## Loop through counts in previous generation and add up how
    ## much each contributes to counts in the next generation on average
    for ii in prange(psi_set.size):
        result += sfs_prev[ii]*np.exp(bin_coeff +
                                      new_counts*np.log(psi_set[ii]) +
                                      (2*N2-new_counts)*np.log1p(-psi_set[ii]))
    return result

def evol_del(N0, ss, NN_set, theta, warmup=100, max_size=10000):
    sfs_start = approx_eq_sfs_del(N0, ss, theta)
    warmup_mat = make_transition_mat_del(N0, N0, ss)
    for ii in range(warmup):
        sfs_start[0] += theta
        sfs_start = np.matmul(sfs_start, warmup_mat)
    NN_prev = N0
    transition_mat = warmup_mat
    sfs_end = sfs_start
    over_max = False
    for ii, NN in enumerate(NN_set):
        sfs_end[0] += theta*(NN/N0)/2
        if NN > max_size:
            over_max = True
            print("Over max size: making new matrix {}->{}".format(NN_prev, NN), end="...")
            sfs_end = transition_loop_del(NN_prev, NN, ss, sfs_end)
            print("done")
        else:
            if ((transition_mat.shape[0] != (2*NN_prev-1)) or
                (transition_mat.shape[1] != (2*NN-1) or over_max)):
                print("making new matrix {}->{}".format(NN_prev, NN), end="...")
                del(transition_mat)
                transition_mat = make_transition_mat_del(NN_prev, NN, ss)
                print("done")
            sfs_end = np.matmul(sfs_end, transition_mat)
            over_max = False
        sfs_end[0] += theta*(NN/N0)/2
        NN_prev = NN
    del(transition_mat)
    return sfs_end

def evol_wf(N0, ss, ss_ud, NN_set, theta, warmup=100, max_size=10000):
    sfs_start = approx_eq_sfs_full(N0, ss, ss_ud, theta)
    w11 = 1
    w12 = max(1 + ss + ss_ud/2, 0)
    w22 = max(1 + 2*ss, 0)
    ## warmup stage doesn't check that N0 is reasonably sized, so be careful here
    warmup_mat = make_transition_mat(N0, N0, w11, w12, w22)
    for ii in range(warmup):
        sfs_start[0] += theta
        sfs_start = np.matmul(sfs_start, warmup_mat)
    NN_prev = N0
    transition_mat = warmup_mat
    sfs_end = sfs_start
    over_max = False
    for ii, NN in enumerate(NN_set):
        sfs_end[0] += theta*(NN/N0)/2
        ## Check whether N is the same as in the previous generation
        next_check = False
        if ii < (len(NN_set)-1):
            if NN == NN_set[ii+1]: ## Popsize will be constant in the next generation
                next_check = True ## Popsize temporarily constant?
                ## If we already have the transition matrix, use it
                if ((transition_mat.shape[0] == (2*NN_prev-1)) and
                    (transition_mat.shape[1] == (2*NN-1))):
                    print("{}->{}".format(NN_prev, NN), end="...")
                    sfs_end = np.matmul(sfs_end, transition_mat)
                elif (NN > max_size) or (NN_prev != NN): ## Too big, don't want to make new transition matrix
                    over_max = True ## Is N large AND changing?
                    print("{}->{}".format(NN_prev, NN), end="...")
                    sfs_end = transition_loop(NN_prev, NN, w11, w12, w22, sfs_end)
                else: ## N is small, do make a new transition matrix
                    del(transition_mat)
                    print("making new matrix {}->{}".format(NN_prev, NN), end="...")
                    transition_mat = make_transition_mat(NN_prev, NN, w11, w12, w22)
                    sfs_end = np.matmul(sfs_end, transition_mat)
            else:
                ## If we already have the transition matrix, use it
                if ((transition_mat.shape[0] == (2*NN_prev-1)) and
                    (transition_mat.shape[1] == (2*NN-1))):
                    print("{}->{}".format(NN_prev, NN), end="...")
                    sfs_end = np.matmul(sfs_end, transition_mat)
                else: ## Don't bother making a new transition matrix if we won't use it
                    print("{}->{}".format(NN_prev, NN), end="...")
                    sfs_end = transition_loop(NN_prev, NN, w11, w12, w22, sfs_end)
        else:
            ## If the currently stored transition matrix is not applicable, use the loop construction
            if ((transition_mat.shape[0] != (2*NN_prev-1)) or
                (transition_mat.shape[1] != (2*NN-1))):
                print("{}->{}".format(NN_prev, NN), end="...")
                sfs_end = transition_loop(NN_prev, NN, w11, w12, w22, sfs_end)
            ## If we can use the current transition matrix, use it
            else:
                print("{}->{}".format(NN_prev, NN), end="...")
                sfs_end = np.matmul(sfs_end, transition_mat)
        sfs_end[0] += theta*(NN/N0)/2
        NN_prev = NN
    del(transition_mat)
    return sfs_end

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

#TODO: consider making this go out from 0.5 instead
def truncate_sfs_vals(WF_pile, theta, Ne, mu, L_target):
    neut_index = np.where(WF_pile["s_set"]==0)[0][0]
    L_standard = theta / (4*Ne*mu)
    N_last = WF_pile["tenn_N"][-1]
    ii_start = math.ceil(0.01 * 2 * N_last)
    ii_end = math.floor(0.99 * 2 * N_last)
    ii_set = np.arange(ii_start, ii_end + 1)
    max_jj_set = np.full_like(WF_pile["s_ud_set"], ii_end+1)
    for ii, _ in enumerate(WF_pile["s_ud_set"]):
        jj = len(ii_set) - 1
        seg_rate = np.sum(np.interp(ii_set[jj:] / (2*N_last),
                                    WF_pile["interp_x"], WF_pile["sfs_grid"][neut_index,ii])/
                          L_standard * L_target)
        while seg_rate < 1 and jj > 0:
            jj -= 1
            seg_rate = np.sum(np.interp(ii_set[jj:] / (2*N_last),
                                        WF_pile["interp_x"], WF_pile["sfs_grid"][neut_index,ii])/
                              L_standard * L_target)
        max_jj_set[ii] = jj
        if jj == 0:
            max_jj_set[(ii+1):] = 0
            break
    return np.where(max_jj_set>0, ii_set[np.array(max_jj_set, dtype=np.int)] / (2*N_last), 0)

def zero_sfs_grid(WF_pile, max_freq_set, zero_val=0):
    result = deepcopy(WF_pile)
    for ii, _ in enumerate(WF_pile["s_ud_set"]):
        result["sfs_grid"][:, ii, WF_pile["interp_x"]>=max_freq_set[ii]] = zero_val
    return result
