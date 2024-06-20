import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from copy import deepcopy

from . import statistics as smile_stats
from . import simulation as sim
from . import posterior as post

def _zero_nan(arr):
    arr[np.where(np.isnan(arr))] = 0

def combine_post_samps(neut_f, stab_f, dir_f, full_f, plei_f, nplei_f, weights=False, use_ZZ_0=False,
                       simple=False, stab_db=False):
    result = {}
    result["neut"] = smile_stats.combine_samps(neut_f, weights=weights, use_ZZ_0=use_ZZ_0)
    result["stab"] = smile_stats.combine_samps(stab_f, weights=weights, use_ZZ_0=use_ZZ_0)
    if stab_db:
        result["stab_db"] = smile_stats.combine_samps(stab_f, weights=weights, use_ZZ_0=use_ZZ_0,
                                                         sel_model="stab_db")
        result["plei_db"] = smile_stats.combine_samps(plei_f, weights=weights, use_ZZ_0=use_ZZ_0,
                                                         sel_model="plei_db")
    result["dir"] = smile_stats.combine_samps(dir_f, weights=weights, use_ZZ_0=use_ZZ_0, sel_model="dir_db")
    if not simple:
        result["full"] = smile_stats.combine_samps(full_f, weights=weights, use_ZZ_0=use_ZZ_0,
                                                      sel_model="full_db")
        result["nplei"] = smile_stats.combine_samps(nplei_f, weights=weights, use_ZZ_0=use_ZZ_0)
        if stab_db:
            result["nplei_db"] = smile_stats.combine_samps(nplei_f, weights=weights, use_ZZ_0=use_ZZ_0,
                                                           sel_model="nplei_db")
    result["plei"] = smile_stats.combine_samps(plei_f, weights=weights, use_ZZ_0=use_ZZ_0)
    return result

def get_all_param_ranges(all_samp_vals, grid_size_1d, grid_size_2d, grid_size_Ip,
                         grid_size_I2, grid_size_nn, nn_max):

    I1_set_1d, I2_set_1d, _, _, _ = smile_stats.choose_param_range(all_samp_vals, grid_size_1d)
    I1_set_2d, I2_set_2d, _, _, _ = smile_stats.choose_param_range(all_samp_vals, grid_size_2d)

    _, _, Ip_set, _, _ = smile_stats.choose_param_range(all_samp_vals, grid_size_Ip)
    _, _, _, I2_set, _ = smile_stats.choose_param_range(all_samp_vals, grid_size_I2, nn_max=nn_max)
    _, _, _, _, nn_set = smile_stats.choose_param_range(all_samp_vals, grid_size_nn, nn_max=nn_max)

    return I1_set_1d, I2_set_1d, I1_set_2d, I2_set_2d, Ip_set, I2_set, nn_set

def get_all_samp_vals(samp_fnames, cut_rows):
    all_samp_vals = []
    for samp_set in samp_fnames:
        with open(samp_set, "r") as handle:
            samp_vals_tmp = handle.readlines()
        samp_vals_tmp = np.abs(np.array([float(bb.strip()) for bb in samp_vals_tmp])[cut_rows])
        all_samp_vals.append(samp_vals_tmp)
    all_samp_vals = np.concatenate(all_samp_vals)
    return all_samp_vals

def setup_data(fname, p_thresh, min_x):
    trait_data = cut_data(pd.read_csv(fname, sep="\t", compression="gzip"), p_thresh, min_x)
    beta_hat = trait_data.rbeta.to_numpy()
    cut_rows = get_cut_rows(pd.read_csv(fname, sep="\t", compression="gzip"), p_thresh, min_x)
    trait_data_full = pd.read_csv(fname, sep="\t", compression="gzip")
    trait_data_full = deepcopy(trait_data_full[trait_data_full.var_exp > 0])
    v_cutoff = smile_stats.calc_cutoffs(trait_data_full)[repr(p_thresh)]
    d_x_set = np.array([np.maximum(smile_stats.discov_x(beta, v_cutoff), min_x) for beta in beta_hat])
    return trait_data, beta_hat, d_x_set

def setup_data_cojo(fname, p_thresh, min_x, sep=None, compression="infer",
                    beta_col="orig_b", freq_col="freq", pp="orig_p", p_cutoff=np.inf,
                    alt_freq_col=None, var_inflation_cutoff=None):
    """
    Cut data according to p_thresh, min_x, and p_cutoff.

    Return the cut data, the absolute value of the beta column, and the 
    discovery maf for each SNP.
    """
    trait_data = cut_data_cojo(post.read_trait_data(fname, sep=sep, compression=compression,
                                                    beta_col=beta_col, freq_col=freq_col,
                                                    alt_freq_col=alt_freq_col,
                                                    var_inflation_cutoff=var_inflation_cutoff),
                               p_thresh, min_x, pp=pp, p_cutoff=p_cutoff)
    beta_hat = np.abs(trait_data[beta_col].to_numpy())

    trait_data_full = deepcopy(post.read_trait_data(fname, sep=sep, compression=compression,
                                                    beta_col=beta_col, freq_col=freq_col,
                                                    alt_freq_col=alt_freq_col,
                                                    var_inflation_cutoff=var_inflation_cutoff))
    v_cutoff = smile_stats.calc_cutoffs_new(trait_data_full.orig_var_exp,
                                              trait_data_full[pp])[repr(p_thresh)]
    d_x_set = np.array([np.maximum(smile_stats.discov_x(beta, v_cutoff), min_x) for beta in beta_hat])
    return trait_data, beta_hat, d_x_set

def setup_sim_data(fname, sample_fname, p_thresh, min_x):
    trait_data = cut_data(pd.read_csv(fname, sep="\t", compression="gzip"), p_thresh, min_x)
    beta_hat = trait_data.rbeta.to_numpy()
    cut_rows = get_cut_rows(pd.read_csv(fname, sep="\t", compression="gzip"), p_thresh, min_x)
    beta_true = get_post_samps(sample_fname, cut_rows)
    trait_data_full = pd.read_csv(fname, sep="\t", compression="gzip")
    trait_data_full = deepcopy(trait_data_full[trait_data_full.var_exp > 0])
    v_cutoff = smile_stats.calc_cutoffs(trait_data_full)[repr(p_thresh)]
    d_x_set = np.array([np.maximum(smile_stats.discov_x(beta, v_cutoff), min_x) for beta in beta_hat])
    return trait_data, beta_hat, beta_true, d_x_set

def get_v_cutoff(fname, p_thresh, cojo=False, sep="\t", compression="gzip",
                 beta_col="orig_b", freq_col="freq", pp="orig_p",
                 alt_freq_col=None, var_inflation_cutoff=None):
    if not cojo:
        trait_data_full = pd.read_csv(fname, sep="\t", compression="gzip")
        trait_data_full = deepcopy(trait_data_full[trait_data_full.var_exp > 0])
        v_cutoff = smile_stats.calc_cutoffs(trait_data_full)[repr(p_thresh)]
    if cojo:
        trait_data_full = deepcopy(post.read_trait_data(fname, sep=sep, compression=compression,
                                                        beta_col=beta_col, freq_col=freq_col,
                                                        alt_freq_col=alt_freq_col,
                                                        var_inflation_cutoff=var_inflation_cutoff))
        v_cutoff = smile_stats.calc_cutoffs_new(trait_data_full.orig_var_exp,
                                                  trait_data_full[pp])[repr(p_thresh)]
    return v_cutoff

def cut_data(trait_data, p_thresh, min_x, p_cutoff=np.inf):
    trait_data_tmp = deepcopy(trait_data[trait_data.var_exp > 0])
    v_cutoff = smile_stats.calc_cutoffs(trait_data_tmp)[repr(p_thresh)]
    cut_rows = (np.array(trait_data_tmp.var_exp > v_cutoff) &
                np.array(trait_data_tmp.maf >= min_x) &
                np.array(trait_data_tmp.pval <= p_cutoff))
    assert np.sum(np.isnan(cut_rows)) == 0
    assert np.sum(np.isnan(trait_data.var_exp[cut_rows])) == 0
    return trait_data_tmp.iloc[cut_rows].reset_index().copy(deep=True)

def cut_data_cojo(trait_data, p_thresh, min_x, pp="orig_p", p_cutoff=np.inf):
    v_cutoff = smile_stats.calc_cutoffs_new(trait_data.orig_var_exp, trait_data[pp])[repr(p_thresh)]
    # np.nan < x should evaluate to False
    cut_rows = (np.array(trait_data.orig_var_exp > v_cutoff) &
                np.array(trait_data.maf >= min_x) &
                np.array(trait_data[pp]<=p_cutoff))
    assert np.sum(np.isnan(cut_rows)) == 0
    assert np.sum(np.isnan(trait_data.orig_var_exp[cut_rows])) == 0
    return trait_data.iloc[cut_rows].reset_index().copy(deep=True)

def get_cut_rows(trait_data, p_thresh, min_x, cojo=False, pp="orig_p", p_cutoff=np.inf):
    if not cojo:
        trait_data_tmp = deepcopy(trait_data[trait_data.var_exp > 0])
        v_cutoff = smile_stats.calc_cutoffs(trait_data_tmp)[repr(p_thresh)]
        return (np.array(trait_data_tmp.var_exp > v_cutoff) &
                np.array(trait_data_tmp.maf >= min_x) &
                np.array(trait_data_tmp[pp] <= p_cutoff))
    else:
        v_cutoff = smile_stats.calc_cutoffs_new(trait_data.orig_var_exp, trait_data[pp])[repr(p_thresh)]
        return (np.array(trait_data.orig_var_exp > v_cutoff) &
                np.array(trait_data.maf >= min_x) &
                np.array(trait_data[pp] <= p_cutoff))

def get_post_samps(fname, cut_rows):
    with open(fname, "r") as handle:
        samp_vals = handle.readlines()
    return np.array([float(bb.strip()) for bb in samp_vals])[cut_rows]

def run_ashr(beta_hats, SEs, RDS_fname):
    robjects.r('''
    library(\"ashr\")
    run_ashr <- function(betahats, sebetahats, RDS.fname){
    gg <- readRDS(RDS.fname)
    result <- ash(betahats, sebetahats, g=gg, fixg=TRUE)
    return(result)
    }
    ''')
    run_ashr = robjects.globalenv["run_ashr"]
    ashr_output = run_ashr(robjects.FloatVector(beta_hats),
                           robjects.FloatVector(SEs),
                           RDS_fname)
    result = pandas2ri.rpy2py_dataframe(ashr_output[4])
    return result

def sample_from_vec(vec):
    uu = random.uniform(0, 1)
    return np.argmax(uu <= np.cumsum(vec)/np.sum(vec))

def sample_x_sfs(x_set, sfs_vals_tid, sfs_vals_tia, pi):
    x_weights = smile_stats.get_weights(x_set)
    sfs_weighted = (pi*sfs_vals_tid + (1-pi)*np.flip(sfs_vals_tia))*x_weights
    sfs_weighted_sum = np.sum(sfs_weighted)
    if sfs_weighted_sum == 0:
        print("Selection too strong, SFS too close to zero")
        if np.sum(sfs_vals_tid) == 0 and np.sum(sfs_vals_tia) > 0:
            return np.min(x_set)
        elif np.sum(sfs_vals_tia) == 0 and np.sum(sfs_vals_tid) > 0:
            return np.max(x_set)
        elif np.random.randint(2)==1:
            return np.max(x_set)
        else:
            return np.min(x_set)
    ## Sample using uniform rv bb and grabbing first CDF entry >= bb
    bb = random.uniform(0, 1)
    x_samp = x_set[np.argmax(bb <= np.cumsum(sfs_weighted)/sfs_weighted_sum)]
    return x_samp

def sample_raf_neut(x_mins, pi=0.5, n_points=2000, epsilon=1e-8, WF_pile=None):
    if WF_pile is not None:
        S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
        S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
        sfs_neut = WF_pile["sfs_grid"][S_0_ii, S_ud_0_ii]
    result = np.zeros_like(x_mins, dtype=np.float64)
    for ii, x_min in enumerate(x_mins):
        x_set = smile_stats.adjusted_x_set(x_mins[ii]+epsilon, 10, n_points)
        if WF_pile is None:
            sfs_vals = 1/x_set
        else:
            sfs_vals = np.interp(x_set, WF_pile["interp_x"], sfs_neut)
        result[ii] = sample_x_sfs(x_set, sfs_vals, sfs_vals, pi)
    return result

def sample_raf_dir_db(beta_true, x_mins, I1, Ne, n_points=2000, epsilon=1e-8, WF_pile=None):
    if WF_pile is not None:
        S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
    S_dir_set = 2*Ne*beta_true*I1
    pi_set = sim.pi_dir_db(S_dir_set)
    result = np.zeros_like(beta_true, dtype=np.float64)
    for ii, beta in enumerate(beta_true):
        x_set = smile_stats.adjusted_x_set(x_mins[ii]+epsilon, 10, n_points)
        if WF_pile is None:
            sfs_vals_tid = sim.sfs_dir_params(x_set, 1, S_dir_set[ii])
            sfs_vals_tia = sim.sfs_dir_params(x_set, 1, -S_dir_set[ii])
        else:
            S_dir = S_dir_set[ii]
            s_dir_comp = S_dir/(2*WF_pile["tenn_N"][0]) # Find s that gives same 2*Ne*s in tennessen traj
            if s_dir_comp >= np.max(WF_pile["s_set"]):
                sfs_vals_tid = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][-1, S_ud_0_ii])
                sfs_vals_tia = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][0, S_ud_0_ii])
            if s_dir_comp <= np.min(WF_pile["s_set"]):
                sfs_vals_tid = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][0, S_ud_0_ii])
                sfs_vals_tia = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][-1, S_ud_0_ii])
            else:
                s_ii_upper = np.argmax(WF_pile["s_set"] >= s_dir_comp)
                s_ii_lower = s_ii_upper - 1
                s_upper = WF_pile["s_set"][s_ii_upper]
                s_lower = WF_pile["s_set"][s_ii_lower]
                weight_lower = (s_upper - s_dir_comp)/(s_upper - s_lower)
                weight_upper = 1 - weight_lower
                sfs_upper = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, S_ud_0_ii])
                sfs_lower = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, S_ud_0_ii])
                sfs_vals_tid = weight_upper*sfs_upper + weight_lower*sfs_lower

                s_ii_upper = np.argmax(WF_pile["s_set"] >= -s_dir_comp)
                s_ii_lower = s_ii_upper - 1
                s_upper = WF_pile["s_set"][s_ii_upper]
                s_lower = WF_pile["s_set"][s_ii_lower]
                weight_lower = (s_upper + s_dir_comp)/(s_upper - s_lower) # + because flipped sign of s
                weight_upper = 1 - weight_lower
                sfs_upper = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, S_ud_0_ii])
                sfs_lower = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, S_ud_0_ii])
                sfs_vals_tia = weight_upper*sfs_upper + weight_lower*sfs_lower

        _zero_nan(sfs_vals_tid)
        _zero_nan(sfs_vals_tia)
        result[ii] = sample_x_sfs(x_set, sfs_vals_tid, sfs_vals_tia, pi_set[ii])
    return result

def sample_raf_stab(beta_true, x_mins, I2, Ne, pi=0.5, n_points=2000, epsilon=1e-8, WF_pile=None):
    if WF_pile is not None:
        S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
    S_ud_set = 2*Ne*beta_true**2*I2
    result = np.zeros_like(beta_true, dtype=np.float64)
    for ii, beta in enumerate(beta_true):
        x_set = smile_stats.adjusted_x_set(x_mins[ii]+epsilon, 10, n_points)
        if WF_pile is None:
            sfs_vals = sim.sfs_ud_params_sigma(x_set, 1, S_ud_set[ii])
        else:
            S_ud = S_ud_set[ii]
            s_ud_comp = S_ud/(2*WF_pile["tenn_N"][0])
            s_ud_wf = WF_pile["s_ud_set"]
            if np.min(s_ud_wf) < 0:
                s_ud_wf = -1*s_ud_wf
            if s_ud_comp >= np.max(s_ud_wf):
                sfs_vals = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][S_0_ii, -1])
            else:
                s_ud_ii_upper = np.argmax(s_ud_wf > s_ud_comp)
                s_ud_ii_lower = s_ud_ii_upper - 1
                s_ud_upper = s_ud_wf[s_ud_ii_upper]
                s_ud_lower = s_ud_wf[s_ud_ii_lower]
                weight_lower = (s_ud_upper - s_ud_comp)/(s_ud_upper - s_ud_lower)
                weight_upper = 1 - weight_lower
                sfs_upper = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][S_0_ii, s_ud_ii_upper])
                sfs_lower = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][S_0_ii, s_ud_ii_lower])
                sfs_vals = weight_upper*sfs_upper + weight_lower*sfs_lower
        _zero_nan(sfs_vals)
        result[ii] = sample_x_sfs(x_set, sfs_vals, sfs_vals, pi)
    return result

def sample_raf_full_db(beta_true, x_mins, I1, I2, Ne, n_points=2000, epsilon=1e-8, WF_pile=None):
    S_dir_set = 2*Ne*beta_true*I1
    pi_set = sim.pi_dir_db(S_dir_set)
    S_ud_set = 2*Ne*beta_true**2*I2
    result = np.zeros_like(beta_true, dtype=np.float64)
    for ii, beta in enumerate(beta_true):
        x_set = smile_stats.adjusted_x_set(x_mins[ii]+epsilon, 10, n_points)
        if WF_pile is None:
            sfs_vals_tid = sim.sfs_full_params_stable_sig(x_set, 1, S_dir_set[ii], S_ud_set[ii])
            sfs_vals_tia = sim.sfs_full_params_stable_sig(x_set, 1, -S_dir_set[ii], S_ud_set[ii])
        else:
            s_ud_wf = WF_pile["s_ud_set"]
            if np.min(s_ud_wf) < 0:
                s_ud_wf = -1*s_ud_wf

            S_dir = S_dir_set[ii]
            s_dir_comp = S_dir/(2*WF_pile["tenn_N"][0])

            S_ud = S_ud_set[ii]
            s_ud_comp = S_ud/(2*WF_pile["tenn_N"][0])

            if s_ud_comp >= np.max(s_ud_wf):
                s_ii_upper = np.argmax(WF_pile["s_set"] >= s_dir_comp)
                s_ii_lower = s_ii_upper - 1
                s_upper = WF_pile["s_set"][s_ii_upper]
                s_lower = WF_pile["s_set"][s_ii_lower]
                weight_upper = (s_upper - s_dir_comp)/(s_upper - s_lower)
                weight_lower = (s_dir_comp - s_lower)/(s_upper - s_lower)
                sfs_upper = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, -1])
                sfs_lower = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, -1])
                sfs_vals_tid = weight_upper*sfs_upper + weight_lower*sfs_lower

                s_ii_upper = np.argmax(WF_pile["s_set"] >= -s_dir_comp)
                s_ii_lower = s_ii_upper - 1
                s_upper = WF_pile["s_set"][s_ii_upper]
                s_lower = WF_pile["s_set"][s_ii_lower]
                weight_upper = (s_upper + s_dir_comp)/(s_upper - s_lower)
                weight_lower = (-s_dir_comp - s_lower)/(s_upper - s_lower)
                sfs_upper = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, -1])
                sfs_lower = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, -1])
                sfs_vals_tia = weight_upper*sfs_upper + weight_lower*sfs_lower
            elif (s_dir_comp >= np.max(WF_pile["s_set"])) or (s_dir_comp <= np.min(WF_pile["s_set"])):
                s_ud_ii_upper = np.argmax(s_ud_wf >= s_ud_comp)
                s_ud_ii_lower = s_ud_ii_upper - 1
                s_ud_upper = s_ud_wf[s_ud_ii_upper]
                s_ud_lower = s_ud_wf[s_ud_ii_lower]
                weight_upper = (s_ud_upper - s_ud_comp)/(s_ud_upper - s_ud_lower)
                weight_lower = (s_ud_comp - s_lower)/(s_ud_upper - s_ud_lower)
                sfs_upper = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][-1, s_ud_ii_upper])
                sfs_lower = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][-1, s_ud_ii_lower])
                sfs_vals_tid = weight_upper*sfs_upper + weight_lower*sfs_lower
                sfs_upper = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][0, s_ud_ii_upper])
                sfs_lower = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][0, s_ud_ii_lower])
                sfs_vals_tia = weight_upper*sfs_upper + weight_lower*sfs_lower
                if s_dir_comp <= np.min(WF_pile["s_set"]):
                    sfs_vals_tid, sfs_vals_tia = sfs_vals_tia, sfs_vals_tid
            else:
                s_ii_upper = np.argmax(WF_pile["s_set"] >= s_dir_comp)
                s_ii_lower = s_ii_upper - 1
                s_upper = WF_pile["s_set"][s_ii_upper]
                s_lower = WF_pile["s_set"][s_ii_lower]

                s_ud_ii_upper = np.argmax(s_ud_wf >= s_ud_comp)
                s_ud_ii_lower = s_ud_ii_upper - 1
                s_ud_upper = s_ud_wf[s_ud_ii_upper]
                s_ud_lower = s_ud_wf[s_ud_ii_lower]

                sfs_11 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, s_ud_ii_lower])
                sfs_21 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, s_ud_ii_lower])
                sfs_12 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, s_ud_ii_upper])
                sfs_22 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, s_ud_ii_upper])

                sfs_vals_tid = (s_upper - s_dir_comp)*(s_ud_upper - s_ud_comp)*sfs_11
                sfs_vals_tid += (s_dir_comp - s_lower)*(s_ud_upper - s_ud_comp)*sfs_21
                sfs_vals_tid += (s_upper - s_dir_comp)*(s_ud_comp - s_ud_lower)*sfs_12
                sfs_vals_tid += (s_dir_comp - s_lower)*(s_ud_comp - s_ud_lower)*sfs_22
                sfs_vals_tid /= (s_upper - s_lower)*(s_ud_upper - s_ud_lower)

                s_ii_upper = np.argmax(WF_pile["s_set"] >= -s_dir_comp)
                s_ii_lower = s_ii_upper - 1
                s_upper = WF_pile["s_set"][s_ii_upper]
                s_lower = WF_pile["s_set"][s_ii_lower]

                sfs_11 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, s_ud_ii_lower])
                sfs_21 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, s_ud_ii_lower])
                sfs_12 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_lower, s_ud_ii_upper])
                sfs_22 = np.interp(x_set, WF_pile["interp_x"], WF_pile["sfs_grid"][s_ii_upper, s_ud_ii_upper])

                sfs_vals_tia = (s_upper - s_dir_comp)*(s_ud_upper - s_ud_comp)*sfs_11
                sfs_vals_tia += (s_dir_comp - s_lower)*(s_ud_upper - s_ud_comp)*sfs_21
                sfs_vals_tia += (s_upper - s_dir_comp)*(s_ud_comp - s_ud_lower)*sfs_12
                sfs_vals_tia += (s_dir_comp - s_lower)*(s_ud_comp - s_ud_lower)*sfs_22
                sfs_vals_tia /= (s_upper - s_lower)*(s_ud_upper - s_ud_lower)

        _zero_nan(sfs_vals_tid)
        _zero_nan(sfs_vals_tia)
        result[ii] = sample_x_sfs(x_set, sfs_vals_tid, sfs_vals_tia, pi_set[ii])
    return result

def sample_raf_plei(beta_true, x_mins, Ip, Ne, pi=0.5, n_points=2000, epsilon=1e-8, n_s=200, WF_pile=None):
    S_ud_set = np.logspace(-3, 2.5, n_s)
    result = np.zeros_like(beta_true, dtype=np.float64)
    if WF_pile is not None:
        sfs_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile)
        _zero_nan(sfs_grid)
    for ii, beta in enumerate(beta_true):
        x_set = smile_stats.adjusted_x_set(x_mins[ii]+epsilon, 10, n_points)
        if WF_pile is None:
            sfs_grid = sim.sfs_ud_params_sigma(x_set[:,np.newaxis], 1, S_ud_set)
        sfs_int = np.trapz(sfs_grid * sim.levy_density(S_ud_set, 2*Ne*Ip*beta**2), S_ud_set)
        if WF_pile is None:
            lower_sfs = (sim.levy_cdf(np.abs(S_ud_set[0]), 2*Ne*Ip*beta**2)*
                         sim.sfs_ud_params_sigma(x_set, 1, S_ud_set[0]))
            upper_sfs = ((1-sim.levy_cdf(np.abs(S_ud_set[-1]), 2*Ne*Ip*beta**2))*
                         sim.sfs_ud_params_sigma(x_set, 1, S_ud_set[-1]))
        else:
            lower_sfs = (sim.levy_cdf(np.abs(S_ud_set[0]), 2*Ne*Ip*beta**2)*
                         sim.sfs_ud_WF_single(S_ud_set[0], WF_pile))
            upper_sfs = ((1-sim.levy_cdf(np.abs(S_ud_set[-1]), 2*Ne*Ip*beta**2))*
                         sim.sfs_ud_WF_single(S_ud_set[-1], WF_pile))
        _zero_nan(lower_sfs)
        _zero_nan(upper_sfs)
        sfs_vals = sfs_int + lower_sfs + upper_sfs

        if WF_pile is not None:
            sfs_vals = np.interp(x_set, WF_pile["interp_x"], sfs_vals)
        result[ii] = sample_x_sfs(x_set, sfs_vals, sfs_vals, pi)
    return result

def sample_raf_nplei(beta_true, x_mins, I2, nn, Ne, pi=0.5, n_points=2000, epsilon=1e-8, n_s=200, WF_pile=None):
    S_ud_set = np.logspace(-3, 2.5, n_s)
    result = np.zeros_like(beta_true, dtype=np.float64)
    if WF_pile is not None:
        sfs_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile)
        _zero_nan(sfs_grid)
    for ii, beta in enumerate(beta_true):
        x_set = smile_stats.adjusted_x_set(x_mins[ii]+epsilon, 10, n_points)
        if WF_pile is None:
            sfs_grid = sim.sfs_ud_params_sigma(x_set[:,np.newaxis], 1, np.abs(S_ud_set))
        ss_ud_set = np.abs(S_ud_set / (2*Ne))
        sfs_int = np.trapz(sfs_grid * sim.nplei_density(ss_ud_set, beta, I2, nn), ss_ud_set)
        if WF_pile is None:
            lower_sfs = (sim.nplei_cdf(ss_ud_set[0], beta, I2, nn)*
                         sim.sfs_ud_params_sigma(x_set, 1, S_ud_set[0]))
            upper_sfs = ((1-sim.nplei_cdf(ss_ud_set[-1], beta, I2, nn))*
                         sim.sfs_ud_params_sigma(x_set, 1, S_ud_set[-1]))
        else:
            lower_sfs = (sim.nplei_cdf(ss_ud_set[0], beta, I2, nn)*
                         sim.sfs_ud_WF_single(S_ud_set[0], WF_pile))
            upper_sfs = ((1-sim.nplei_cdf(ss_ud_set[-1], beta, I2, nn))*
                         sim.sfs_ud_WF_single(S_ud_set[-1], WF_pile))
        _zero_nan(lower_sfs)
        _zero_nan(upper_sfs)
        sfs_vals = sfs_int + lower_sfs + upper_sfs
        if WF_pile is not None:
            sfs_vals = np.interp(x_set, WF_pile["interp_x"], sfs_vals)
        result[ii] = sample_x_sfs(x_set, sfs_vals, sfs_vals, pi)
    return result

def extract_ash_prior(RDS_fname):
    base = importr("base")
    ashr = importr("ashr")
    gg = base.readRDS(RDS_fname)

    pi = np.array(gg[0])
    sd = np.array(gg[2])

    nonzero_pi = pi[(pi!=0) & (sd!=0)]
    nonzero_sd = sd[(pi!=0) & (sd!=0)]

    zero_pi = pi[sd==0]

    return {"zero_pi":zero_pi, "nonzero_pi":nonzero_pi, "nonzero_sd":nonzero_sd}

def make_prior_sample_t(nsamp, RDS_fname, df=1):
    ash = extract_ash_prior(RDS_fname)
    all_pi = np.concatenate(([ash["zero_pi"]], ash["nonzero_pi"]))
    all_sd = np.concatenate(([0], ash["nonzero_sd"]))
    entries = np.random.choice(a=np.arange(len(ash["nonzero_pi"])+1),
                               size=nsamp, replace=True,
                               p=all_pi)
    return stats.t.rvs(df=df, loc=0, sd=all_sd[entries])


def make_prior_sample(nsamp, RDS_fname):
    robjects.r('''
    library(\"ashr\")
    run_prior_sample <- function(nsamp, RDS.fname){
    gg <- readRDS(RDS.fname)
    mix.ids <- sample.int(n=length(gg$pi), prob=gg$pi, size=nsamp, replace=TRUE)
    result <- rnorm(n=nsamp, mean=gg$mean[mix.ids], sd=gg$sd[mix.ids])
    return(result)
    }
    ''')
    run_prior_sample = robjects.globalenv["run_prior_sample"]
    result = run_prior_sample(nsamp, RDS_fname)
    return np.array(result)

def generate_frequencies(sfs_func, x_set, nsamp, RDS_fname, beta_min):
    x_spaces = np.concatenate((x_set, [x_set[-1]])) - np.concatenate(([x_set[0]], x_set))
    x_weights = (x_spaces[:-1] + x_spaces[1:])/2

    min_sfs_weighted = sfs_func(beta_min)*x_weights

    result_beta = np.zeros(nsamp)
    result_x = np.zeros(nsamp)
    ii = 0
    def prior_freqs(ii):
        prior_betas = make_prior_sample(nsamp, RDS_fname)
        prior_betas = prior_betas[np.abs(prior_betas) > beta_min]
        for prior_beta in prior_betas:
            sfs_weighted = sfs_func(prior_beta)*x_weights
            if random.uniform(0,1) < (np.sum(sfs_weighted)/np.sum(min_sfs_weighted)):
                if ii < nsamp:
                    result_beta[ii] = prior_beta
                    result_x[ii] = random.choices(x_set, sfs_weighted)[0]
                    ii += 1
                else:
                    break
        return ii
    while ii < nsamp:
        ii = prior_freqs(ii)

    return result_beta, result_x

def sample_beta_x(RDS_fname, WF_pile, nsamp, sfs_func, obs_weight=False, df=None):
    x_set = WF_pile["interp_x"]
    x_spaces = np.concatenate((x_set, [x_set[-1]])) - np.concatenate(([x_set[0]], x_set))
    x_weights = (x_spaces[:-1] + x_spaces[1:])/2
    prior_betas = make_prior_sample_t(nsamp, RDS_fname, df=df) if df else make_prior_sample(nsamp, RDS_fname)
    sfs_grid = sfs_func(prior_betas)
    ## If we treat the distribution of betas as input from de novo mutations
    ## then we need to downsample based on the probability a variant makes it
    ## into the maf range. `sfs_func` should already truncate to this range.
    if obs_weight:
        max_weight = np.sum(sfs_func(np.array([0]))[:,0] * x_weights)
        samp_weights = np.sum(x_weights * sfs_grid, axis=0) / max_weight
        beta_unif = np.random.uniform(0, 1, len(prior_betas))
        prior_betas = prior_betas[samp_weights >= beta_unif]
        while len(prior_betas) < nsamp:
            prior_betas_add = make_prior_sample_t(nsamp, RDS_fname, df=df) if df else make_prior_sample(nsamp, RDS_fname)
            sfs_grid = sfs_func(prior_betas_add)
            samp_weights = np.sum(x_weights[:,None] * sfs_grid, axis=0) / max_weight
            beta_unif = np.random.uniform(0, 1, len(prior_betas_add))
            prior_betas = np.concatenate((prior_betas, prior_betas_add[samp_weights >= beta_unif]))
        prior_betas = prior_betas[0:nsamp]
        sfs_grid = sfs_func(prior_betas)

    sfs_grid_w = x_weights[:,None] * sfs_grid
    x_samp = np.zeros(sfs_grid.shape[1])
    for ii in range(sfs_grid.shape[1]):
        if sfs_grid_w[:,ii].sum() > 0:
            if sfs_grid_w[:,ii].sum() == 0:
                x_samp[ii] = 0
            else:
                probs = sfs_grid_w[:,ii]/sfs_grid_w[:,ii].sum()
                x_samp[ii] = np.random.choice(WF_pile["interp_x"], size=1, p=probs)[0]
    return prior_betas, x_samp

def add_GWAS_noise(beta_set, x_set, Neff):
    SEs = np.sqrt(1/(2*Neff*x_set*(1-x_set)))
    return stats.norm.rvs(loc=beta_set, scale=SEs), SEs

def make_full_sample(RDS_fname, WF_pile, nsamp, sfs_func, neff, v_cutoff, itersamp=10000, obs_weight=False, asc=True):
    beta_result = np.array([])
    hat_result = np.array([])
    xx_result = np.array([])
    se_result = np.array([])
    while len(beta_result) < nsamp:
        # print(RDS_fname, WF_pile, itersamp, sfs_func)
        beta, xx = sample_beta_x(RDS_fname, WF_pile, itersamp, sfs_func, obs_weight=obs_weight)
        beta = np.atleast_1d(beta[xx>0])
        xx = np.atleast_1d(xx[xx>0])
        if len(xx) == 0:
            print("Severe sampling difficulty: selection too strong")
        beta_hat, SEs = add_GWAS_noise(beta, xx, neff)
        keep_inds = 2*beta_hat**2*xx*(1-xx) >= v_cutoff if asc else 2*beta**2*xx*(1-xx) >= v_cutoff
        beta_result = np.concatenate((beta_result, beta[keep_inds]))
        hat_result = np.concatenate((hat_result, beta_hat[keep_inds]))
        xx_result = np.concatenate((xx_result, xx[keep_inds]))
        se_result = np.concatenate((se_result, SEs[keep_inds]))
    return (np.abs(beta_result[:nsamp]), np.abs(hat_result[:nsamp]),
            np.where(beta_result[:nsamp] > 0, xx_result[:nsamp], 1 - xx_result[:nsamp]), se_result[:nsamp])

def make_sfs_func_stab(I2, Ne, WF_pile):
    def sfs_func(prior_betas):
        S_ud_set = 2 * Ne * prior_betas**2 * I2
        sfs_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile)
        return 0.5 * (sfs_grid + np.flip(sfs_grid, axis=0))
    return sfs_func

def make_sfs_func_plei(Ip, Ne, WF_pile):
    def sfs_func(prior_betas):
        S_p_set = 2 * Ne * prior_betas**2 * Ip
        S_ud_set = stats.levy.rvs(loc=0, scale=np.abs(S_p_set))
        sfs_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile)
        return 0.5 * (sfs_grid + np.flip(sfs_grid, axis=0))
    return sfs_func

def make_sfs_func_dir(I1, Ne, WF_pile):
    def sfs_func(prior_betas):
        S_dir_set = 4 * Ne * prior_betas * I1
        pi_set = sim.pi_dir_db(S_dir_set)
        sfs_grid_tid = sim.sfs_dir_WF_grid(S_dir_set, WF_pile)
        sfs_grid_tia = sim.sfs_dir_WF_grid(-S_dir_set, WF_pile)
        sfs_grid = sfs_grid_tid * pi_set + np.flip(sfs_grid_tia * (1-pi_set), axis=0)
        return sfs_grid
    return sfs_func

def make_sfs_func_plei_(Ip, Ne, WF_pile, n_s=4000):
    S_ud_set = np.logspace(-3, 2.5, n_s)
    WF_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile)
    def sfs_func(prior_betas):
        S_p_set = 2 * Ne * prior_betas**2 * Ip
        sfs_grid = np.trapz(WF_grid[...,None] * stats.levy.pdf(S_ud_set[:,None], 0, S_p_set),
                            S_ud_set, axis=1)
        sfs_grid += WF_grid[:,0,None] * stats.levy.cdf(S_ud_set[0], 0, S_p_set)
        sfs_grid += WF_grid[:,-1,None] * stats.levy.sf(S_ud_set[-1], 0, S_p_set)
        return sfs_grid
    return sfs_func