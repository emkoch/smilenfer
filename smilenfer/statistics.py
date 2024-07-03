import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import pickle

from . import simulation as sim
from . import dfe as dfe

###
def calc_cutoffs(data, p_threshes=[1e-05, 1e-06, 1e-07, 5e-08, 1e-08, 1e-09], use_log=False, n_eff = None):
    if n_eff is None:
        n_eff = calc_n_eff(data.var_exp, data.pval, use_log)
    result = {}
    for p_thresh in p_threshes:
        result[str(p_thresh)] = stats.chi2.ppf(q=1-p_thresh, df=1)/n_eff
    return result

###
def calc_cutoffs_new(var_exp, pval, p_threshes=[1e-05, 1e-06, 1e-07, 5e-08, 1e-08, 1e-09],
                     use_log=False, n_eff = None):
    if n_eff is None:
        n_eff = calc_n_eff(var_exp, pval, use_log)
    result = {}
    for p_thresh in p_threshes:
        result[str(p_thresh)] = stats.chi2.ppf(q=1-p_thresh, df=1)/n_eff
    return result

###
def calc_n_eff(vv, pp, use_log=False, gws=False, gws_thresh=5e-08):
    vv = np.asarray(vv)
    pp = np.asarray(pp)
    valid_sites = ~np.isnan(vv) & np.array(vv>0)
    vv = vv[valid_sites]
    pp = pp[valid_sites]
    if gws:
        gws_loci = pp <= gws_thresh
        vv = vv[gws_loci]
        pp = pp[gws_loci]
    inv_vals = stats.chi2.ppf(q=1-np.array(pp), df=1)
    vals_keep = ~np.isinf(inv_vals)
    if use_log:
        return np.mean(np.log(inv_vals[vals_keep]) - np.log(vv[vals_keep]))
    else:
        fit_nolog = LinearRegression(fit_intercept=False).fit(inv_vals[vals_keep, np.newaxis], vv[vals_keep])
        return 1/fit_nolog.coef_[0]
    
###
def discov_x(beta, vv):
    """
    Calculate the minimum frequency to discover a variant
    with effect size beta given variance cutoff vv."""
    return (beta**2. - np.sqrt(beta**4. - 2*beta**2. * vv))/(2*beta**2.)

###
def beta_cutoff_sd(xx, vv, neff):
    beta_cutoff = np.sqrt(vv/(2*xx*(1-xx)))
    beta_sd = np.sqrt(1/(2*neff*xx*(1-xx)))
    return beta_cutoff, beta_sd

###
def trad_x_set(min_x, n_points):
    return 1/(1+np.exp(-np.linspace(np.log(min_x/(1-min_x)), np.log((1-min_x)/min_x), n_points)))

###
def adjusted_x_set(min_x, min_z, n_points):
    min_z = -np.abs(min_z)
    LL = (1-2*min_x)*(1+np.exp(min_z))/(1-np.exp(min_z))
    CC = 1 - min_x - (1 - 2*min_x)/(1-np.exp(min_z))
    z_set = np.linspace(min_z, -min_z, n_points)
    return LL/(1 + np.exp(-z_set)) + CC

###
def freq_grid_upper_lower(x_data, x_grid):
    x_grid = np.round(x_grid, 10)
    x_data = np.round(x_data, 10)
    x_data_flip = np.round(1-x_data, 10)
    max_grid = np.nanmax(x_grid)
    min_grid = np.min(x_grid)
    max_data = np.nanmax(np.maximum(x_data, x_data_flip))
    min_data = np.min(np.minimum(x_data, x_data_flip))

    assert max_grid >= max_data, "frequency data outside of grid"
    assert min_grid <= min_data, "frequency data outside of grid"

    xx_low_set_tid = [x_grid[x_grid < xx][-1] if xx not in x_grid else xx for xx in x_data]
    xx_low_set_tia = [x_grid[x_grid < xx][-1] if xx not in x_grid else xx for xx in x_data_flip]
    xx_high_set_tid = [x_grid[x_grid > xx][0] if xx not in x_grid else xx for xx in x_data]
    xx_high_set_tia = [x_grid[x_grid > xx][0] if xx not in x_grid else xx for xx in x_data_flip]

    weight_low_set_tid = [(xx-xx_low_set_tid[ii])/(xx_high_set_tid[ii]-xx_low_set_tid[ii])
                          if xx not in x_grid else 0.5 for ii, xx in enumerate(x_data)]
    weight_low_set_tia = [(xx-xx_low_set_tia[ii])/(xx_high_set_tia[ii]-xx_low_set_tia[ii])
                          if xx not in x_grid else 0.5 for ii, xx in enumerate(x_data_flip)]

    index_low_set_tid = [np.where(x_grid == xx_low_set_tid[ii])[0][0] for ii, _ in enumerate(x_data)]
    index_low_set_tia = [np.where(x_grid == xx_low_set_tia[ii])[0][0] for ii, _ in enumerate(x_data_flip)]
    index_high_set_tid = [np.where(x_grid == xx_high_set_tid[ii])[0][0] for ii, _ in enumerate(x_data)]
    index_high_set_tia = [np.where(x_grid == xx_high_set_tia[ii])[0][0] for ii, _ in enumerate(x_data_flip)]

    return (xx_low_set_tid, xx_low_set_tia,
            xx_high_set_tid, xx_high_set_tia,
            weight_low_set_tid, weight_low_set_tia,
            index_low_set_tid, index_low_set_tia,
            index_high_set_tid, index_high_set_tia)

#TODO: remove xx since unused in current version
###
def tau_x_int(tau_grid, x_set, xx, beta, vv, min_x=0, asc_probs=None):
    d_x = np.maximum(discov_x(beta, vv), min_x) if asc_probs is None else min_x
    obs_x = (x_set >= d_x) & (x_set <= (1-d_x))
    tau_tmp = tau_grid[:,obs_x]
    residual_lower = [np.interp(d_x, x_set, tau_grid[ii,:]) for ii in range(tau_grid.shape[0])]
    residual_upper = [np.interp(1-d_x, x_set, tau_grid[ii,:]) for ii in range(tau_grid.shape[0])]
    # Concatenate the residuals to the tau tmp grid
    tau_tmp = np.concatenate((np.array(residual_lower)[:,None], tau_tmp, 
                              np.array(residual_upper)[:,None]), axis=1)
    x_tmp = np.concatenate(([d_x], x_set[obs_x], [1-d_x]))
    # Integrate over the tau tmp grid using trapz
    if asc_probs is None:
        result = np.trapz(tau_tmp, x_tmp, axis=1)
    else:
        # Interpolate asc_probs to x_tmp
        asc_probs_tmp = np.interp(x_tmp, x_set, asc_probs)
        result = np.trapz(tau_tmp*asc_probs_tmp, x_tmp, axis=1)

    return result

###
def max_S_ud_setup(min_x, ee):
    d_x = np.linspace(min_x, 0.49, 200)
    x_sets = [1/(1+np.exp(-np.linspace(np.log(d_x_i/(1-d_x_i)), np.log((1-d_x_i)/d_x_i), 2000)))
              for d_x_i in d_x]
    max_SS_ud = np.zeros(len(x_sets))
    for ii, x_set in enumerate(x_sets):
        SS_ud = 0
        ZZ = np.trapz(sim.sfs_full_params_stable_sig(x_set, 1, 0, SS_ud), x_set)
        while ZZ > ee:
            SS_ud -= 1
            ZZ = np.trapz(sim.sfs_full_params_stable_sig(x_set, 1, 0, SS_ud), x_set)
        max_SS_ud[ii] = SS_ud
    return d_x, max_SS_ud

###
def max_S_ud(min_x, ee, deg=10):
    d_x, max_SS_ud = max_S_ud_setup(min_x, ee)
    fit_coeffs = np.polyfit(d_x, np.log10(-max_SS_ud), deg=deg)
    return fit_coeffs

###
def quick_full_ZZ_WF_grid(x_set, S_set, S_ud_set, WF_pile, asc_probs=None):
    sfs_grid = sim.sfs_full_WF_grid(S_set, S_ud_set, WF_pile, x_set)
    return (np.trapz(sfs_grid*asc_probs[:,np.newaxis], x_set, axis=0)
            if (asc_probs is not None)
            else np.trapz(sfs_grid, x_set, axis=0))

###
def quick_full_ZZ(x_set, S_dir_set, S_ud_set, neut_val, max_S, min_S):
    result = np.zeros_like(S_dir_set)
    for ii in range(len(S_dir_set)):
        if (2*S_dir_set[ii] - np.abs(S_ud_set[ii])) > max_S:
            if (np.abs(S_dir_set[ii]) < min_S) & (np.abs(S_ud_set[ii]) < min_S):
                result[ii] = neut_val
            else:
                result[ii] = np.trapz(sim.sfs_full_params_stable_sig(x_set, 1, S_dir_set[ii],
                                                                     -np.abs(S_ud_set[ii])), x_set)
    return result

###
def quick_dir_ZZ_WF_grid(x_set, S_set, WF_pile, asc_probs=None):
    sfs_grid = sim.sfs_dir_WF_grid(S_set, WF_pile, x_set)
    return (np.trapz(sfs_grid*asc_probs[:,np.newaxis], x_set, axis=0) if (asc_probs is not None) else
            np.trapz(sfs_grid, x_set, axis=0))

###
def quick_dir_ZZ(x_set, S_dir_set, neut_val, max_S, min_S, asc_probs=None):
    result = np.zeros_like(S_dir_set)
    for ii in range(len(S_dir_set)):
        if 2*S_dir_set[ii] > max_S:
            if np.abs(S_dir_set[ii]) < min_S:
                result[ii] = neut_val
            else:
                result[ii] = (np.trapz(sim.sfs_dir_params(x_set, 1, S_dir_set[ii])*asc_probs, x_set)
                              if (asc_probs is not None) else
                              np.trapz(sim.sfs_dir_params(x_set, 1, S_dir_set[ii]), x_set))
    return result

###
def quick_stab_ZZ_WF_grid(x_set, S_ud_set, WF_pile, asc_probs=None):
    sfs_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, x_set)
    return (np.trapz(sfs_grid*asc_probs[:,np.newaxis], x_set, axis=0) if (asc_probs is not None) else
            np.trapz(sfs_grid, x_set, axis=0))

###
def quick_stab_ZZ(x_set, S_ud_set, neut_val, max_S, min_S, asc_probs=None):
    result = np.zeros_like(S_ud_set)
    for ii in range(len(S_ud_set)):
        if np.abs(2*S_ud_set[ii]) < np.abs(max_S):
            if np.abs(S_ud_set[ii]) < min_S:
                result[ii] = neut_val
            else:
                result[ii] = (np.trapz(sim.sfs_ud_params_sigma(x_set, 1, np.abs(S_ud_set[ii]))*asc_probs, x_set)
                              if (asc_probs is not None) else
                              np.trapz(sim.sfs_ud_params_sigma(x_set, 1, np.abs(S_ud_set[ii])), x_set))
    return result

###
def quick_plei_ZZ(x_set, S_p_set, filt_tau_x_int, S_ud_set, ssd=False, Ne=10000):
    S_ud_set = np.abs(S_ud_set)
    S_p_set = np.abs(S_p_set)
    result = np.zeros_like(S_p_set)
    for ii in range(len(S_p_set)):
        if ssd:
            S_ud_density, S_ud_lower_mass, S_ud_upper_mass = dfe.simons_ssd_posterior_density(S_p_set[ii], Ne, S_ud_set)
        else:
            S_ud_density = sim.levy_density(S_ud_set, S_p_set[ii])
            S_ud_lower_mass = sim.levy_cdf(S_ud_set[0], S_p_set[ii])
            S_ud_upper_mass = 1 - sim.levy_cdf(S_ud_set[-1], S_p_set[ii])
        ## Integrate over possible S_ud values for each S_p value
        result[ii] = (np.trapz(filt_tau_x_int*S_ud_density, S_ud_set) + 
                      filt_tau_x_int[0]*S_ud_lower_mass + 
                      filt_tau_x_int[-1]*S_ud_upper_mass)
    return result

###
def quick_nplei_ZZ(x_set, I2_set, nn_set, filt_tau_x_int, ss_set, beta):
    result = np.zeros_like(I2_set)
    for ii in range(len(I2_set)):
        if beta == 0:
            result[ii] = filt_tau_x_int[0]
        else:
            result[ii] = np.trapz(filt_tau_x_int*
                                  sim.nplei_density(np.abs(ss_set), beta, np.abs(I2_set[ii]), nn_set[ii]),
                                  np.abs(ss_set))
            result[ii] += (filt_tau_x_int[0]*
                           sim.nplei_cdf(np.abs(ss_set[0]), beta, np.abs(I2_set[ii]), nn_set[ii]) +
                           filt_tau_x_int[-1]*
                           (1-sim.nplei_cdf(np.abs(ss_set[-1]), beta, np.abs(I2_set[ii]), nn_set[ii])))
    return result

###
def quick_plei_num(S_p_set, tau_grid, weight_low, index_low, index_high, S_ud_set, ssd=False, Ne=10000):
    S_ud_set = np.abs(S_ud_set)
    S_p_set = np.abs(S_p_set)
    avg_grid = weight_low*tau_grid[:, index_low] + (1-weight_low)*tau_grid[:,index_high]
    result = np.zeros_like(S_p_set)
    for ii in range(len(S_p_set)):
        if ssd:
            S_ud_density, S_ud_lower_mass, S_ud_upper_mass = dfe.simons_ssd_posterior_density(S_p_set[ii], Ne, S_ud_set)
        else:
            S_ud_density = sim.levy_density(S_ud_set, S_p_set[ii])
            S_ud_lower_mass = sim.levy_cdf(S_ud_set[0], S_p_set[ii])
            S_ud_upper_mass = 1 - sim.levy_cdf(S_ud_set[-1], S_p_set[ii])

        result[ii] = np.trapz(avg_grid*S_ud_density, S_ud_set) + S_ud_lower_mass*avg_grid[0] + S_ud_upper_mass*avg_grid[-1]
    return result

###
def quick_nplei_num(I2_set, nn_set, tau_grid, weight_low, index_low, index_high, ss_set, beta):
    result = np.zeros_like(I2_set)
    for ii in range(len(I2_set)):
        if beta == 0:
            result[ii] = weight_low*tau_grid[0, index_low] + (1-weight_low)*tau_grid[0, index_high]
        else:
            result[ii] = np.trapz((weight_low*tau_grid[:, index_low] +
                                   (1-weight_low)*tau_grid[:,index_high])*
                                  sim.nplei_density(np.abs(ss_set), beta, np.abs(I2_set[ii]), nn_set[ii]),
                                  np.abs(ss_set))
            result[ii] += ((weight_low*tau_grid[0, index_low] +
                            (1-weight_low)*tau_grid[0, index_high]) *
                           sim.nplei_cdf(np.abs(ss_set[0]), beta, np.abs(I2_set[ii]), nn_set[ii]))
            result[ii] += ((weight_low*tau_grid[-1, index_low] +
                            (1-weight_low)*tau_grid[-1, index_high]) *
                           (1-sim.nplei_cdf(np.abs(ss_set[-1]), beta, np.abs(I2_set[ii]), nn_set[ii])))
    return result

###
def neut_llhood(xx, beta, vv, Ne, pi, n_x=2000, min_x=0, neff=None):
    neut_ss = 0.001/Ne
    d_x = np.maximum(discov_x(beta, vv), min_x) if neff == None else min_x
    # sfs if trait-increasing allele is derived
    if neff == None:
        numer_tid = pi*sim.sfs_del_params(xx=xx, theta=1, ss=neut_ss, Ne=Ne)
    else:
        beta_cutoff = np.sqrt(vv/(2*xx*(1-xx)))
        beta_sd = np.sqrt(1/(2*neff*xx*(1-xx)))
        numer_tid = (pi*sim.sfs_del_params(xx=xx, theta=1, ss=neut_ss, Ne=Ne) *
                     stats.norm.sf(x=beta_cutoff, loc=beta, scale=beta_sd))
    # sfs if trait-increasing allele is ancestral
    if neff == None:
        numer_tia = (1-pi)*sim.sfs_del_params(xx=1-xx, theta=1, ss=neut_ss, Ne=Ne)
    else:
        numer_tia = ((1-pi)*sim.sfs_del_params(xx=1-xx, theta=1, ss=neut_ss, Ne=Ne) *
                     stats.norm.sf(x=beta_cutoff, loc=beta, scale=beta_sd))
    # evenly space frequencies on logit scale
    x_set = 1/(1+np.exp(-np.linspace(np.log(d_x/(1-d_x)), np.log((1-d_x)/d_x), n_x)))
    if neff == None:
        denom_tid = np.trapz(sim.sfs_del_params(xx=x_set, theta=1, ss=neut_ss, Ne=Ne), x_set)
        denom_tia = np.trapz(sim.sfs_del_params(xx=1-x_set, theta=1, ss=neut_ss, Ne=Ne), x_set)
    else:
        beta_cutoffs = np.sqrt(vv/(2*x_set*(1-x_set)))
        beta_sds = np.sqrt(1/(2*neff*x_set*(1-x_set)))
        denom_tid = np.trapz(sim.sfs_del_params(xx=x_set, theta=1, ss=neut_ss, Ne=Ne)*
                             stats.norm.sf(x=beta_cutoffs, loc=beta, scale=beta_sds), x_set)
        denom_tia = np.trapz(sim.sfs_del_params(xx=1-x_set, theta=1, ss=neut_ss, Ne=Ne)*
                             stats.norm.sf(x=beta_cutoffs, loc=beta, scale=beta_sds), x_set)
    return np.log(np.where(np.isnan(numer_tid/denom_tid), 0, numer_tid/denom_tid) +
                  np.where(np.isnan(numer_tia/denom_tia), 0, numer_tia/denom_tia))

###
def neut_llhood_WF(xx, beta, vv, Ne, pi, WF_pile, n_x=2000, min_x=0, neff=None):
    S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
    S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
    sfs_neut = WF_pile["sfs_grid"][S_0_ii, S_ud_0_ii]
    d_x = np.maximum(discov_x(beta, vv), min_x) if neff == None else min_x
    # sfs if trait-increasing allele is derived
    if neff == None:
        numer_tid = pi*np.interp(xx, WF_pile["interp_x"], sfs_neut)
    else:
        beta_cutoff = np.sqrt(vv/(2*xx*(1-xx)))
        beta_sd = np.sqrt(1/(2*neff*xx*(1-xx)))
        numer_tid = (pi*np.interp(xx, WF_pile["interp_x"], sfs_neut)*
                     stats.norm.sf(x=beta_cutoff, loc=beta, scale=beta_sd))
    # sfs if trait-increasing allele is ancestral
    if neff == None:
        numer_tia = (1-pi)*np.interp(1-xx, WF_pile["interp_x"], sfs_neut)
    else:
        numer_tia = ((1-pi)*np.interp(1-xx, WF_pile["interp_x"], sfs_neut)*
                     stats.norm.sf(x=beta_cutoff, loc=beta, scale=beta_sd))
    # evenly space frequencies on logit scale
    x_set = 1/(1+np.exp(-np.linspace(np.log(d_x/(1-d_x)), np.log((1-d_x)/d_x), n_x)))

    if neff == None:
        denom_tid = np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut), x_set)
        denom_tia = denom_tid
    else:
        beta_cutoffs = np.sqrt(vv/(2*x_set*(1-x_set)))
        beta_sds = np.sqrt(1/(2*neff*x_set*(1-x_set)))
        denom_tid = np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut)*
                             stats.norm.sf(x=beta_cutoffs, loc=beta, scale=beta_sds), x_set)
        denom_tia = denom_tid

    return np.log(np.where(np.isnan(numer_tid/denom_tid), 0, numer_tid/denom_tid) +
                  np.where(np.isnan(numer_tia/denom_tia), 0, numer_tia/denom_tia))

###
def data_llhood_neut(xx_data, beta_data, vv, Ne, pi, n_x=2000, min_x=0.01, WF_pile=None, neff=None, neff_set=None):
    """Calculate the total log-likelihood under neutrality for a set of
    observed effects sizes and frequencies."""
    try:
        vv_size = len(vv)
        assert len(xx_data) == vv_size
        vv_set = vv
    except TypeError:
        vv_size = 1
        vv_set = vv * np.ones_like(xx_data)
    
    result = np.array([0.])
    if WF_pile is None:
        if neff_set is not None:
            for ii in range(xx_data.shape[0]):
                result += neut_llhood(xx_data[ii], beta_data[ii], vv, Ne, pi, n_x, min_x=min_x, neff=neff_set[ii])
        else:
            for ii in range(xx_data.shape[0]):
                result += neut_llhood(xx_data[ii], beta_data[ii], vv_set[ii], Ne, pi, n_x, min_x=min_x, neff=neff)     
    else:
        if neff_set is not None:
            for ii in range(xx_data.shape[0]):
                result += neut_llhood_WF(xx_data[ii], beta_data[ii], vv, Ne, pi,
                                        WF_pile=WF_pile, n_x=n_x, min_x=min_x, neff=neff_set[ii])
        else:
            for ii in range(xx_data.shape[0]):
                result += neut_llhood_WF(xx_data[ii], beta_data[ii], vv_set[ii], Ne, pi,
                                        WF_pile=WF_pile, n_x=n_x, min_x=min_x, neff=neff)
    return result

###
def get_weights(param_vals):
    hangover_1 = param_vals[1] - param_vals[0]
    hangover_2 = param_vals[-1] - param_vals[-2]
    return (np.diff(param_vals, prepend=param_vals[0] - hangover_1) +
            np.diff(param_vals, append=param_vals[-1] + hangover_2))/2

###
def get_pi_weights(pi_vals):
    pi_diffs = np.diff(pi_vals)
    return (np.concatenate(([0], pi_diffs)) + np.concatenate((pi_diffs, [0])))/2

###
def choose_param_range(beta_data, n_points, Ne=10000, S_dir_max=1000, S_ud_max=1000,
                       S_dir_min=0.01, S_ud_min=0.01, p_min=0.99, p_max=0.01, nn_max=50):
    beta_data = np.array(beta_data)
    nonzero_beta = beta_data[beta_data > 0]
    if nonzero_beta.size == 0:
        nonzero_min = 1e-4
        nonzero_max = 1e-3
    else:
        nonzero_min = np.min(beta_data[beta_data > 0])
        nonzero_max = np.max(beta_data[beta_data > 0])
    assert nonzero_max > nonzero_min
    I1_max = S_dir_max/(2*Ne*nonzero_min)
    I1_min = S_dir_min/(2*Ne*nonzero_max)
    I2_max = S_ud_max/(2*Ne*nonzero_min**2)
    I2_min = S_ud_min/(2*Ne*nonzero_max**2)
    Ip_max = S_ud_max/(2*Ne*nonzero_min**2)*stats.norm.ppf(1-p_max/2)**2
    Ip_min = S_ud_min/(2*Ne*nonzero_max**2)*stats.norm.ppf(1-p_min/2)**2
    I2_nn_max = Ip_max
    I2_nn_min = Ip_min/nn_max
    return (-np.logspace(np.log10(I1_max), np.log10(I1_min), n_points),
            -np.logspace(np.log10(I2_max), np.log10(I2_min), n_points),
            np.logspace(np.log10(Ip_min), np.log10(Ip_max), n_points),
            np.logspace(np.log10(I2_nn_min), np.log10(I2_nn_max), n_points),
            np.linspace(1.01, nn_max, n_points))

###
def llhood_post_neut(x_data, beta_data, v_cutoff, Ne, min_x=0.01, WF_pile=None):
    pointwise_llhood = np.array([data_llhood_neut(np.array([raf]), np.array([beta_data[ii]]),
                                                          v_cutoff, Ne, pi=0.5,
                                                          min_x=min_x, WF_pile=WF_pile)[0]
                                 for ii, raf in enumerate(x_data)])
    return pointwise_llhood

###
def llhood_grid_neut_setup(x_data, beta_data, v_cutoff, pi_set, Ne, min_x=0.01, beta_obs=None,
                           return_ZZ=False, return_ZZ_0=False, WF_pile=None, neff=None, n_x=200):
    if beta_obs is None:
        if WF_pile is None:
            llhood_surface = np.array([data_llhood_neut(x_data, beta_data,
                                                                v_cutoff, Ne, pi, min_x=min_x, neff=neff, n_x=n_x)[0]
                                       for pi in  pi_set])
        else:
            llhood_surface = np.array([data_llhood_neut(x_data, beta_data,
                                                                v_cutoff, Ne, pi,
                                                                min_x=min_x, WF_pile=WF_pile, neff=neff, n_x=n_x)[0]
                                       for pi in  pi_set])
    else:
        beta_obs = beta_data if neff != None else beta_obs
        ## True betas don't even matter in the absence of selection, only observation
        if WF_pile is None:
            llhood_surface = np.array([data_llhood_neut(x_data, beta_obs,
                                                                v_cutoff, Ne, pi, min_x=min_x, neff=neff, n_x=n_x)[0]
                                       for pi in  pi_set])
        else:
            llhood_surface = np.array([data_llhood_neut(x_data, beta_obs,
                                                                v_cutoff, Ne, pi,
                                                                min_x=min_x, WF_pile=WF_pile, neff=neff, n_x=n_x)[0]
                                       for pi in  pi_set])

    pi_diffs = np.diff(pi_set)
    pi_weights = (np.concatenate(([0], pi_diffs)) + np.concatenate((pi_diffs, [0])))/2
    ## Under neutrality Z values won't actually be used.
    ## The following is bad to do, because they aren't actually all 1, but used for laziness.
    if return_ZZ:
        log_ZZ_vals = np.zeros_like(llhood_surface)
        if return_ZZ_0:
            log_ZZ_0_vals = np.zeros_like(llhood_surface)
            return llhood_surface, log_ZZ_vals, log_ZZ_0_vals, pi_set, pi_weights
        return llhood_surface, log_ZZ_vals, pi_set, pi_weights
    return llhood_surface, pi_set, pi_weights

###
def tau_int_fast(tau_grid, global_x_set, x_data, beta_data, v_cutoff, min_x=0, asc_prob_grid=None):
    try:
        v_size = len(v_cutoff)
        assert len(x_data) == v_size
        v_set = v_cutoff
    except TypeError:
        v_size = 1
        v_set = v_cutoff * np.ones_like(x_data)

    result = [np.zeros(tau_grid.shape[0]) for ii in range(len(x_data))]
    for ii in range(len(x_data)):
        if asc_prob_grid is not None:
            result[ii] = tau_x_int(tau_grid, global_x_set, x_data[ii], beta_data[ii], v_set[ii],
                                           min_x=min_x, asc_probs=asc_prob_grid[ii,:])
        else:
            result[ii] = tau_x_int(tau_grid, global_x_set, x_data[ii], beta_data[ii], v_set[ii], 
                                           min_x=min_x)
    return result

###
def llhood_post_plei(x_data, beta_data, v_cutoff, Ip, Ne,
                     min_x=0.01, neut_min=0.01, n_x=4000, n_s=4000, beta_obs=None, WF_pile=None):
    ## All possible x values which variants might be discovered at
    global_x_set = trad_x_set(min_x, n_points=n_x)

    S_ud_set = np.logspace(-3, 2.5, n_s)
    if WF_pile is None:
        tau_grid = sim.sfs_ud_params_sigma(global_x_set, 1, -S_ud_set[:,np.newaxis])
    else:
        tau_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, x_set = global_x_set).T
    if beta_obs is None:
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_data, v_cutoff)
    else:
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_obs, v_cutoff)

    (xx_low_set_tid, xx_low_set_tia,   # unused
     xx_high_set_tid, xx_high_set_tia, # unused
     weight_low_set_tid, weight_low_set_tia,
     index_low_set_tid, index_low_set_tia,
     index_high_set_tid, index_high_set_tia) = freq_grid_upper_lower(x_data, global_x_set)

    pointwise_llhood = np.zeros_like(x_data)

    for ii in range(len(beta_data)):
        S_p_set = np.array([2*Ne*beta_data[ii]**2 * Ip]*2)
        ZZ = quick_plei_ZZ(global_x_set, S_p_set, filt_tau_x_int_set[ii], S_ud_set)
        num_tid =  quick_plei_num(S_p_set, tau_grid, weight_low_set_tid[ii],
                                  index_low_set_tid[ii], index_high_set_tid[ii],  S_ud_set)
        num_tia =  quick_plei_num(S_p_set, tau_grid, weight_low_set_tia[ii],
                                  index_low_set_tia[ii], index_high_set_tia[ii], S_ud_set)

        probs_tid = np.where(ZZ==0, 0, num_tid/ZZ)
        probs_tia = np.where(ZZ==0, 0, num_tia/ZZ)
        pointwise_llhood[ii] = np.log(probs_tid * 0.5 + probs_tia * 0.5)[0]
    return pointwise_llhood

###
def llhood_grid_plei_setup(x_data, beta_data, v_cutoff, Ip_set, pi_set, Ne,
                           min_x=0.01, neut_min=0.01, n_x=4000, n_s=4000, beta_obs=None,
                           return_ZZ=False, return_ZZ_0=False, min_x_ash=None, db=False, WF_pile=None,
                           verbose=False, neff=None, ssd=False):
    # Set up grids for likelihood calculations based on various settings
    Ip_weights = get_weights(Ip_set)
    if not db:
        pi_weights = get_pi_weights(pi_set)
        param_volume = Ip_weights[:, np.newaxis] * pi_weights

        llhood_surface = np.zeros((len(Ip_set), len(pi_set)))
        if return_ZZ:
            ZZ_surface = np.zeros((len(Ip_set), len(pi_set)))
            if return_ZZ_0:
                ZZ_0_surface = np.zeros((len(Ip_set), len(pi_set)))
    else:
        llhood_surface = np.zeros(len(Ip_set))
        if return_ZZ:
            ZZ_surface = np.zeros(len(Ip_set))
            if return_ZZ_0:
                ZZ_0_surface = np.zeros(len(Ip_set))

    ## All possible x values which variants might be discovered at
    global_x_set = trad_x_set(min_x, n_points=n_x)
    if min_x_ash is not None:
        global_x_set_ash = trad_x_set(min_x_ash, n_points=n_x)

    if neff != None:
        beta_cutoff, beta_sd = beta_cutoff_sd(x_data, v_cutoff, neff)
        asc_probs = (stats.norm.sf(x=beta_cutoff, loc=np.abs(beta_data), scale=beta_sd) +
                     stats.norm.cdf(x=-beta_cutoff, loc=np.abs(beta_data), scale=beta_sd))
        beta_cutoff_global, beta_sd_global = beta_cutoff_sd(global_x_set, v_cutoff, neff)
        # dim asc_probs_global = (len(beta_data), len(global_x_set))
        asc_probs_global = (stats.norm.sf(x=beta_cutoff_global, loc=np.abs(beta_data[:,np.newaxis]),
                                          scale=beta_sd_global) +
                            stats.norm.cdf(x=-beta_cutoff_global, loc=np.abs(beta_data[:,np.newaxis]),
                                           scale=beta_sd_global))
    else:
        asc_probs = None
        asc_probs_global = None
        asc_probs_globa_ash = None

    S_ud_set = np.logspace(-2, 3, n_s)
    if verbose: print("making tau grid...")
    if WF_pile is None:
        tau_grid = sim.sfs_ud_params_sigma(global_x_set, 1, -S_ud_set[:,np.newaxis])
    else:
        tau_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, x_set = global_x_set).T

    if min_x_ash is not None:
        if WF_pile is None:
            tau_grid_ash = sim.sfs_ud_params_sigma(global_x_set_ash, 1, -S_ud_set[:,np.newaxis])
        else:
            tau_grid_ash = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, x_set = global_x_set_ash).T
    # shape tau_grid: (n_S, n_x)
    if verbose: print("doing tau integration...")
    if beta_obs is None:
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_data,
                                          v_cutoff, min_x=min_x, asc_prob_grid=asc_probs_global)
    else:
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_obs,
                                          v_cutoff, min_x=min_x, asc_prob_grid=asc_probs_global)

    if return_ZZ_0:
        #TODO: Check that this is actually working correctly generally
        #TODO: What to do with ascertainment probabilities here?
        ## xx does nothing here
        ## beta doesn't matter because we are using cutoff at zero
        ## Will just integrate over global_x_set for each S_ud value, this doesn't depend on individual beta.
        ## Individual beta will matter when we have different weightings for S_ud based on the prob. density.
        if min_x_ash is None:
            full_tau_x_int = tau_x_int(tau_grid, global_x_set, xx=0, beta=0.1, vv=0)
        else:
            full_tau_x_int = tau_x_int(tau_grid_ash, global_x_set_ash, xx=0, beta=0.1, vv=0)

    ## We have precomputed tau for a grid of x and S_ud to interpolate to observed x values
    ## calculate the lower (low) and upper (high) x values in the grid that flank each observed x

    (_, _, _, _,
     weight_low_set_tid, weight_low_set_tia,
     index_low_set_tid, index_low_set_tia,
     index_high_set_tid, index_high_set_tia) = freq_grid_upper_lower(x_data, global_x_set)

    if verbose: print("real stuff...")
    for ii in range(len(beta_data)):
        if neff != None:
            tau_grid_asc = tau_grid * asc_probs_global[ii,:]
        else:
            tau_grid_asc = tau_grid
        if verbose: print(ii, end=".")
        S_p_set = 2*Ne*beta_data[ii]**2 * Ip_set
        ZZ = quick_plei_ZZ(global_x_set, S_p_set, filt_tau_x_int_set[ii], S_ud_set, ssd=ssd, Ne=Ne)
        if return_ZZ_0:
            if min_x_ash is not None:
                ZZ_0 = quick_plei_ZZ(global_x_set_ash, S_p_set, full_tau_x_int, S_ud_set)
            else:
                ZZ_0 = quick_plei_ZZ(global_x_set, S_p_set, full_tau_x_int, S_ud_set)
        num_tid =  quick_plei_num(S_p_set, tau_grid_asc, weight_low_set_tid[ii],
                                  index_low_set_tid[ii], index_high_set_tid[ii], S_ud_set, ssd=ssd, Ne=Ne)
        num_tia =  quick_plei_num(S_p_set, tau_grid_asc, weight_low_set_tia[ii],
                                  index_low_set_tia[ii], index_high_set_tia[ii], S_ud_set, ssd=ssd, Ne=Ne)

        if not db:
            probs_tid = np.where(ZZ==0, 0, num_tid/ZZ)[:,np.newaxis]
            probs_tia = np.where(ZZ==0, 0, num_tia/ZZ)[:,np.newaxis]
            llhood_surface += np.log(probs_tid * pi_set + probs_tia * (1-pi_set))
            if return_ZZ:
                ZZ_surface += np.log(ZZ)[:, np.newaxis] * np.ones_like(pi_set)
                if return_ZZ_0:
                    ZZ_0_surface += np.log(ZZ_0)[:, np.newaxis] * np.ones_like(pi_set)
        else:
            llhood_surface += np.where(ZZ==0, -np.inf, np.log(num_tid  + num_tia) - np.log(ZZ) - np.log(2))
            if return_ZZ:
                ZZ_surface += np.log(ZZ)
                if return_ZZ_0:
                    ZZ_0_surface += np.log(ZZ_0)

    if not db:
        if return_ZZ:
            if return_ZZ_0:
                return llhood_surface, ZZ_surface, ZZ_0_surface, Ip_set, pi_set, param_volume
            return llhood_surface, ZZ_surface, Ip_set, pi_set, param_volume
        return llhood_surface, Ip_set, pi_set, param_volume
    else:
        if return_ZZ:
            if return_ZZ_0:
                return llhood_surface, ZZ_surface, ZZ_0_surface, Ip_set, Ip_weights
            return llhood_surface, ZZ_surface, Ip_set, Ip_weights
        return llhood_surface, Ip_set, Ip_weights

###
def llhood_post_nplei(x_data, beta_data, v_cutoff, I2, nn, Ne,
                      min_x=0.01, neut_min=0.01, n_x=4000, n_s=4000, beta_obs=None, WF_pile=None, verbose=False):
    I2_grid_set = np.array([I2]*2)
    nn_grid_set = np.array([nn]*2)

    global_x_set = trad_x_set(min_x, n_points=n_x)

    ss_set = np.logspace(-3, 2.5, n_s)/(2*Ne)
    S_ud_set = ss_set*2*Ne
    if verbose: print("making tau grid...")
    if WF_pile is None:
        tau_grid = sim.sfs_ud_params_sigma(global_x_set, 1, -S_ud_set[:,np.newaxis])
    else:
        tau_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, x_set = global_x_set).T

    if beta_obs is None:
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_data, v_cutoff)
    else:
        # Observed betas determine where the cutoffs are put
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_obs, v_cutoff)

    (xx_low_set_tid, xx_low_set_tia,
     xx_high_set_tid, xx_high_set_tia,
     weight_low_set_tid, weight_low_set_tia,
     index_low_set_tid, index_low_set_tia,
     index_high_set_tid, index_high_set_tia) = freq_grid_upper_lower(x_data, global_x_set)

    pointwise_llhood = np.zeros_like(x_data)

    for ii in range(len(beta_data)):
        ZZ = quick_nplei_ZZ(global_x_set, I2_grid_set, nn_grid_set, filt_tau_x_int_set[ii],
                            ss_set, beta_data[ii])
        num_tid = quick_nplei_num(I2_grid_set, nn_grid_set, tau_grid,
                                  weight_low_set_tid[ii], index_low_set_tid[ii], index_high_set_tid[ii],
                                  ss_set, beta_data[ii])
        num_tia = quick_nplei_num(I2_grid_set, nn_grid_set, tau_grid,
                                  weight_low_set_tia[ii], index_low_set_tia[ii], index_high_set_tia[ii],
                                  ss_set, beta_data[ii])
        probs_tid = np.where(ZZ==0, 0, num_tid/ZZ)
        probs_tia = np.where(ZZ==0, 0, num_tia/ZZ)
        pointwise_llhood[ii] = np.log(probs_tid * 0.5 + probs_tia * 0.5)[0]
    return pointwise_llhood

###
def llhood_grid_nplei_setup(x_data, beta_data, v_cutoff, I2_set, nn_set, pi_set, Ne,
                            min_x=0.01, neut_min=0.01, n_x=4000, n_s=4000, beta_obs=None,
                            return_ZZ=False, return_ZZ_0=False, min_x_ash=None, db=False, WF_pile=None, verbose=False):
    I2_grid, nn_grid = np.meshgrid(I2_set, nn_set)
    I2_grid_set = I2_grid.flatten()
    nn_grid_set = nn_grid.flatten()

    I2_weights = get_weights(I2_set)
    nn_weights = get_weights(nn_set)
    if not db:
        pi_weights = get_pi_weights(pi_set)

    I2_w_grid, nn_w_grid = np.meshgrid(I2_weights, nn_weights)
    I2_w_flat = I2_w_grid.flatten()
    nn_w_flat = nn_w_grid.flatten()

    II_area = I2_w_flat * nn_w_flat
    if not db:
        param_volume = II_area[:, np.newaxis] * pi_weights
        llhood_surface = np.zeros(I2_grid_set.shape + pi_set.shape)
        if return_ZZ:
            ZZ_surface = np.zeros(I2_grid_set.shape + pi_set.shape)
            if return_ZZ_0:
                ZZ_0_surface = np.zeros(I2_grid_set.shape + pi_set.shape)
    else:
        llhood_surface = np.zeros(I2_grid_set.shape)
        if return_ZZ:
            ZZ_surface = np.zeros(I2_grid_set.shape)
            if return_ZZ_0:
                ZZ_0_surface = np.zeros(I2_grid_set.shape)

    global_x_set = trad_x_set(min_x, n_points=n_x)
    if min_x_ash is not None:
        global_x_set_ash = trad_x_set(min_x_ash, n_points=n_x)

    ss_set = np.logspace(-3, 2.5, n_s)/(2*Ne)
    S_ud_set = ss_set*2*Ne
    if verbose: print("making tau grid...")
    if WF_pile is None:
        tau_grid = sim.sfs_ud_params_sigma(global_x_set, 1, -S_ud_set[:,np.newaxis])
    else:
        tau_grid = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, x_set = global_x_set).T
    if min_x_ash is not None:
        if WF_pile is None:
            tau_grid_ash = sim.sfs_ud_params_sigma(global_x_set_ash, 1, -S_ud_set[:,np.newaxis])
        else:
            tau_grid_ash = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, x_set = global_x_set_ash).T
    if verbose: print("done")
    if verbose: print("doing tau integration...")
    if beta_obs is None:
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_data, v_cutoff)
    else:
        # Observed betas determine where the cutoffs are put
        filt_tau_x_int_set = tau_int_fast(tau_grid, global_x_set, x_data, beta_obs, v_cutoff)
    if return_ZZ_0:
        ## xx does actual nothing here
        ## beta doesn't matter because we are using cutoff at zero
        ## Will just integrate over global_x_set for each S_ud value, this doesn't depend on individual beta.
        ## Individual beta will matter when we have different weightings for S_ud based on the prob. density.
        if min_x_ash is None:
            full_tau_x_int = tau_x_int(tau_grid, global_x_set, xx=0, beta=0.1, vv=0)
        else:
            full_tau_x_int = tau_x_int(tau_grid_ash, global_x_set_ash, xx=0, beta=0.1, vv=0)
    if verbose: print("done")

    (xx_low_set_tid, xx_low_set_tia,
     xx_high_set_tid, xx_high_set_tia,
     weight_low_set_tid, weight_low_set_tia,
     index_low_set_tid, index_low_set_tia,
     index_high_set_tid, index_high_set_tia) = freq_grid_upper_lower(x_data, global_x_set)

    if verbose: print("real stuff...")
    for ii in range(len(beta_data)):
        if verbose: print(ii, end=".")
        ZZ = quick_nplei_ZZ(global_x_set, I2_grid_set, nn_grid_set, filt_tau_x_int_set[ii],
                            ss_set, beta_data[ii])
        if return_ZZ_0:
            if min_x_ash is not None:
                ZZ_0 = quick_nplei_ZZ(global_x_set_ash, I2_grid_set, nn_grid_set, full_tau_x_int,
                                      ss_set, beta_data[ii])
            else:
                ZZ_0 = quick_nplei_ZZ(global_x_set, I2_grid_set, nn_grid_set, full_tau_x_int,
                                      ss_set, beta_data[ii])
        num_tid = quick_nplei_num(I2_grid_set, nn_grid_set, tau_grid,
                                  weight_low_set_tid[ii], index_low_set_tid[ii], index_high_set_tid[ii],
                                  ss_set, beta_data[ii])
        num_tia = quick_nplei_num(I2_grid_set, nn_grid_set, tau_grid,
                                  weight_low_set_tia[ii], index_low_set_tia[ii], index_high_set_tia[ii],
                                  ss_set, beta_data[ii])
        if not db:
            probs_tid = np.where(ZZ==0, 0, num_tid/ZZ)[:,np.newaxis]
            probs_tia = np.where(ZZ==0, 0, num_tia/ZZ)[:,np.newaxis]
            llhood_surface += np.log(probs_tid * pi_set + probs_tia * (1-pi_set))
            if return_ZZ:
                ZZ_surface += np.log(ZZ)[:, np.newaxis] * np.ones_like(pi_set)
                if return_ZZ_0:
                    ZZ_0_surface += np.log(ZZ_0)[:, np.newaxis] * np.ones_like(pi_set)
        else:
            probs_tid = np.where(ZZ==0, 0, num_tid/ZZ)
            probs_tia = np.where(ZZ==0, 0, num_tia/ZZ)
            llhood_surface += np.log(probs_tid * 0.5 + probs_tia * 0.5)
            if return_ZZ:
                ZZ_surface += np.log(ZZ)
                if return_ZZ_0:
                    ZZ_0_surface += np.log(ZZ_0)
    if not db:
        if return_ZZ:
            if return_ZZ_0:
                return llhood_surface, ZZ_surface, ZZ_0_surface, I2_grid_set, nn_grid_set, pi_set, param_volume
            return llhood_surface, ZZ_surface, I2_grid_set, nn_grid_set, pi_set, param_volume
        return llhood_surface, I2_grid_set, nn_grid_set, pi_set, param_volume
    else:
        if return_ZZ:
            if return_ZZ_0:
                return llhood_surface, ZZ_surface, ZZ_0_surface, I2_grid_set, nn_grid_set, II_area
            return llhood_surface, ZZ_surface, I2_grid_set, nn_grid_set, II_area
        return llhood_surface, I2_grid_set, nn_grid_set, II_area

###
def llhood_post_ud(x_data, beta_data, v_cutoff, I2, Ne,
                   min_x=0.01, neut_min=0.01, beta_obs=None, WF_pile=None, n_x=200,
                   d_x_set = None, x_sets = None):

    # If ""observed"" betas are given, these are what the discovery regions should be based off of
    if d_x_set is None:
        if beta_obs is None:
            d_x_set = [np.maximum(discov_x(beta, v_cutoff), min_x) for beta in beta_data]
        else:
            d_x_set = [np.maximum(discov_x(beta, v_cutoff), min_x) for beta in beta_obs]
    if x_sets is None:
        x_sets = [adjusted_x_set(d_x, 10, n_x) for d_x in d_x_set]

    if WF_pile is None:
        neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1), x_set) for x_set in x_sets]

    pointwise_llhood = np.zeros_like(x_data)

    for ii in range(len(beta_data)):
        S_ud_set = 2*Ne*beta_data[ii]**2 * I2
        if WF_pile is None:
            ZZ = quick_stab_ZZ(x_sets[ii], S_ud_set, neut_vals[ii], -np.inf, neut_min)
            num_tid = sim.sfs_ud_params_sigma(x_data[ii], 1., np.abs(S_ud_set))
            num_tia = sim.sfs_ud_params_sigma(1-x_data[ii], 1., np.abs(S_ud_set))
        else:
            sfs_grid = sim.sfs_ud_WF_single(S_ud_set, WF_pile, x_sets[ii])
            ZZ = np.trapz(sfs_grid, x_sets[ii], axis=0)
            num_tid = np.interp(x_data[ii], x_sets[ii], sfs_grid)
            num_tia = np.interp(1-x_data[ii], x_sets[ii], sfs_grid)

        if ZZ == 0:
            pointwise_llhood[ii] = -np.inf
        else:
            probs_tid = num_tid/ZZ
            probs_tia = num_tia/ZZ
            pointwise_llhood[ii] = np.log(probs_tid * 0.5 + probs_tia * 0.5)
    return pointwise_llhood

# Wrapper for llhood_grid_ud_setup for the case of a single s value
def llhood_grid_s_setup(x_data, beta_data, v_cutoff, s_set, Ne,
                         min_x=0.01, neut_min=0.01, beta_obs=None,
                         return_ZZ=False, return_ZZ_0=False, min_x_ash=None,
                         db=True, WF_pile=None, verbose=False, neff=None, n_x=200):
    return llhood_grid_ud_setup(x_data, beta_data, v_cutoff, I2_set=None, pi_set=None, Ne=Ne,
                                min_x=min_x, neut_min=neut_min, beta_obs=beta_obs,
                                return_ZZ=return_ZZ, return_ZZ_0=return_ZZ_0, min_x_ash=min_x_ash,
                                db=db, WF_pile=WF_pile, verbose=verbose, neff=neff, n_x=n_x, s_set=s_set)

###
def llhood_grid_ud_setup(x_data, beta_data, v_cutoff, I2_set, pi_set, Ne,
                         min_x=0.01, neut_min=0.01, beta_obs=None,
                         return_ZZ=False, return_ZZ_0=False, min_x_ash=None,
                         db=False, WF_pile=None, verbose=False, neff=None, n_x=200,
                         s_set=None):
    if min_x_ash is None:
        min_x_ash = min_x

    param_set = I2_set if s_set is None else s_set
    param_weights = get_weights(param_set)
    n_grid = len(param_set)

    if not db:
        pi_weights = get_pi_weights(pi_set)
        param_volume = param_weights[:, np.newaxis] * pi_weights

        llhood_surface = np.zeros((n_grid, len(pi_set)))
        if return_ZZ:
            ZZ_surface = np.zeros((n_grid, len(pi_set)))
            if return_ZZ_0:
                ZZ_0_surface = np.zeros((n_grid, len(pi_set)))
    else:
        llhood_surface = np.zeros(n_grid)
        if return_ZZ:
            ZZ_surface = np.zeros(n_grid)
            if return_ZZ_0:
                ZZ_0_surface = np.zeros(n_grid)

    try:
        v_size = len(v_cutoff)
        assert v_size == len(x_data)
        v_set = v_cutoff
    except TypeError:
        v_size = 1
        v_set = np.ones_like(beta_data)*v_cutoff

    if v_size == 1:
        # If ""observed"" betas are given, these are what the discovery regions should be based off of
        if beta_obs is None:
            d_x_set = [np.maximum(discov_x(beta, v_cutoff), min_x) for beta in beta_data]
        else:
            d_x_set = [np.maximum(discov_x(beta, v_cutoff), min_x) for beta in beta_obs]
    else:
        # All use of different v_cutoffs is through the x_sets
        if beta_obs is None:
            d_x_set = [np.maximum(discov_x(beta, v_set[ii]), min_x) for ii, beta in enumerate(beta_data)]
        else:
            d_x_set = [np.maximum(discov_x(beta, v_set[ii]), min_x) for ii, beta in enumerate(beta_obs)]

    if neff == None:
        x_sets = [trad_x_set(d_x, n_x) for d_x in d_x_set] # (n_beta, n_x)
    else:
        x_sets = [trad_x_set(min_x, n_x) for _ in d_x_set]

    # If we have been given neff values, we can calculate the ascertainment probabilities
    # This is only really applicable to simulations at the moment
    # TODO: implement v_cutoff differences here
    if neff != None:
        beta_cutoff, beta_sd = beta_cutoff_sd(x_data, v_cutoff, neff)
        asc_probs = (stats.norm.sf(x=beta_cutoff, loc=np.abs(beta_data), scale=beta_sd) +
                     stats.norm.cdf(x=-beta_cutoff, loc=np.abs(beta_data), scale=beta_sd))
        x_set_a = np.array(x_sets) #TODO: fix since all x_set are the same so this is probably inefficient
        beta_cutoff_grid, beta_sd_grid = beta_cutoff_sd(x_set_a, v_cutoff, neff)
        asc_prob_grid = (stats.norm.sf(x=beta_cutoff_grid, loc=np.abs(beta_data[:,np.newaxis]), scale=beta_sd_grid) +
                         stats.norm.cdf(x=-beta_cutoff_grid, loc=np.abs(beta_data[:,np.newaxis]), scale=beta_sd_grid))

    if return_ZZ_0:
        x_set_ash = adjusted_x_set(min_x_ash, 10, n_x)
        if neff != None:
            beta_cutoff_ash, beta_sd_ash = beta_cutoff_sd(x_set_ash, v_cutoff, neff)
            asc_probs_ash = (stats.norm.sf(x=beta_cutoff_ash, loc=np.abs(beta_data), scale=beta_sd_ash) +
                             stats.norm.cdf(x=-beta_cutoff_ash, loc=np.abs(beta_data), scale=beta_sd_ash))

    if WF_pile is None:
        if neff == None:
            neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1), x_set) for x_set in x_sets]
        else:

            neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1)*asc_prob_grid[ii,:], x_set)
                         for ii, x_set in enumerate(x_sets)]
    else:
        S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
        S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
        sfs_neut = WF_pile["sfs_grid"][S_0_ii, S_ud_0_ii]
        if neff == None:
            neut_vals = [np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut), x_set) for x_set in x_sets]
        else:
            neut_vals = [np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut)*asc_prob_grid[ii,:], x_set)
                         for ii, x_set in enumerate(x_sets)]
    if return_ZZ_0:
        if WF_pile is None:
            if neff == None:
                neut_val_ash = np.trapz(sim.sfs_neut_params(x_set_ash, 1), x_set_ash)
            else:
                neut_val_ash = np.trapz(sim.sfs_neut_params(x_set_ash, 1)*asc_probs_ash, x_set_ash)
        else:
            if neff == None:
                neut_val_ash = np.trapz(np.interp(x_set_ash, WF_pile["interp_x"], sfs_neut), x_set_ash)
            else:
                neut_val_ash = np.trapz(np.interp(x_set_ash, WF_pile["interp_x"], sfs_neut)*asc_probs_ash, x_set_ash)

    if verbose: print("Calculating appropriate maximum S", end=".")
    if verbose: print(".", end=".")
    if verbose: print(" done")

    for ii in range(len(beta_data)):
        if verbose: print(ii, end=".")
        S_ud_set = 2*Ne*beta_data[ii]**2 * I2_set if s_set is None else 2*Ne*s_set
        if WF_pile is None:
            if neff == None:
                ZZ = quick_stab_ZZ(x_sets[ii], S_ud_set, neut_vals[ii], -np.inf, neut_min)
            else:
                ZZ = quick_stab_ZZ(x_sets[ii], S_ud_set, neut_vals[ii], -np.inf, neut_min, asc_prob_grid[ii,:])
        else:
            if neff == None:
                ZZ = quick_stab_ZZ_WF_grid(x_sets[ii], np.abs(S_ud_set), WF_pile)
            else:
                ZZ = quick_stab_ZZ_WF_grid(x_sets[ii], np.abs(S_ud_set), WF_pile, asc_prob_grid[ii,:])
        if return_ZZ_0:
            if WF_pile is None:
                ZZ_0 = quick_stab_ZZ(x_set_ash, S_ud_set, neut_val_ash, -np.inf, neut_min)
            else:
                if neff == None:
                    ZZ_0 = quick_stab_ZZ_WF_grid(x_set_ash, np.abs(S_ud_set), WF_pile)
                else:
                    ZZ_0 = quick_stab_ZZ_WF_grid(x_set_ash, np.abs(S_ud_set), WF_pile, asc_prob_grid[ii,:])
        if WF_pile is None:
            num_tid = sim.sfs_ud_params_sigma(x_data[ii], 1., np.abs(S_ud_set))
            num_tia = sim.sfs_ud_params_sigma(1-x_data[ii], 1., np.abs(S_ud_set))
        else:
            num_grid = sim.sfs_ud_WF_grid(np.abs(S_ud_set), WF_pile)
            num_tid = np.array([np.interp(x_data[ii], WF_pile["interp_x"], num_grid[:,jj])
                                for jj in range(num_grid.shape[1])])
            num_tia = np.array([np.interp(1-x_data[ii], WF_pile["interp_x"], num_grid[:,jj])
                                for jj in range(num_grid.shape[1])])
        if neff != None:
            num_tid *= asc_probs[ii]
            num_tia *= asc_probs[ii]

        if not db:
            probs_tid = np.where(ZZ==0, 0, num_tid/ZZ)[:,np.newaxis]
            probs_tia = np.where(ZZ==0, 0, num_tia/ZZ)[:,np.newaxis]
            llhood_surface += np.log(probs_tid * pi_set + probs_tia * (1-pi_set))
            if return_ZZ:
                ZZ_surface += np.log(ZZ)[:, np.newaxis] * np.ones_like(pi_set)
                if return_ZZ_0:
                    ZZ_0_surface += np.log(ZZ_0)[:, np.newaxis] * np.ones_like(pi_set)
        else:
            probs_tid = np.where(ZZ==0, 0, num_tid/ZZ)
            probs_tia = np.where(ZZ==0, 0, num_tia/ZZ)
            llhood_surface += np.log(probs_tid * 0.5 + probs_tia * 0.5)
            if return_ZZ:
                ZZ_surface += np.log(ZZ)
                if return_ZZ_0:
                    ZZ_0_surface += np.log(ZZ_0)
    if not db:
        if return_ZZ:
            if return_ZZ_0:
                return llhood_surface, ZZ_surface, ZZ_0_surface, param_set, pi_set, param_volume
            return llhood_surface, ZZ_surface, param_set, pi_set, param_volume
        return llhood_surface, param_set, pi_set, param_volume
    else:
        if return_ZZ:
            if return_ZZ_0:
                return llhood_surface, ZZ_surface, ZZ_0_surface, param_set, param_weights
            return llhood_surface, ZZ_surface, param_set, param_weights
        return llhood_surface, param_set, param_weights

###
def x_grid_setup(beta_data, beta_obs, min_x, min_x_ash, v_cutoff, return_ZZ_0=False, n_x=200):
    try:
        v_size = len(v_cutoff)
        assert v_size == len(beta_data)
        v_set = v_cutoff
    except TypeError:
        v_set = np.ones_like(beta_data)*v_cutoff

    if min_x_ash is None:
        min_x_ash = min_x
    if beta_obs is None:
        d_x_set = [np.maximum(discov_x(beta, v_set[ii]), min_x) for ii, beta in enumerate(beta_data)]
    else:
        d_x_set = [np.maximum(discov_x(beta, v_set[ii]), min_x) for ii, beta in enumerate(beta_obs)]
    x_sets = [adjusted_x_set(d_x, 10, n_x) for d_x in d_x_set]
    if return_ZZ_0:
        x_set_ash = adjusted_x_set(min_x_ash, 10, n_x)
    else:
        x_set_ash = None

    return(d_x_set, x_sets, x_set_ash, min_x_ash)

###
def llhood_post_dir_db(x_data, beta_data, v_cutoff, I1, Ne,
                       min_x=0.01, neut_min=0.01, beta_obs=None, WF_pile=None):
    _, x_sets, _, _ = x_grid_setup(beta_data, beta_obs, min_x, None, v_cutoff, False)

    if WF_pile is None:
        neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1), x_set) for x_set in x_sets]
    else:
        S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
        S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
        sfs_neut = WF_pile["sfs_grid"][S_0_ii, S_ud_0_ii]
        neut_vals = [np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut), x_set) for x_set in x_sets]

    pointwise_llhood = np.zeros_like(x_data)
    ZZ_tid_set = []
    ZZ_tia_set = []

    for ii in range(len(beta_data)):
        S_dir_set = 2*Ne*beta_data[ii]*np.array([I1]*2)
        pi_set = sim.pi_dir_db(S_dir_set)

        if WF_pile is None:
            ZZ_tid = quick_dir_ZZ(x_sets[ii],  S_dir_set, neut_vals[ii], -np.inf, neut_min)
            ZZ_tia = quick_dir_ZZ(x_sets[ii], -S_dir_set, neut_vals[ii], -np.inf, neut_min)
        else:
            ZZ_tid = quick_dir_ZZ_WF_grid(x_sets[ii], S_dir_set, WF_pile)
            ZZ_tia = quick_dir_ZZ_WF_grid(x_sets[ii], -S_dir_set, WF_pile)
        ZZ_tid_set.append(ZZ_tid)
        ZZ_tia_set.append(ZZ_tia)

        if WF_pile is None:
            num_tid = sim.sfs_dir_params_multi_S(x_data[ii], 1., S_dir_set)
            num_tia = sim.sfs_dir_params_multi_S(1-x_data[ii], 1., -S_dir_set)
        else:
            num_grid_tid = sim.sfs_dir_WF_grid(S_dir_set, WF_pile)
            num_grid_tia = sim.sfs_dir_WF_grid(-S_dir_set, WF_pile)
            num_tid = np.array([np.interp(x_data[ii], WF_pile["interp_x"], num_grid_tid[:,jj])
                                for jj in range(num_grid_tid.shape[1])])
            num_tia = np.array([np.interp(1-x_data[ii], WF_pile["interp_x"], num_grid_tia[:,jj])
                                for jj in range(num_grid_tia.shape[1])])

        numer = num_tid * pi_set + num_tia * (1-pi_set)
        denom = ZZ_tid * pi_set + ZZ_tia * (1-pi_set)

        pointwise_llhood[ii] = np.where(denom==0, -np.inf, np.log(numer) - np.log(denom))[0]

    return pointwise_llhood

###
def llhood_grid_dir_db_setup(x_data, beta_data, v_cutoff, I1_set, Ne,
                             min_x=0.01, neut_min=0.01, beta_obs=None,
                             return_ZZ=False, return_ZZ_0=False, min_x_ash=None, WF_pile=None, verbose=False,
                             neff=None, n_x=200):
    beta_data = np.abs(beta_data)
    if beta_obs is not None:
        beta_obs = np.abs(beta_obs)
    I1_set_full = np.concatenate((I1_set, -np.flip(I1_set)))
    I1_weights = get_weights(I1_set_full)

    llhood_surface = np.zeros(len(I1_set_full))

    if return_ZZ:
        ZZ_surface = np.zeros(len(I1_set_full))
        if return_ZZ_0:
            ZZ_0_surface = np.zeros(len(I1_set_full))

    _, x_sets, x_set_ash, _ = x_grid_setup(beta_data, beta_obs, min_x, # all v_cutoff differences are in x_sets
                                                   min_x_ash, v_cutoff, return_ZZ_0, n_x=n_x)
    if neff != None:
        x_sets = [adjusted_x_set(min_x, 10, n_x) for _ in x_data]
        x_set_ash = adjusted_x_set(min_x, 10, n_x)

    if neff != None:
        # TODO: implement v_cutoff differences here
        beta_cutoff, beta_sd = beta_cutoff_sd(x_data, v_cutoff, neff)
        asc_probs = stats.norm.sf(x=beta_cutoff, loc=beta_data, scale=beta_sd)
        x_set_a = np.array(x_sets) #TODO: all x_set equal
        beta_cutoff_grid, beta_sd_grid = beta_cutoff_sd(x_set_a, v_cutoff, neff)
        asc_prob_grid = stats.norm.sf(x=beta_cutoff_grid, loc=beta_data[:,np.newaxis], scale=beta_sd_grid)
        if x_set_ash is not None:
            beta_cutoff_ash, beta_sd_ash = beta_cutoff_sd(x_set_ash, v_cutoff, neff)
            asc_probs_ash = stats.norm.sf(x=beta_cutoff_ash, loc=beta_data[:,None], scale=beta_sd_ash)
    else:
        asc_probs, asc_prob_grid, asc_probs_ash = [None]*3

    if WF_pile is None:
        if neff == None:
            neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1), x_set) for x_set in x_sets]
        else:
            neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1)*asc_prob_grid[ii,:], x_set)
                         for ii, x_set in enumerate(x_sets)]
    else:
        S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
        S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
        sfs_neut = WF_pile["sfs_grid"][S_0_ii, S_ud_0_ii]
        if neff == None:
            neut_vals = [np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut), x_set) for x_set in x_sets]
        else:
            neut_vals = [np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut)*asc_prob_grid[ii,:], x_set)
                         for ii, x_set in enumerate(x_sets)]
    if return_ZZ_0:
        if WF_pile is None:
            if neff == None:
                neut_val_ash = np.trapz(sim.sfs_neut_params(x_set_ash, 1), x_set_ash)
            else:
                neut_val_ash = np.trapz(sim.sfs_neut_params(x_set_ash, 1)*asc_probs_ash, x_set_ash)
        else:
            if neff == None:
                neut_val_ash = np.trapz(np.interp(x_set_ash, WF_pile["interp_x"], sfs_neut), x_set_ash)
            else:
                neut_val_ash = np.trapz(np.interp(x_set_ash, WF_pile["interp_x"], sfs_neut)*asc_probs_ash, x_set_ash)

    pointwise_llhoods = []
    ZZ_tid_set = []
    ZZ_tia_set = []

    for ii in range(len(beta_data)):
        if verbose: print(ii, end=".")

        S_dir_set = 2*Ne*beta_data[ii]*I1_set_full
        pi_set = sim.pi_dir_db(S_dir_set)

        asc_probs_tmp = asc_prob_grid[ii,:] if neff else None

        if WF_pile is None:
            ZZ_tid = quick_dir_ZZ(x_sets[ii],  S_dir_set, neut_vals[ii], -np.inf, neut_min,
                                  asc_probs=asc_probs_tmp)
            ZZ_tia = quick_dir_ZZ(x_sets[ii], -S_dir_set, neut_vals[ii], -np.inf, neut_min,
                                  asc_probs=asc_probs_tmp)
        else:
            ZZ_tid = quick_dir_ZZ_WF_grid(x_sets[ii], S_dir_set, WF_pile, asc_probs=asc_probs_tmp)
            ZZ_tia = quick_dir_ZZ_WF_grid(x_sets[ii], -S_dir_set, WF_pile, asc_probs=asc_probs_tmp)
        ZZ_tid_set.append(ZZ_tid)
        ZZ_tia_set.append(ZZ_tia)
        if return_ZZ_0:
            if WF_pile is None:
                ZZ_0_tid = quick_dir_ZZ(x_set_ash, S_dir_set, neut_val_ash, -np.inf, neut_min, asc_probs=asc_probs_tmp)
                ZZ_0_tia = quick_dir_ZZ(x_set_ash, -S_dir_set, neut_val_ash, -np.inf, neut_min, asc_probs=asc_probs_tmp)
            else:
                ZZ_0_tid = quick_dir_ZZ_WF_grid(x_set_ash, S_dir_set, WF_pile, asc_probs=asc_probs_tmp)
                ZZ_0_tia = quick_dir_ZZ_WF_grid(x_set_ash, -S_dir_set, WF_pile, asc_probs=asc_probs_tmp)

        if WF_pile is None:
            num_tid = sim.sfs_dir_params_multi_S(x_data[ii], 1., S_dir_set)
            num_tia = sim.sfs_dir_params_multi_S(1-x_data[ii], 1., -S_dir_set)
        else:
            num_grid_tid = sim.sfs_dir_WF_grid(S_dir_set, WF_pile)
            num_grid_tia = sim.sfs_dir_WF_grid(-S_dir_set, WF_pile)
            num_tid = np.array([np.interp(x_data[ii], WF_pile["interp_x"], num_grid_tid[:,jj])
                                for jj in range(num_grid_tid.shape[1])])
            num_tia = np.array([np.interp(1-x_data[ii], WF_pile["interp_x"], num_grid_tia[:,jj])
                                for jj in range(num_grid_tia.shape[1])])
        if neff != None:
            num_tid *= asc_probs[ii]
            num_tia *= asc_probs[ii]

        numer = num_tid * pi_set + num_tia * (1-pi_set)
        denom = ZZ_tid * pi_set + ZZ_tia * (1-pi_set)
        if return_ZZ_0:
            denom_0 = ZZ_0_tid * pi_set + ZZ_0_tia * (1-pi_set)
        llhood_surface += np.where(denom==0, -np.inf, np.log(numer) - np.log(denom))
        pointwise_llhoods.append(np.where(denom==0, -np.inf, np.log(numer) - np.log(denom)))
        if return_ZZ:
            ZZ_surface += np.log(denom)
            if return_ZZ_0:
                ZZ_0_surface += np.log(denom_0)

    if return_ZZ:
        if return_ZZ_0:
            return (llhood_surface, ZZ_surface, ZZ_0_surface, I1_set_full,
                    I1_weights, pointwise_llhoods, ZZ_tid_set, ZZ_tia_set)
        return llhood_surface, ZZ_surface, I1_set_full, I1_weights
    return llhood_surface, I1_set_full, I1_weights

###
def llhood_post_full_db(x_data, beta_data, v_cutoff, I1, I2, Ne,
                        min_x=0.01, neut_min=0.01, beta_obs=None, WF_pile=None, n_x=200):
    # Order data descending on risk (+) allele effect sizes
    data_order = np.argsort(beta_data)[::-1]
    x_data = x_data[data_order]
    beta_data = beta_data[data_order]

    if beta_obs is None:
        d_x_set = [np.maximum(discov_x(beta, v_cutoff), min_x) for beta in beta_data]
    else:
        beta_obs = beta_obs[data_order]
        d_x_set = [np.maximum(discov_x(beta, v_cutoff), min_x) for beta in beta_obs]

    x_sets = [adjusted_x_set(d_x, 10, n_x) for d_x in d_x_set]

    if WF_pile is None:
        neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1), x_set) for x_set in x_sets]
    else:
        S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
        S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
        sfs_neut = WF_pile["sfs_grid"][S_0_ii, S_ud_0_ii]
        neut_vals = [np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut), x_set) for x_set in x_sets]

    pointwise_llhood = np.zeros_like(x_data)
    for ii in range(len(beta_data)):
        S_dir_set = 2*Ne*beta_data[ii] * np.array([I1]*2)
        S_ud_set = 2*Ne*beta_data[ii]**2 * np.array([I2]*2)
        ZZ_tid = np.full_like(S_dir_set, 0)
        ZZ_tia = np.full_like(S_dir_set, 0)
        num_tid = np.full_like(S_dir_set, 0)
        num_tia = np.full_like(S_dir_set, 0)
        pi_set = sim.pi_dir_db(S_dir_set)
        if WF_pile is None:
            ZZ_tid = quick_full_ZZ(x_sets[ii], S_dir_set, S_ud_set,
                                   neut_vals[ii], -np.inf, neut_min)
            ZZ_tia = quick_full_ZZ(x_sets[ii], -S_dir_set, S_ud_set,
                                   neut_vals[ii], -np.inf, neut_min)
        else:
            ZZ_tid = quick_full_ZZ_WF_grid(x_sets[ii], S_dir_set, S_ud_set, WF_pile)
            ZZ_tia = quick_full_ZZ_WF_grid(x_sets[ii], -S_dir_set, S_ud_set, WF_pile)

        if WF_pile is None:
            num_tid = sim.sfs_full_params_stable_vec(x_data[ii], 1., S_dir_set, S_ud_set)
            num_tia = sim.sfs_full_params_stable_vec(1-x_data[ii], 1., -S_dir_set, S_ud_set)
        else:
            num_grid_tid = sim.sfs_full_WF_grid(S_dir_set, S_ud_set, WF_pile)
            num_grid_tia = sim.sfs_full_WF_grid(-S_dir_set, S_ud_set, WF_pile)
            num_tid = np.array([np.interp(x_data[ii], WF_pile["interp_x"], num_grid_tid[:,jj])
                                for jj in range(num_grid_tid.shape[1])])
            num_tia = np.array([np.interp(1-x_data[ii], WF_pile["interp_x"], num_grid_tia[:,jj])
                                for jj in range(num_grid_tia.shape[1])])

        numer = num_tid * pi_set + num_tia * (1-pi_set)
        denom = ZZ_tid * pi_set + ZZ_tia * (1-pi_set)

        pointwise_llhood[ii] = np.where(denom==0, -np.inf, np.log(numer) - np.log(denom))[0]

    return pointwise_llhood[np.argsort(data_order)] # return in the correct order

###
def llhood_grid_full_db_setup(x_data, beta_data, v_cutoff, I1_set, I2_set, Ne,
                              min_x=0.01, kill=True, neut_min=0.01, beta_obs=None,
                              return_ZZ=False, return_ZZ_0=False, min_x_ash=None, WF_pile=None, verbose=False,
                              n_x=200):
    beta_data = np.abs(beta_data)
    if beta_obs is not None:
        beta_obs = np.abs(beta_obs)
        
    if min_x_ash is None:
        min_x_ash = min_x

    I1_set_full = np.concatenate((I1_set, -np.flip(I1_set)))
    I1_grid, I2_grid = np.meshgrid(I1_set_full, I2_set)
    I1_grid_set = I1_grid.flatten()
    I2_grid_set = I2_grid.flatten()

    I1_weights = get_weights(I1_set_full)
    I2_weights = get_weights(I2_set)

    I1_w_grid, I2_w_grid = np.meshgrid(I1_weights, I2_weights)
    I1_w_flat = I1_w_grid.flatten()
    I2_w_flat = I2_w_grid.flatten()

    II_area = I1_w_flat * I2_w_flat

    data_order = np.argsort(beta_data)[::-1]
    x_data = x_data[data_order]
    beta_data = beta_data[data_order]

    try:
        v_size = len(v_cutoff)
        assert v_size == len(beta_data)
        v_set = v_cutoff
    except TypeError:
        v_size = 1
        v_set = np.ones_like(beta_data)*v_cutoff

    if beta_obs is None:
        d_x_set = [np.maximum(discov_x(beta, v_set[ii]), min_x) for ii, beta in enumerate(beta_data)]
    else:
        beta_obs = beta_obs[data_order]
        d_x_set = [np.maximum(discov_x(beta, v_set[ii]), min_x) for ii, beta in enumerate(beta_obs)]

    x_sets = [adjusted_x_set(d_x, 10, n_x) for d_x in d_x_set]
    if return_ZZ_0:
        x_set_ash = adjusted_x_set(min_x_ash, 10, n_x)
    if WF_pile is None:
        neut_vals = [np.trapz(sim.sfs_neut_params(x_set, 1), x_set) for x_set in x_sets]
    else:
        S_0_ii = np.where(WF_pile["s_set"] == 0)[0][0]
        S_ud_0_ii = np.where(WF_pile["s_ud_set"] == 0)[0][0]
        sfs_neut = WF_pile["sfs_grid"][S_0_ii, S_ud_0_ii]
        neut_vals = [np.trapz(np.interp(x_set, WF_pile["interp_x"], sfs_neut), x_set) for x_set in x_sets]
    if return_ZZ_0:
        if WF_pile is None:
            neut_val_ash = np.trapz(sim.sfs_neut_params(x_set_ash, 1), x_set_ash)
        else:
            neut_val_ash = np.trapz(np.interp(x_set_ash, WF_pile["interp_x"], sfs_neut), x_set_ash)

    llhood_surface = np.zeros(I1_grid_set.shape)
    if return_ZZ:
        ZZ_surface = np.zeros(I1_grid_set.shape)
        if return_ZZ_0:
            ZZ_0_surface = np.zeros(I1_grid_set.shape)

    kill_params = np.full_like(I1_grid_set, True, dtype=np.bool)
    for ii in range(len(beta_data)):
        if verbose: print(ii, end="-")
        S_dir_set = 2*Ne*beta_data[ii] * I1_grid_set
        S_ud_set = 2*Ne*beta_data[ii]**2 * I2_grid_set
        ZZ_tid = np.full_like(S_dir_set, 0)
        ZZ_tia = np.full_like(S_dir_set, 0)
        if return_ZZ_0:
            ZZ_0_tid = np.full_like(S_dir_set, 0)
            ZZ_0_tia = np.full_like(S_dir_set, 0)
        num_tid = np.full_like(S_dir_set, 0)
        num_tia = np.full_like(S_dir_set, 0)
        pi_set = sim.pi_dir_db(S_dir_set)
        if kill:
            if WF_pile is not None:
                print("Don't try to kill params with WF grid!!")
                return None
            ZZ_tid[kill_params] = quick_full_ZZ(x_sets[ii], S_dir_set[kill_params],
                                                S_ud_set[kill_params],
                                                neut_vals[ii], -np.inf, neut_min)
            ZZ_tia[kill_params] = quick_full_ZZ(x_sets[ii], -S_dir_set[kill_params],
                                                S_ud_set[kill_params],
                                                neut_vals[ii], -np.inf, neut_min)
            if return_ZZ_0:
                ZZ_0_tid[kill_params] = quick_full_ZZ(x_set_ash, S_dir_set[kill_params],
                                                      S_ud_set[kill_params],
                                                      neut_val_ash, -np.inf, neut_min)
                ZZ_0_tia[kill_params] = quick_full_ZZ(x_set_ash, -S_dir_set[kill_params],
                                                      S_ud_set[kill_params],
                                                      neut_val_ash, -np.inf, neut_min)


            num_tid[kill_params] = sim.sfs_full_params_stable_vec(x_data[ii], 1.,
                                                                  S_dir_set[kill_params],
                                                                  S_ud_set[kill_params])
            num_tia[kill_params] = sim.sfs_full_params_stable_vec(1-x_data[ii], 1.,
                                                                  -S_dir_set[kill_params],
                                                                  S_ud_set[kill_params])
        else:
            if WF_pile is None:
                ZZ_tid = quick_full_ZZ(x_sets[ii], S_dir_set, S_ud_set,
                                       neut_vals[ii], -np.inf, neut_min)
                ZZ_tia = quick_full_ZZ(x_sets[ii], -S_dir_set, S_ud_set,
                                       neut_vals[ii], -np.inf, neut_min)
            else:
                ZZ_tid = quick_full_ZZ_WF_grid(x_sets[ii], S_dir_set, S_ud_set, WF_pile)
                ZZ_tia = quick_full_ZZ_WF_grid(x_sets[ii], -S_dir_set, S_ud_set, WF_pile)
            if return_ZZ_0:
                if WF_pile is None:
                    ZZ_0_tid = quick_full_ZZ(x_set_ash, S_dir_set, S_ud_set,
                                             neut_val_ash, -np.inf, neut_min)
                    ZZ_0_tia = quick_full_ZZ(x_set_ash, -S_dir_set, S_ud_set,
                                             neut_val_ash, -np.inf, neut_min)
                else:
                    ZZ_0_tid = quick_full_ZZ_WF_grid(x_set_ash, S_dir_set, S_ud_set, WF_pile)
                    ZZ_0_tia = quick_full_ZZ_WF_grid(x_set_ash, -S_dir_set, S_ud_set, WF_pile)
            if WF_pile is None:
                num_tid = sim.sfs_full_params_stable_vec(x_data[ii], 1., S_dir_set, S_ud_set)
                num_tia = sim.sfs_full_params_stable_vec(1-x_data[ii], 1., -S_dir_set, S_ud_set)
            else:
                num_grid_tid = sim.sfs_full_WF_grid(S_dir_set, S_ud_set, WF_pile)
                num_grid_tia = sim.sfs_full_WF_grid(-S_dir_set, S_ud_set, WF_pile)
                num_tid = np.array([np.interp(x_data[ii], WF_pile["interp_x"], num_grid_tid[:,jj])
                                    for jj in range(num_grid_tid.shape[1])])
                num_tia = np.array([np.interp(1-x_data[ii], WF_pile["interp_x"], num_grid_tia[:,jj])
                                    for jj in range(num_grid_tia.shape[1])])

        numer = num_tid * pi_set + num_tia * (1-pi_set)
        denom = ZZ_tid * pi_set + ZZ_tia * (1-pi_set)

        llhood_surface += np.where(denom==0, -np.inf, np.log(numer) - np.log(denom))

        if return_ZZ:
            ZZ_surface += np.log(denom)
            if return_ZZ_0:
                denom_ash = ZZ_0_tid * pi_set + ZZ_0_tia * (1-pi_set)
                ZZ_0_surface += np.log(denom_ash)

        kill_params[((ZZ_tid*pi_set)==0) & ((ZZ_tia*(1-pi_set))==0)] = False

    if return_ZZ:
        if return_ZZ_0:
            return llhood_surface, ZZ_surface, ZZ_0_surface, I1_grid_set, I2_grid_set, II_area
        return llhood_surface, ZZ_surface, I1_grid_set, I2_grid_set, II_area
    return llhood_surface, I1_grid_set, I2_grid_set, II_area

###
def make_pi_grid(xmin, xmax, pi_size):
    if not pi_size % 2:
        pi_size += 1
    assert ((xmin < 0.5) & (xmax > 0.5)), "Inappropriate x range"
    pi_grid = np.concatenate((np.linspace(xmin, 0.5, int(np.floor(pi_size/2))),
                              np.linspace(0.5, xmax, int(np.floor(pi_size/2)))[1:]))
    return pi_grid

###
def llhood_all_db(x_data, beta_data, v_cutoff, Ne, grid_size_1d, grid_size_2d, pi_size,
                  min_x=0.01, kill=True, neut_min=0.01, beta_obs=None,
                  I1_set_1d=None, I2_set_1d=None, I1_set_2d=None, I2_set_2d=None,
                  xmin=0.001, xmax=0.999, simple=False, neut_db=True, stab_db=True, WF_pile=None,
                  S_dir_max=1000, S_dir_min=0.01, S_ud_max=1000, S_ud_min=0.01, neff=None, n_x=200, single_s=False):
    pi_grid = make_pi_grid(xmin, xmax, pi_size)
    if I1_set_1d is None:
        I1_set_1d, I2_set_1d, _, _, _ = choose_param_range(beta_data, grid_size_1d, Ne=Ne,
                                                           S_dir_max=S_dir_max, S_dir_min=S_dir_min,
                                                           S_ud_max=S_ud_max, S_ud_min=S_ud_min)
        I1_set_2d, I2_set_2d, _, _, _ = choose_param_range(beta_data, grid_size_2d, Ne=Ne,
                                                           S_dir_max=S_dir_max, S_dir_min=S_dir_min,
                                                           S_ud_max=S_ud_max, S_ud_min=S_ud_min)
    
    llhood_neut, pi_set, pi_weights = llhood_grid_neut_setup(x_data, beta_data, v_cutoff,
                                                             pi_grid, Ne, min_x=min_x, beta_obs=beta_obs,
                                                             WF_pile=WF_pile, neff=neff, n_x=n_x)
    llhood_stab, I2_stab, pi_set, w_stab = llhood_grid_ud_setup(x_data, beta_data, v_cutoff,
                                                                I2_set_1d, pi_grid, Ne, min_x=min_x,
                                                                neut_min=neut_min, beta_obs=beta_obs,
                                                                WF_pile=WF_pile, neff=neff,
                                                                n_x=n_x)
    llhood_dir_db, I1_dir_db, w_dir_db = llhood_grid_dir_db_setup(x_data, beta_data, v_cutoff,
                                                                  I1_set_1d, Ne,
                                                                  min_x=min_x, neut_min=neut_min,
                                                                  beta_obs=beta_obs,
                                                                  WF_pile=WF_pile, neff=neff,
                                                                  n_x=n_x)

    if single_s:
        s_min = S_ud_min / (2*Ne)
        s_max = S_ud_max / (2*Ne)
        s_set = np.logspace(np.log10(s_min), np.log10(s_max), grid_size_1d)
        llhood_s, s_set, w_s = llhood_grid_s_setup(x_data, beta_data, v_cutoff, s_set, Ne,
                                                   min_x=min_x, neut_min=neut_min, beta_obs=beta_obs,
                                                   WF_pile=WF_pile, neff=neff, n_x=n_x)

    if not simple:
        if WF_pile is not None:
            kill = False
        llhood_full_db, I1_full_db, I2_full_db, w_full_db = llhood_grid_full_db_setup(x_data, beta_data,
                                                                                      v_cutoff,
                                                                                      I1_set_2d, I2_set_2d,
                                                                                      Ne,
                                                                                      min_x=min_x, kill=kill,
                                                                                      neut_min=neut_min,
                                                                                      beta_obs=beta_obs,
                                                                                      WF_pile=WF_pile,
                                                                                      n_x=n_x)
        result = {"llhood_neut":llhood_neut, "pi_weights":pi_weights,
                  "llhood_stab":llhood_stab, "I2_stab":I2_stab, "w_stab":w_stab,
                  "llhood_dir_db":llhood_dir_db, "I1_dir_db":I1_dir_db, "w_dir_db":w_dir_db,
                  "llhood_full_db":llhood_full_db, "I1_full_db":I1_full_db,
                  "I2_full_db":I2_full_db, "w_full_db":w_full_db,
                  "pi_grid":pi_grid}
    if simple:
        result = {"llhood_neut":llhood_neut, "pi_weights":pi_weights,
                  "llhood_stab":llhood_stab, "I2_stab":I2_stab, "w_stab":w_stab,
                  "llhood_dir_db":llhood_dir_db, "I1_dir_db":I1_dir_db, "w_dir_db":w_dir_db,
                  "pi_grid":pi_grid}
    if stab_db:
        stab_db = reduce_pi({"llhood_stab":llhood_stab,
                             "pi_grid":pi_set,
                             "I2_stab":I2_stab},
                            pi_val=0.5,
                            sel_model="stab")
        result["llhood_stab_db"] = stab_db["llhood_stab_db"]
        result["I2_stab_db"] = stab_db["I2_stab_db"]
        result["w_stab_db"] = stab_db["w_stab_db"]
    if neut_db:
        neut_db = reduce_pi({"llhood_neut":llhood_neut,
                             "pi_grid":pi_set},
                            pi_val=0.5, sel_model="neut")
        result["llhood_neut_db"] = neut_db["llhood_neut_db"]

    if single_s:
        # raise not implemented error if stab_db is not True
        if not stab_db:
            raise NotImplementedError("stab_db must be True to use single_s")
        result["llhood_s"] = llhood_s
        result["s_set"] = s_set
        result["w_s"] = w_s

    return result

###
def llhood_plei(x_data, beta_data, v_cutoff, Ne, grid_size_Ip, grid_size_I2,
                grid_size_nn, pi_size, beta_obs=None, nn_max=20, min_x=0.01,
                Ip_set=None, I2_set=None, nn_set=None, xmin=0.001, xmax=0.999,
                simple=False, stab_db=False, WF_pile=None,
                S_dir_max=1000, S_dir_min=0.01, S_ud_max=1000, S_ud_min=0.01, neff=None, 
                n_x=200, n_s=4000, ssd=False):
    pi_grid = make_pi_grid(xmin, xmax, pi_size)
    if Ip_set is None:
        _, _, Ip_set, _, _ = choose_param_range(beta_data, grid_size_Ip, Ne=Ne,
                                                           S_dir_max=S_dir_max, S_dir_min=S_dir_min,
                                                           S_ud_max=S_ud_max, S_ud_min=S_ud_min)
        _, _, _, I2_set, _ = choose_param_range(beta_data, grid_size_I2, nn_max=nn_max, Ne=Ne,
                                                           S_dir_max=S_dir_max, S_dir_min=S_dir_min,
                                                           S_ud_max=S_ud_max, S_ud_min=S_ud_min)
        _, _, _, _, nn_set = choose_param_range(beta_data, grid_size_nn, nn_max=nn_max, Ne=Ne,
                                                           S_dir_max=S_dir_max, S_dir_min=S_dir_min,
                                                           S_ud_max=S_ud_max, S_ud_min=S_ud_min)

    llhood_plei, Ip_plei, _, w_plei = llhood_grid_plei_setup(x_data,
                                                             beta_data,
                                                             v_cutoff,
                                                             Ip_set,
                                                             pi_grid, Ne,
                                                             min_x=min_x,
                                                             beta_obs=beta_obs,
                                                             WF_pile=WF_pile,
                                                             neff=neff, n_x=n_x, n_s=n_s, ssd=ssd)
    if not simple:
        llhood_nplei, I2_nplei, nn_nplei, _, w_nplei = llhood_grid_nplei_setup(x_data,
                                                                                beta_data,
                                                                                v_cutoff,
                                                                                I2_set,
                                                                                nn_set,
                                                                                pi_grid, Ne,
                                                                                min_x=min_x,
                                                                                beta_obs=beta_obs,
                                                                                WF_pile=WF_pile, n_x=n_x)
    if not simple:
        result = {"llhood_plei":llhood_plei, "Ip_plei":Ip_plei, "w_plei":w_plei,
                  "llhood_nplei":llhood_nplei, "I2_nplei":I2_nplei, "nn_nplei":nn_nplei,
                  "w_nplei":w_nplei, "pi_grid":pi_grid}
        if stab_db:
            nplei_db = reduce_pi({"llhood_nplei":llhood_nplei,
                                  "I2_nplei":I2_nplei,
                                  "nn_nplei":nn_nplei,
                                  "pi_grid":pi_grid,
                                  "w_nplei":w_nplei},
                                 pi_val=0.5,
                                 sel_model="nplei")
            result["llhood_nplei_db"] = nplei_db["llhood_nplei_db"]
            result["I2_nplei_db"] = nplei_db["I2_nplei_db"]
            result["nn_nplei_db"] = nplei_db["nn_nplei_db"]
            result["w_nplei_db"] = nplei_db["w_nplei_db"]
    if simple:
        result = {"llhood_plei":llhood_plei, "Ip_plei":Ip_plei, "w_plei":w_plei,
                  "pi_grid":pi_grid}
    if stab_db:
        plei_db = reduce_pi({"llhood_plei":llhood_plei,
                             "pi_grid":pi_grid,
                             "Ip_plei":Ip_plei},
                            pi_val=0.5,
                            sel_model="plei")
        result["llhood_plei_db"] = plei_db["llhood_plei_db"]
        result["Ip_plei_db"] = plei_db["Ip_plei_db"]
        result["w_plei_db"] = plei_db["w_plei_db"]

    return result

###
def update_s(max_surface, rr, llhood_surface):
    new_max_surface = np.maximum(max_surface, llhood_surface)
    new_rr = np.exp(max_surface - new_max_surface)*rr + np.exp(llhood_surface - new_max_surface)
    return new_max_surface, np.where(np.isnan(new_rr), rr+1, new_rr)

###
def combine_samps(samp_llhoods, sel_model=None, weights=False, use_ZZ_0=False):
    n_reps = len(samp_llhoods)
    with open(samp_llhoods[0], "rb") as handle:
        ex_ll = pickle.load(handle)
        if sel_model in ["neut_db", "stab_db", "plei_db", "nplei_db"]:
            ex_ll = reduce_pi(ex_ll, 0.5, sel_model=sel_model[0:-3])

    if sel_model is None:
        sel_model = [key.split("_", 1)[1] for key in ex_ll.keys() if "llhood_" in key][0]

    if sel_model == "neut":
        max_surf = np.full_like(ex_ll["pi_grid"], -np.inf)
        rr = np.full_like(ex_ll["pi_grid"], 0)
        if weights:
            max_surf_ZZ = np.full_like(ex_ll["pi_grid"], -np.inf)
            rr_ZZ = np.full_like(ex_ll["pi_grid"], 0)
            ZZ_name = "log_ZZ_neut"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full_like(ex_ll["pi_grid"], -np.inf)
                rr_ZZ_0 = np.full_like(ex_ll["pi_grid"], 0)
                ZZ_0_name = "log_ZZ_0_neut"
        ll_name = "llhood_neut"

    elif sel_model == "neut_db":
        return ex_ll# {"llhood_neut_db":ex_ll}

    elif sel_model == "stab":
        max_surf = np.full(ex_ll["I2_stab"].shape + ex_ll["pi_grid"].shape, -np.inf)
        rr = np.full(ex_ll["I2_stab"].shape + ex_ll["pi_grid"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I2_stab"].shape + ex_ll["pi_grid"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I2_stab"].shape + ex_ll["pi_grid"].shape, 0)
            ZZ_name = "log_ZZ_stab"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I2_stab"].shape + ex_ll["pi_grid"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I2_stab"].shape + ex_ll["pi_grid"].shape, 0)
                ZZ_0_name = "log_ZZ_0_stab"
        ll_name = "llhood_stab"

    elif sel_model == "stab_db":
        max_surf = np.full(ex_ll["I2_stab_db"].shape, -np.inf)
        rr = np.full(ex_ll["I2_stab_db"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I2_stab_db"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I2_stab_db"].shape, 0)
            ZZ_name = "log_ZZ_stab_db"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I2_stab_db"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I2_stab_db"].shape, 0)
                ZZ_0_name = "log_ZZ_0_stab_db"
        ll_name = "llhood_stab_db"

    elif sel_model == "dir":
        max_surf = np.full(ex_ll["I1_dir"].shape + ex_ll["pi_grid"].shape, -np.inf)
        rr = np.full(ex_ll["I1_dir"].shape + ex_ll["pi_grid"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I1_dir"].shape + ex_ll["pi_grid"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I1_dir"].shape + ex_ll["pi_grid"].shape, 0)
            ZZ_name = "log_ZZ_dir"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I1_dir"].shape + ex_ll["pi_grid"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I1_dir"].shape + ex_ll["pi_grid"].shape, 0)
                ZZ_0_name = "log_ZZ_0_dir"
        ll_name = "llhood_dir"

    elif sel_model == "dir_db":
        max_surf = np.full(ex_ll["I1_dir_db"].shape, -np.inf)
        rr = np.full(ex_ll["I1_dir_db"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I1_dir_db"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I1_dir_db"].shape, 0)
            ZZ_name = "log_ZZ_dir_db"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I1_dir_db"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I1_dir_db"].shape, 0)
                ZZ_0_name = "log_ZZ_0_dir_db"
        ll_name = "llhood_dir_db"

    elif sel_model == "full":
        max_surf = np.full(ex_ll["I2_full"].shape + ex_ll["pi_grid"].shape, -np.inf)
        rr = np.full(ex_ll["I2_full"].shape + ex_ll["pi_grid"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I2_full"].shape + ex_ll["pi_grid"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I2_full"].shape + ex_ll["pi_grid"].shape, 0)
            ZZ_name = "log_ZZ_full"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I2_full"].shape + ex_ll["pi_grid"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I2_full"].shape + ex_ll["pi_grid"].shape, 0)
                ZZ_0_name = "log_ZZ_0_full"
        ll_name = "llhood_full"

    elif sel_model == "full_db":
        max_surf = np.full(ex_ll["I2_full_db"].shape, -np.inf)
        rr = np.full(ex_ll["I2_full_db"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I2_full_db"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I2_full_db"].shape, 0)
            ZZ_name = "log_ZZ_full_db"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I2_full_db"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I2_full_db"].shape, 0)
                ZZ_0_name = "log_ZZ_0_full_db"
        ll_name = "llhood_full_db"

    elif sel_model == "plei":
        max_surf = np.full(ex_ll["Ip_plei"].shape + ex_ll["pi_grid"].shape, -np.inf)
        rr = np.full(ex_ll["Ip_plei"].shape + ex_ll["pi_grid"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["Ip_plei"].shape + ex_ll["pi_grid"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["Ip_plei"].shape + ex_ll["pi_grid"].shape, 0)
            ZZ_name = "log_ZZ_plei"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["Ip_plei"].shape + ex_ll["pi_grid"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["Ip_plei"].shape + ex_ll["pi_grid"].shape, 0)
                ZZ_0_name = "log_ZZ_0_plei"
        ll_name = "llhood_plei"

    elif sel_model == "plei_db":
        max_surf = np.full(ex_ll["Ip_plei_db"].shape, -np.inf)
        rr = np.full(ex_ll["Ip_plei_db"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["Ip_plei_db"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["Ip_plei_db"].shape, 0)
            ZZ_name = "log_ZZ_plei_db"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["Ip_plei_db"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["Ip_plei_db"].shape, 0)
                ZZ_0_name = "log_ZZ_0_plei_db"
        ll_name = "llhood_plei_db"

    elif sel_model == "nplei":
        max_surf = np.full(ex_ll["I2_nplei"].shape + ex_ll["pi_grid"].shape, -np.inf)
        rr = np.full(ex_ll["I2_nplei"].shape + ex_ll["pi_grid"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I2_nplei"].shape + ex_ll["pi_grid"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I2_nplei"].shape + ex_ll["pi_grid"].shape, 0)
            ZZ_name = "log_ZZ_nplei"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I2_nplei"].shape + ex_ll["pi_grid"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I2_nplei"].shape + ex_ll["pi_grid"].shape, 0)
                ZZ_0_name = "log_ZZ_0_nplei"
        ll_name = "llhood_nplei"

    elif sel_model == "nplei_db":
        max_surf = np.full(ex_ll["I2_nplei_db"].shape, -np.inf)
        rr = np.full(ex_ll["I2_nplei_db"].shape, 0)
        if weights:
            max_surf_ZZ = np.full(ex_ll["I2_nplei_db"].shape, -np.inf)
            rr_ZZ = np.full(ex_ll["I2_nplei_db"].shape, 0)
            ZZ_name = "log_ZZ_nplei_db"
            if use_ZZ_0:
                max_surf_ZZ_0 = np.full(ex_ll["I2_nplei_db"].shape, -np.inf)
                rr_ZZ_0 = np.full(ex_ll["I2_nplei_db"].shape, 0)
                ZZ_0_name = "log_ZZ_0_nplei_db"
        ll_name = "llhood_nplei_db"

    ## The variables (max_surf, rr), and (max_surf_ZZ, rr_ZZ) do have different meanings
    ## depending on what weighting is used.

    for ii in range(n_reps):
        with open(samp_llhoods[ii], "rb") as handle:
            ll = pickle.load(handle)
            if sel_model in ["stab_db", "plei_db", "nplei_db"]:
                ll = reduce_pi(ll, 0.5, sel_model=sel_model[0:-3])
            if weights:
                if not use_ZZ_0:
                    max_surf, rr = update_s(max_surf, rr, ll[ll_name] + ll[ZZ_name])
                    max_surf_ZZ, rr_ZZ = update_s(max_surf_ZZ, rr_ZZ, ll[ZZ_name])
                else:
                    max_surf, rr = update_s(max_surf, rr, np.where(np.isnan(ll[ll_name] +
                                                                            ll[ZZ_name] - ll[ZZ_0_name]),
                                                                   -np.inf,
                                                                   ll[ll_name] +
                                                                   ll[ZZ_name] - ll[ZZ_0_name]))
                    max_surf_ZZ, rr_ZZ = update_s(max_surf_ZZ, rr_ZZ,
                                                  np.where(np.isnan(ll[ZZ_name] - ll[ZZ_0_name]),
                                                           -np.inf,
                                                           ll[ZZ_name] - ll[ZZ_0_name]))
                    max_surf_ZZ_0, rr_ZZ_0 = update_s(max_surf_ZZ_0, rr_ZZ_0, ll[ZZ_0_name])
            else:
                max_surf, rr = update_s(max_surf, rr, ll[ll_name])

    if weights:
        ## Combination procedure is the same regardless of which weighting is used
        ex_ll[ll_name] = np.where(np.isinf(max_surf + np.log(rr)),
                                  -np.inf,
                                  max_surf + np.log(rr) - max_surf_ZZ - np.log(rr_ZZ))
        ex_ll[ZZ_name] = max_surf_ZZ + np.log(rr_ZZ)
        if use_ZZ_0:
            ## Save this just in case it is needed
            ex_ll[ZZ_0_name] = max_surf_ZZ_0 + np.log(rr_ZZ_0)
    else:
        ex_ll[ll_name] = max_surf + np.log(rr) - np.log(n_reps)

    return ex_ll

###
def get_ml(ll, model):
    if model == "neut":
        pi_neut = ll["pi_grid"][np.nanargmax(ll["llhood_neut"])]
        ll_neut = np.max(ll["llhood_neut"])
        return pi_neut, ll_neut
    if model == "neut_db":
        return ll["llhood_neut_db"],
    elif model == "dir":
        max_ind_dir = np.unravel_index(np.nanargmax(ll["llhood_dir"]), ll["llhood_dir"].shape)
        pi_dir = (np.ones(ll["I1_dir"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_dir]
        I1_dir = (ll["I1_dir"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_dir]
        ll_dir = np.max(ll["llhood_dir"])
        return pi_dir, I1_dir, ll_dir
    elif model == "dir_db":
        I1_dir_db = ll["I1_dir_db"][np.nanargmax(ll["llhood_dir_db"])]
        ll_dir_db = np.max(ll["llhood_dir_db"])
        return I1_dir_db, ll_dir_db
    elif model == "stab":
        max_ind_stab = np.unravel_index(np.nanargmax(ll["llhood_stab"]), ll["llhood_stab"].shape)
        pi_stab = (np.ones(ll["I2_stab"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_stab]
        I2_stab = (ll["I2_stab"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_stab]
        ll_stab = np.max(ll["llhood_stab"])
        return pi_stab, I2_stab, ll_stab
    elif model == "stab_db":
        I2_stab_db = ll["I2_stab_db"][np.nanargmax(ll["llhood_stab_db"])]
        ll_stab_db = np.max(ll["llhood_stab_db"])
        return I2_stab_db, ll_stab_db
    elif model == "full":
        max_ind_full = np.unravel_index(np.nanargmax(ll["llhood_full"]), ll["llhood_full"].shape)
        pi_full = (np.ones(ll["I2_full"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_full]
        I1_full = (ll["I1_full"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_full]
        I2_full = (ll["I2_full"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_full]
        ll_full = np.max(ll["llhood_full"])
        return pi_full, I1_full, I2_full, ll_full
    elif model == "full_db":
        I1_full_db = ll["I1_full_db"][np.nanargmax(ll["llhood_full_db"])]
        I2_full_db = ll["I2_full_db"][np.nanargmax(ll["llhood_full_db"])]
        ll_full_db = np.max(ll["llhood_full_db"])
        return I1_full_db, I2_full_db, ll_full_db
    elif model == "plei":
        max_ind_plei = np.unravel_index(np.nanargmax(ll["llhood_plei"]), ll["llhood_plei"].shape)
        pi_plei = (np.ones(ll["Ip_plei"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_plei]
        Ip_plei = (ll["Ip_plei"][:, np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_plei]
        ll_plei = np.max(ll["llhood_plei"])
        return pi_plei, Ip_plei, ll_plei
    elif model == "plei_db":
        Ip_plei_db = ll["Ip_plei_db"][np.nanargmax(ll["llhood_plei_db"])]
        ll_plei_db = np.max(ll["llhood_plei_db"])
        return Ip_plei_db, ll_plei_db
    elif model == "nplei":
        max_ind_nplei = np.unravel_index(np.nanargmax(ll["llhood_nplei"]), ll["llhood_nplei"].shape)
        pi_nplei = (np.ones(ll["I2_nplei"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_nplei]
        I2_nplei = (ll["I2_nplei"][:, np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_nplei]
        nn_nplei = (ll["nn_nplei"][:, np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_nplei]
        ll_nplei = np.max(ll["llhood_nplei"])
        return pi_nplei, I2_nplei, nn_nplei, ll_nplei
    elif model == "nplei_db":
        I2_nplei_db = ll["I2_nplei_db"][np.nanargmax(ll["llhood_nplei_db"])]
        nn_nplei_db = ll["nn_nplei_db"][np.nanargmax(ll["llhood_nplei_db"])]
        ll_nplei_db = np.nanmax(ll["llhood_nplei_db"])
        return I2_nplei_db, nn_nplei_db, ll_nplei_db
    else:
        return None

###
def llhood_to_maximums(ll):
    pi_neut = ll["pi_grid"][np.nanargmax(ll["llhood_neut"])]
    ll_neut = np.nanmax(ll["llhood_neut"])

    max_ind_dir = np.unravel_index(np.nanargmax(ll["llhood_dir"]), ll["llhood_dir"].shape)
    pi_dir = (np.ones(ll["I1_dir"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_dir]
    I1_dir = (ll["I1_dir"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_dir]
    ll_dir = np.nanmax(ll["llhood_dir"])

    I1_dir_db = ll["I1_dir_db"][np.nanargmax(ll["llhood_dir_db"])]
    ll_dir_db = np.nanmax(ll["llhood_dir_db"])

    max_ind_stab = np.unravel_index(np.nanargmax(ll["llhood_stab"]), ll["llhood_stab"].shape)
    pi_stab = (np.ones(ll["I2_stab"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_stab]
    I2_stab = (ll["I2_stab"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_stab]
    ll_stab = np.nanmax(ll["llhood_stab"])

    max_ind_full = np.unravel_index(np.nanargmax(ll["llhood_full"]), ll["llhood_full"].shape)
    pi_full = (np.ones(ll["I2_full"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_full]
    I1_full = (ll["I1_full"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_full]
    I2_full = (ll["I2_full"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_full]
    ll_full = np.nanmax(ll["llhood_full"])

    I1_full_db = ll["I1_full_db"][np.nanargmax(ll["llhood_full_db"])]
    I2_full_db = ll["I2_full_db"][np.nanargmax(ll["llhood_full_db"])]
    ll_full_db = np.nanmax(ll["llhood_full_db"])

    return {"pi_ml_neut":pi_neut, "ll_ml_neut": ll_neut,
            "pi_ml_dir": pi_dir, "I1_ml_dir":I1_dir, "ll_ml_dir": ll_dir,
            "I1_ml_dir_db":I1_dir_db, "ll_ml_dir_db": ll_dir_db,
            "pi_ml_stab": pi_stab, "I2_ml_stab":I2_stab, "ll_ml_stab": ll_stab,
            "pi_ml_full": pi_full, "I1_ml_full":I1_full, "I2_ml_full":I2_full, "ll_ml_full": ll_full,
            "I1_ml_full_db":I1_full_db, "I2_ml_full_db":I2_full_db, "ll_ml_full_db": ll_full_db}

###
def llhood_to_maximums_db(ll, simple=False):
    pi_neut = ll["pi_grid"][np.nanargmax(ll["llhood_neut"])]
    ll_neut = np.nanmax(ll["llhood_neut"])

    ll_neut_db = ll["llhood_neut_db"]

    I1_dir_db = ll["I1_dir_db"][np.nanargmax(ll["llhood_dir_db"])]
    ll_dir_db = np.nanmax(ll["llhood_dir_db"])

    max_ind_stab = np.unravel_index(np.nanargmax(ll["llhood_stab"]), ll["llhood_stab"].shape)
    pi_stab = (np.ones(ll["I2_stab"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_stab]
    I2_stab = (ll["I2_stab"][:,np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_stab]
    ll_stab = np.nanmax(ll["llhood_stab"])

    I2_stab_db = ll["I2_stab_db"][np.nanargmax(ll["llhood_stab_db"])]
    ll_stab_db = np.nanmax(ll["llhood_stab_db"])

    if not simple:
        I1_full_db = ll["I1_full_db"][np.nanargmax(ll["llhood_full_db"])]
        I2_full_db = ll["I2_full_db"][np.nanargmax(ll["llhood_full_db"])]
        ll_full_db = np.nanmax(ll["llhood_full_db"])

        result =  {"pi_ml_neut":pi_neut, "ll_ml_neut":ll_neut, "ll_ml_neut_db":ll_neut_db,
                    "I1_ml_dir_db":I1_dir_db, "ll_ml_dir_db":ll_dir_db,
                    "pi_ml_stab": pi_stab, "I2_ml_stab":I2_stab, "ll_ml_stab":ll_stab,
                    "I2_ml_stab_db":I2_stab_db, "ll_ml_stab_db":ll_stab_db,
                    "I1_ml_full_db":I1_full_db, "I2_ml_full_db":I2_full_db, "ll_ml_full_db":ll_full_db}
    else:
        result =  {"pi_ml_neut":pi_neut, "ll_ml_neut":ll_neut, "ll_ml_neut_db":ll_neut_db,
                    "I1_ml_dir_db":I1_dir_db, "ll_ml_dir_db":ll_dir_db,
                    "pi_ml_stab": pi_stab, "I2_ml_stab":I2_stab, "ll_ml_stab":ll_stab,
                    "I2_ml_stab_db":I2_stab_db, "ll_ml_stab_db":ll_stab_db}

    if "llhood_s" in ll.keys():
        s_stab_db = ll["s_set"][np.nanargmax(ll["llhood_s"])]
        ll_s = np.nanmax(ll["llhood_s"])
        result["s_ml"] = s_stab_db
        result["ll_ml_s"] = ll_s

    return result

###
def llhood_to_maximums_plei_db(ll, simple=False):
    max_ind_plei = np.unravel_index(np.nanargmax(ll["llhood_plei"]), ll["llhood_plei"].shape)
    pi_plei = (np.ones(ll["Ip_plei"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_plei]
    Ip_plei = (ll["Ip_plei"][:, np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_plei]
    ll_plei = np.nanmax(ll["llhood_plei"])

    Ip_plei_db = ll["Ip_plei_db"][np.nanargmax(ll["llhood_plei_db"])]
    ll_plei_db = np.nanmax(ll["llhood_plei_db"])

    if not simple:
        max_ind_nplei = np.unravel_index(np.nanargmax(ll["llhood_nplei"]), ll["llhood_nplei"].shape)
        pi_nplei = (np.ones(ll["I2_nplei"].shape + ll["pi_grid"].shape)*ll["pi_grid"])[max_ind_nplei]
        I2_nplei = (ll["I2_nplei"][:, np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_nplei]
        nn_nplei = (ll["nn_nplei"][:, np.newaxis]*np.ones_like(ll["pi_grid"]))[max_ind_nplei]
        ll_nplei = np.nanmax(ll["llhood_nplei"])

        I2_nplei_db = ll["I2_nplei_db"][np.nanargmax(ll["llhood_nplei_db"])]
        nn_nplei_db = ll["nn_nplei_db"][np.nanargmax(ll["llhood_nplei_db"])]
        ll_nplei_db = np.nanmax(ll["llhood_nplei_db"])

        return {"pi_ml_plei": pi_plei, "Ip_ml_plei":Ip_plei, "ll_ml_plei": ll_plei,
                "Ip_ml_plei_db":Ip_plei_db, "ll_ml_plei_db": ll_plei_db,
                "pi_ml_nplei": pi_nplei, "I2_ml_nplei":I2_nplei, "nn_ml_nplei":nn_nplei,
                "ll_ml_nplei":ll_nplei,
                "I2_ml_nplei_db":I2_nplei_db, "nn_ml_nplei_db":nn_nplei_db,
                "ll_ml_nplei_db":ll_nplei_db}
    else:
        return {"pi_ml_plei": pi_plei, "Ip_ml_plei":Ip_plei, "ll_ml_plei": ll_plei,
                "Ip_ml_plei_db":Ip_plei_db, "ll_ml_plei_db": ll_plei_db}

###
def reduce_pi(ll, pi_val, sel_model):
    pi_grid = ll["pi_grid"]
    if np.sum(pi_grid == pi_val) < 1:
        print("pi value not present in grid")
        return None
    ii_pi = np.min(np.where(pi_grid == pi_val)) # min just gest first instance here
    result = None
    if sel_model == "neut":
        result = {}
        result["llhood_neut_db"] = ll["llhood_neut"][ii_pi]
    if sel_model == "stab":
        result = {}
        result["llhood_stab_db"] = ll["llhood_stab"][:,ii_pi]
        if "log_ZZ_stab" in ll.keys():
            result["log_ZZ_stab_db"] = ll["log_ZZ_stab"][:,ii_pi]
        if "log_ZZ_0_stab" in ll.keys():
            result["log_ZZ_0_stab_db"] = ll["log_ZZ_0_stab"][:,ii_pi]
        result["I2_stab_db"] = ll["I2_stab"]
        result["w_stab_db"] = get_weights(ll["I2_stab"])
    elif sel_model == "plei":
        result = {}
        result["llhood_plei_db"] = ll["llhood_plei"][:,ii_pi]
        if "log_ZZ_plei" in ll.keys():
            result["log_ZZ_plei_db"] = ll["log_ZZ_plei"][:,ii_pi]
        if "log_ZZ_0_plei" in ll.keys():
            result["log_ZZ_0_plei_db"] = ll["log_ZZ_0_plei"][:,ii_pi]
        result["Ip_plei_db"] = ll["Ip_plei"]
        result["w_plei_db"] = get_weights(ll["Ip_plei"])
    elif sel_model == "nplei":
        result = {}
        result["llhood_nplei_db"] = ll["llhood_nplei"][:,ii_pi]
        if "log_ZZ_nplei" in ll.keys():
            result["log_ZZ_nplei_db"] = ll["log_ZZ_nplei"][:,ii_pi]
        if "log_ZZ_0_nplei" in ll.keys():
            result["log_ZZ_0_nplei_db"] = ll["log_ZZ_0_nplei"][:,ii_pi]
        result["I2_nplei_db"] = ll["I2_nplei"]
        result["nn_nplei_db"] = ll["nn_nplei"]
        result["w_nplei_db"] = ll["w_nplei"][:,ii_pi]
    return result

###
def posterior_median_plei(raf_set, beta_set, Ip, Ne, n_s=200):
    S_ud_set = np.logspace(-2, 3.0, n_s)
    S_ud_medians = np.zeros_like(beta_set)
    for ii, beta in enumerate(beta_set):
        raf = raf_set[ii]
        sfs_vals = (sim.sfs_ud_params_sigma(raf, 1, S_ud_set) +
                        sim.sfs_ud_params_sigma(1-raf, 1, S_ud_set))
        lower_val = (sim.levy_cdf(np.abs(S_ud_set[0]), 2*Ne*Ip*beta**2)*
                         sim.sfs_ud_params_sigma(raf, 1, S_ud_set[0]))
        lower_val += (sim.levy_cdf(np.abs(S_ud_set[0]), 2*Ne*Ip*beta**2)*
                          sim.sfs_ud_params_sigma(1-raf, 1, S_ud_set[0]))
        upper_val = ((1-sim.levy_cdf(np.abs(S_ud_set[-1]), 2*Ne*Ip*beta**2))*
                         sim.sfs_ud_params_sigma(raf, 1, S_ud_set[-1]))
        upper_val += ((1-sim.levy_cdf(np.abs(S_ud_set[-1]), 2*Ne*Ip*beta**2))*
                          sim.sfs_ud_params_sigma(1-raf, 1, S_ud_set[-1]))
        x_find_weights = np.concatenate(([lower_val],
                                             (get_weights(S_ud_set)*sfs_vals*
                                              sim.levy_density(S_ud_set, 2*Ne*Ip*beta**2)),
                                             [upper_val]))
        S_expand = np.concatenate(([S_ud_set[0]], S_ud_set, [S_ud_set[-1]]))
        ## Get S value of first point not less than 0.5
        median_S_x = S_expand[np.nanargmax(np.logical_not(np.cumsum(x_find_weights)/np.sum(x_find_weights) < 0.5))]
        S_ud_medians[ii] = median_S_x
    return S_ud_medians

###
def posterior_median_plei_WF(raf_set, beta_set, Ip, Ne, WF_pile, n_s=200):
    S_ud_set = np.logspace(-2.0, 3.0, n_s)
    S_ud_medians = np.zeros_like(beta_set)
    for ii, beta in enumerate(beta_set):
        raf = raf_set[ii]
        sfs = sim.sfs_ud_WF_grid(S_ud_set, WF_pile, np.array([raf]))[0]
        sfs += sim.sfs_ud_WF_grid(S_ud_set, WF_pile, np.array([1-raf]))[0] # len n_s
        lower_val = (sim.levy_cdf(np.abs(S_ud_set[0]), 2*Ne*Ip*beta**2)*sfs[0])
        upper_val = ((1-sim.levy_cdf(np.abs(S_ud_set[-1]), 2*Ne*Ip*beta**2))*sfs[-1])
        x_find_weights = np.concatenate(([lower_val],
                                                (get_weights(S_ud_set)*sfs*
                                                    sim.levy_density(S_ud_set, 2*Ne*Ip*beta**2)),
                                                [upper_val]))
        S_expand = np.concatenate(([S_ud_set[0]], S_ud_set, [S_ud_set[-1]]))
        ## Get S value of first point not less than 0.5
        median_S_x = S_expand[np.nanargmax(np.logical_not(np.cumsum(x_find_weights)/np.sum(x_find_weights) < 0.5))]
        S_ud_medians[ii] = median_S_x
    return S_ud_medians