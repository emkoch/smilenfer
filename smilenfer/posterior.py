from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.stats as stats

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

from . import statistics as smile_stats

import scipy.stats as stats
import scipy.optimize as opt

def inv_logsf_chi2(neg_log10_p, df=1):
    ## Solve for the chi2 value that corresponds to a given p-value
    ## using the inverse survival function of the chi2 distribution

    # define the chi2 inverse survivaf function
    log10sf = lambda x: -stats.chi2.logsf(x, df) / np.log(10)
    
    if neg_log10_p > 305:
        return 4.59116667840005 * neg_log10_p - 5.1735856012059855

    # solve for the chi2 value that corresponds to the p-value
    chi2_val = opt.newton(lambda x: log10sf(x) - neg_log10_p, 1)

    return chi2_val

inv_logsf_chi2_vec = np.vectorize(inv_logsf_chi2)

def scientific_to_log10(scientific_str):
    # Split the string into mantissa and exponent parts
    mantissa_str, exponent_str = scientific_str.split('E')

    # Convert mantissa and exponent parts to float
    mantissa = float(mantissa_str)
    exponent = float(exponent_str)

    # Calculate the log10 value
    log10_value = np.log10(mantissa) + exponent

    return log10_value

def samp_setup(trait_fname, p_thresh, min_x, post_samps, post_samp_set,
               sep=None, compression="infer", beta_col="orig_b", freq_col="freq", pp="orig_p", p_cutoff=np.inf,
               alt_freq_col=None, var_inflation_cutoff=None, trait_data=None):
    if trait_data is None:
        trait_data = read_trait_data(trait_fname, sep=sep, compression=compression,
                                    beta_col=beta_col, freq_col=freq_col, alt_freq_col=alt_freq_col,
                                    var_inflation_cutoff=var_inflation_cutoff)
    non_nan = ~np.isnan(trait_data[freq_col]) & np.array(trait_data.orig_var_exp>0)
    v_cutoff = smile_stats.calc_cutoffs_new(trait_data.orig_var_exp[non_nan],
                                              trait_data[pp][non_nan])[p_thresh]
    # np.nan < x should evaluate to False
    cut_rows = np.array(trait_data.orig_var_exp > v_cutoff) & np.array(trait_data.maf >= min_x)
    cut_rows = cut_rows & np.array(trait_data[pp] <= p_cutoff)

    with open(post_samps, "r") as handle:
        samp_vals = handle.readlines()
    samp_vals = np.array([float(bb.strip()) for bb in samp_vals])[cut_rows]

    all_samp_vals = get_all_samp_vals(post_samp_set, cut_rows)

    beta_obs = np.abs(trait_data[cut_rows][beta_col].to_numpy())
    x_data = trait_data[cut_rows].raf.to_numpy()

    # Flip the raf if we sample a negative value by chance (rbeta always positive)
    x_data = np.where(samp_vals < 0, 1-x_data, x_data)

    return x_data, beta_obs, samp_vals, all_samp_vals, v_cutoff

def custom_converter(value):
    try:
        float_value = float(value)
        if abs(float_value) <= 0:
            return -scientific_to_log10(value)  # Return the value as string
        else:
            return -np.log10(float_value)  # Return the value as float
    except ValueError:
        if value == "0.0":
            return np.inf
        return value  # Return the original value if conversion fails

def read_and_process_trait_data(trait_fname, cut_top=0, cojo_filter=False, rel_n_eff_min=-np.inf, rel_n_eff_max=np.inf,
                                style="finngen", sep=None, compression="infer",
                                beta_col="orig_b", freq_col="freq", alt_freq_col=None,
                                var_inflation_cutoff=None, RDS=None, max_r2_cutoff=0.2):
    """
    Read and process trait data from a file.

    Parameters:
    trait_fname (str): The file path of the trait data file.
    style (str, optional): The style of the trait data. Default is "finngen".
    sep (str, optional): The separator used in the trait data file. Default is None.
    compression (str, optional): The compression format of the trait data file. Default is "infer".
    beta_col (str, optional): The column name for the beta values. Default is "orig_b".
    freq_col (str, optional): The column name for the frequency values. Default is "freq".
    alt_freq_col (str, optional): The column name for the alternate frequency values. Default is None.
    var_inflation_cutoff (float, optional): The cutoff value for variant inflation. Default is None.

    Returns:
    pandas.DataFrame: The processed trait data.

    Processing Steps:
    1. Read the trait data from the specified file.
    2. If the style is "finngen":
        - Convert the p-values to mlogp values.
        - Remove rows where the raf is missing or the variant is not significant in the meta-analysis.
        - Replace zero values in the PosteriorMean column with the original beta values.
        - Calculate the minor allele frequency (maf).
        - Calculate the variance explained (var_exp).
        - Calculate the effective sample size (n_eff).
        - Calculate the median effective sample size (median_n_eff).
        - Calculate the relative effective sample size (n_eff_rel).
    3. If the style is "old":
        - Read the trait data using the old function.
    """
    
    if style == "finngen":
        trait_data = pd.read_csv(trait_fname, sep="\t", compression=compression, converters={'pval': custom_converter})
        if not "mlogp" in trait_data.columns:
            trait_data["mlogp"] = trait_data.pval.to_numpy()
        if "neglog10_pval" in trait_data.columns:
            trait_data["mlogp"] = trait_data.neglog10_pval.to_numpy()
        if "neglog10p" in trait_data.columns:
            trait_data["mlogp"] = trait_data.neglog10p.to_numpy()
        trait_data["pval"] = 10**(-trait_data.mlogp)
        # Remove rows where the following columns are missing: raf, rbeta, pval, se
        trait_data = trait_data.loc[~np.isnan(trait_data.raf), :]
        trait_data = trait_data.loc[~np.isnan(trait_data.rbeta), :]
        trait_data = trait_data.loc[~np.isnan(trait_data.mlogp), :]
        trait_data = trait_data.loc[~np.isnan(trait_data.se), :]
        if "max_r2" in trait_data.columns:
            n_variants = trait_data.shape[0]
            trait_data = trait_data.loc[trait_data.max_r2 < max_r2_cutoff, :]
            print(trait_fname)
            # print(f"Removed {n_variants-trait_data.shape[0]} variants with max_r2 > {max_r2_cutoff}")
            print(f"Remaining variants: {trait_data.shape[0]} out of {n_variants} after removing variants with max_r2 >= {max_r2_cutoff}")
        # Filter the MHC
        mhc_set = ((trait_data.chr.to_numpy(dtype=int) == 6) & 
                   (trait_data.pos.to_numpy(dtype=int) > 24e6) & 
                   (trait_data.pos.to_numpy(dtype=int) < 36e6))
        trait_data = trait_data.loc[~mhc_set, :]
        print(f"Removed {np.sum(mhc_set)} variants near the MHC")
            
        # Remove rows where cojo_locus is True
        if cojo_filter:
            trait_data = trait_data.loc[~trait_data.cojo_locus, :]
        # Where PosteriorMean is zero, replace with the original beta
        trait_data["PosteriorMean"] = np.where((trait_data.PosteriorMean.to_numpy() == 0) | (trait_data.se.to_numpy() == 0),
                                                trait_data.rbeta,
                                                trait_data.PosteriorMean)
        trait_data["PosteriorMean"] = np.abs(trait_data.PosteriorMean)
        # calculate MAF
        trait_data["maf"] = np.minimum(trait_data.raf, 1-trait_data.raf)
        # calculate var_exp
        trait_data["var_exp"] = 2 * trait_data.raf * (1-trait_data.raf) * trait_data.rbeta**2
        # calculate n_eff
        trait_data["n_eff"] = inv_logsf_chi2_vec(trait_data.mlogp) / trait_data.var_exp
        # calculate median n_eff
        trait_data["median_n_eff"] = np.median(trait_data.n_eff)
        # calculate the relative n_eff
        trait_data["n_eff_rel"] = trait_data.n_eff / trait_data.median_n_eff
        # Filter out variants with n_eff_rel outside of the specified range
        trait_data = trait_data.loc[(trait_data.n_eff_rel >= rel_n_eff_min) & (trait_data.n_eff_rel <= rel_n_eff_max), :]
    elif style == "old":
        trait_data = read_trait_data(trait_fname, sep=sep, compression=compression,
                                     beta_col=beta_col, freq_col=freq_col, alt_freq_col=alt_freq_col,
                                     var_inflation_cutoff=var_inflation_cutoff)
    else:
        raise ValueError("style not yet implemented: must be one of 'finngen' or 'old'")
    
    # check if a1, a2 in columns
    if "A1" in trait_data.columns and "A2" in trait_data.columns:
        pass
    elif "a1" in trait_data.columns and "a2" in trait_data.columns:
        trait_data["A1"] = trait_data.a1.to_numpy(dtype=str)
        trait_data["A2"] = trait_data.a2.to_numpy(dtype=str)
    elif "ref" in trait_data.columns and "alt" in trait_data.columns:
        trait_data["A1"] = trait_data.ref.to_numpy(dtype=str)
        trait_data["A2"] = trait_data.alt.to_numpy(dtype=str)
    else:
        raise ValueError("(A1, A2), (a1, a2), or (ref, alt) columns must be present in the data")
    
    if cut_top > 0:
        beta_data = trait_data.rbeta.to_numpy()
        x_data = trait_data.raf.to_numpy()
        v_set = 2*beta_data**2*x_data*(1-x_data)
        v_top = np.sort(v_set)[-cut_top]
        # Remove the top variants
        trait_data = trait_data.loc[v_set < v_top, :]

    return trait_data.reset_index(drop=True).copy(deep=True)

def read_trait_data(trait_fname, sep=None, compression="infer",
                    beta_col="orig_b", freq_col="freq", alt_freq_col=None,
                    var_inflation_cutoff=None):
    """
    Read trait data from a file and perform data preprocessing.

    Args:
        trait_fname (str): The file path of the trait data.
        sep (str, optional): The delimiter used in the file. Defaults to None.
        compression (str, optional): The compression type of the file. Defaults to "infer".
        beta_col (str, optional): The column name for the beta values. Defaults to "orig_b".
        freq_col (str, optional): The column name for the frequency values. Defaults to "freq".
        alt_freq_col (str, optional): The column name for the alternative frequency values. Defaults to None.
        var_inflation_cutoff (float, optional): The cutoff value for variance inflation. Defaults to None.

    Returns:
        pandas.DataFrame: The preprocessed trait data.
    """
    trait_data = pd.read_csv(trait_fname, sep=sep, compression=compression)

    if alt_freq_col is not None:
        # Replace missing eaf (distinct from the column named "eaf") values in freq_col with alt_freq_col
        # Mostly this will be used to replace missing "topmed_af" values with "eaf"
        eaf_missing = np.isnan(trait_data[freq_col])
        trait_data.loc[eaf_missing, freq_col] = trait_data[alt_freq_col][eaf_missing].to_numpy()

    if "topmed_af" in trait_data.keys():
        assert "eaf" in trait_data.keys() or "af" in trait_data.keys(), "eaf column not present in data"
        # Align topmed allele frequencies with original eaf
        # If not topmed_af, given freq_col is assumed to be eaf
        if not "eaf" in trait_data.keys():
            trait_data["eaf"] = trait_data["af"]
        eaf = trait_data.eaf.to_numpy()
        topmed_af = trait_data.topmed_af.to_numpy()
        # Flip topmed_af if it is closer to 1-eaf than eaf
        flip_topmed = (np.abs(eaf - topmed_af > np.abs(1 - eaf - topmed_af)))
        trait_data["topmed_af"] = np.where(flip_topmed, 1-topmed_af, topmed_af)
    
    # Calculate RAF now that we have gotten eaf aligned
    # Warning: this overwrites original raf values in the data frame so that they 
    #          reflect the freq_col rathern than "eaf"
    trait_data["raf"] = np.where(trait_data[beta_col] > 0,
                                 trait_data[freq_col],
                                 1 - trait_data[freq_col])
    # Same for MAF
    trait_data["maf"] = np.minimum(trait_data.raf, 1-trait_data.raf)
    # This was for removing variants where COJO caused a large variance change, 
    # but these should be filtered out ahead of time in the data preprocessing now
    if var_inflation_cutoff is not None:
        over_cutoff = (np.array(2*trait_data.raf*(1-trait_data.raf) * trait_data[beta_col]**2) >
                       (var_inflation_cutoff * trait_data.orig_var_exp.to_numpy()))
        # Remove points where the COJO var_exp is > a * the original, do this before we overwrite the original
        trait_data = trait_data.loc[~over_cutoff, :]
    # Overwrite orig_var_exp
    trait_data["orig_var_exp"] = 2 * trait_data.raf * (1-trait_data.raf) * trait_data[beta_col]**2
    # if orig_se in trait_data.columns, overwrite PosteriorMean using orig_beta where orig_se is zero
    # then overwrite orig_se to some small value
    if "orig_se" in trait_data.columns:
        trait_data["PosteriorMean"] = np.where(trait_data.orig_se > 0,
                                               trait_data.orig_beta,
                                               trait_data.PosteriorMean)
        trait_data["orig_se"] = np.where(trait_data.orig_se > 0,
                                         trait_data.orig_se,
                                         1e-6)

    return trait_data

def get_all_samp_vals(samp_fnames, cut_rows):
    all_samp_vals = []
    for samp_set in samp_fnames:
        with open(samp_set, "r") as handle:
            samp_vals_tmp = handle.readlines()
        samp_vals_tmp = np.abs(np.array([float(bb.strip()) for bb in samp_vals_tmp])[cut_rows])
        all_samp_vals.append(samp_vals_tmp)
    all_samp_vals = np.concatenate(all_samp_vals)
    return all_samp_vals

def make_post_sample(beta, se, nsamp, RDS_fname):
    robjects.r('''
    library(\"ashr\")
    run_post_sample <- function(RDS.fname, xx, ss, nsamp){
    gg <- readRDS(RDS.fname)
    if(class(gg)==\"ash\"){
        result <- ashr::post_sample(gg$fitted_g, list(x=xx, s=ss, lik=list(name=\"normal\")), nsamp=nsamp)
    } else if(class(gg)==\"normalmix\"){
        result <- ashr::post_sample(gg, list(x=xx, s=ss, lik=list(name=\"normal\")), nsamp=nsamp)   
    }
    return(result)
    }
    ''')
    run_post_sample = robjects.globalenv["run_post_sample"]
    result = run_post_sample(RDS_fname,
                             robjects.FloatVector(beta),
                             robjects.FloatVector(se), nsamp=nsamp)
    return np.array(result)

def make_post_mean_sd(beta, se, RDS_fname):
    robjects.r('''
    library(\"ashr\")
    post_mean <- function(RDS.fname, xx, ss){
    gg <- readRDS(RDS.fname)
    if(class(gg)==\"ash\"){
        result <- ashr::postmean(gg$fitted_g, list(x=xx, s=ss, lik=list(name=\"normal\")))
    } else if(class(gg)==\"normalmix\"){
        result <- ashr::postmean(gg, list(x=xx, s=ss, lik=list(name=\"normal\")))   
    }
    return(result)
    }
               
    post_sd <- function(RDS.fname, xx, ss){
    gg <- readRDS(RDS.fname)
    if(class(gg)==\"ash\"){
        result <- ashr::postsd(gg$fitted_g, list(x=xx, s=ss, lik=list(name=\"normal\")))
    } else if(class(gg)==\"normalmix\"){
        result <- ashr::postsd(gg, list(x=xx, s=ss, lik=list(name=\"normal\")))   
    }
    return(result)
    }
    ''')
    post_mean = robjects.globalenv["post_mean"]
    post_sd = robjects.globalenv["post_sd"]

    mean = post_mean(RDS_fname, robjects.FloatVector(beta), robjects.FloatVector(se))
    sd = post_sd(RDS_fname, robjects.FloatVector(beta), robjects.FloatVector(se))
    return np.array(mean), np.array(sd)

def make_prior_sample(nsamp, RDS_fname):
    base = importr("base")
    ashr = importr("ashr")
    gg = base.readRDS(RDS_fname)
    sds_samp = np.random.choice(gg[2], nsamp, gg[0])
    return stats.norm.rvs(0, sds_samp)