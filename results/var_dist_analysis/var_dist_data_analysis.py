import os
import math

import smilenfer.var_dist as vd
import smilenfer.statistics as smile_stats
import smilenfer.plotting as splot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

data_dir = "../data"

_, _, _, _, _, _, all_traits, all_labels, data_traits_update = splot.read_trait_files(os.path.join(data_dir, "clumped_ash"))

# Do some filtering on maf and p-value
for trait in all_traits:
    trait_data = data_traits_update[trait]
    trait_data = trait_data.loc[trait_data.maf.to_numpy() > 0.01, :]
    trait_data = trait_data.loc[trait_data.pval.to_numpy() < 1e-5, :]
    trait_data["var_exp_ash"] = 2*trait_data.raf*(1-trait_data.raf)*trait_data.PosteriorMean**2
    data_traits_update[trait] = trait_data

# Calculate approximate p-value cutoffs for variance explained
p_cutoffs = {}
for trait in all_traits:
    trait_data = data_traits_update[trait]
    v_cuts = smile_stats.calc_cutoffs_new(trait_data.var_exp.to_numpy(), trait_data.pval.to_numpy(), 
                                 n_eff=trait_data.median_n_eff.to_numpy()[0])
    p_cutoffs[trait] = v_cuts

# Function to fit the 1d and hD distributions of variance explained
# given a set of variance explained values and a set of variance explained cutoffs
def fit_var_dist(vv, vv_cuts):
    # Fit the 1d distribution
    eta_1d_set = [vd.fit_stab_1D((vv[vv > vv_cut] - vv_cut)/vv_cut) for vv_cut in vv_cuts]
    # Fit the hD distribution
    eta_hd_set = [vd.fit_stab_hD((vv[vv > vv_cut] - vv_cut)/vv_cut) for vv_cut in vv_cuts]
    # Calculate the ll values
    eta_1d_ll = np.array([np.sum(vd.ll_stab_1D(vv[vv>vv_cut], eta)) 
                          for eta, vv_cut in zip(eta_1d_set, vv_cuts)])
    eta_hd_ll = np.array([np.sum(vd.ll_stab_hD(vv[vv>vv_cut], eta))
                            for eta, vv_cut in zip(eta_hd_set, vv_cuts)])
    return np.array(eta_1d_set), np.array(eta_hd_set), eta_1d_ll, eta_hd_ll
    
# Fit the distribution of variance contributions for each trait
var_fits = {}
var_fits_ash = {}
var_fits_2x = {}
var_fits_ash_2x = {}

var_fits_5e8 = {}
var_fits_ash_5e8 = {}
var_fits_2x_5e8 = {}
var_fits_ash_2x_5e8 = {}

for trait in all_traits:
    trait_data = data_traits_update[trait]
    vv = trait_data.var_exp.to_numpy()
    vv_ash = trait_data.var_exp_ash.to_numpy()
    # check if the trait one of the UKBB/FinnGen traits
    vv_min = np.min(vv)
    vv_max = np.max(vv)
    vv_min_ash = np.min(vv_ash)
    vv_max_ash = np.max(vv_ash)

    vv_set = np.logspace(np.log10(vv_min*0.99), np.log10(vv_max*0.9), 100)
    vv_cutoff_5e8 = p_cutoffs[trait]["5e-08"]

    eta_1d_set, eta_hd_set, eta_1d_ll, eta_hd_ll = fit_var_dist(vv, vv_set)
    var_fits[trait] = {"vv_cutoff": vv_set, "eta_1d": eta_1d_set,
                       "eta_hd": eta_hd_set, "1d_ll": eta_1d_ll,
                       "hd_ll": eta_hd_ll}
    
    # Do it just for the 5e-08 cutoff
    eta_1d_5e8, eta_hd_5e8, eta_1d_ll_5e8, eta_hd_ll_5e8 = fit_var_dist(vv[trait_data.pval<5e-08], [vv_cutoff_5e8])
    var_fits_5e8[trait] = {"vv": vv[trait_data.pval<5e-08].copy(), "vv_cutoff": vv_cutoff_5e8, "eta_1d": eta_1d_5e8[0],
                            "eta_hd": eta_hd_5e8[0], "1d_ll": eta_1d_ll_5e8[0],
                            "hd_ll": eta_hd_ll_5e8[0]}
    
    # Do the same for the ASH variance contributions
    vv_set_ash = np.logspace(np.log10(vv_min_ash*0.99), np.log10(vv_max_ash*0.9), 100)
    eta_1d_set_ash, eta_hd_set_ash, eta_1d_ll_ash, eta_hd_ll_ash = fit_var_dist(vv_ash, vv_set_ash)
    var_fits_ash[trait] = {"vv_cutoff": vv_set_ash, "eta_1d": eta_1d_set_ash,
                            "eta_hd": eta_hd_set_ash, "1d_ll": eta_1d_ll_ash,
                            "hd_ll": eta_hd_ll_ash}
    
    eta_1d_ash_5e8, eta_hd_ash_5e8, eta_1d_ll_ash_5e8, eta_hd_ll_ash_5e8 = fit_var_dist(vv_ash, [vv_cutoff_5e8])
    var_fits_ash_5e8[trait] = {"vv":vv_ash, "vv_cutoff": vv_cutoff_5e8, "eta_1d": eta_1d_ash_5e8[0],
                                "eta_hd": eta_hd_ash_5e8[0], "1d_ll": eta_1d_ll_ash_5e8[0],
                                "hd_ll": eta_hd_ll_ash_5e8[0]}

# Make a QQ plot for the variance distribution against each fit
def plot_qq(var_fits, title=""):
    n_traits = len(var_fits)
    n_rows = math.ceil(n_traits/5)
    fig, axs = plt.subplots(n_rows, 5, figsize=(30, 20))
    axs = axs.flatten()
    for i, trait in enumerate(all_traits):
        ax = axs[i]
        vv = var_fits[trait]["vv"][(var_fits[trait]["vv"] > var_fits[trait]["vv_cutoff"])]
        vv_sorted = (np.sort(vv) - var_fits[trait]["vv_cutoff"])/var_fits[trait]["vv_cutoff"]
        qq = (np.arange(len(vv_sorted)) + 1) / (len(vv_sorted) + 1)
        eta_1d = var_fits[trait]["eta_1d"]
        eta_hd = var_fits[trait]["eta_hd"]
        qq_1d = vd.inv_F_stab_1D(qq, eta_1d)
        qq_hd = vd.inv_F_stab_hD(qq, eta_hd)
        ax.plot(qq_1d, vv_sorted, "o", label="single-trait", alpha=0.7)
        ax.plot(qq_hd, vv_sorted, "o", label="pleiotropic", alpha=0.7)
        ax.plot([np.min(vv_sorted), np.max(vv_sorted)], [np.min(vv_sorted), np.max(vv_sorted)], "--", color="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(trait)
        ax.text(0.1, 0.9, r"$\Delta \ell_\mathrm{PLEI-1T}: $" + "{:.1f}".format(-var_fits[trait]["1d_ll"] + var_fits[trait]["hd_ll"]),
                transform=ax.transAxes, fontsize=20, verticalalignment="top")

    axs[0].legend(loc="lower right", bbox_to_anchor=(0, 1.1), ncol=1, fontsize=20)
    fig.text(0.5, 0.01, "Theoretical quantiles", ha="center", va="center", fontsize=50)
    fig.text(0.01, 0.5, "Variance contribution", ha="center", va="center", rotation="vertical", fontsize=50)
    for i in range(len(all_traits), len(axs)):
        axs[i].axis("off")
    fig.tight_layout()
    fig.savefig("qq_plot_{}.pdf".format(title), bbox_inches="tight")

plot_qq(var_fits_5e8, "5e8")
plot_qq(var_fits_ash_5e8, "5e8_ash")

# Function to plot the log likelihood difference between the 1D and hD models
# at each variance cutoff, for each trait
def plot_ll_diff(var_fits, var_fits_ash):
    n_traits = len(var_fits)
    n_rows = math.ceil(n_traits/5)
    fig, axs = plt.subplots(n_rows, 5, figsize=(30, 20))
    axs = axs.flatten()
    for i, trait in enumerate(all_traits):
        ax = axs[i]
        ax.plot(var_fits[trait]["vv_cutoff"], 
                var_fits[trait]["hd_ll"] - var_fits[trait]["1d_ll"], "-", label="RAW")
        ax.plot(var_fits_ash[trait]["vv_cutoff"], 
                var_fits_ash[trait]["hd_ll"] - var_fits_ash[trait]["1d_ll"], "-", label="ASH")
        ax.set_title(trait)
        ax.set_xscale("log")
        ax.axvline(p_cutoffs[trait]["5e-08"], color="k", linestyle="--", alpha=0.5)
    for ax in axs[len(all_traits):]:
        ax.axis("off")
    for ax in axs[::5]:
        ax.set_ylabel("LL(PLEI) - LL(1T)")
    for ax in axs[-5:]:
        ax.set_xlabel("Variance cutoff")
    # Add a legend to the top left subplot, set the font size to be smaller
    axs[0].legend(fontsize=10)
    return fig, axs

fig, _ = plot_ll_diff(var_fits, var_fits_ash)
fig.tight_layout()
fig.savefig("ll_diff.pdf", bbox_inches="tight")
