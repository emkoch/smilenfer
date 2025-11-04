import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from sklearn.linear_model import LinearRegression
from . import simulation as sim
from . import statistics as smile_stats
from . import posterior as smile_post

import rpy2.robjects as robjects

from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter

def _plot_params():
    matplotlib.rcParams.update({'font.size': 22})
    matplotlib.rcParams["figure.facecolor"] = '#ffffff'
    matplotlib.rcParams["axes.facecolor"] = "#ffffff"
    matplotlib.rcParams["savefig.facecolor"] = "#ffffff"
    plt.style.use('bmh')
    # make the figure background white
    matplotlib.rcParams.update({'figure.facecolor': 'white'})
    # make the box background white
    matplotlib.rcParams.update({'axes.facecolor': 'white'})

_plot_params()

def plot_se_raf(raf, se, trait_name="", ax_given=None):
    if ax_given is None:
        fig, ax = plt.subplots(figsize=(14,8))
    else:
        ax = ax_given
    lower = raf < 0.5
    ax.scatter(1-raf[lower], se[lower], alpha=0.4, s=50, label="risk-minor, flip raf")
    ax.scatter(1-raf[~lower], se[~lower], alpha=0.4, s=50, label="risk-major, flip raf")
    ax.plot(raf, se, "+", c="black")
    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_ylabel("SE")
    ax.set_xlabel("RAF")
    ax.set_title(trait_name)
    ax.legend()
    if ax_given is None:
        return fig, ax

def plot_ash_prior(RDS_fname, trait_name="", max_percentile=0.99, ax_given=None):
    robjects.r('''
    library(\"ashr\")
    get_prior <- function(RDS.fname){               
    gg <- readRDS(RDS.fname)
    if(class(gg)==\"ash\"){
        return(gg$fitted_g)
    } else if(class(gg)==\"normalmix\"){
        return(gg)
    }
    }
    ''')
    ash_prior = robjects.globalenv["get_prior"](RDS_fname)
    ash_probs = np.array(ash_prior[0])
    ash_sds = np.array(ash_prior[2])
    zero_entry = np.where((ash_probs > 0) & (ash_sds == 0))[0]
    zero_prob = 0 if len(zero_entry)==0 else ash_probs[zero_entry[0]]
    nonzero_entries = np.where((ash_probs > 0) & (ash_sds > 0))[0]
    nonzero_probs = ash_probs[nonzero_entries]
    nonzero_sds = ash_sds[nonzero_entries]
    norm = scipy.stats.norm
    max_betas = [norm.isf(1-max_percentile, loc=0, scale=sd) for sd in nonzero_sds]
    max_beta = np.max(max_betas)
    beta_set = np.linspace(0, max_beta*2, int(1e5))
    mixture_density = 2*sum([nonzero_probs[ii]*norm.pdf(x=beta_set, loc=0, scale=sd)
                             for ii, sd in enumerate(nonzero_sds)])

    if ax_given is None:
        fig, ax = plt.subplots(figsize=(7,6))
    else:
        ax = ax_given
    ax.plot(beta_set, mixture_density, color="black")
    for ii, sd in enumerate(nonzero_sds):
        ax.plot(beta_set, 2*nonzero_probs[ii]*norm.pdf(x=beta_set, loc=0, scale=sd), "--", color="orange")
    ax.plot([0, 0], [min(mixture_density), 2*max(mixture_density)], color="grey", alpha=0.5)
    ax.set_ylim((min(mixture_density), 2*max(mixture_density)))
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\beta|$")
    ax.set_ylabel("Density")
    ax.text(max_beta/2, max(mixture_density)/2,  r"$P(\beta=0)={}$".format(np.around(zero_prob, 2)))
    ax.set_title(trait_name)

    if ax_given is None:
        return fig, ax
    else: 
        return ax

def plot_ML_table(ML_table, trait_groups, trait_group_labels, ss=100, fit="post", logy=False):
    ML_tmp = ML_table.copy()
    group_sizes = [len(group) for group in trait_groups.values()]
    trait_cats = list(trait_groups.keys())
    n_traits = sum(group_sizes)
    trait_mult = 1.5

    fig, axes = plt.subplots(nrows=1, ncols=len(trait_groups), sharey=True,
                             figsize=(max(n_traits*trait_mult, 6), 6),
                             gridspec_kw={'width_ratios': group_sizes})

    marker_dir = ">"
    marker_stab = "s"
    marker_full = "D"
    marker_plei = "o"

    c_dir = "#a65628"
    c_stab = "#984ea3"
    c_full = "#999999"
    c_plei = "#377eb8"

    if len(trait_groups) == 1:
        axes = [axes]

    # Convert to log10
    for model in ["neut", "dir", "stab", "full", "plei"]:
        if model == "neut":
            adjust = 0
        elif model == "full":
            adjust = 2
        else:
            adjust = 1
        ML_tmp["ll_" + model] = -(2*adjust - 2*ML_tmp["ll_" + model].to_numpy()) / np.log(10)

    all_ML = np.concatenate([ML_tmp["ll_" + model] - ML_tmp["ll_neut"] for model in ["dir", "stab", "full", "plei"]])
    y_max = np.max(all_ML)
    y_min = np.min(all_ML)

    for ii, ax in enumerate(axes):
        ax.set_xlim([-0.5, group_sizes[ii]-0.5])
        ax.set_xticks(range(group_sizes[ii]))
        xticklabels = trait_groups[trait_cats[ii]]
        xticklabels = [xticklabel.replace("_", " ") for xticklabel in xticklabels]
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='x', labelrotation = 80)
        ax.set_title(trait_group_labels[ii])
        for jj, trait in enumerate(trait_groups[trait_cats[ii]]):
            ML_trait = ML_tmp.loc[np.logical_and(ML_tmp.trait==trait, ML_tmp.beta==fit)]
            
            ax.scatter(jj-0.2, ML_trait.ll_dir-ML_trait.ll_neut,
                       marker=marker_dir, c=c_dir, s=ss, label="directional")
            ax.scatter(jj-0.2/3, ML_trait.ll_stab-ML_trait.ll_neut,
                       marker=marker_stab, c=c_stab, s=ss, label="1D Stabilizing")
            ax.scatter(jj+0.2/3, ML_trait.ll_full-ML_trait.ll_neut,
                           marker=marker_full, c=c_full, s=ss, label="dir. + 1-D stab.")
            ax.scatter(jj+0.2, ML_trait.ll_plei-ML_trait.ll_neut,
                       marker=marker_plei, c=c_plei, s=ss, label="hD pleiotropic")
            
    if logy:
        ax.set_ylim(y_min-0.25, y_max*1.2)
        ax.set_yscale("symlog", linthresh=2)
        ax.set_yticks([0, 0.5, 1., 2.] + list(5*2**np.arange(0, np.log2(y_max/5))))
        ax.set_yticklabels([0, 0.5, 1., 2.] + list(5*2**np.arange(0, np.log2(y_max/5))))
    else:
        y1, y2 = axes[0].get_ylim()
        if y1 > 0:
            y1 = -1/np.log(10)
        axes[0].set_ylim(y1, y2)

    axes[0].set_ylabel(r"$-\Delta \mathrm{AIC}_{\mathrm{model} - \mathrm{neut}} (\log_{10})$")

    axes[-1].legend(labels=["Directional", "1D Stabilizing", "Dir. + Stab.", "hD Stabilizing"],
                    loc="upper left", bbox_to_anchor=(1.0, 0.9), ncol=1)
    return fig, axes

def fit_plot_llhood_diff(raf, beta_obs, beta_post, llhood_1, llhood_2,
                         model_1="1", model_2="2", figsize=(8,9)):
    _plot_params()

    llhood_diffs = llhood_1 - llhood_2
    llhood_diff_max = np.max(np.abs(llhood_diffs))

    fig, ax = plt.subplots(figsize=figsize)
    if beta_post is None:
        foo = ax.scatter(x=raf, y=beta_obs, c=llhood_diffs,
                         vmin=-llhood_diff_max, vmax=llhood_diff_max,
                         cmap=cm.BrBG, edgecolor="black", alpha=0.7, s=120)
    else:
        foo = ax.scatter(x=raf, y=beta_post, c=llhood_diffs,
                         vmin=-llhood_diff_max, vmax=llhood_diff_max,
                         cmap=cm.BrBG, edgecolor="black", alpha=0.7, s=120)
        ax.scatter(x=raf, y=beta_obs, color="black", marker="+", alpha=0.5)
        for ii, raf_i in enumerate(raf):
            ax.plot([raf_i]*2, [beta_post[ii], beta_obs[ii]],
                    color="black", linewidth=0.5, alpha=0.5)
    ax.set_yscale("log")
    if beta_post is None:
        ax.set_ylim(np.min(beta_obs)/1.25, np.max(beta_obs)*1.25)
    else:
        ax.set_ylim(np.min(beta_post)/1.25, np.max(beta_obs)*1.25)
    y1, y2 = ax.get_ylim()
    yticks1 = np.round(np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5), 3)
    ax.set_yticks(yticks1)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    clb = plt.colorbar(foo, ax=ax, orientation="horizontal")
    clb.ax.set_xlabel(r"$l_{{{}}} - l_{{{}}}$".format(model_1, model_2))
    ax.set_xlim([0,1])
    ax.set_xlabel("raf")
    ax.set_ylabel(r"$\beta$")

    return fig, ax

def sim_plot(raf_true, raf_sim, beta_hat, beta_true_sim, figsize=(8, 9)):
    _plot_params()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(raf_true, beta_hat, color="darkred", marker="o")
    ax.scatter(raf_sim, beta_hat, color="goldenrod", marker="o")
    ax.scatter(raf_sim, beta_true_sim, color="goldenrod", marker="+")
    for ii in range(len(raf_sim)):
        ax.plot([raf_sim[ii], raf_sim[ii]], [beta_hat[ii], beta_true_sim[ii]], color="goldenrod",
                linewidth=0.5, alpha=0.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\beta$")
    return fig, ax

def plot_smile_fit(raf, beta_hat, beta_post, v_cut, model, params, WF_pile=None, min_x=0.01, hat_as_true=False,
                   figsize=(8, 9), title=None, ylabel=r"$\beta$", xlabel="raf",  point_size=120,
                   fig=None, ax_1=None, no_cbar=False, return_cbar=False):
    """
    Plot the estimated selection strength on top of the GWAS smile plot.

    Parameters
    ----------
    raf : array
        The risk allele frequency.
    beta_hat : array
        The estimated effect size from GWAS
    beta_post : array
        The shrunk effect size from the posterior distribution
    v_cut : float
        The variance contributino cutoff
    model : str
        The model name
    params : dict
        The model parameters
    min_x : float
        The minimum x value for ascertainment
    hat_as_true : bool
        Whether to use the estimated effect size as the true effect size even when we color it by beta_post
    figsize : tuple
        The figure size
    title : str
        The title of the plot
    ylabel : str
        The y-axis label
    xlabel : str
        The x-axis label
    point_size : int
        The size of the points
    fig : matplotlib figure
        The figure object
    ax_1 : matplotlib axis
        The axis object
    no_cbar : bool
        Whether to plot the colorbar
    return_cbar : bool
        Whether to return the colorbar object

    Returns
    -------
    fig : matplotlib figure
        The figure object
    ax_1 : matplotlib axis
        The axis object
    cbar : matplotlib colorbar
        The colorbar object, only if return_cbar is True
    """

    # Make sure effect sizes are non-negative
    beta_hat = np.abs(beta_hat)
    if beta_post is not None:
        beta_post = np.abs(beta_post)
        all_beta = np.concatenate((beta_hat, beta_post))
        min_beta = np.min(all_beta)
        max_beta = np.max(all_beta)
    else:
        min_beta = np.min(beta_hat)
        max_beta = np.max(beta_hat)

    _plot_params()
    if fig is None or ax_1 is None:
        fig, ax_1 = plt.subplots(figsize=figsize)

    # Set and store y axis limits and ticks
    ax_1.set(yscale="log", xlabel=xlabel, ylabel=ylabel, title=title)
    ax_1.set_ylim(min_beta/1.25, max_beta*1.25)
    y1, y2 = ax_1.get_ylim()
    yticks1 = np.round(np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5), 3)
    print(yticks1)
    ax_1.set_yticks(yticks1)
    frmt = matplotlib.ticker.ScalarFormatter()
    ax_1.get_yaxis().set_major_formatter(frmt)
    ax_1.yaxis.set_minor_formatter(NullFormatter())

    # Plot the variance contribution cutoff
    if v_cut is not None:
        x_set = np.arange(min_x, 1, min_x)
        discov_betas = np.sqrt(v_cut/(2*x_set*(1-x_set)))
        ax_1.plot(np.concatenate(([min_x], x_set, [1-min_x])),
                  np.concatenate(([max_beta*1.25], discov_betas, [max_beta*1.25])),
                  color="darkslategrey", linestyle="dashed", linewidth=4)
        ax_1.set_xlim(-0.02, 1.02)

    # Make a dataframe for plotting
    trait_df = pd.DataFrame({"raf": raf, "beta_hat": beta_hat})
    if beta_post is not None:
        if hat_as_true:
            trait_df["beta_show"] = beta_hat
        else:
            trait_df["beta_show"] = beta_post
    else:
        trait_df["beta_show"] = beta_hat

    # If showing shrunk values, also plot the originals
    if beta_post is not None and not hat_as_true:
        for ii, x in enumerate(raf):
            ax_1.plot([x, x], [beta_hat[ii], beta_post[ii]],
                      color="goldenrod", alpha=0.7, zorder=1)
        ax_1.scatter(raf, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.7, zorder=3)

    if model == "neut":
        sns.scatterplot(x="raf", y="beta_show", data=trait_df, ax=ax_1, s=point_size, edgecolor="black")
    else:
        # Make a colormap if things are non-neutral
        cc = sns.color_palette("Spectral", as_cmap=True)
        cc = cc.reversed()
        norm = plt.Normalize(-1, 2, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cc)
        sm.set_array([])
        # Also grab Ne if non-neutral
        Ne = params["Ne"]
    
    # Possible selection models
    if model == "dir":
        I1 = params["I1"]
        if beta_post is not None:
            S_dir = np.abs(2*Ne*I1*beta_post)
        else:
            S_dir = np.abs(2*Ne*I1*beta_hat)
        trait_df["S"] = np.log10(S_dir)
        if I1 > 0: # Selection to increase trait value -- positive selection on trait incr alleles
            clabel = r"$+S_{dir}: \longrightarrow$"
        else:
            clabel = r"$-S_{dir}: \longleftarrow$"
    elif model == "stab":
        I2 = params["I2"]
        if beta_post is not None:
            S_ud = np.abs(2*Ne*I2*beta_post**2)
        else:
            S_ud = np.abs(2*Ne*I2*beta_hat**2)
        trait_df["S"] = np.log10(S_ud)
        clabel = r"$S_{ud}$"
    elif model == "full":
        I1 = params["I1"] + 1e-8
        I2 = params["I2"]
        if beta_post is not None:
            S_dir = np.abs(2*Ne*I1*beta_post)
            S_ud = np.abs(2*Ne*I2*beta_post**2)
        else:
            S_dir = np.abs(2*Ne*I1*beta_hat)
            S_ud = np.abs(2*Ne*I2*beta_hat**2)
        trait_df["S"] = np.log10(S_ud)
        clabel = r"$S_{ud}$"
        if I1 > 0: # Selection to increase trait value -- positive selection on trait incr alleles
            ylabel2 = r"$+S_{dir}$"
        else:
            ylabel2 = r"$-S_{dir}$"
        ax_2 = ax_1.twinx()
        ax_2.set(yscale="log", ylabel=ylabel2)
        ax_2.set_ylim(np.abs(I1*y1*Ne*2), np.abs(I1*y2*Ne*2))
        ax_2.set_yticks(np.abs(I1*yticks1*Ne*2))
        ax_2.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
        ax_2.yaxis.set_minor_formatter(NullFormatter())

    elif model == "plei":
        Ip = params["Ip"]
        if WF_pile is None:
            if beta_post is not None:
                S_ud_median = smile_stats.posterior_median_plei(raf, beta_post, Ip, Ne)
            else:
                S_ud_median = smile_stats.posterior_median_plei(raf, beta_hat, Ip, Ne)
        else:
            if beta_post is not None:
                S_ud_median_eq = smile_stats.posterior_median_plei(raf, beta_post, Ip, Ne)
                S_ud_median = smile_stats.posterior_median_plei_WF(raf, beta_post, Ip, Ne, WF_pile)
            else:
                S_ud_median_eq = smile_stats.posterior_median_plei(raf, beta_hat, Ip, Ne)
                S_ud_median = smile_stats.posterior_median_plei_WF(raf, beta_hat, Ip, Ne, WF_pile)
            trait_df["S_eq"] = np.log10(S_ud_median_eq)
        trait_df["S"] = np.log10(S_ud_median)
        clabel = r"$S_{ud}$"

    # Plot the selection strength
    if model != "neut":
        points = sns.scatterplot(x="raf", y="beta_show", hue="S", data=trait_df, 
                                 ax=ax_1, palette=cc, edgecolor="black", hue_norm=norm, s=point_size, zorder=2)
        ax_1.get_legend().remove()
        if not no_cbar:
            sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
            cbar = ax_1.figure.colorbar(sm, orientation="horizontal", ticks=norm(sel_ticks))
            cbar.ax.set_xlabel(clabel)
            cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))
        else:
            cbar = None
    
    if return_cbar:
        return fig, ax_1, cbar, trait_df
    return fig, ax_1, trait_df

def sim_plot_truebeta(raf_true, raf_sim, beta_hat, beta_post=None, beta_true_sim=None, model="neut",
                      params=None, figsize=(8, 9),
                      incl_raf_true=True, title=None, ylabel=r"$\beta$", point_size=120, color_only=False,
                      v_cut=None, min_x=0.01, true_color="darkred", sim_raf_color="black", 
                      hat_as_true=False, return_cbar=False, fig=None, ax_1=None, 
                      no_cbar=False):
    _plot_params()
    if fig is None or ax_1 is None:
        fig, ax_1 = plt.subplots(figsize=figsize)

    if v_cut is not None:
        x_set = np.arange(min_x, 1, min_x)
        discov_betas = np.sqrt(v_cut/(2*x_set*(1-x_set)))
        ax_1.plot(np.concatenate(([min_x], x_set, [1-min_x])),
                  np.concatenate(([np.max(beta_hat)*1.25], discov_betas, [np.max(beta_hat)*1.25])),
                  color="darkslategrey", linestyle="dashed", linewidth=4)

    ax_1.set_xlim(-0.02, 1.02)
    if incl_raf_true:
        ax_1.scatter(raf_true, beta_hat, color=true_color, marker="+", s=point_size)
    if model == "neut":
        if beta_post is not None:
            for ii, raf in enumerate(raf_sim):
                ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.7, zorder=3)
            ax_1.scatter(raf_sim, beta_post, color="goldenrod", marker="o", s=point_size)
            ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.7)
        else:
            ax_1.scatter(raf_sim, beta_hat, color="goldenrod", marker="o", s=point_size)
            if beta_true_sim is not None:
                ax_1.scatter(raf_sim, beta_true_sim, color="goldenrod", marker="+")
                for ii in range(len(raf_sim)):
                    ax_1.plot([raf_sim[ii], raf_sim[ii]], [beta_hat[ii], beta_true_sim[ii]], color="goldenrod",
                            linewidth=0.5, alpha=0.7)
        ax_1.set_xlabel(r"$x$")
        ax_1.set_ylabel(ylabel)
        ax_1.set_title(title)

        ax_1.set_ylim(np.min(beta_hat)/1.25, np.max(beta_hat)*1.25)
        y1, y2 = ax_1.get_ylim()
        ax_1.set_yscale("log")
        yticks1 = np.round(np.logspace(np.log10(y1), np.log10(y2), 5), 3)
        ax_1.set_yticks(yticks1)
        frmt = matplotlib.ticker.ScalarFormatter()
        ax_1.get_yaxis().set_major_formatter(frmt)
        ax_1.yaxis.set_minor_formatter(NullFormatter())
    elif model == "dir":
        if beta_post is not None:
            for ii, raf in enumerate(raf_sim):
                ax_1.plot([raf, raf], [np.abs(beta_hat[ii]), np.abs(beta_post[ii])],
                          color="goldenrod", alpha=0.7, zorder=1)
            if not color_only:
                ax_1.scatter(raf_sim, np.abs(beta_post), color="goldenrod", marker="o", s=point_size)
            ax_1.scatter(raf_sim, np.abs(beta_hat), color="darkred", marker="+", s=point_size, alpha=0.7)
        else:
            if not color_only:
                ax_1.scatter(raf_sim, np.abs(beta_hat), color="goldenrod", marker="o", s=point_size)
                if beta_true_sim is not None:
                    ax_1.scatter(raf_sim, beta_true_sim, color="goldenrod", marker="+")
                    for ii in range(len(raf_sim)):
                        ax_1.plot([raf_sim[ii], raf_sim[ii]], [beta_hat[ii], beta_true_sim[ii]], color="goldenrod",
                                  linewidth=0.5, alpha=0.7)
            else:
                if beta_true_sim is not None:
                    ax_1.scatter(raf_sim, beta_hat, color=sim_raf_color, marker="+", s=point_size)
                    for ii in range(len(raf_sim)):
                        ax_1.plot([raf_sim[ii], raf_sim[ii]], [beta_hat[ii], beta_true_sim[ii]], color=sim_raf_color,
                                  alpha=0.7)
        I1 = params["I1"]
        Ne = params["Ne"]
        ax_1.set(yscale="log", ylabel=ylabel, xlabel="x", title=title)
        if beta_post is not None:
            ax_1.set_ylim(np.min(beta_post)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5)
        elif beta_true_sim is not None:
            ax_1.set_ylim(np.min(np.concatenate((beta_hat, beta_true_sim)))/1.25,
                          np.max(np.concatenate((beta_hat, beta_true_sim)))*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5)
        else:
            ax_1.set_ylim(np.min(beta_hat)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.round(np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5), 3)

        if I1 > 0: # Selection to increase trait value -- positive selection on trait incr alleles
            if color_only:
                ylabel2 = r"$+S_{dir}: \longrightarrow$"
            else:
                ylabel2 = r"$+S_{dir}$"
        else:
            if color_only:
                ylabel2 = r"$-S_{dir}: \longleftarrow$"
            else:
                ylabel2 = r"$-S_{dir}$"

        ax_1.set_yticks(yticks1)
        frmt = matplotlib.ticker.ScalarFormatter()
        ax_1.get_yaxis().set_major_formatter(frmt)
        ax_1.yaxis.set_minor_formatter(NullFormatter())

        if not color_only:
            ax_2 = ax_1.twinx()
            ax_2.set(yscale="log", ylabel=ylabel2)

            ax_2.set_ylim(np.abs(I1*y1*Ne*2), np.abs(I1*y2*Ne*2))
            ax_2.set_yticks(np.abs(I1*yticks1*Ne*2))
            ax_2.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
            ax_2.yaxis.set_minor_formatter(NullFormatter())
        else:
            cc = sns.color_palette("Spectral", as_cmap=True)
            norm = plt.Normalize(-1, 2, clip=True)
            sm = plt.cm.ScalarMappable(cmap=cc)
            sm.set_array([])

            if beta_post is not None:
                for ii, raf in enumerate(raf_sim):
                    ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.5, zorder=1)
                ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.5, zorder=3)
                S_dir = np.abs(2*Ne*I1*beta_post)
                df = pd.DataFrame({"x":raf_sim, "beta":beta_post, "S_dir":np.log10(S_dir)})
            else:
                if beta_true_sim is None:
                    S_dir = np.abs(2*Ne*I1*beta_hat)
                    df = pd.DataFrame({"x":raf_sim, "beta":beta_hat, "S_dir":np.log10(S_dir)})
                else:
                    S_dir = np.abs(2*Ne*I1*beta_true_sim)
                    df = pd.DataFrame({"x":raf_sim, "beta":beta_true_sim, "S_dir":np.log10(S_dir)})
            points = sns.scatterplot(x="x", y="beta", hue="S_dir",
                                     data=df,
                                     ax=ax_1, palette=cc, edgecolor="black", s=120, hue_norm=norm, zorder=2)

            ax_2 = ax_1.twinx()
            # remove the legend if not None:
            if ax_1.get_legend() is not None:
                ax_1.get_legend().remove()

            ax_2.set(ylabel=None, yticks=[])

            sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
            cbar = ax_2.figure.colorbar(sm, orientation="horizontal",
                                        ticks=norm(sel_ticks))
            cbar.ax.set_xlabel(ylabel2)
            cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))

    elif model == "stab":
        if beta_post is not None:
            for ii, raf in enumerate(raf_sim):
                ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.7, zorder=1)
            if not color_only:
                ax_1.scatter(raf_sim, np.abs(beta_post), color="goldenrod", marker="o", s=point_size)
            ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.7)
        else:
            if not color_only:
                ax_1.scatter(raf_sim, beta_hat, color="goldenrod", marker="+", s=point_size)
                if beta_true_sim is not None:
                    ax_1.scatter(raf_sim, beta_true_sim, color="goldenrod", marker="o")
                    for ii in range(len(raf_sim)):
                        ax_1.plot([raf_sim[ii], raf_sim[ii]], [beta_hat[ii], beta_true_sim[ii]], color="goldenrod",
                                  linewidth=0.5, alpha=0.7)
            else:
                if beta_true_sim is not None:
                    ax_1.scatter(raf_sim, beta_hat, color=sim_raf_color, marker="+", s=point_size)
                    for ii in range(len(raf_sim)):
                        ax_1.plot([raf_sim[ii], raf_sim[ii]], [beta_hat[ii], beta_true_sim[ii]], color=sim_raf_color,
                                  alpha=0.7)

        I2 = params["I2"]
        Ne = params["Ne"]

        ax_1.set(ylabel=ylabel, xlabel="x", title=title, yscale="log")
        # Set limits and axis ticks
        if beta_post is not None:
            ax_1.set_ylim(np.min(beta_post)/1.25, np.max(beta_hat)*1.25)
        elif beta_true_sim is not None:
            ax_1.set_ylim(np.min(np.concatenate((beta_hat, beta_true_sim)))/1.25,
                          np.max(np.concatenate((beta_hat, beta_true_sim)))*1.25)
        else:
            ax_1.set_ylim(np.min(beta_hat)/1.25, np.max(beta_hat)*1.25)
        y1, y2 = ax_1.get_ylim()
        yticks1 = np.round(np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5), 3)
        ax_1.set_yticks(yticks1)
        ax_1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_1.yaxis.set_minor_formatter(NullFormatter())

        ax_2 = ax_1.twinx()
        if not color_only:
            # Set up the second y axis
            ylabel2 = r"$S_{ud}$"
            ax_2.set(yscale="log", ylabel=ylabel2)
            ax_2.set_ylim(np.abs(I2*y1**2*Ne*2), np.abs(I2*y2**2*Ne*2))
            ax_2.set_yticks(np.abs(I2*yticks1**2*Ne*2))
            ax_2.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
            ax_2.yaxis.set_minor_formatter(NullFormatter())
        else:
            cc = sns.color_palette("Spectral", as_cmap=True)
            norm = plt.Normalize(-1, 2, clip=True)
            sm = plt.cm.ScalarMappable(cmap=cc)
            sm.set_array([])

            if beta_post is not None:
                for ii, raf in enumerate(raf_sim):
                    ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.5, zorder=1)
                ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.5, zorder=3)
                S_ud = np.abs(2*Ne*I2*beta_post**2)
                df = pd.DataFrame({"x":raf_sim, "beta":beta_post, "S_ud":np.log10(S_ud)})
            else:
                if beta_true_sim is None:
                    S_ud = np.abs(2*Ne*I2*beta_hat**2)
                    df = pd.DataFrame({"x":raf_sim, "beta":beta_hat, "S_ud":np.log10(S_ud)})
                else:
                    S_ud = np.abs(2*Ne*I2*beta_true_sim**2)
                    df = pd.DataFrame({"x":raf_sim, "beta":beta_true_sim, "S_ud":np.log10(S_ud)})

            points = sns.scatterplot(x="x", y="beta", hue="S_ud",
                                     data=df,
                                     ax=ax_1, palette=cc, edgecolor="black", s=120, hue_norm=norm, zorder=2)

            ax_1.get_legend().remove()
            

            sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
            cbar = ax_2.figure.colorbar(sm, orientation="horizontal",
                                        ticks=norm(sel_ticks))
            cbar.ax.set_xlabel(r"$S_{ud}$")
            cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))
    elif model == "full":
        if beta_post is not None:
            for ii, raf in enumerate(raf_sim):
                ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.7, zorder=1)
            ax_1.scatter(raf_sim, beta_post, color="goldenrod", marker="o", s=point_size)
            ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.7)
        else:
            ax_1.scatter(raf_sim, beta_hat, color="goldenrod", marker="o", s=point_size)

        I1 = params["I1"] + 1e-8
        I2 = params["I2"]
        Ne = params["Ne"]
        ax_1.set(ylabel=ylabel, xlabel="x", title=title, yscale="log")

        if beta_post is not None:
            ax_1.set_ylim(np.min(beta_post)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5)
            # S_combo = np.abs(2*Ne*I1*beta_post - 2*Ne*I2*beta_post**2)
            S_ud = np.abs(2*Ne*I2*beta_post**2)
        else:
            ax_1.set_ylim(np.min(beta_hat)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.round(np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5), 3)
            # S_combo = np.abs(2*Ne*I1*beta_hat - 2*Ne*I2*beta_hat**2)
            S_ud = np.abs(2*Ne*I2*beta_hat**2)

        ax_1.set_yticks(yticks1)
        frmt = matplotlib.ticker.ScalarFormatter()
        ax_1.get_yaxis().set_major_formatter(frmt)
        ax_1.yaxis.set_minor_formatter(NullFormatter())

        if I1 > 0: # Selection to increase trait value -- positive selection on trait incr alleles
            ylabel2 = r"$+S_{dir}$"
        else:
            ylabel2 = r"$-S_{dir}$"
        ax_2 = ax_1.twinx()

        if not color_only:
            ax_2.set(yscale="log", ylabel=ylabel2)
            ax_2.set_ylim(np.abs(I1*y1*Ne*2), np.abs(I1*y2*Ne*2))
            ax_2.set_yticks(np.abs(I1*yticks1*Ne*2))
            ax_2.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
            ax_2.yaxis.set_minor_formatter(NullFormatter())

            ax_3 = ax_1.twinx()
            ax_3.spines["right"].set_position(("axes", 1.2))
            ylabel3 = r"$S_{ud}$"
            ax_3.set(yscale="log", ylabel=ylabel3)
            ax_3.set_ylim(np.abs(I2*y1**2*Ne*2), np.abs(I2*y2**2*Ne*2))
            ax_3.set_yticks(np.abs(I2*yticks1**2*Ne*2))
            ax_3.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
            ax_3.yaxis.set_minor_formatter(NullFormatter())
        else:
            cc = sns.color_palette("Spectral", as_cmap=True)
            norm = plt.Normalize(-1, 2, clip=True)
            sm = plt.cm.ScalarMappable(cmap=cc)
            sm.set_array([])

            if beta_post is not None:
                for ii, raf in enumerate(raf_sim):
                    ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.5, zorder=1)
                ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.5, zorder=3)
                df = pd.DataFrame({"x":raf_sim, "beta":beta_post, "S_ud":np.log10(S_ud)})
            else:
                df = pd.DataFrame({"x":raf_sim, "beta":beta_hat, "S_ud":np.log10(S_ud)})

            points = sns.scatterplot(x="x", y="beta", hue="S_ud",
                                     data=df,
                                     ax=ax_1, palette=cc, edgecolor="black", s=120, hue_norm=norm, zorder=2)

            ax_1.get_legend().remove()
            ax_2.set(yscale="log", ylabel=ylabel2)
            ax_2.set_ylim(np.abs(I1*y1*Ne*2), np.abs(I1*y2*Ne*2))
            ax_2.set_yticks(np.abs(I1*yticks1*Ne*2))
            ax_2.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
            ax_2.yaxis.set_minor_formatter(NullFormatter())

            sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
            cbar = ax_2.figure.colorbar(sm, orientation="horizontal",
                                        ticks=norm(sel_ticks))
            cbar.ax.set_xlabel(r"$S_{ud}$")
            cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))
    elif model == "plei":
        Ip = params["Ip"]
        Ne = params["Ne"]
        n_s = 200
        S_ud_set = np.logspace(-3, 2.5, n_s)
        S_ud_medians = np.zeros_like(beta_hat)
        for ii, beta in enumerate(beta_hat):
            if beta_post is not None:
                beta = beta_post[ii]
            if beta_true_sim is not None:
                beta = beta_true_sim[ii]
            sfs_vals = (sim.sfs_ud_params_sigma(raf_sim[ii], 1, S_ud_set) +
                        sim.sfs_ud_params_sigma(1-raf_sim[ii], 1, S_ud_set))
            lower_val = (sim.levy_cdf(np.abs(S_ud_set[0]), 2*Ne*Ip*beta**2)*
                         sim.sfs_ud_params_sigma(raf_sim[ii], 1, S_ud_set[0]))
            lower_val += (sim.levy_cdf(np.abs(S_ud_set[0]), 2*Ne*Ip*beta**2)*
                          sim.sfs_ud_params_sigma(1-raf_sim[ii], 1, S_ud_set[0]))
            upper_val = ((1-sim.levy_cdf(np.abs(S_ud_set[-1]), 2*Ne*Ip*beta**2))*
                         sim.sfs_ud_params_sigma(raf_sim[ii], 1, S_ud_set[-1]))
            upper_val += ((1-sim.levy_cdf(np.abs(S_ud_set[-1]), 2*Ne*Ip*beta**2))*
                          sim.sfs_ud_params_sigma(1-raf_sim[ii], 1, S_ud_set[-1]))
            x_find_weights = np.concatenate(([lower_val],
                                             (smile_stats.get_weights(S_ud_set)*sfs_vals*
                                              sim.levy_density(S_ud_set, 2*Ne*Ip*beta**2)),
                                             [upper_val]))
            S_expand = np.concatenate(([S_ud_set[0]], S_ud_set, [S_ud_set[-1]]))
            ## Get S value of first point not less than 0.5
            median_S_x = S_expand[np.argmax(np.logical_not(np.cumsum(x_find_weights)/
                                                           np.sum(x_find_weights) < 0.5))]
            S_ud_medians[ii] = median_S_x
        cc = sns.color_palette("Spectral", as_cmap=True)
        cc = cc.reversed()
        norm = plt.Normalize(-1, 2, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cc)
        sm.set_array([])

        if beta_post is not None:
            if hat_as_true:
                pleistuff = pd.DataFrame({"x":raf_sim, "beta":beta_hat,
                                         "S_ud_median":np.log10(S_ud_medians)})
            else:
                for ii, raf in enumerate(raf_sim):
                    ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.5, zorder=1)
                ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.5, zorder=3)
                pleistuff = pd.DataFrame({"x":raf_sim, "beta":beta_post,
                                          "S_ud_median":np.log10(S_ud_medians)})
        else:
            if beta_true_sim is None:
                pleistuff = pd.DataFrame({"x":raf_sim, "beta":beta_hat,
                                          "S_ud_median":np.log10(S_ud_medians)})
            else:
                if beta_true_sim is not None:
                    pleistuff = pd.DataFrame({"x":raf_sim, "beta":beta_true_sim,
                                              "S_ud_median":np.log10(S_ud_medians)})
                    ax_1.scatter(raf_sim, beta_hat, color=sim_raf_color, marker="+", s=point_size)
                    for ii in range(len(raf_sim)):
                        ax_1.plot([raf_sim[ii], raf_sim[ii]], [beta_hat[ii], beta_true_sim[ii]], 
                                  color=sim_raf_color, alpha=0.7)

        points = sns.scatterplot(x="x", y="beta", hue="S_ud_median",
                                     data=pleistuff,
                                     ax=ax_1, palette=cc, edgecolor="black", s=120, hue_norm=norm, zorder=2)

        ax_2 = ax_1.twinx()
        ax_1.get_legend().remove()
        ax_1.set(yscale="log", ylabel=ylabel, xlabel="x", title=title)
        ax_2.set(ylabel=None, yticks=[])

        if beta_post is not None:
            ax_1.set_ylim(np.min(beta_post)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5)
        elif beta_true_sim is not None:
            ax_1.set_ylim(np.min(np.concatenate((beta_hat, beta_true_sim)))/1.25,
                          np.max(np.concatenate((beta_hat, beta_true_sim)))*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5)
        else:
            ax_1.set_ylim(np.min(beta_hat)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.round(np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5), 3)

        ax_1.set_yticks(yticks1)
        frmt = matplotlib.ticker.ScalarFormatter()
        ax_1.get_yaxis().set_major_formatter(frmt)
        ax_1.yaxis.set_minor_formatter(NullFormatter())
        if not no_cbar:
            sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
            cbar = ax_2.figure.colorbar(sm, orientation="horizontal",
                                        ticks=norm(sel_ticks))
            cbar.ax.set_xlabel(r"Median $S_{ud}$")
            cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))
        else:
            cbar = None
    elif model == "nplei":
        I2 = params["I2"]
        nn = params["nn"]
        Ne = params["Ne"]
        n_s = 200
        S_ud_set = np.logspace(-3, 2.5, n_s)
        S_ud_medians = np.zeros_like(beta_hat)
        for ii, beta in enumerate(beta_hat):
            if beta_post is not None:
                beta = beta_post[ii]
            sfs_vals = (sim.sfs_ud_params_sigma(raf_sim[ii], 1, S_ud_set) +
                        sim.sfs_ud_params_sigma(1-raf_sim[ii], 1, S_ud_set))
            lower_val = (sim.nplei_cdf(S_ud_set[0]/(2*Ne), beta, I2, nn) *
                         sim.sfs_ud_params_sigma(raf_sim[ii], 1, S_ud_set[0]))
            lower_val += (sim.nplei_cdf(S_ud_set[0]/(2*Ne), beta, I2, nn)*
                          sim.sfs_ud_params_sigma(1-raf_sim[ii], 1, S_ud_set[0]))
            upper_val = ((1-sim.nplei_cdf(S_ud_set[-1]/(2*Ne), beta, I2, nn)) *
                         sim.sfs_ud_params_sigma(raf_sim[ii], 1, S_ud_set[-1]))
            upper_val += ((1-sim.nplei_cdf(S_ud_set[-1]/(2*Ne), beta, I2, nn))*
                          sim.sfs_ud_params_sigma(1-raf_sim[ii], 1, S_ud_set[-1]))
            x_find_weights = np.concatenate(([lower_val],
                                             (smile_stats.get_weights(S_ud_set)*sfs_vals*
                                              sim.nplei_density(S_ud_set/(2*Ne), beta, I2, nn)),
                                             [upper_val]))
            S_expand = np.concatenate(([S_ud_set[0]], S_ud_set, [S_ud_set[-1]]))
            ## Get S value of first point not less than 0.5
            median_S_x = S_expand[np.argmax(np.logical_not(np.cumsum(x_find_weights)/
                                                           np.sum(x_find_weights) < 0.5))]
            S_ud_medians[ii] = median_S_x
        cc = sns.color_palette("Spectral", as_cmap=True)
        norm = plt.Normalize(-1, 2, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cc)
        sm.set_array([])

        if beta_post is not None:
            for ii, raf in enumerate(raf_sim):
                ax_1.plot([raf, raf], [beta_hat[ii], beta_post[ii]], color="goldenrod", alpha=0.7, zorder=1)
            ax_1.scatter(raf_sim, beta_hat, color="darkred", marker="+", s=point_size, alpha=0.7)
            pleistuff = pd.DataFrame({"x":raf_sim, "beta":beta_post,
                                      "S_ud_median":np.log10(S_ud_medians)})
        else:
            pleistuff = pd.DataFrame({"x":raf_sim, "beta":beta_hat,
                                      "S_ud_median":np.log10(S_ud_medians)})

        points = sns.scatterplot(x="x", y="beta", hue="S_ud_median",
                                 data=pleistuff,
                                 ax=ax_1, palette=cc, edgecolor="black", s=120, hue_norm=norm, zorder=2)

        ax_2 = ax_1.twinx()
        ax_1.get_legend().remove()
        ax_1.set(yscale="log", ylabel=ylabel, xlabel="x", title=title)
        ax_2.set(ylabel=None, yticks=[])

        if beta_post is not None:
            ax_1.set_ylim(np.min(beta_post)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5)
        else:
            ax_1.set_ylim(np.min(beta_hat)/1.25, np.max(beta_hat)*1.25)
            y1, y2 = ax_1.get_ylim()
            yticks1 = np.round(np.logspace(np.log10(y1*1.1), np.log10(y2/1.1), 5), 3)

        ax_1.set_yticks(yticks1)
        frmt = matplotlib.ticker.ScalarFormatter()
        ax_1.get_yaxis().set_major_formatter(frmt)
        ax_1.yaxis.set_minor_formatter(NullFormatter())

        sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
        cbar = ax_2.figure.colorbar(sm, orientation="horizontal",
                                    ticks=norm(sel_ticks))
        cbar.ax.set_xlabel(r"Median $S_{ud}$")
        cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))

    if return_cbar:
        return fig, ax_1, cbar
    return fig, ax_1

def add_genes(ax, x_vals, y_vals, genes, distances, in_exons, consequences=None):
    def adjust_x(xx):
        return xx-(xx-0.5)*(0.02/0.5)

    def alpha_distance(dist):
        if dist ==0:
            return 1
        else:
            return np.minimum(1, 2/np.log10(dist))

    for ii, xx in enumerate(x_vals):
        if xx <= 0.1:
            ha = "left"
        elif xx >= 0.9:
            ha = "right"
        else:
            xx = adjust_x(xx)
            ha = "center"
        yy = np.abs(y_vals[ii])
        yy_plot = np.exp(np.log(yy) + 0.04)
        gene = genes[ii]
        dist = distances[ii]
        alpha = alpha_distance(dist)
        color = "black"
        if consequences is None:
            in_exon = in_exons[ii]
            if in_exon:
                alpha = 1
                if dist == 0 and not in_exon:
                    color = "navy"
                if in_exon:
                    color = "red"
        else:
            consequence = consequences[ii]
            if consequence == "nonsynonymous":
                alpha = 1
                color = "red"
        ax.annotate(text=gene,
                    xy=(xx, yy_plot),
                    fontsize=10, ha=ha, alpha=alpha, color=color)
    return ax

def plot_local_neff(neg_log10_pval, n_eff, ax, trait_name=""):
    _plot_params()
    neg_log10_pval = np.array(neg_log10_pval)
    n_eff = np.array(n_eff)
    gw_sig = neg_log10_pval > -np.log10(5e-08)
    median_n_eff = np.median(n_eff[gw_sig])
    ax.axhline(median_n_eff, color="black", linestyle="--")
    ax.plot(neg_log10_pval[~gw_sig], n_eff[~gw_sig], ".", alpha=0.5)
    ax.plot(neg_log10_pval[gw_sig], n_eff[gw_sig], "o", alpha=0.5, color="red")
    ax.text(np.max(neg_log10_pval[gw_sig]), np.min(n_eff), r"$n_{{eff}} = {:.0f}$".format(median_n_eff), fontsize=24, 
            horizontalalignment="right", verticalalignment="bottom")    
    ax.set_title(trait_name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\log_{10}(\mathrm{p{-}value})$")
    ax.set_ylabel("Effective sample size")

def plot_n_eff(v_ests, p_vals, trait_name=""):
    _plot_params()
    # Don't use calc_n_eff() here since we want vals_keep anyway
    inv_vals = scipy.stats.chi2.ppf(q=1-np.array(p_vals), df=1)
    vals_keep = (~np.isinf(inv_vals)) & (inv_vals > 0)
    fit_nolog = LinearRegression(fit_intercept=False).fit(inv_vals[vals_keep, np.newaxis], v_ests[vals_keep])
    neff_nolog = 1/fit_nolog.coef_[0]
    neff_log = np.mean(np.log(inv_vals[vals_keep]) - np.log(v_ests[vals_keep]))

    v_min = np.min(v_ests[vals_keep])
    v_max = np.max(v_ests[vals_keep])
    inv_min = np.min(inv_vals[vals_keep])
    inv_max = np.max(inv_vals[vals_keep])

    bound_lin_v = v_max/10
    bound_lin_inv = inv_max/10

    interval_log_v = (np.log(v_max) - np.log(v_min))/10
    interval_log_inv = (np.log(inv_max) - np.log(inv_min))/10

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(19, 9))

    axes[0].scatter(v_ests[vals_keep], inv_vals[vals_keep], s=1)
    axes[0].set_xlim([-bound_lin_v, v_max+bound_lin_v])
    axes[0].set_ylim([-bound_lin_inv, inv_max+bound_lin_inv])
    axes[0].text(v_min, inv_max, r"$n_{{eff}} = {:.0f}$".format(neff_nolog))
    axes[0].set_xlabel(r"$\hat{v}$")
    axes[0].set_ylabel(r"$F^{-1}(p$-$value)$")

    axes[1].scatter(np.log(v_ests[vals_keep]), np.log(inv_vals[vals_keep]), s=1)
    axes[1].set_xlim([np.log(v_min)-interval_log_v, np.log(v_max) + interval_log_v])
    axes[1].set_ylim([np.log(inv_min)-interval_log_inv, np.log(inv_max) + interval_log_inv])
    axes[1].text(np.log(v_min), np.log(inv_max), r"$n_{{eff}} = {:.0f}$".format(np.exp(neff_log)))
    axes[1].set_xlabel(r"$\log(\hat{v})$")
    axes[1].set_ylabel(r"$\log(F^{-1}(p$-$value))$")

    xpoints = ypoints = axes[0].get_xlim()
    axes[0].plot(xpoints, np.array(xpoints)*neff_nolog,
                 linestyle='--', color='k', lw=1, scalex=False, scaley=False, alpha=0.5)

    xpoints = ypoints = axes[1].get_xlim()
    axes[1].plot(xpoints, np.array(xpoints) + neff_log,
                 linestyle='--', color='k', lw=1, scalex=False, scaley=False, alpha=0.5)

    axes[0].ticklabel_format(axis="x", style="scientific",scilimits=(-2,2))
    axes[1].ticklabel_format(axis="x", style="scientific",scilimits=(-2,2))

    fig.suptitle(trait_name)

    return fig, axes

def p_val_cuts(p_vals, cut_points):
    result = ["> " + str(cut_points[0])]*len(p_vals)
    for ii, p_val in enumerate(p_vals):
        for cut_point in cut_points:
            if p_val < cut_point:
                result[ii] = str(cut_point)
    return result

def var_cuts(var_vals, cut_points, p_cuts):
    result = ["> " + str(p_cuts[0])]*len(var_vals)
    for ii, var_val in enumerate(var_vals):
        for jj, cut_point in enumerate(cut_points):
            if var_val > cut_point:
                result[ii] = str(p_cuts[jj])
    return result

def plot_cuts(ash_data, p_threshes=[1e-05, 1e-06, 1e-07, 5e-08, 1e-08], trait_name=""):
    _plot_params()

    data_tmp = deepcopy(ash_data)
    data_tmp["p_thresh"] = p_val_cuts(data_tmp.pval, p_threshes)
    max_label = "> " + str(p_threshes[0])
    data_strong = data_tmp[np.array(data_tmp.p_thresh) != max_label]

    n_eff = smile_stats.calc_n_eff(ash_data.var_exp, ash_data.pval)
    v_cuts = [scipy.stats.chi2.ppf(q=1-p_thresh, df=1)/n_eff for p_thresh in p_threshes]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(23, 9))

    sns.scatterplot(x="raf", y="var_exp", hue="p_thresh", data=data_strong,
                    alpha=0.7,  palette=sns.color_palette("colorblind", data_strong.p_thresh.nunique()),
                    hue_order=[str(p_thresh) for p_thresh in p_threshes], s=70, ax=axes[0])
    sns.scatterplot(x="raf", y="var_exp", data=data_tmp[np.array(data_tmp.p_thresh) == max_label],
                    alpha=0.1, color="black", ax=axes[0])

    sns.scatterplot(x="raf", y="var_exp_ash", hue="p_thresh", data=data_strong,
                    alpha=0.7,  palette=sns.color_palette("colorblind", data_strong.p_thresh.nunique()),
                    hue_order=[str(p_thresh) for p_thresh in p_threshes], s=70, ax=axes[1])
    sns.scatterplot(x="raf", y="var_exp_ash", data=data_tmp[np.array(data_tmp.p_thresh) == max_label],
                    alpha=0.1, color="black", ax=axes[1])

    axes[0].set(ylabel=r"$\hat{v}$", ylim=[-v_cuts[0]/10, np.max(data_tmp.var_exp)*2], title="pre-ash")
    axes[0].set_yscale("symlog", linthreshy=v_cuts[0]/2)
    axes[1].set(ylabel=r"$\hat{v}_{ash}$", ylim=[-v_cuts[0]/10, np.max(data_tmp.var_exp)*2], title="post-ash")
    axes[1].set_yscale("symlog", linthreshy=v_cuts[0]/2)
    axes[0].legend().remove()

    for ii, v_cut in enumerate(v_cuts):
        axes[0].axhline(y=v_cut, color=sns.color_palette("colorblind", data_strong.p_thresh.nunique())[ii])
        axes[1].axhline(y=v_cut, color=sns.color_palette("colorblind", data_strong.p_thresh.nunique())[ii])

    plt.legend(ncol=3)
    fig.suptitle(trait_name)

    return fig, axes

def plot_ash_smiles(ash_data, p_threshes=[1e-05, 1e-06, 1e-07, 5e-08, 1e-08], trait_name="", min_x=0.01):
    _plot_params()

    n_eff = smile_stats.calc_n_eff(ash_data.var_exp, ash_data.pval)
    v_cuts = [scipy.stats.chi2.ppf(q=1-p_thresh, df=1)/n_eff for p_thresh in p_threshes]

    data_tmp = deepcopy(ash_data)
    data_tmp["rbeta_ash"] = np.array(np.abs(data_tmp.beta_ash))
    data_tmp["p_thresh"] = p_val_cuts(data_tmp.pval, p_threshes)
    data_tmp["var_thresh"] = var_cuts(data_tmp.var_exp, v_cuts, p_threshes)
    data_tmp["var_thresh_ash"] = var_cuts(data_tmp.var_exp_ash, v_cuts, p_threshes)
    max_label = "> " + str(p_threshes[0])
    data_strong = data_tmp[np.array(data_tmp.var_thresh) != max_label]
    data_strong_ash = data_tmp[np.array(data_tmp.var_thresh_ash) != max_label]

    x_set = np.arange(min_x, 1, min_x)
    beta_sets = [np.sqrt(v_cut/(2*x_set*(1-x_set))) for v_cut in reversed(v_cuts)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(23, 9))

    pal_use = sns.color_palette("colorblind", data_strong.var_thresh.nunique())

    for ii, beta_set in enumerate(beta_sets):
        axes[0].plot(x_set, beta_set, color=pal_use[ii], alpha=0.5)
        axes[1].plot(x_set, beta_set, color=pal_use[ii], alpha=0.5)

    sns.scatterplot(x="raf", y="rbeta", hue="var_thresh", data=data_strong,
                    alpha=0.7,  palette=pal_use,
                    hue_order=sorted(data_strong_ash.var_thresh_ash.unique(),
                                     key=lambda e: float(e.split("> ")[-1])), s=50, ax=axes[0])
    sns.scatterplot(x="raf", y="rbeta", data=data_tmp[np.array(data_tmp.var_thresh) == max_label],
                    alpha=0.05, color="black", ax=axes[0], s=20)

    sns.scatterplot(x="raf", y="rbeta_ash", hue="var_thresh_ash", data=data_strong_ash,
                    alpha=0.7,  palette=pal_use,
                    hue_order=sorted(data_strong_ash.var_thresh_ash.unique(),
                                     key=lambda e: float(e.split("> ")[-1])),
                    s=50, ax=axes[1])
    sns.scatterplot(x="raf", y="rbeta_ash", data=data_tmp[np.array(data_tmp.var_thresh_ash) == max_label],
                    alpha=0.05, color="black", ax=axes[1], s=20)

    axes[0].set(ylabel=r"$\hat{\beta}$", ylim=[min(data_strong.rbeta)/2,
                                               np.max(data_tmp.rbeta)*2], title="pre-ash")
    axes[0].set_yscale("log")
    #axes[0].set_yscale("symlog", linthreshy=min(data_strong.rbeta))
    axes[1].set(ylabel=r"$\hat{\beta}_{ash}$", ylim=[min(data_strong.rbeta)/2,
                                                     np.max(data_tmp.rbeta)*2], title="post-ash")
    axes[1].set_yscale("log")
    #axes[1].set_yscale("symlog", linthreshy=min(data_strong.rbeta))

    axes[0].legend().remove()

    plt.legend(ncol=3)
    fig.suptitle(trait_name)

    return fig, axes

def plot_smiles_noash(ash_data, p_threshes=[1e-05, 1e-06, 1e-07, 5e-08, 1e-08], trait_name="", min_x=0.01):
    _plot_params()

    n_eff = smile_stats.calc_n_eff(ash_data.var_exp, ash_data.pval)
    v_cuts = [scipy.stats.chi2.ppf(q=1-p_thresh, df=1)/n_eff for p_thresh in p_threshes]

    data_tmp = deepcopy(ash_data)
    data_tmp["p_thresh"] = p_val_cuts(data_tmp.pval, p_threshes)
    data_tmp["var_thresh"] = var_cuts(data_tmp.var_exp, v_cuts, p_threshes)
    max_label = "> " + str(p_threshes[0])
    data_strong = data_tmp[np.array(data_tmp.var_thresh) != max_label]

    x_set = np.arange(min_x, 1, min_x)
    beta_sets = [np.sqrt(v_cut/(2*x_set*(1-x_set))) for v_cut in reversed(v_cuts)]

    fig, ax = plt.subplots(figsize=(10, 9))

    pal_use = sns.color_palette("colorblind", data_strong.var_thresh.nunique())

    for ii, beta_set in enumerate(beta_sets):
        ax.plot(x_set, beta_set, color=pal_use[ii], alpha=0.5)

    sns.scatterplot(x="raf", y="rbeta", hue="var_thresh", data=data_strong,
                    alpha=0.7,  palette=pal_use,
                    hue_order=sorted(data_strong.var_thresh.unique(),
                                     key=lambda e: float(e.split("> ")[-1])), s=50, ax=ax)
    sns.scatterplot(x="raf", y="rbeta", data=data_tmp[np.array(data_tmp.var_thresh) == max_label],
                    alpha=0.05, color="black", ax=ax, s=20)

    ax.set(ylabel=r"$\hat{\beta}$", ylim=[min(data_strong.rbeta)/2,
                                               np.max(data_tmp.rbeta)*2], title=trait_name)
    ax.set_yscale("log")


    plt.legend(ncol=3)

    return fig, ax

def plot_smile(raf, rbeta, pvals, shrunk_beta=None, alpha=0.8, trait="", figsize=(10,8), ax_given=None):
    x_set = smile_stats.trad_x_set(0.01, 2000)
    if ax_given is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax_given
    colors = ['#EC7063', '#AF7AC5', '#5499C7', '#48C9B0', '#F7DC6F', '#F5B041', '#EB984E', '#BDC3C7'] + ["grey"]*10
    var_exp = 2*rbeta**2*raf*(1-raf)
    v_cuts = smile_stats.calc_cutoffs_new(var_exp, pvals)
    p_cutoffs = np.sort([float(key) for key in v_cuts.keys()])
    for ii, p_cutoff in enumerate(p_cutoffs):
        use = pvals <= p_cutoff
        if ii > 0:
            use = use & (pvals>p_cutoffs[ii-1])
        ax.scatter(raf[use], rbeta[use], color=colors[ii], alpha=alpha, label=r"$\leq$ "+str(p_cutoff))
        v_cut = v_cuts[str(p_cutoff)]
        beta_cut = np.sqrt(v_cut/(2*x_set*(1-x_set)))
        ax.plot(x_set, beta_cut, color=colors[ii], alpha=0.7)
    use = pvals>np.max(p_cutoffs)
    ax.scatter(raf[use], rbeta[use], color=colors[ii+1], alpha=alpha, label="$>$ "+str(np.max(p_cutoffs)))

    min_beta = np.min(rbeta)
    max_beta = np.max(rbeta)
    ax.plot([0.01]*2, [min_beta, max_beta], "--", color="black")
    ax.plot([0.99]*2, [min_beta, max_beta], "--", color="black")

    ax.set_yscale("log")
    ax.set_xscale("logit")
    legend = ax.legend(ncol=2, fontsize=14)
    legend.set_title("p-value")
    ax.set_xlabel("Risk / trait-increasing allele frequency")
    ax.set_ylabel("Effect size")
    ax.set_title(trait)
    if ax_given is None:
        return fig, ax

def plot_vexp_pval(varexp, pval, alpha=0.3, trait=""):
    min_pval = np.min(pval[pval>0])
    pval[pval==0] = min_pval
    neff = smile_stats.calc_n_eff(varexp, pval, use_log=False)
    vmin = np.min(varexp)
    vmax = np.max(varexp)
    v_set = np.linspace(vmin, vmax, 2000)
    p_fit_log = scipy.stats.chi2.logsf(v_set * neff, df=1) / np.log(10)

    fig, axes = plt.subplots(1,2,figsize=(20,8))
    signif = pval <= 5e-08
    for ax in axes:
        ax.scatter(varexp[~signif], -np.log10(pval)[~signif], alpha=alpha, color="black")
        ax.scatter(varexp[signif], -np.log10(pval)[signif], alpha=alpha, color="orange",
                   label=r"genome-wide significant \n($p\leq 5\times10^{-8}$)")
        ax.plot(v_set, -p_fit_log)
        ax.set_xscale("log")
        #ax.set_yscale("symlog", linthresh=-np.log10(5e-08))
        #ax.set_ylim(0, np.max(-np.log10(pval))+5)
        ax.set_xlabel("variance explained")
        ax.set_ylabel(r"$-\log_{10}(pvalue)$")
    axes[0].set_title(trait)
    axes[1].set_yscale("log")
    axes[1].set_ylim(np.min(-np.log10(pval))-2, np.max(-np.log10(pval))+20)
    return fig, ax

def to_vep(fname, chromosomes, positions, A1, A2):
    out_data = {'chr':chromosomes,
                'pos1':positions,
                'pos2':positions,
                "change":["/".join([str(x[0]), str(x[1])]) for x in zip(A1, A2)],
                "trash":[1]*len(positions)}
    out_data = pd.DataFrame(out_data)
    out_data.to_csv(fname, sep=" ", index=False, header=False)

import re

def extract_gene_name(attribute_string):
    match = re.search(r'gene_name "([^"]+)"', attribute_string)
    if match:
        return match.group(1)
    return None

def calculate_distance(pos, start, end, use_start=True):
    assert np.sum(end < start) == 0
    if use_start:
        distance = np.abs(start - pos)
    else:
        start_dist = np.maximum(0, start - pos)
        end_dist = np.maximum(0, pos - end)
        distance = np.maximum(start_dist, end_dist)
    return distance

def get_nearest_protein_coding_gene(gtf_file, chromosomes, positions, use_start=True):
    # Load GTF file into a pandas dataframe
    df_gtf = pd.read_csv(gtf_file, sep='\t', comment='#', header=None,
                         names=['seqname', 'source', 'feature', 'start', 'end',
                                'score', 'strand', 'frame', 'attribute'])

    # Filter for protein-coding genes
    df_genes = df_gtf[df_gtf['feature'] == 'gene']
    df_genes = df_genes[df_genes['attribute'].str.contains('protein_coding')]

    nearest_genes = []
    distances = []
    in_exon = []
    chrom_prev = None

    df_exons = df_gtf.loc[df_gtf['feature'] == 'exon'].copy()

    for ii, chrom in enumerate(chromosomes):
        pos = positions[ii]
        # Filter for the specified chromosome
        if (ii==0) or (chrom != chrom_prev):
            df_chrom = df_genes.loc[df_genes['seqname'] == chrom].copy()

        # Calculate distance to genes
        df_chrom['distance'] = calculate_distance(pos, df_chrom['start'].to_numpy(),
                                                       df_chrom['end'].to_numpy(),
                                                  use_start=use_start)
        closest_ii = np.argmin(df_chrom['distance'])

        # Find the nearest gene and its distance
        nearest_gene = extract_gene_name(df_chrom.iloc[closest_ii]['attribute'])
        distance = df_chrom.iloc[closest_ii]['distance']
        nearest_genes.append(nearest_gene)
        distances.append(distance)

        # Find the exons for the nearest gene
        gene_id = extract_gene_name(df_chrom.iloc[0]['attribute'])
        df_gene_exons = df_exons[df_exons['attribute'].str.contains(nearest_gene)]
        exon_distances = calculate_distance(pos, df_gene_exons['start'].to_numpy(),
                                                 df_gene_exons['end'].to_numpy(), use_start=False)
        in_exon.append(np.min(exon_distances)==0)
        chrom_prev = chrom

    return nearest_genes, distances, in_exon

def read_trait_files(data_dir, ash_type="genome_wide_ash", fname="clumped.{ash_type}.{trait}.max_r2.tsv"):
    main_traits = ["bc", "bmi", "cad", "dbp", "hdl", "height", "ibd", "ldl", "rbc", 
                   "sbp", "scz", "t2d", "triglycerides", "urate", "wbc"]
    update_traits = ["asthma", "arthrosis", "diverticulitis", "fvc", "gallstones", "glaucoma", "grip_strength", 
                     "hypothyroidism", "malignant_neoplasms", "pulse_rate", "uterine_fibroids", "varicose_veins"]
    
    main_traits_labels = ["Breast cancer", "BMI", "CAD", "Diastolic BP", "HDL levels", "Standing height",
                            "IBD", "LDL levels", "RBC", "Systolic BP", "SCZ", "T2D", "Triglycerides", "Urate", "WBC"]
    update_traits_labels = ["Asthma", "Arthrosis", "Diverticulitis", "FVC", "Gallstones", "Glaucoma", "Grip strength",
                            "Hypothyroidism", "Malignant neoplasms", "Pulse rate", "Uterine fibroids", "Varicose veins"]

    # Create a merged list of traits and sort them
    all_traits = np.concatenate([main_traits, update_traits])
    all_labels = np.concatenate([main_traits_labels, update_traits_labels])
    all_order = np.argsort(all_labels)
    all_traits = all_traits[all_order]
    all_labels = all_labels[all_order]

    # fname = "clumped.{ash_type}.{trait}.tsv"

    data_main_traits = {trait: smile_post.read_and_process_trait_data(os.path.join(data_dir, fname.format(ash_type=ash_type, trait=trait))) 
                        for trait in main_traits}
    data_update_traits = {trait: smile_post.read_and_process_trait_data(os.path.join(data_dir, fname.format(ash_type=ash_type, trait=trait)))
                            for trait in update_traits}

    # make a separate dictionary with the data for all traits
    data_all_traits = {trait: data_main_traits[trait] for trait in main_traits}
    data_all_traits.update(data_update_traits)

    # return everything
    return (main_traits, main_traits_labels, data_main_traits,
            update_traits, update_traits_labels, data_update_traits, 
            all_traits, all_labels, data_all_traits)

def plot_basic_smiles(all_traits, all_labels, data_traits_update, min_x, p_thresh, p_cutoff, 
                      plot_name="basic_smiles_all_update.pdf", n_cols=4, offset = -0.01, col_size=10, row_size=6, labelsize=16,
                      loci_count=False):
    """
    Plot scatterplots of per-locus effect sizes versus trait-increasing allele frequency
    for multiple traits, overlaid with a discovery (power) boundary, and save the
    composite panel figure as both PDF and PNG.
    For each trait, loci are filtered prior to plotting by:
    1. Minor allele frequency (MAF) >= min_x
    2. A variance explained threshold derived from a chi-square cutoff:
        v_cut = chi2.ppf(1 - p_thresh, df=1) / median_n_eff
    3. P-value <= p_cutoff
    Only variants passing all filters are retained. A dashed "discovery boundary"
    curve is drawn to represent the minimum detectable effect size across a grid
    of allele frequencies given v_cut: beta_detectable = sqrt(v_cut / (2 * p * (1 - p))).
    Parameters
    ----------
    all_traits : Sequence[str]
        Iterable of trait identifiers (keys into data_traits_update).
    all_labels : Sequence[str]
        Human-readable labels corresponding one-to-one with all_traits. Underscores
        are replaced with spaces in plot annotations.
    data_traits_update : Mapping[str, pandas.DataFrame]
        Dictionary mapping each trait name to a DataFrame containing at least the
        following columns:
          - median_n_eff : Effective sample size (can be scalar-repeated; first
            element is used).
          - var_exp      : Variance explained per locus.
          - maf          : Minor allele frequency (used for filtering).
          - pval         : Association p-value.
          - rbeta        : Effect size estimate aligned to the trait-increasing allele.
          - raf          : Frequency of the trait-increasing (reference) allele
            used on the x-axis.
    min_x : float
        Minimum allele frequency considered both for filtering (maf >= min_x) and
        as the grid step size for constructing the discovery boundary (np.arange(min_x, 1, min_x)).
    p_thresh : float
        Tail probability (alpha) used to derive the chi-square critical value
        (1 - p_thresh quantile) for the variance explained cutoff.
    p_cutoff : float
        Maximum allowed p-value for a locus to be included (pval <= p_cutoff).
    plot_name : str, default="basic_smiles_all.pdf"
        Output filename (PDF). A PNG of the same base name is also written.
    n_cols : int, default=4
        Number of columns in the facet grid. Rows are computed to fit all traits.
    offset : float, default=-0.01
        Positional offset used when placing the global x and y axis labels in figure
        coordinates (left/bottom margins).
    col_size : float, default=10
        Width (in inches) allocated per column (multiplied by n_cols for total width).
    row_size : float, default=6
        Height (in inches) allocated per row (multiplied by number of rows).
    labelsize : int, default=16
        Font size for tick labels on individual subplots.
    loci_count : bool, default=False
        If True, append the number of plotted loci for each trait inside its panel label.
    Behavior
    --------
    - Produces a multi-panel matplotlib figure (one panel per trait).
    - Each panel:
        * Sets x-limits to [-0.02, 1.02].
        * Uses log scaling on the y-axis (effect sizes).
        * Overlays a dashed discovery boundary curve.
        * Plots individual loci as semi-transparent points with black edges.
        * Annotates with the trait label (and locus count if loci_count=True).
    - Unused subplot axes (if trait count not filling the grid) are hidden.
    - Saves the figure twice:
        * PDF: plot_name
        * PNG: plot_name with .pdf replaced by .png (dpi=300)
    Returns
    -------
    None
        The function has the side effect of writing figure files to disk.
    Notes
    -----
    - The discovery boundary uses the maximum observed effect size (scaled by 1.25)
      at the edges (min_x and 1 - min_x) to "cap" the dashed line visually.
    - Ensure that median_n_eff is present and consistent across loci; only the
      first value is used. If per-locus effective sample size varies, this method
      may misrepresent the true detection boundary.
    - Setting an excessively small min_x may densify the boundary grid and slow
      rendering.
    Raises
    ------
    KeyError
        If a trait in all_traits is missing from data_traits_update.
    AttributeError
        If required DataFrame columns are absent.
    Example
    -------
    plot_basic_smiles(
        all_traits=["height", "bmi", "whr"],
        all_labels=["Height", "BMI", "WHR"],
        data_traits_update=trait_df_dict,
        min_x=0.01,
        p_thresh=5e-8,
        p_cutoff=5e-8,
        plot_name="polygenic_effects.pdf",
        n_cols=3,
        loci_count=True
    )
    """
    num_traits = len(all_traits)
    num_rows = math.ceil(num_traits / n_cols)

    fig, axes = plt.subplots(num_rows, n_cols, figsize=(col_size * n_cols, row_size * num_rows))
    for i, trait in enumerate(all_traits):
        trait_df = data_traits_update[trait].copy()
        ax = axes.flatten()[i]
        ax.set_xlim(-0.02, 1.02)
        x_set = np.arange(min_x, 1, min_x)
        v_cut = scipy.stats.chi2.ppf(q=1-p_thresh, df=1)/trait_df.median_n_eff[0]

        cut_rows = np.array(trait_df.var_exp > v_cut) & np.array(trait_df.maf >= min_x)
        cut_rows = cut_rows & np.array(trait_df.pval <= p_cutoff)

        # filter out the rows that don't meet the cutoff
        trait_df = trait_df[cut_rows]

        discov_betas = np.sqrt(v_cut/(2*x_set*(1-x_set)))
        beta_hat = trait_df.rbeta.to_numpy()
        if np.max(beta_hat)*1.25 > np.max(discov_betas):
            ax.plot(np.concatenate(([min_x], x_set, [1-min_x])),
                      np.concatenate(([np.max(beta_hat)*1.25], discov_betas, [np.max(beta_hat)*1.25])),
                      color="darkslategrey", linestyle="dashed", linewidth=4)
        else:
            ax.plot(x_set, discov_betas,
                    color="darkslategrey", linestyle="dashed", linewidth=4)

        sns.scatterplot(x=trait_df.raf.to_numpy(), y=trait_df.rbeta.to_numpy(), data=trait_df, ax=ax, 
                        edgecolor="black", s=120, alpha=0.7)
        
        # make tick labels larger
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.tick_params(axis='both', which='minor', labelsize=labelsize)
        
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("")
        if loci_count:
            ax.text(0.2, 0.95, all_labels[i].replace("_", " ") + ", {} loci".format(len(trait_df.raf.to_numpy())), 
                    transform=ax.transAxes, fontsize=30,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax.text(0.2, 0.95, all_labels[i].replace("_", " "), transform=ax.transAxes, fontsize=30,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # make any unused axes invisible
    for i in range(num_traits, num_rows * n_cols):
        axes.flatten()[i].axis("off")

    # grand x axis label
    fig.text(0.55, offset * 2, "Trait-increasing allele frequency", ha='center', va='center', fontsize=35)
    # grand y axis label
    fig.text(offset, 0.5, "Effect size", ha='center', va='center', rotation='vertical', fontsize=35)

    fig.tight_layout()
    fig.savefig(plot_name, bbox_inches="tight")
    fig.savefig(plot_name.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)