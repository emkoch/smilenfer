## Code for Figure 1
## Show examples of different types of selection and the ability to detect them in simulated data

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Patch
from scipy.stats import chi2

# TODO: move this stuff to smilenfer.plotting
from contour_plots import *

import smilenfer.plotting as splot
from smilenfer.statistics import trad_x_set
import smilenfer.statistics as stats_bf
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

# Define 4 bright, colorblind-friendly colors
# COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"] # old colors
COLORS2 = ["#F0E442", "#009E73", "#F0E442", "#CC79A7"]

def log10_t(x): return x / np.log(10)

V_CUTOFF = 0.005246 # calculated using old file "ibd.5e-5.cojo.normal.no_mhc.tsv", just used to calibrate sims

# Selection intensity parameters corresponding to simulations, just put here manually so I can see them
# Can be obtained from ML_table_IBD_*_1e-08_nsamp_200.tsv files
I2_ex = 0.023
Ip_ex = 0.045
I1_ex = 0.00012
x_set = trad_x_set(0.01, 2000)
beta_set = np.logspace(np.log10(0.08), np.log10(0.8), 200)
beta_cut = np.sqrt(V_CUTOFF / (2*x_set*(1-x_set)))

with open("../data/WF_pile.pkl", "rb") as f:
    WF_pile = pickle.load(f)

discovery_x = np.maximum(stats_bf.discov_x(beta_set, V_CUTOFF), 0.01)
neut_eq = neutral_conditional_densities(beta_set, x_set, discovery_x)
neut_wf = neutral_conditional_densities(beta_set, x_set, discovery_x, pile=WF_pile)
dir_eq = directional_conditional_densities(beta_set, x_set, discovery_x, -I1_ex, 10000)
dir_wf = directional_conditional_densities(beta_set, x_set, discovery_x, -I1_ex, 10000, pile=WF_pile)
stab_eq = stabilizing_conditional_densities(beta_set, x_set, discovery_x, I2_ex, 10000)
stab_wf = stabilizing_conditional_densities(beta_set, x_set, discovery_x, I2_ex, 10000, pile=WF_pile)
plei_eq = pleiotropic_conditional_densities(beta_set, x_set, discovery_x, Ip_ex, 10000)
plei_wf = pleiotropic_conditional_densities(beta_set, x_set, discovery_x, Ip_ex, 10000, pile=WF_pile)

MODELS = [("directional", "dir"), ("stabilizing", "stab"), ("pleiotropic", "plei")]
PARAMETERS = {"directional": ("I1", r"$I_1$"), "stabilizing": ("I2", r"$I_2$"), "pleiotropic": ("Ip", r"$I_p$")}

EXAMPLE_SIMS = {"neutral": "IBD_1e-08_plei_param_combo_0_nsamp_200_rep_0.tsv",
                "directional": "IBD_1e-08_dir_param_combo_6_nsamp_200_rep_1.tsv",
                "stabilizing": "IBD_1e-08_stab_param_combo_71_nsamp_200_rep_1.tsv",
                "pleiotropic": "IBD_1e-08_plei_param_combo_75_nsamp_200_rep_1.tsv"}

PLOT_NAMES = {"neutral": "Neutral", "directional": "Directional", "stabilizing": "1-Trait Stabilizing", "pleiotropic": "Pleiotropic Stabilizing"}

I2_MAX = 5e-2 # maximum I2 to plot in the stabilizing model, gets wonky above this when variants are really too rare to ever appear in the sample
ASC_DIR = "../data/sims/trait_sims/ASCERTAINMENT_SIMS_SIMPLE/"
NO_ASC_DIR = "../data/sims/trait_sims/ASCERTAINMENT_SIMS_SIMPLE_NOASC/"
BASE_ML_TABLES = {"beta_hat":  os.path.join(ASC_DIR, "ML_table_IBD_{}_1e-08_nsamp_200.tsv"),
                  "beta_ash":  os.path.join(ASC_DIR, "ML_table_beta_ash_IBD_{}_1e-08_nsamp_200.tsv"),
                  "beta_true": os.path.join(ASC_DIR, "ML_table_beta_true_IBD_{}_1e-08_nsamp_200.tsv"),
                  "no_asc":    os.path.join(NO_ASC_DIR, "ML_table_IBD_{}_1e-08_nsamp_200.tsv")}
ML_TABLE_LABELS = {"beta_hat": r"GWAS effect size", "beta_ash": r"Shrunk effect size", "beta_true": r"True effect size", "no_asc": r"No ascertainment"}
ML_FILES = {m[0]: {k: v.format(m[1]) for k, v in BASE_ML_TABLES.items()} for m in MODELS}
ML_TABLES = {m[0]: {k: pd.read_csv(v, sep="\t") for k, v in ML_FILES[m[0]].items()} for m in MODELS}
# Remove points in stabilizing tables where I2 > I2_MAX
for k, v in ML_TABLES["stabilizing"].items():
    ML_TABLES["stabilizing"][k] = v[v.I2 < I2_MAX]

# Load the data from EXAMPLE_SIMS
data = {k: pd.read_csv(os.path.join(NO_ASC_DIR, v), sep="\t") for k, v in EXAMPLE_SIMS.items()}
# Add a column to each dataframe with the type of selection (neutral, directional, etc.)
for k, v in data.items():
    v["selection"] = k

# Read in neutral simulations
NEUT_DIR = "../data/sims/trait_sims/ASCERTAINMENT_SIMS_SIMPLE_NEUT/"
NEUT_FILES = ["ML_table_beta_true_IBD_dir_1e-08_nsamp_200.tsv", 
              "ML_table_beta_true_IBD_stab_1e-08_nsamp_200.tsv", 
              "ML_table_beta_true_IBD_plei_1e-08_nsamp_200.tsv"]
NEUT_FILES_ASC = ["ML_table_IBD_dir_1e-08_nsamp_200.tsv",
                  "ML_table_IBD_stab_1e-08_nsamp_200.tsv",
                  "ML_table_IBD_plei_1e-08_nsamp_200.tsv"]
NEUT_FILES_ASH = ["ML_table_beta_ash_IBD_dir_1e-08_nsamp_200.tsv",
                  "ML_table_beta_ash_IBD_stab_1e-08_nsamp_200.tsv",
                  "ML_table_beta_ash_IBD_plei_1e-08_nsamp_200.tsv"]
NEUT_TABLES = [pd.read_csv(os.path.join(NEUT_DIR, f), sep="\t") for f in NEUT_FILES]
NEUT_TABLES_ASC = [pd.read_csv(os.path.join(NEUT_DIR, f), sep="\t") for f in NEUT_FILES_ASC]
NEUT_TABLES_ASH = [pd.read_csv(os.path.join(NEUT_DIR, f), sep="\t") for f in NEUT_FILES_ASH]

# Calculate -2*Delta log likelihood for each model, AIC scale
for t in NEUT_TABLES:
    t["diff_dir"] = 2*(t.ll_dir - t.ll_neut)
    t["diff_stab"] = 2*(t.ll_stab - t.ll_neut)
    t["diff_plei"] = 2*(t.ll_plei - t.ll_neut)
    # Also compute p-values
    t["p_dir"] = chi2.logsf(t.diff_dir, 1)/-np.log(10)
    t["p_stab"] = chi2.logsf(t.diff_stab, 1)/-np.log(10)
    t["p_plei"] = chi2.logsf(t.diff_plei, 1)/-np.log(10)
for t in NEUT_TABLES_ASC:
    t["diff_dir"] = 2*(t.ll_dir - t.ll_neut)
    t["diff_stab"] = 2*(t.ll_stab - t.ll_neut)
    t["diff_plei"] = 2*(t.ll_plei - t.ll_neut)
    # Also compute p-values
    t["p_dir"] = chi2.logsf(t.diff_dir, 1)/-np.log(10)
    t["p_stab"] = chi2.logsf(t.diff_stab, 1)/-np.log(10)
    t["p_plei"] = chi2.logsf(t.diff_plei, 1)/-np.log(10)
for t in NEUT_TABLES_ASH:
    t["diff_dir"] = 2*(t.ll_dir - t.ll_neut)
    t["diff_stab"] = 2*(t.ll_stab - t.ll_neut)
    t["diff_plei"] = 2*(t.ll_plei - t.ll_neut)
    # Also compute p-values
    t["p_dir"] = chi2.logsf(t.diff_dir, 1)/-np.log(10)
    t["p_stab"] = chi2.logsf(t.diff_stab, 1)/-np.log(10)
    t["p_plei"] = chi2.logsf(t.diff_plei, 1)/-np.log(10)
# Concatenate the tables in NEUT_TABLES into a single table
NEUT_TABLE = pd.concat(NEUT_TABLES, ignore_index=True)
NEUT_TABLE_ASC = pd.concat(NEUT_TABLES_ASC, ignore_index=True)
NEUT_TABLE_ASH = pd.concat(NEUT_TABLES_ASH, ignore_index=True)

# Set up a plot with 4 columns and two rows in matplotlib
fig, axes = plt.subplots(3, 4, figsize=(17.5, 9))
# Plot beta_cut in each column of the top row
for ax in axes[0]:
    ax.plot(x_set, beta_cut, color="black", linestyle="--")
    # add vertical dashed lines at x=0.01 and x=0.99
    ax.plot([0.01, 0.01], [np.max(beta_cut), np.max(beta_set)], color="black", linestyle="--")
    ax.plot([0.99, 0.99], [np.max(beta_cut), np.max(beta_set)], color="black", linestyle="--")

# Plot beta_cut in each column of the top row
for ax in axes[1]:
    ax.plot(x_set, beta_cut, color="black", linestyle="--")
    # add vertical dashed lines at x=0.01 and x=0.99
    ax.plot([0.01, 0.01], [np.max(beta_cut), np.max(beta_set)], color="black", linestyle="--")
    ax.plot([0.99, 0.99], [np.max(beta_cut), np.max(beta_set)], color="black", linestyle="--")

# Plot contours
cmap = "cividis_r"# plt.get_cmap('jet')
# from matplotlib import cm
# cividis_reversed = cm.cividis.reversed()
# cmap = cm.cividis.reversed()

contours, density_min, density_max = conditional_contour_plot(axes[0, 2], x_set, beta_set, stab_wf, add_colorbar=False, cdf_min=0.025, cdf_max=0.99, 
                                                              use_contourf=True, cmap=cmap, xlabel="", ylabel="", density_min=-5, density_max=4, n_levels=None)
_, _, _ = conditional_contour_plot(axes[0,1], x_set, beta_set, dir_wf, add_colorbar=False, levels=contours.levels, 
                                                       use_contourf=True, cmap=cmap, xlabel="", ylabel="")
_, _, _ = conditional_contour_plot(axes[0,0], x_set, beta_set, neut_wf, add_colorbar=False, levels=contours.levels, 
                                                       use_contourf=True,    cmap=cmap, xlabel="", ylabel="")
_, _, _ = conditional_contour_plot(axes[0,3], x_set, beta_set, plei_wf, add_colorbar=False, levels=contours.levels, 
                                                       use_contourf=True, cmap=cmap, xlabel="", ylabel="")

# Plot simulation results in each column of the top rowolor="lightgreen",
for ax, (k, v) in zip(axes[0], data.items()):
    if k=="directional":
        plot_raf = 1 - v.raf # flip raf for directional so it looks like selection against risk
    else:
        plot_raf = v.raf
    # Plot the simulated values of beta against raf, make the edge black and the inside transparent
    # ax.scatter(plot_raf, v.beta, marker=".", facecolors="None", edgecolor="crimson", s=50, alpha=0.5, linewidths=2) # color="lightgreen"
    # Add a vertical line from the maximum beta_cut to the maximum beta in the data
    ax.plot([0.01, 0.01], [np.max(beta_cut), v.beta.max()], color="black", linestyle="--")
    ax.plot([0.99, 0.99], [np.max(beta_cut), v.beta.max()], color="black", linestyle="--")

    # Add some italic text in the top center of the plot
    ax.text(0.5, 0.85, PLOT_NAMES[k], transform=ax.transAxes, size=16, ha="center", va="bottom", style="italic", color="black")#color="bisque")

    # log scale the y axis and remove all ticks and tick labels
    ax.set_yscale("log")
    # Remove the y ticks and tick labels taking into account the fact that we have set yscale to log
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Set ticks and ticklabels for the x-axis
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([0, 0.5, 1])
    # Set the axes limits
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.08, 0.8])

# Title the y-axis of the left-most column, make the font bold
axes[0, 0].set_ylabel("Effect size", fontweight="bold")

# Add horizontal text to the left of the first column, first row, just left of the x-axis tick labels
axes[0, 0].text(-0.25, -0.1, "RAF", transform=axes[0, 0].transAxes, size=20, fontweight="bold")

# Add a colorbar to the right of all three rows of plots, move it up a bit so it is next to the top two rows
cbar = plt.colorbar(contours, ax=axes.ravel().tolist(), fraction=0.05, pad=0.05, aspect=20, shrink=4)
# move the color bar up a bit
cbar.ax.set_position([0.85, 0.5, 1, 0.3])
# label the colorbar log2 density and put the label to the left of the colorbar
cbar.ax.set_ylabel(r"$\log_{2}$ density", rotation=270, labelpad=20, fontsize=16)

# Plot simulation results in each column of the top row
for ax, (k, v) in zip(axes[1], data.items()):
    if k=="directional":
        plot_raf = 1 - v.raf # flip raf for directional so it looks like selection against risk
    else:
        plot_raf = v.raf
    # Plot the simulated values of beta against raf, make the edge black and the inside transparent
    ax.scatter(plot_raf, v.beta, marker=".", facecolors="None", edgecolor="crimson", s=50, alpha=0.5, linewidths=2) # color="lightgreen"
    # Add a vertical line from the maximum beta_cut to the maximum beta in the data
    ax.plot([0.01, 0.01], [np.max(beta_cut), v.beta.max()], color="black", linestyle="--")
    ax.plot([0.99, 0.99], [np.max(beta_cut), v.beta.max()], color="black", linestyle="--")

    # Add some italic text in the top center of the plot
    # ax.text(0.5, 0.9, PLOT_NAMES[k], transform=ax.transAxes, size=16, ha="center", va="bottom", style="italic", color="black")#color="bisque")

    # log scale the y axis and remove all ticks and tick labels
    ax.set_yscale("log")
    # Remove the y ticks and tick labels taking into account the fact that we have set yscale to log
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Set ticks and ticklabels for the x-axis
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([0, 0.5, 1])
    # Set the axes limits
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.08, 0.8])

# Title the y-axis of the left-most column, make the font bold
axes[1, 0].set_ylabel("Effect size", fontweight="bold")

# Add horizontal text to the left of the first column, first row, just left of the x-axis tick labels
axes[1, 0].text(-0.25, -0.1, "RAF", transform=axes[1, 0].transAxes, size=20, fontweight="bold")

# previously -0.28

axes[0, 0].text(-0.38, 0.90, "a", transform=axes[0, 0].transAxes, size=20, fontweight="bold")
axes[1, 0].text(-0.38, 0.90, "b", transform=axes[1, 0].transAxes, size=20, fontweight="bold")
axes[2, 0].text(-0.38, 0.97, "C", transform=axes[2, 0].transAxes, size=20, fontweight="bold")

# axes[1, 1].text(-0.1, 0.95, "c", transform=axes[1, 1].transAxes, size=20, fontweight="bold")
# axes[1, 2].text(-0.1, 0.95, "d", transform=axes[1, 2].transAxes, size=20, fontweight="bold")
# axes[1, 3].text(-0.1, 0.95, "e", transform=axes[1, 3].transAxes, size=20, fontweight="bold")


# Plot paired boxplots in the bottom left plot using NEUT_TABLE and NEUT_TABLE_ASC
# Pair "diff_dir", "diff_stab", and "diff_plei" from NEUT_TABLE and NEUT_TABLE_ASC
tables = [(NEUT_TABLE[model], NEUT_TABLE_ASC[model], NEUT_TABLE_ASH[model]) for model in ["p_dir", "p_stab", "p_plei"]]
positions = np.arange(1, len(tables)*3+1, 3)
color_reorder = [3, 0, 1]
artists = []
# Plot the paired boxplots
for i, (t1, t2, t3) in enumerate(tables):
    boxplot = axes[2,0].boxplot([t1, t2, t3], positions=[positions[i], positions[i]+0.7, positions[i]+1.4], 
                      labels=["", "", ""], whis=(0,95), widths=0.65, showfliers=False, patch_artist=True)
    
    # Set different colors for the boxplots in each pair
    for j, patch in enumerate(boxplot['boxes']):
        patch.set_facecolor(COLORS2[color_reorder[j]])
        if i==0:
            artists.append(Patch(facecolor=COLORS2[color_reorder[j]], label=list(ML_TABLE_LABELS.values())[color_reorder[j]]))
# axes[2,0].legend(handles=artists, fontsize=10)

# Add x-axis ticks and tick labels to the bottom left plot one for each triplet of boxplots
axes[2,0].set_xticks(positions+0.7)
axes[2,0].set_xticklabels(["Dir.", "1T Stab.", "Plei. Stab."], ha="center", fontsize=13)

# Plot three boxplots in the bottom left plot using NEUT_TABLE
axes[2,0].set_yscale("symlog", linthresh=2)
axes[2,0].set_ylim([-0.5, 200])
axes[2,0].set_yticks([-0.5, 0, 0.5, 1, 2, 10, 100])
axes[2,0].set_yticklabels(["", 0, "", 1, 2, 10, 100])
axes[2,0].set_ylabel(r"$-\log_{10} \mathrm{p-value}$", fontweight="bold")

# Plot the non-neutral simulation tables in the second row
for j, (ax, (k, v)) in enumerate(zip(axes[2, 1:], ML_TABLES.items())):
    # Plot three sets of p-values for each model
    for i, (k2, v2) in enumerate(v.items()):
        if k2=="beta_true":
            continue
        ax.scatter(v2[PARAMETERS[k][0]], chi2.logsf(2*(v2["ll_"+MODELS[j][1]] - v2.ll_neut), 1)/-np.log(10), 
                   marker="o", edgecolor="black", s=50, alpha=0.4, color=COLORS2[i])
        # Calculate a smoothed curve using LOESS and plot that on top
        loess = sm.nonparametric.lowess(chi2.logsf(2*(v2["ll_"+MODELS[j][1]] - v2.ll_neut), 1)/-np.log(10), v2[PARAMETERS[k][0]], frac=0.5)
        ax.plot(loess[:,0], loess[:,1], linestyle="-", linewidth=3, label=ML_TABLE_LABELS[k2], color=COLORS2[i])
    # Log scale the x axis
    ax.set_xscale("log")
    # Sym-log scale the y axis, with linear threshold at 2
    ax.set_yscale("symlog", linthresh=2)
    ax.set_ylim([-0.5, 500])
    ax.set_yticks([-0.5, 0, 0.5, 1, 2, 10, 100])
    ax.set_yticklabels([])
    ax.text(-0.15, -0.1, PARAMETERS[k][1], transform=ax.transAxes, size=20, fontweight="bold")

# Set y tick labels at -0.5, 0, 0.5, 1, 2, 10, 100
#axes[1,1].set_yticklabels(["", 0, "", 1, 2, 10, 100])
axes[2,1].legend(fontsize=10, loc="lower right")
# Reorder legend entries
handles, labels = axes[2,1].get_legend_handles_labels()
handles = [handles[2], handles[0], handles[1]]
labels = [labels[2], labels[0], labels[1]]
#axes[2,1].legend(handles, labels, fontsize=10, loc="outside")

plt.legend(handles, labels, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# hide the  legend in axes[2,1]
axes[2,1].legend().set_visible(False)


# add a legend to the right of all plots on the bottom row and beneath the colorbar
# plt.legend() 

# Save Figure 1 as a pdf
fig.savefig("figure_1_ml_c_tmp.pdf", bbox_inches="tight")
# Also save as a png
fig.savefig("figure_1_ml_c_tmp.png", bbox_inches="tight", dpi=300)

#################################################################
#################################################################
# Make a plot showing when the high-dimensional model is prefered
# in the simulations used to make Figure 1

colors = ["#E69F00", "#009E73"]


fig, ax = plt.subplots(figsize=(8, 6))

ms = 20

ax.scatter(2*(ML_TABLES["stabilizing"]["no_asc"].ll_stab - ML_TABLES["stabilizing"]["no_asc"].ll_neut-1), 
           2*(ML_TABLES["stabilizing"]["no_asc"].ll_plei - ML_TABLES["stabilizing"]["no_asc"].ll_stab), 
           color=colors[0], alpha=0.5, s=ms)
# Add loess fit
loess = sm.nonparametric.lowess
x = 2*(ML_TABLES["stabilizing"]["no_asc"].ll_stab - ML_TABLES["stabilizing"]["no_asc"].ll_neut-1)
y = 2*(ML_TABLES["stabilizing"]["no_asc"].ll_plei - ML_TABLES["stabilizing"]["no_asc"].ll_stab)
z = loess(y, x, frac=0.5)
ax.plot(z[:, 0], z[:, 1], color=colors[0], linewidth=4, label="1T sim.")

# Do the same for pleiotropic
ax.scatter(2*(ML_TABLES["pleiotropic"]["no_asc"].ll_plei - ML_TABLES["pleiotropic"]["no_asc"].ll_neut-1),
           2*(ML_TABLES["pleiotropic"]["no_asc"].ll_plei - ML_TABLES["pleiotropic"]["no_asc"].ll_stab), 
           color=colors[1], alpha=0.35, s=ms)
# add loess fit
x = 2*(ML_TABLES["pleiotropic"]["no_asc"].ll_plei - ML_TABLES["pleiotropic"]["no_asc"].ll_neut-1)
y = 2*(ML_TABLES["pleiotropic"]["no_asc"].ll_plei - ML_TABLES["pleiotropic"]["no_asc"].ll_stab)
z = loess(y, x, frac=0.5)
ax.plot(z[:, 0], z[:, 1], color=colors[1], linewidth=4, label="Plei. sim.")

# Do the same for beta_hat rather than no_asc
ax.scatter(2*(ML_TABLES["stabilizing"]["beta_hat"].ll_stab - ML_TABLES["stabilizing"]["beta_hat"].ll_neut-1),
           2*(ML_TABLES["stabilizing"]["beta_hat"].ll_plei - ML_TABLES["stabilizing"]["beta_hat"].ll_stab),
           color=colors[0], alpha=0.35, marker="^", s=ms)
# add loess fit
x = 2*(ML_TABLES["stabilizing"]["beta_hat"].ll_stab - ML_TABLES["stabilizing"]["beta_hat"].ll_neut-1)
y = 2*(ML_TABLES["stabilizing"]["beta_hat"].ll_plei - ML_TABLES["stabilizing"]["beta_hat"].ll_stab)
z = loess(y, x, frac=0.5)
ax.plot(z[:, 0], z[:, 1], color=colors[0], linewidth=4, linestyle="--", label="1T sim. (GWAS noise)")

# Do the same for pleiotropic
ax.scatter(2*(ML_TABLES["pleiotropic"]["beta_hat"].ll_plei - ML_TABLES["pleiotropic"]["beta_hat"].ll_neut-1),
              2*(ML_TABLES["pleiotropic"]["beta_hat"].ll_plei - ML_TABLES["pleiotropic"]["beta_hat"].ll_stab),
                color=colors[1], alpha=0.5, marker="^", s=ms)
# add loess fit
x = 2*(ML_TABLES["pleiotropic"]["beta_hat"].ll_plei - ML_TABLES["pleiotropic"]["beta_hat"].ll_neut-1)
y = 2*(ML_TABLES["pleiotropic"]["beta_hat"].ll_plei - ML_TABLES["pleiotropic"]["beta_hat"].ll_stab)
z = loess(y, x, frac=0.5)
ax.plot(z[:, 0], z[:, 1], color=colors[1], linewidth=4, linestyle="--", label="Plei. sim. (GWAS noise)")

ax.set_xscale("symlog", linthresh=2)
ax.set_xlim([-5, 500])
ax.set_xticks([-2, 0, 2, 10, 100])
ax.set_xticklabels([-2, 0, 2, 10, 100])
ax.set_xlabel(r"Evidence for selection $(-\Delta \mathrm{AIC}_{\mathrm{model} - \mathrm{neut}})$", fontweight="bold")

ax.set_yscale("symlog", linthresh=2)
ax.set_ylim([-50, 100])
ax.set_yticks([-10, -2, 0, 2, 10])
ax.set_yticklabels([-10, -2, 0, 2, 10])
ax.set_ylabel(r"Evidence for pleiotropic model $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$", fontweight="bold")

ax.legend(fontsize=14)

fig.tight_layout()
fig.savefig("IBD_PRIOR_STAB_PLEI.PDF", bbox_inches="tight")