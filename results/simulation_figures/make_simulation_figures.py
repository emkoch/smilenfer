import os
from os.path import join

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from adjustText import adjust_text

import smilenfer.plotting as splot
import smilenfer.var_dist as vd
from smilenfer.statistics import trad_x_set
import random
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data/sims"

V_CUTOFF = 0.005246 # calculated using old file "ibd.5e-5.cojo.normal.no_mhc.tsv", just used to calibrate sims
x_set = trad_x_set(0.01, 2000)
beta_cut = np.sqrt(V_CUTOFF / (2*x_set*(1-x_set)))

MODELS = [("directional", "dir"), ("stabilizing", "stab"), ("pleiotropic", "plei")]
PARAMETERS = {"directional": ("I1", r"$I_1$"), "stabilizing": ("I2", r"$I_2$"), "pleiotropic": ("Ip", r"$I_p$")}

I2_MAX = 4e-2

asc_set = ["asc", "no_asc"]
plei_set = ["plei", "stab"]
sfs_set = ["eq", "WF"]

ML_BASENAME = "ML_table_{asc}_{plei}_{sfs}.tsv"
# Load the ML tables into a dictionary
DFE_ML_TABLES = {(asc, plei, sfs): pd.read_csv(os.path.join(data_dir, "DFE_sims/large/",
                                                        ML_BASENAME.format(asc=asc, plei=plei, sfs=sfs)), sep="\t")
              for asc in asc_set for plei in plei_set for sfs in sfs_set}

# Define one bright and one dark color that are colorblind friendly
colors = ["#E69F00", "#009E73"]

def log10_t(x): return x / np.log(10)

for key in DFE_ML_TABLES.keys():
    dd_ml = DFE_ML_TABLES[key]
    for col in ["neut", "stab", "plei", "plei_ssd"]:
        dd_ml["ll_{}".format(col)] = dd_ml["ll_{}".format(col)]

# calculated using old file "ibd.5e-5.cojo.normal.no_mhc.tsv", 
# just used to calibrate sims
x_set = trad_x_set(0.01, 2000)
beta_cut = np.sqrt(V_CUTOFF / (2*x_set*(1-x_set)))

# Load a simulation
no_asc_dir = os.path.join(data_dir, "trait_sims/ASCERTAINMENT_SIMS_SIMPLE_NOASC/individual_sims")
ex_sim_fname = "IBD_1e-08_stab_param_combo_0_nsamp_200_rep_1.tsv.gz"
d_test = pd.read_csv(join(no_asc_dir, ex_sim_fname), sep="\t", compression="gzip")

# Load the entire set of simulations
stab_sims = []
plei_sims = []
for combo in range(100):
    for rep in range(3):
        fname = "IBD_1e-08_stab_param_combo_{}_nsamp_200_rep_{}.tsv.gz".format(combo, rep)
        dd = pd.read_csv(join(no_asc_dir, fname), sep="\t", compression="gzip")
        dd["zz"] = (2*dd.beta**2 * dd.raf * (1-dd.raf) - V_CUTOFF) / V_CUTOFF
        stab_sims.append(dd)
        fname = "IBD_1e-08_plei_param_combo_{}_nsamp_200_rep_{}.tsv.gz".format(combo, rep)
        dd = pd.read_csv(join(no_asc_dir, fname), sep="\t", compression="gzip")
        dd["zz"] = (2*dd.beta**2 * dd.raf * (1-dd.raf) - V_CUTOFF) / V_CUTOFF
        plei_sims.append(dd)

# Load the entire set of DFE simulations
DFE_DIR = "../data/sims/DFE_sims/large/samples"
stab_dfe_sims = []
plei_dfe_sims = []
fname = "sample_no_asc_plei_WF_0_{}.csv.gz".format(rep)
dd = pd.read_csv(join(DFE_DIR, fname), sep=",", compression="gzip")
dd["vv"] = 2*dd.b**2 * dd.x * (1-dd.x) - 1

## v cutoff is 1 for these simulations
for rep in range(100):
    fname = "sample_no_asc_stab_WF_0_{}.csv.gz".format(rep)
    dd = pd.read_csv(join(DFE_DIR, fname), sep=",", compression="gzip")
    dd["vv"] = 2*dd.b**2 * dd.x * (1-dd.x) - 1
    stab_dfe_sims.append(dd)
    fname = "sample_no_asc_plei_WF_0_{}.csv.gz".format(rep)
    dd = pd.read_csv(join(DFE_DIR, fname), sep=",", compression="gzip")
    dd["vv"] = 2*dd.b**2 * dd.x * (1-dd.x) - 1
    plei_dfe_sims.append(dd)

stab_ml = pd.read_csv(join("../data/sims/trait_sims/ASCERTAINMENT_SIMS_SIMPLE_NOASC", 
                           "ML_table_IBD_stab_1e-08_nsamp_200.tsv"), sep="\t")
plei_ml = pd.read_csv(join("../data/sims/trait_sims/ASCERTAINMENT_SIMS_SIMPLE_NOASC", 
                           "ML_table_IBD_plei_1e-08_nsamp_200.tsv"), sep="\t")


# Fit the variance distribution for the entire set of simulations
eta_1d_stab = [vd.fit_stab_1D(dd.zz) for dd in stab_sims]
eta_hd_stab = [vd.fit_stab_hD(dd.zz) for dd in stab_sims]
eta_1d_plei = [vd.fit_stab_1D(dd.zz) for dd in plei_sims]
eta_hd_plei = [vd.fit_stab_hD(dd.zz) for dd in plei_sims]

# Fit the variance distribution for the entire set of DFE simulations
eta_1d_stab_dfe = [vd.fit_stab_1D(dd.vv) for dd in stab_dfe_sims]
eta_hd_stab_dfe = [vd.fit_stab_hD(dd.vv) for dd in stab_dfe_sims]
eta_1d_plei_dfe = [vd.fit_stab_1D(dd.vv) for dd in plei_dfe_sims]
eta_hd_plei_dfe = [vd.fit_stab_hD(dd.vv) for dd in plei_dfe_sims]


# Calculate the log-likelihood for the set of simulations
ll_1d_stab = np.array([np.sum(vd.ll_stab_1D(dd.zz, eta_1d_stab[ii])) 
                       for ii, dd in enumerate(stab_sims)])
ll_hd_stab = np.array([np.sum(vd.ll_stab_hD(dd.zz, eta_hd_stab[ii])) 
                       for ii, dd in enumerate(stab_sims)])
ll_1d_plei = np.array([np.sum(vd.ll_stab_1D(dd.zz, eta_1d_plei[ii])) 
                       for ii, dd in enumerate(plei_sims)])
ll_hd_plei = np.array([np.sum(vd.ll_stab_hD(dd.zz, eta_hd_plei[ii])) 
                       for ii, dd in enumerate(plei_sims)])

# Calculate the log-likelihood for the set of DFE simulations
ll_1d_stab_dfe = np.array([np.sum(vd.ll_stab_1D(dd.vv, eta_1d_stab_dfe[ii]))
                            for ii, dd in enumerate(stab_dfe_sims)])
ll_hd_stab_dfe = np.array([np.sum(vd.ll_stab_hD(dd.vv, eta_hd_stab_dfe[ii]))
                            for ii, dd in enumerate(stab_dfe_sims)])
ll_1d_plei_dfe = np.array([np.sum(vd.ll_stab_1D(dd.vv, eta_1d_plei_dfe[ii]))
                            for ii, dd in enumerate(plei_dfe_sims)])
ll_hd_plei_dfe = np.array([np.sum(vd.ll_stab_hD(dd.vv, eta_hd_plei_dfe[ii]))
                            for ii, dd in enumerate(plei_dfe_sims)])

# Make violin plots of AIC values computed from likelihood differences between 1d and hd models
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
parts = ax.violinplot([2*(ll_hd_stab_dfe - ll_1d_stab_dfe), 2*(ll_hd_plei_dfe - ll_1d_plei_dfe)], 
                      showmeans=False, showmedians=False, showextrema=False)
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)
# put boxplots on top of violin plots
ax.boxplot([2*(ll_hd_stab_dfe - ll_1d_stab_dfe), 2*(ll_hd_plei_dfe - ll_1d_plei_dfe)], showfliers=False, widths=0.1)


ax.set_xticks([1, 2])
ax.set_xticklabels(["1T simulation", "Plei simulation"])
ax.set_ylabel(r"Simons variance dist. $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$")
fig.tight_layout()
fig.savefig("dfe_1_simons_violin.pdf", bbox_inches="tight")

# Get unique cuts
cuts = np.array([1, 3, 5, 7]) 
LLs = {key:[] for key in DFE_ML_TABLES.keys()}
LLs_ssd = {key:[] for key in DFE_ML_TABLES.keys()}
LLs_stab = {key:[] for key in DFE_ML_TABLES.keys()}
LLs_plei = {key:[] for key in DFE_ML_TABLES.keys()}
for cut in cuts:
    for key in DFE_ML_TABLES.keys():
        LL_CUT = DFE_ML_TABLES[key].loc[DFE_ML_TABLES[key]["cut"] == cut]
        LL_CUT_PLEI = 2*(LL_CUT.ll_plei.to_numpy() - LL_CUT.ll_stab.to_numpy())
        LL_CUT_PLEI_SSD = 2*(LL_CUT.ll_plei_ssd.to_numpy() - LL_CUT.ll_stab.to_numpy())
        LL_NEUT_PLEI = 2*(LL_CUT.ll_plei.to_numpy() - LL_CUT.ll_neut.to_numpy())
        LL_NEUT_STAB = 2*(LL_CUT.ll_stab.to_numpy() - LL_CUT.ll_neut.to_numpy())
        LLs[key].append(LL_CUT_PLEI)
        LLs_ssd[key].append(LL_CUT_PLEI_SSD)
        LLs_stab[key].append(LL_NEUT_STAB)
        LLs_plei[key].append(LL_NEUT_PLEI)

boxwidth = 0.4

no_asc_c = "#CC79A7"
asc_c = "#F0E442"

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

ms = 20

ax.axhline(0, c="k", ls="--")

# Make a boxplot
ax.boxplot(LLs[("no_asc", "stab", "WF")], positions=cuts, widths=boxwidth, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs[("asc", "stab", "WF")], positions=cuts+0.4, widths=boxwidth, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs[("no_asc", "plei", "WF")], positions=cuts+9, widths=boxwidth, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs[("asc", "plei", "WF")], positions=cuts+9.4, widths=boxwidth, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.plot([], c=no_asc_c, label="No ascertainment")
ax.plot([], c=asc_c, label="GWAS effect size")
ax.legend(loc="lower right", frameon=True, facecolor="white")


ax.set_ylabel(r"Evidence for pleiotropy $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$", fontweight="bold")
ax.set_yscale("symlog", linthresh=2)
ax.set_ylim([-50, 50])
ax.set_yticks([-40, -10, -2, -1,  0,  1, 2, 10, 40])
ax.set_yticklabels([-40, -10, -2, -1, 0, 1, 2, 10, 40])

ax.set_xticks(list(cuts+0.2) + [cut+9.2 for cut in cuts])
ax.set_xticklabels(list(int(cut) for cut in cuts) + list(int(cut) for cut in cuts))
ax.set_xlabel("Variance threshold", fontweight="bold")

ax.axvline(8.8, c="k", ls="-")

ax.text(7.5, 20, "1T Stabilizing", ha="right", va="bottom", rotation=0, fontsize=16)
ax.text(10.5, 20, "Plei Stabilizing", ha="left", va="bottom", rotation=0, fontsize=16)

fig.tight_layout()
fig.savefig("Figure_5_leftonly.pdf", bbox_inches="tight")
fig.savefig("Figure_5_leftonly.png", bbox_inches="tight", dpi=300)

fig, ax = plt.subplots(figsize=(8, 6))
boxwidth = 0.4

ax.axhline(0, c="k", ls="--")

ax.boxplot(LLs[("no_asc", "stab", "eq")], positions=cuts, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs[("asc", "stab", "eq")], positions=cuts+0.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs[("no_asc", "plei", "eq")], positions=cuts+9, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs[("asc", "plei", "eq")], positions=cuts+9.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.set_title("Equilibrium population model", fontweight="bold")
ax.plot([], c=no_asc_c, label="No ascertainment")
ax.plot([], c=asc_c, label="GWAS effect size")
ax.legend(loc="lower right", frameon=True, facecolor="white")

ax.set_ylabel(r"Evidence for pleiotropy $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$", fontweight="bold")
ax.set_yscale("symlog", linthresh=2)
ax.set_ylim([-50, 70])
ax.set_yticks([-40, 10, -2, -1, 0,  1, 2, 10, 40])
ax.set_yticklabels([40, -10, -2, -1, 0, 1, 2, 10, 40])

ax.set_xticks(list(cuts+0.2) + [cut+9.2 for cut in cuts])
ax.set_xticklabels(list(int(cut) for cut in cuts) + list(int(cut) for cut in cuts))
ax.set_xlabel("Variance threshold", fontweight="bold")

ax.axvline(8.8, c="k", ls="-")

ax.text(5.0, 30, "1T Stabilizing", ha="right", va="bottom", rotation=0, fontsize=16)
ax.text(9.5, 30, "Plei Stabilizing", ha="left", va="bottom", rotation=0, fontsize=16)

fig.tight_layout()
fig.savefig("supp_eq_effect.pdf", bbox_inches="tight")


fig, ax = plt.subplots(figsize=(8, 6))
boxwidth = 0.4

ax.axhline(0, c="k", ls="--")

ax.boxplot(LLs[("no_asc", "stab", "eq")], positions=cuts, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs_ssd[("no_asc", "stab", "eq")], positions=cuts+0.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs[("no_asc", "plei", "eq")], positions=cuts+9, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs_ssd[("no_asc", "plei", "eq")], positions=cuts+9.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.set_title("Equilibrium population model", fontweight="bold")
ax.plot([], c=no_asc_c, label="Flat prior")
ax.plot([], c=asc_c, label="SSD prior")
ax.legend(loc="lower right", frameon=True, facecolor="white")

ax.set_ylabel(r"Evidence for pleiotropy $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$", fontweight="bold")
ax.set_yscale("symlog", linthresh=2)
ax.set_ylim([-30, 70])
ax.set_yticks([-10, -2, -1, 0, 1, 2, 10])
ax.set_yticklabels([-10, -2, -1, 0, 1, 2, 10])
ax.set_xticks(list(cuts+0.2) + [cut+9.2 for cut in cuts])
ax.set_xticklabels(list(int(cut) for cut in cuts) + list(int(cut) for cut in cuts))
ax.set_xlabel("Variance threshold", fontweight="bold")


ax.axvline(8.8, c="k", ls="-")

ax.text(5.0, 30, "1T Stabilizing", ha="right", va="bottom", rotation=0, fontsize=16)
ax.text(9.5, 30, "Plei Stabilizing", ha="left", va="bottom", rotation=0, fontsize=16)

fig.tight_layout()
fig.savefig("supp_SSD_effect_eq.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(8, 6))
boxwidth = 0.4

ax.axhline(0, c="k", ls="--")

ax.boxplot(LLs[("no_asc", "stab", "WF")], positions=cuts, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs_ssd[("no_asc", "stab", "WF")], positions=cuts+0.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs[("no_asc", "plei", "WF")], positions=cuts+9, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs_ssd[("no_asc", "plei", "WF")], positions=cuts+9.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.set_title("Non-equilibrium population model", fontweight="bold")
ax.plot([], c=no_asc_c, label="Flat prior")
ax.plot([], c=asc_c, label="SSD prior")
ax.legend(loc="lower right", frameon=True, facecolor="white")
ax.set_title("Non-equilibrium population model", fontweight="bold")

ax.set_ylabel(r"Evidence for pleiotropy $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$", fontweight="bold")
ax.set_yscale("symlog", linthresh=2)
ax.set_ylim([-20, 30])
ax.set_yticks([-10, -2, -1, 0,  1, 2, 10])
ax.set_yticklabels([-10, -2, -1,  0,  1, 2, 10])
ax.set_xticks(list(cuts+0.2) + [cut+9.2 for cut in cuts])
ax.set_xticklabels(list(int(cut) for cut in cuts) + list(int(cut) for cut in cuts))
ax.set_xlabel("Variance threshold", fontweight="bold")

ax.axvline(8.8, c="k", ls="-")
ax.text(5.0, 13, "1T Stabilizing", ha="right", va="bottom", rotation=0, fontsize=16)
ax.text(9.5, 13, "Plei Stabilizing", ha="left", va="bottom", rotation=0, fontsize=16)

fig.tight_layout()
fig.savefig("supp_SSD_effect_wf.pdf", bbox_inches="tight")

# Make boxplots to show the effect of ascertainment on the overall evidence for selection
fig, ax = plt.subplots(figsize=(8, 6))
boxwidth = 0.4

ax.boxplot(LLs_plei[("no_asc", "stab", "eq")], positions=cuts, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs_plei[("asc", "stab", "eq")], positions=cuts+0.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs_stab[("no_asc", "plei", "eq")], positions=cuts+9, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs_stab[("asc", "plei", "eq")], positions=cuts+9.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.plot([], c=no_asc_c, label="No ascertainment")
ax.plot([], c=asc_c, label="GWAS effect size")
ax.legend(loc="upper right", frameon=True, facecolor="white")

ax.set_title("Equilibrium population model", fontweight="bold")
ax.set_ylabel(r"Evidence for selection $(-\Delta \mathrm{AIC}_{\mathrm{Plei} - \mathrm{neut}})$", fontweight="bold")
ax.set_yticks([200, 400, 800])
ax.set_yticklabels([ 200, 400, 800 ])

ax.set_xticks(list(cuts+0.2) + [cut+9.2 for cut in cuts])
ax.set_xticklabels(list(int(cut) for cut in cuts) + list(int(cut) for cut in cuts))
ax.set_xlabel("Variance threshold", fontweight="bold")

ax.axvline(8.8, c="k", ls="-")
ax.text(5.0, 220, "1T Stabilizing", ha="right", va="bottom", rotation=0, fontsize=16)
ax.text(9.5, 700, "Plei Stabilizing", ha="left", va="bottom", rotation=0, fontsize=16)

fig.tight_layout()
fig.savefig("supp_asc_effect_eq.pdf", bbox_inches="tight")


# Make boxplots to show the effect of ascertainment on the overall evidence for selection
fig, ax = plt.subplots(figsize=(8, 6))
boxwidth = 0.4

ax.boxplot(LLs_plei[("no_asc", "stab", "WF")], positions=cuts, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs_plei[("asc", "stab", "WF")], positions=cuts+0.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.boxplot(LLs_stab[("no_asc", "plei", "WF")], positions=cuts+9, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
              boxprops=dict(facecolor=no_asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=no_asc_c, markeredgecolor=no_asc_c),
                medianprops=dict(color="black"))
ax.boxplot(LLs_stab[("asc", "plei", "WF")], positions=cuts+9.4, widths=boxwidth, showfliers=False, patch_artist=True, whis=[2.5, 97.5],
                boxprops=dict(facecolor=asc_c, color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color=asc_c, markeredgecolor=asc_c),
                medianprops=dict(color="black"))

ax.plot([], c=no_asc_c, label="No ascertainment")
ax.plot([], c=asc_c, label="GWAS effect size")
ax.legend(loc="upper right", frameon=True, facecolor="white")
ax.set_title("Non-equilibrium population model", fontweight="bold")


ax.set_ylabel(r"Evidence for selection $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{neut}})$", fontweight="bold")
ax.set_yticks([50, 100, 200, 400])
ax.set_yticklabels([50, 100, 200, 400])

ax.set_xticks(list(cuts+0.2) + [cut+9.2 for cut in cuts])
ax.set_xticklabels(list(int(cut) for cut in cuts) + list(int(cut) for cut in cuts))
ax.set_xlabel("Variance threshold", fontweight="bold")

ax.axvline(8.8, c="k", ls="-")
ax.text(5.0, 300, "1T Stabilizing", ha="right", va="bottom", rotation=0, fontsize=16)
ax.text(9.5, 300, "Plei Stabilizing", ha="left", va="bottom", rotation=0, fontsize=16)

fig.tight_layout()
fig.savefig("supp_asc_effect.pdf", bbox_inches="tight")




########################################
########################################
########################################
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(2*(stab_ml.ll_stab - stab_ml.ll_neut-1), 2*(ll_hd_stab-ll_1d_stab), label="1T", color=colors[0], alpha=0.75, s=ms)
loess = sm.nonparametric.lowess
x = 2*(stab_ml.ll_stab - stab_ml.ll_neut-1)
y = 2*(ll_hd_stab-ll_1d_stab)
z = loess(y, x, frac=0.6)
ax.plot(z[:, 0], z[:, 1], color=colors[0], linewidth=4)
ax.set_xscale("symlog", linthresh=2)
ax.set_xlim([-3, 500])
ax.set_xticks([-2, 0,  2, 10, 100])
ax.set_xticklabels([-2, 0, 2, 10, 100])
ax.set_xlabel("Variance threshold", fontweight="bold")

ax.scatter(2*(plei_ml.ll_plei - plei_ml.ll_neut-1), 2*(ll_hd_plei-ll_1d_plei), label="Plei", color=colors[1], alpha=0.75, s=ms)
loess = sm.nonparametric.lowess
x = 2*(plei_ml.ll_plei - plei_ml.ll_neut-1)
y = 2*(ll_hd_plei-ll_1d_plei)
z = loess(y, x, frac=0.6)
ax.plot(z[:, 0], z[:, 1], color=colors[1], linewidth=4)
ax.set_xlabel(r"Evidence for selection $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{neut}})$", fontweight="bold")
ax.set_ylabel(r"Simons variance dist $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$", fontweight="bold")
ax.set_yscale("symlog", linthresh=2)
ax.set_ylim([-5, 200])
ax.set_yticks([-2, 0, 2, 10, 100])
ax.set_yticklabels([-2, 0, 2, 10, 100])
ax.legend(fontsize=14, loc="lower left")

fig.tight_layout()
fig.savefig("IBD_PRIOR_STAB_PLEI_VARTEST.pdf", bbox_inches="tight")


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(2*(stab_ml.ll_stab - stab_ml.ll_neut-1), 2*(stab_ml.ll_plei - stab_ml.ll_stab), label="1T", color=colors[0], alpha=0.75, s=ms)
loess = sm.nonparametric.lowess
x = 2*(stab_ml.ll_stab - stab_ml.ll_neut-1)
y = 2*(stab_ml.ll_plei - stab_ml.ll_stab)
z = loess(y, x, frac=0.6)
ax.plot(z[:, 0], z[:, 1], color=colors[0], linewidth=4)
ax.set_xscale("symlog", linthresh=2)
ax.set_xlim([-3, 500])
ax.set_xticks([-2, 0,  2, 10, 100])
ax.set_xticklabels([-2, 0, 2, 10, 100])
ax.set_xlabel("Variance threshold", fontweight="bold")

ax.scatter(2*(plei_ml.ll_plei - plei_ml.ll_neut-1), 2*(plei_ml.ll_plei - plei_ml.ll_stab), label="Plei", color=colors[1], alpha=0.75, s=ms)
loess = sm.nonparametric.lowess
x = 2*(plei_ml.ll_plei - plei_ml.ll_neut-1)
y = 2*(plei_ml.ll_plei - plei_ml.ll_stab)
z = loess(y, x, frac=0.6)
ax.plot(z[:, 0], z[:, 1], color=colors[1], linewidth=4)
ax.set_xlabel(r"Evidence for selection $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{neut}})$", fontweight="bold")
ax.set_ylabel(r"Evidence for pleiotropy $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$", fontweight="bold")
ax.set_yscale("symlog", linthresh=1)
ax.set_ylim([-40, 40])
ax.set_yticks([-10, -2, 0, 2, 10])
ax.set_yticklabels([-10, -2, 0, 2, 10])
ax.legend(fontsize=14, loc="lower left")

fig.tight_layout()
fig.savefig("IBD_PRIOR_STAB_PLEI_new.pdf", bbox_inches="tight")
