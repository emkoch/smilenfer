import smilenfer.var_dist as vd
from smilenfer.statistics import trad_x_set
import smilenfer.plotting as splot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from os.path import join

splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

def log10_t(x): return x / np.log(10)

# calculated using old file "ibd.5e-5.cojo.normal.no_mhc.tsv", 
# just used to calibrate sims
V_CUTOFF = 0.005246 
x_set = trad_x_set(0.01, 2000)
beta_cut = np.sqrt(V_CUTOFF / (2*x_set*(1-x_set)))

# Load a simulation
no_asc_dir = "no_asc_sims"
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

# Load BF_tables
stab_bf = pd.read_csv(join("../figure_1", "BF_table_IBD_stab_1e-08_nsamp_200_noasc.tsv"), sep="\t")
plei_bf = pd.read_csv(join("../figure_1", "BF_table_IBD_plei_1e-08_nsamp_200_noasc.tsv"), sep="\t")

# Plot the example simulation
fig, ax = plt.subplots()
ax.plot(d_test.raf, d_test.beta, "k.")
ax.plot(x_set, beta_cut, "r--")
fig.savefig("var_dist_sim_example.pdf", bbox_inches="tight")

# Compute the variance distribution for the example simulation
vd_test = 2*d_test.beta**2 * d_test.raf * (1-d_test.raf)
fig, ax = plt.subplots()
ax.hist(vd_test, bins=100, density=True)
ax.plot([V_CUTOFF, V_CUTOFF], [0, 200], "r--")
fig.savefig("var_dist_sim_example_hist.pdf", bbox_inches="tight")

eta_set = np.linspace(1e-4, 100, 10000)
z_test = (vd_test-V_CUTOFF)/V_CUTOFF

# Fit the variance distribution for the test example
eta_1d = vd.fit_stab_1D(z_test)
eta_hd = vd.fit_stab_hD(z_test)

# Fit the variance distribution for the entire set of simulations
eta_1d_stab = [vd.fit_stab_1D(dd.zz) for dd in stab_sims]
eta_hd_stab = [vd.fit_stab_hD(dd.zz) for dd in stab_sims]
eta_1d_plei = [vd.fit_stab_1D(dd.zz) for dd in plei_sims]
eta_hd_plei = [vd.fit_stab_hD(dd.zz) for dd in plei_sims]

# Calculate the log-likelihood for the set of simulations
ll_1d_stab = np.array([np.sum(vd.ll_stab_1D(dd.zz, eta_1d_stab[ii])) 
                       for ii, dd in enumerate(stab_sims)])
ll_hd_stab = np.array([np.sum(vd.ll_stab_hD(dd.zz, eta_hd_stab[ii])) 
                       for ii, dd in enumerate(stab_sims)])
ll_1d_plei = np.array([np.sum(vd.ll_stab_1D(dd.zz, eta_1d_plei[ii])) 
                       for ii, dd in enumerate(plei_sims)])
ll_hd_plei = np.array([np.sum(vd.ll_stab_hD(dd.zz, eta_hd_plei[ii])) 
                       for ii, dd in enumerate(plei_sims)])

# Plot the log-likelihood for the set of simulations
fig, axes = plt.subplots(1,2,figsize=(12,6), sharey=True)
axes[0].plot(stab_bf.I2, ll_hd_stab-ll_1d_stab, "k.", label="1-D")
axes[0].set_xscale("log")
axes[0].set_xlabel(r"$I_2$")
axes[0].set_ylabel(r"$\ell_{hD} - \ell_{1D}$")

axes[1].plot(plei_bf.Ip, ll_hd_plei-ll_1d_plei, "k.", label="h-D")
axes[1].set_xscale("log")
axes[1].set_xlabel(r"$I_p$")
axes[1].set_ylabel(r"$\ell_{hD} - \ell_{1D}$")
fig.savefig("var_dist_sims_ll.pdf", bbox_inches="tight")

fig, axes = plt.subplots(1,2,figsize=(12,6), sharey=True)
axes[0].plot(log10_t(stab_bf.BF_stab - stab_bf.BF_neut), ll_hd_stab-ll_1d_stab, "k.", label="1-D")
axes[0].set_xlabel("evidence for selection")
axes[0].set_ylabel(r"$\ell_{hD} - \ell_{1D}$")
axes[0].set_xscale("symlog", linthresh=2)

axes[1].plot(log10_t(plei_bf.BF_plei - plei_bf.BF_neut), ll_hd_plei-ll_1d_plei, "k.", label="h-D")
axes[1].set_xlabel("evidence for selection")
axes[1].set_ylabel(r"$\ell_{hD} - \ell_{1D}$")
axes[1].set_xscale("symlog", linthresh=2)
fig.savefig("var_dist_sims_ll_alt.pdf", bbox_inches="tight")

z_set = np.linspace(1e-3, 25, 10000)
f_z_1d = vd.f_stab_1D(z_set, eta_1d)
f_z_hd = vd.f_stab_hD(z_set, eta_hd)
fig, ax = plt.subplots()
ax.hist(z_test, bins=100, density=True)
ax.plot(z_set, f_z_1d, "k-", label="1-D")
ax.plot(z_set, f_z_hd, "r-", label="h-D")
ax.legend()
fig.savefig("var_dist_sim_example_hist_fit.pdf", bbox_inches="tight")

# Calculate the emprical CDF of vd_test_norm
z_test_sort = np.sort(z_test)
z_test_ecdf = np.arange(1, len(z_test_sort)+1) / len(z_test_sort)

# Calculate the theoretical CDF of vd_test_norm
F_z_1d = vd.F_stab_1D(z_set, eta_1d)
F_z_hd = vd.F_stab_hD(z_set, eta_hd)

# Plot empirical vs theoretical CDF
fig, ax = plt.subplots()
ax.plot(z_test_sort, z_test_ecdf, "o", label="Neutral sim", alpha=0.5)
ax.plot(z_set, F_z_1d, "k-", label="1-D fit")
ax.plot(z_set, F_z_hd, "r-", label="h-D fit")
ax.set_xscale("log")
ax.set_xlabel("Normalized variance contribution")
ax.set_ylabel("F")
ax.legend()
fig.savefig("var_dist_sim_example_ecdf.pdf", bbox_inches="tight")