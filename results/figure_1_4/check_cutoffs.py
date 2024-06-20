import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import smilenfer.plotting as splot
import smilenfer.prior as prior
from smilenfer.statistics import trad_x_set

splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})
trait = "cd"

data_dir = "../data"
fname_base = "clumped_ash.{}.ld_wind_cm_0.6.block_mhc.pval_5e-05.normal.maf_0.0001.tsv.gz"
fname = os.path.join(data_dir, fname_base.format(trait))

p_thresh = 5e-08
p_cutoff = 5e-08
min_x = 0.01

data_1, beta_hat_1, _ = prior.setup_data_cojo(fname, p_thresh=p_thresh, min_x=min_x,
                                              sep="\t", compression="gzip", beta_col="betahat",
                                              freq_col="topmed_af", pp="pval", p_cutoff=p_cutoff,
                                              alt_freq_col=None, var_inflation_cutoff=None)

v_cutoff_1 = prior.get_v_cutoff(fname, p_thresh=p_thresh, cojo=True, sep="\t", compression="gzip",
                                beta_col="betahat", freq_col="topmed_af", pp="pval",
                                alt_freq_col=None, var_inflation_cutoff=None)

print(v_cutoff_1)

data_2, beta_hat_2, _ = prior.setup_data_cojo(fname, p_thresh=p_thresh, min_x=min_x,
                                            sep="\t", compression="gzip", beta_col="betahat",
                                            freq_col="topmed_af", pp="pval", p_cutoff=p_cutoff,
                                            alt_freq_col="eaf", var_inflation_cutoff=None)

v_cutoff_2 = prior.get_v_cutoff(fname, p_thresh=p_thresh, cojo=True, sep="\t", compression="gzip",
                                beta_col="betahat", freq_col="topmed_af", pp="pval",
                                alt_freq_col="eaf", var_inflation_cutoff=None)

# plot data
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(data_1.raf, np.abs(data_1.betahat), marker="o", alpha=1, label="1")
ax.scatter(data_2.raf, np.abs(data_2.betahat), marker="+", alpha=1, label="2")
ax.legend()
x_set = trad_x_set(0.01, 2000)
beta_cut_1 = np.sqrt(v_cutoff_1 / (2*x_set*(1-x_set)))
beta_cut_2 = np.sqrt(v_cutoff_2 / (2*x_set*(1-x_set)))
ax.plot(x_set, beta_cut_1, linestyle="--")
ax.plot(x_set, beta_cut_2, linestyle="--")
ax.set_yscale("log")
ax.set_xscale("logit")
#save
fig.savefig("example_setup.pdf", bbox_inches="tight")

fig, ax = splot.sim_plot_truebeta(raf_true=None, 
                                  raf_sim=data_1.raf.to_numpy(),
                                  beta_hat=beta_hat_1,
                                  beta_post=None,
                                  model="stab",
                                  params={"Ne":10000,"I2":1e-08},
                                  incl_raf_true=False,
                                  title=trait,
                                  ylabel="",
                                  color_only=True,
                                  v_cut=v_cutoff_1)
# save as a pdf
fig.savefig("example_fitplot.pdf", bbox_inches="tight")