## Code for Figure 0
## Will be Figure 1 in the paper but added after discussion among authors
## Example smiles prior to any selection inference
import os
import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.stats as stats

import smilenfer.posterior as post
import smilenfer.plotting as splot
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data"
min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08

_, _, _, _, _, _, all_traits, all_labels, data_traits_all = splot.read_trait_files(os.path.join(data_dir, "clumped_ash"))
# Define a set of 4 traits/diseases to plot as an example in a main figure
traits_main_fig = ["scz", "bc", "sbp", "ldl"]
# get corresponding labels for the main figure traits
labels_main_fig = [all_labels[np.where(all_traits == trait)[0][0]] for trait in traits_main_fig]

def plot_basic_smiles(all_traits, all_labels, data_traits_update, min_x, p_thresh, p_cutoff, 
                      plot_name="basic_smiles_all.pdf", n_cols=4, offset = -0.01, col_size=10, row_size=6, labelsize=16):
    num_traits = len(all_traits)
    num_rows = math.ceil(num_traits / n_cols)

    fig, axes = plt.subplots(num_rows, n_cols, figsize=(col_size * n_cols, row_size * num_rows))
    for i, trait in enumerate(all_traits):
        trait_df = data_traits_update[trait].copy()
        ax = axes.flatten()[i]
        ax.set_xlim(-0.02, 1.02)
        x_set = np.arange(min_x, 1, min_x)
        v_cut = stats.chi2.ppf(q=1-p_thresh, df=1)/trait_df.median_n_eff[0]

        cut_rows = np.array(trait_df.var_exp > v_cut) & np.array(trait_df.maf >= min_x)
        cut_rows = cut_rows & np.array(trait_df.pval <= p_cutoff)

        # filter out the rows that don't meet the cutoff
        trait_df = trait_df[cut_rows]

        discov_betas = np.sqrt(v_cut/(2*x_set*(1-x_set)))
        beta_hat = trait_df.rbeta.to_numpy()
        ax.plot(np.concatenate(([min_x], x_set, [1-min_x])),
                  np.concatenate(([np.max(beta_hat)*1.25], discov_betas, [np.max(beta_hat)*1.25])),
                  color="darkslategrey", linestyle="dashed", linewidth=4)

        sns.scatterplot(x=trait_df.raf.to_numpy(), y=trait_df.rbeta.to_numpy(), data=trait_df, ax=ax, 
                        edgecolor="black", s=120, alpha=0.7)
        
        # make tick labels larger
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.tick_params(axis='both', which='minor', labelsize=labelsize)
        
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(0.2, 0.95, all_labels[i].replace("_", " "), transform=ax.transAxes, fontsize=30,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # make any unused axes invisible
    for i in range(num_traits, num_rows * n_cols):
        axes.flatten()[i].axis("off")

    # grand x axis label
    fig.text(0.5, offset, "Trait-increasing allele frequency", ha='center', va='center', fontsize=55)
    # grand y axis label
    fig.text(offset, 0.5, "Effect size", ha='center', va='center', rotation='vertical', fontsize=55)

    fig.tight_layout()
    fig.savefig(plot_name, bbox_inches="tight")
    fig.savefig(plot_name.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

plot_basic_smiles(all_traits, all_labels, data_traits_all, min_x, p_thresh, p_cutoff, 
                  plot_name="basic_smiles_all.pdf")

plot_basic_smiles(traits_main_fig, labels_main_fig, data_traits_all, min_x, p_thresh, p_cutoff, 
                  plot_name="figure_0_new.pdf", n_cols=2, col_size=8, row_size=4.5, labelsize=32)


## Plot smile fits
WF_path = "../data/WF_pile/"
WF_pile = {}
WF_pile["sfs_grid"] = np.load(os.path.join(WF_path, "SFS_pile.npy"))
WF_pile["interp_x"] = np.load(os.path.join(WF_path, "x_set.npy"))
WF_pile["s_set"]    = np.load(os.path.join(WF_path, "s_set.npy"))
WF_pile["s_ud_set"] = np.load(os.path.join(WF_path, "s_ud_set.npy"))
WF_pile["tenn_N"]   = np.load(os.path.join(WF_path, "tenn_N.npy"))

trait_fits = [trait.lower() for trait in all_traits] # ["scz"]

DATA_DIR = "../data/ML"
# ml_pcut_table_new = pd.read_csv(os.path.join(DATA_DIR, "main_traits_simple_no_filter/ML_all_flat_5e-08_new.csv"), sep=",")
# ml_pcut_table_new = ml_pcut_table_new.iloc[:, :-2]
# ml_pcut_table_diseases = pd.read_csv(os.path.join(DATA_DIR, "new_diseases_simple_no_filter/ML_all_flat_5e-08_new.csv"), sep=",")
# ml_all = pd.concat([ml_pcut_table_new, ml_pcut_table_diseases])

ml_all = pd.read_csv(os.path.join(DATA_DIR, "SIMPLE_ALL_TRAITS_NOFILTER_GENOMEWIDE/ML_all_flat_5e-08_new.csv"), sep=",")

trait_params = {}
for trait in trait_fits:
    trait_row = ml_all[(ml_all.trait == trait.upper()) & (ml_all.beta == "ash")]
    trait_params[trait] = {"Ne":10000, "I1": trait_row.I1_dir.values[0], "Ip":trait_row.Ip_plei.values[0], "I2":trait_row.I2_stab.values[0]}

n_traits = len(trait_fits)
n_rows = math.ceil(n_traits/4)
fig, ax = plt.subplots(n_rows, 4, figsize=(30, 4.5*n_rows))
ax = ax.flatten()

for ii, trait in enumerate(trait_fits):
    trait_df = data_traits_all[trait].copy()
    x_set = np.arange(min_x, 1, min_x)
    v_cut = stats.chi2.ppf(q=1-p_thresh, df=1)/trait_df.median_n_eff[0]

    cut_rows = np.array(trait_df.var_exp > v_cut) & np.array(trait_df.maf >= min_x)
    cut_rows = cut_rows & np.array(trait_df.pval <= p_cutoff)

    # filter out the rows that don't meet the cutoff
    trait_df = trait_df[cut_rows]
    _, _, _ = splot.plot_smile_fit(raf=trait_df.raf.to_numpy(), 
                                       beta_hat=trait_df.rbeta.to_numpy(),
                                       beta_post=trait_df.PosteriorMean.to_numpy(),
                                       v_cut=stats.chi2.ppf(q=1-p_thresh, df=1)/trait_df.median_n_eff[0],
                                       model="plei",
                                       params=trait_params[trait],
                                       WF_pile=WF_pile,
                                       fig=fig, ax_1=ax[ii], no_cbar=True)
    ax[ii].set_title(all_labels[ii].replace("_", " "))

# remove empty axes
for i in range(n_traits, len(ax)):
    fig.delaxes(ax[i])
fig.tight_layout()
fig.savefig("shrink_smiles_{}.pdf".format("all"), bbox_inches="tight")
