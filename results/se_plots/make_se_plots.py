import os
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import smilenfer.plotting as splot
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data"

_, _, _, _, _, _, all_traits, all_labels, data_traits_all = splot.read_trait_files(os.path.join(data_dir, "clumped_ash"))

n_traits = len(all_traits)
n_rows = math.ceil(n_traits/4)
fig, ax = plt.subplots(n_rows, 4, figsize=(30, 4.5*n_rows))
ax = ax.flatten()
for i, trait in enumerate(all_traits):
    data = data_traits_all[trait]
    splot.plot_se_raf(data.raf, data.se, trait_name = all_labels[i], ax_given=ax[i])

# Remove legend from all but top left plot
for i in range(1, n_traits):
    ax[i].get_legend().remove()

# remove empty axes
for i in range(n_traits, len(ax)):
    fig.delaxes(ax[i])

fig.tight_layout()
fig.savefig("all_traits_se.pdf", bbox_inches='tight')