## Code for Figure SX: Basic smiles plots for BBJ data

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
import smilenfer.statistics as sstats
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data"
min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08

traits, labels, data_traits = post.bbj_trait_files()

splot.plot_basic_smiles(traits, labels, data_traits, min_x, p_thresh, p_cutoff, 
                        plot_name="basic_smiles_bbj.pdf", loci_count=True)

# Create clumped version of data_traits
clumped_data_traits = {}
for trait in data_traits:
    clumped_data_traits[trait] = sstats.pval_clump_trait_data(data_traits[trait], dist=500000)

splot.plot_basic_smiles(traits, labels, clumped_data_traits, min_x, p_thresh, p_cutoff,
                    plot_name="basic_smiles_bbj_reclumped.pdf", loci_count=True)