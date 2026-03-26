## Code for Figure 1

import matplotlib

import smilenfer.posterior as spost
import smilenfer.plotting as splot

splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08

original_traits, original_trait_labels, original_trait_data = spost.original_trait_files()
splot.plot_basic_smiles(
    original_traits,
    original_trait_labels,
    original_trait_data,
    min_x=min_x,
    p_thresh=p_thresh,
    p_cutoff=p_cutoff,
    loci_count=True,
    plot_name="basic_smiles_all.pdf",
)

traits_main_fig = ["scz", "t2d", "sbp", "ldl"]
labels_main_fig = ["SCZ", "T2D", "SBP", "LDL"]

splot.plot_basic_smiles(
    traits_main_fig,
    labels_main_fig,
    original_trait_data,
    min_x=min_x,
    p_thresh=p_thresh,
    p_cutoff=p_cutoff,
    n_cols=2,
    col_size=8,
    row_size=4.5,
    labelsize=32,
    loci_count=False,
    plot_name="figure_1.pdf",
)
