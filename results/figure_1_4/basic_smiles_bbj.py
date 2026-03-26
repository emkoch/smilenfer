## Code for Figure SX: Basic smiles plots for BBJ data

import matplotlib

import smilenfer.posterior as spost
import smilenfer.plotting as splot
import smilenfer.statistics as sstats

splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08

traits, labels, data_traits = spost.bbj_trait_files()

splot.plot_basic_smiles(
    traits,
    labels,
    data_traits,
    min_x,
    p_thresh,
    p_cutoff,
    plot_name="basic_smiles_bbj.pdf",
    loci_count=True,
)

reclumped_data_traits = {
    trait: sstats.pval_clump_trait_data(data_traits[trait], dist=500000)
    for trait in traits
}

splot.plot_basic_smiles(
    traits,
    labels,
    reclumped_data_traits,
    min_x,
    p_thresh,
    p_cutoff,
    plot_name="basic_smiles_bbj_reclumped.pdf",
    loci_count=True,
)

high_clumped_data_traits = {
    trait: sstats.high_clump_trait_data(data_traits[trait], dist=500000)
    for trait in traits
}

splot.plot_basic_smiles(
    traits,
    labels,
    high_clumped_data_traits,
    min_x,
    p_thresh,
    p_cutoff,
    plot_name="basic_smiles_bbj_highclumped.pdf",
    loci_count=True,
)
