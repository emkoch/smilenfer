import os
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import smilenfer.posterior as post
import smilenfer.plotting as splot
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data"
ukbb_data_dir = os.path.join(data_dir, "final", "UKBB_susiex")
mvp_data_dir = os.path.join(data_dir, "final", "mvp_finemapping")

ukbb_susiex_traits = ["bmi", "dbp", "hdl", "height", "ldl", "sbp", "triglycerides", "wbc"]
ukbb_susiex_labels = ["BMI", "DBP", "HDL", "Height", "LDL", "SBP", "Triglycerides", "WBC"]

mvp_traits = [
    "Atrial fibrillation",
    "Basal cell carcinoma",
    "Cancer of prostate",
    "Coronary atherosclerosis",
    "Diverticulosis and diverticulitis",
    "Glaucoma",
    "Gout",
    "Hyperlipidemia",
    "Hypertension",
    "Hypothyroidism",
    "Type 2 diabetes",
]
mvp_labels = [
    "AF",
    "Basal cell carcinoma",
    "Prostate cancer",
    "CAD",
    "Diverticulitis",
    "Glaucoma",
    "Gout",
    "Hyperlipidemia",
    "Hypertension",
    "Hypothyroidism",
    "T2D",
]


def load_ukbb_susiex_trait_data():
    data_traits = {}
    for trait in ukbb_susiex_traits:
        data_traits[trait] = pd.read_csv(os.path.join(ukbb_data_dir, f"susiex_cs_table_{trait}.csv"))
    return data_traits


def load_mvp_trait_data():
    data_traits = {}
    for trait in mvp_traits:
        trait_data = pd.read_csv(
            os.path.join(mvp_data_dir, f"{trait.replace(' ', '_')}_mvp_eur_finemapping.tsv"),
            sep="\t",
        )
        trait_data = trait_data[(trait_data["Description"] == trait) & (trait_data["Category"] == "PheCodes")].copy()
        trait_data["raf"] = np.where(
            trait_data["Beta Population"] > 0,
            trait_data["EAF Population"],
            1 - trait_data["EAF Population"],
        )
        trait_data["se"] = trait_data["SE Population"]
        data_traits[trait] = trait_data
    return data_traits


def plot_se_grid(all_traits, all_labels, data_traits_all, plot_name):
    n_traits = len(all_traits)
    n_rows = math.ceil(n_traits/4)
    fig, ax = plt.subplots(n_rows, 4, figsize=(30, 4.5*n_rows))
    ax = ax.flatten()
    for i, trait in enumerate(all_traits):
        data = data_traits_all[trait]
        splot.plot_se_raf(data.raf, data.se, trait_name=all_labels[i], ax_given=ax[i])

    # Remove legend from all but top left plot
    for i in range(1, n_traits):
        ax[i].get_legend().remove()

    # remove empty axes
    for i in range(n_traits, len(ax)):
        fig.delaxes(ax[i])

    fig.tight_layout()
    fig.savefig(plot_name, bbox_inches='tight')


all_traits, all_labels, data_traits_all = post.original_trait_files()
plot_se_grid(all_traits, all_labels, data_traits_all, "all_traits_se.pdf")

bbj_traits, bbj_labels, bbj_data_traits = post.bbj_trait_files()
plot_se_grid(bbj_traits, bbj_labels, bbj_data_traits, "bbj_traits_se.pdf")

ukbb_susiex_data_traits = load_ukbb_susiex_trait_data()
plot_se_grid(ukbb_susiex_traits, ukbb_susiex_labels, ukbb_susiex_data_traits, "ukbb_susiex_traits_se.pdf")

mvp_data_traits = load_mvp_trait_data()
plot_se_grid(mvp_traits, mvp_labels, mvp_data_traits, "mvp_traits_se.pdf")
