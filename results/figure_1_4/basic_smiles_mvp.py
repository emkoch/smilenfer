## Code for Figure SX: Basic smiles plots for MVP fine-mapping data

import os

import numpy as np
import pandas as pd
import matplotlib

import smilenfer.plotting as splot

MVP_DATA_DIR = "../data/final/mvp_finemapping"
MVP_TRAITS = [
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
MVP_LABELS = [
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


splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

min_x = 0.01
p_thresh = 4.6e-11
p_cutoff = 4.6e-11


def sample_mvp_from_cs(trait_df):
    sampled_rows = []
    for locus_cs in trait_df["Locus_CS"].dropna().unique():
        subset = trait_df[trait_df["Locus_CS"] == locus_cs].copy()
        sampled_rows.append(subset.sample(n=1, replace=True, weights=subset["CS-Level Pip"]))
    if not sampled_rows:
        return pd.DataFrame()
    return pd.concat(sampled_rows, ignore_index=True)


def to_risk(eaf, beta):
    return np.where(beta > 0, eaf, 1 - eaf), np.abs(beta)


def load_mvp_trait(label):
    trait_path = os.path.join(MVP_DATA_DIR, f"{label.replace(' ', '_')}_mvp_eur_finemapping.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df = trait_df[(trait_df["Description"] == label) & (trait_df["Category"] == "PheCodes")].copy()
    trait_df["n_eff"] = 1 / (
        2 * trait_df["SE Population"] ** 2 * trait_df["EAF Population"] * (1 - trait_df["EAF Population"])
    )
    n_eff_median = np.nanmedian(trait_df["n_eff"])

    sampled_df = sample_mvp_from_cs(trait_df)
    eaf = sampled_df["EAF Population"].to_numpy()
    beta = sampled_df["Beta Population"].to_numpy()
    raf, rbeta = to_risk(eaf, beta)

    sampled_df["raf"] = raf
    sampled_df["rbeta"] = rbeta
    sampled_df["median_n_eff"] = n_eff_median
    sampled_df["pval"] = 0.0
    sampled_df["maf"] = np.minimum(sampled_df["raf"], 1 - sampled_df["raf"])
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    return sampled_df

data_traits = {
    trait: load_mvp_trait(trait)
    for trait in MVP_TRAITS
}

splot.plot_basic_smiles(
    MVP_TRAITS,
    MVP_LABELS,
    data_traits,
    min_x,
    p_thresh,
    p_cutoff,
    plot_name="basic_smiles_mvp.pdf",
    loci_count=True,
)
