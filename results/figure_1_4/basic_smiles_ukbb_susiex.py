## Code for Figure SX: Basic smiles plots for UKBB SuSiE-X data

import os

import numpy as np
import pandas as pd
import matplotlib

import smilenfer.plotting as splot

UKBB_DATA_DIR = "../data/final/UKBB_susiex"
ORIGINAL_DATA_DIR = "../data/final/original_traits"
UKBB_SUSIEX_TRAITS = [
    "bmi",
    "dbp",
    "hdl",
    "height",
    "ldl",
    "sbp",
    "triglycerides",
    "wbc",
]
UKBB_SUSIEX_LABELS = [
    "BMI",
    "DBP",
    "HDL",
    "Height",
    "LDL",
    "SBP",
    "Triglycerides",
    "WBC",
]


splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08


def sample_finemap(fm_df):
    fm_df = fm_df.copy()

    fm_df.loc[fm_df["locus"].isna(), "raf"] = fm_df["orig_raf"]
    fm_df.loc[fm_df["locus"].isna(), "rbeta"] = fm_df["orig_rbeta"]
    fm_df.loc[fm_df["locus"].isna(), "pip"] = 1

    n_missing = fm_df["locus"].isna().sum()
    if n_missing:
        start_id = int(fm_df["locus"].max(skipna=True)) + 1
        fm_df.loc[fm_df["locus"].isna(), "locus"] = np.arange(
            start_id,
            start_id + n_missing,
            dtype=int,
        )
        fm_df.loc[fm_df["locus"].isna(), "cs_id"] = 1

    fm_df = fm_df[["locus", "cs_id", "raf", "rbeta", "pip"]].copy()
    fm_df = fm_df.drop_duplicates()

    def normalize_pip(group):
        total_pip = group["pip"].sum()
        if total_pip > 0:
            group["pip"] = group["pip"] / total_pip
        else:
            group["pip"] = 1.0 / len(group)
        return group

    fm_df = fm_df.groupby(["locus", "cs_id"], group_keys=False).apply(normalize_pip)
    fm_df = fm_df.groupby(["locus", "cs_id"]).apply(
        lambda x: x.sample(n=1, weights=x["pip"])
    ).reset_index(drop=True)

    return fm_df


def load_ukbb_susiex_trait(trait):
    susiex_path = os.path.join(UKBB_DATA_DIR, f"susiex_cs_table_{trait}.csv")
    original_path = os.path.join(ORIGINAL_DATA_DIR, f"processed.{trait}.snps_low_r2.tsv")

    susiex_df = pd.read_csv(susiex_path)
    original_df = pd.read_csv(original_path, sep="\t")
    n_eff_median = np.nanmedian(original_df["n_eff"])

    sampled_df = sample_finemap(susiex_df)
    sampled_df["median_n_eff"] = n_eff_median
    sampled_df["pval"] = 0.0
    sampled_df["maf"] = np.minimum(sampled_df["raf"], 1 - sampled_df["raf"])
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    return sampled_df


data_traits = {
    trait: load_ukbb_susiex_trait(trait)
    for trait in UKBB_SUSIEX_TRAITS
}

splot.plot_basic_smiles(
    UKBB_SUSIEX_TRAITS,
    UKBB_SUSIEX_LABELS,
    data_traits,
    min_x,
    p_thresh,
    p_cutoff,
    plot_name="basic_smiles_ukbb_susiex.pdf",
    loci_count=True,
)
