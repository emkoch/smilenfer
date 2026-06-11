## Code for Figure SX: Full-model smile plots for traits with AIC(full) < AIC(stab)

import math
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import smilenfer.plotting as splot
import smilenfer.posterior as spost
import smilenfer.simulation as sim
import smilenfer.statistics as sstats

splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

data_dir = os.path.join("..", "data")
all_opt_fit_dir = os.path.join("..", "all_opt_fits")
ukbb_data_dir = os.path.join(data_dir, "final", "UKBB_susiex")
mvp_data_dir = os.path.join(data_dir, "final", "mvp_finemapping")

min_x = 0.01
original_p_thresh = 5e-8
bbj_p_thresh = 5e-8
ukbb_p_thresh = 5e-8
mvp_p_thresh = 4.6e-11
ne = 10000
n_cols = 4

sfs_piles = {}


def get_sfs_pile(dataset):
    if dataset == "bbj":
        pile_path = os.path.join(data_dir, "SFS_pile", "joug_jpt_pile.pkl")
    else:
        pile_path = os.path.join(data_dir, "SFS_pile", "tenn_eur_pile.pkl")

    if pile_path not in sfs_piles:
        with open(pile_path, "rb") as f:
            sfs_piles[pile_path] = sim.truncate_pile(pickle.load(f), 1e-8)
    return sfs_piles[pile_path]


def sample_finemap(fm_df):
    fm_df = fm_df.copy()

    fm_df.loc[fm_df["locus"].isna(), "raf"] = fm_df["orig_raf"]
    fm_df.loc[fm_df["locus"].isna(), "rbeta"] = fm_df["orig_rbeta"]
    fm_df.loc[fm_df["locus"].isna(), "pip"] = 1

    n_missing = fm_df["locus"].isna().sum()
    if n_missing:
        start_id = int(fm_df["locus"].max(skipna=True)) + 1
        fm_df.loc[fm_df["locus"].isna(), "locus"] = np.arange(start_id, start_id + n_missing, dtype=int)
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


def sample_mvp_from_cs(trait_df):
    sampled_rows = []
    for locus_cs in trait_df["Locus_CS"].dropna().unique():
        subset = trait_df[trait_df["Locus_CS"] == locus_cs].copy()
        sampled_rows.append(subset.sample(n=1, replace=True, weights=subset["CS-Level Pip"]))
    if not sampled_rows:
        return pd.DataFrame()
    return pd.concat(sampled_rows, ignore_index=True)


def load_original_trait(trait):
    trait_path = os.path.join(data_dir, "final", "original_traits", f"processed.{trait}.snps_low_r2.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df["median_n_eff"] = np.nanmedian(trait_df["n_eff"])
    trait_df["pval"] = 10 ** (-trait_df["neglog10p"])
    v_cut = stats.chi2.isf(original_p_thresh, df=1) / trait_df["median_n_eff"].iloc[0]
    trait_df = trait_df[trait_df["var_exp"] > v_cut].copy()
    return trait_df, v_cut, trait_df["PosteriorMean"].to_numpy()


def load_bbj_trait(trait):
    trait_path = os.path.join(data_dir, "final", "bbj_traits", f"processed.{trait}.max_r2.bbj.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df["median_n_eff"] = np.nanmedian(trait_df["n_eff"])
    trait_df["pval"] = 10 ** (-trait_df["neglog10p"])
    v_cut = stats.chi2.isf(bbj_p_thresh, df=1) / trait_df["median_n_eff"].iloc[0]
    trait_df = sstats.high_clump_trait_data(trait_df, dist=500000)
    return trait_df, v_cut, trait_df["PosteriorMean"].to_numpy()


def load_ukbb_susiex_trait(trait):
    susiex_path = os.path.join(ukbb_data_dir, f"susiex_cs_table_{trait}.csv")
    original_path = os.path.join(data_dir, "final", "original_traits", f"processed.{trait}.snps_low_r2.tsv")

    trait_df = pd.read_csv(susiex_path)
    original_df = pd.read_csv(original_path, sep="\t")
    n_eff_median = np.nanmedian(original_df["n_eff"])
    v_cut = stats.chi2.isf(ukbb_p_thresh, df=1) / n_eff_median

    sampled_df = sample_finemap(trait_df)
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    sampled_df = sampled_df[sampled_df["var_exp"] > v_cut].copy()
    sampled_df = sampled_df[sampled_df["raf"].between(min_x, 1 - min_x)].copy()
    return sampled_df, v_cut, None


def load_mvp_trait(trait):
    trait_path = os.path.join(mvp_data_dir, f"{trait.replace(' ', '_')}_mvp_eur_finemapping.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df = trait_df[(trait_df["Description"] == trait) & (trait_df["Category"] == "PheCodes")].copy()
    trait_df["n_eff"] = 1 / (
        2 * trait_df["SE Population"] ** 2 * trait_df["EAF Population"] * (1 - trait_df["EAF Population"])
    )
    v_cut = stats.chi2.isf(mvp_p_thresh, df=1) / np.nanmedian(trait_df["n_eff"])

    sampled_df = sample_mvp_from_cs(trait_df)
    sampled_df = sampled_df[sampled_df["EAF Population"].between(min_x, 1 - min_x)].copy()

    eaf = sampled_df["EAF Population"].to_numpy()
    beta = sampled_df["Beta Population"].to_numpy()
    sampled_df["raf"] = np.where(beta > 0, eaf, 1 - eaf)
    sampled_df["rbeta"] = np.abs(beta)
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    sampled_df = sampled_df[sampled_df["var_exp"] > v_cut].copy()
    return sampled_df, v_cut, None


entries = []

fit_df = pd.read_csv(os.path.join(all_opt_fit_dir, "original_traits", "opt_results_original_traits_eur_post.csv"))
fit_df = fit_df[(fit_df["ll_full"] - fit_df["ll_stab"]) > 1].copy()
for _, row in fit_df.iterrows():
    entries.append(
        {
            "dataset": "original",
            "trait": row["trait"],
            "label": spost.original_trait_names[row["trait"]],
            "fit_row": row,
        }
    )

fit_df = pd.read_csv(os.path.join(all_opt_fit_dir, "bbj", "opt_results_high_bbj.csv"))
fit_df = fit_df[(fit_df["ll_full"] - fit_df["ll_stab"]) > 1].copy()
for _, row in fit_df.iterrows():
    entries.append(
        {
            "dataset": "bbj",
            "trait": row["trait"],
            "label": "BBJ " + spost.bbj_trait_names[row["trait"]],
            "fit_row": row,
        }
    )

fit_df = pd.read_csv(os.path.join(all_opt_fit_dir, "ukbb_finemapping", "opt_results_ukbb_susiex.csv"))
fit_df = fit_df.groupby("trait", as_index=False).mean(numeric_only=True)
fit_df = fit_df[(fit_df["ll_full"] - fit_df["ll_stab"]) > 1].copy()
for _, row in fit_df.iterrows():
    label = spost.original_trait_names.get(row["trait"], row["trait"].replace("_", " "))
    entries.append(
        {
            "dataset": "ukbb_susiex",
            "trait": row["trait"],
            "label": "SuSiE-X " + label,
            "fit_row": row,
        }
    )

fit_df = pd.read_csv(os.path.join(all_opt_fit_dir, "mvp", "opt_results_mvp_finemapping_eur.csv"))
fit_df = fit_df.groupby("trait", as_index=False).mean(numeric_only=True)
fit_df = fit_df[(fit_df["ll_full"] - fit_df["ll_stab"]) > 1].copy()
short_labels = {
    "Atrial fibrillation": "AF",
    "Basal cell carcinoma": "Basal cell carcinoma",
    "Cancer of prostate": "Prostate cancer",
    "Coronary atherosclerosis": "CAD",
    "Diverticulosis and diverticulitis": "Diverticulitis",
    "Glaucoma": "Glaucoma",
    "Gout": "Gout",
    "Hyperlipidemia": "Hyperlipidemia",
    "Hypertension": "Hypertension",
    "Hypothyroidism": "Hypothyroidism",
    "Type 2 diabetes": "T2D",
}
for _, row in fit_df.iterrows():
    entries.append(
        {
            "dataset": "mvp",
            "trait": row["trait"],
            "label": "MVP " + short_labels[row["trait"]],
            "fit_row": row,
        }
    )

n_traits = len(entries)
n_rows = math.ceil(n_traits / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 4.5 * n_rows))
axes = np.atleast_1d(axes).flatten()

for ii, entry in enumerate(entries):
    dataset = entry["dataset"]
    trait = entry["trait"]
    fit_row = entry["fit_row"]

    if dataset == "original":
        trait_df, v_cut, beta_post = load_original_trait(trait)
        ylabel = r"Effect size ($\mathrm{OR}-1$)"
        xlabel = "Risk allele frequency"
    elif dataset == "bbj":
        trait_df, v_cut, beta_post = load_bbj_trait(trait)
        ylabel = r"Effect size ($\beta$)"
        xlabel = "Trait-increasing allele frequency"
    elif dataset == "ukbb_susiex":
        trait_df, v_cut, beta_post = load_ukbb_susiex_trait(trait)
        ylabel = r"Effect size ($\beta$)"
        xlabel = "Trait-increasing allele frequency"
    elif dataset == "mvp":
        trait_df, v_cut, beta_post = load_mvp_trait(trait)
        ylabel = r"Effect size ($\mathrm{OR}-1$)"
        xlabel = "Risk allele frequency"

    splot.plot_smile_fit(
        raf=trait_df["raf"].to_numpy(),
        beta_hat=trait_df["rbeta"].to_numpy(),
        beta_post=beta_post,
        v_cut=v_cut,
        model="full",
        params={"Ne": ne, "I1": float(fit_row["I1_full"]), "I2": float(fit_row["I2_full"])},
        WF_pile=get_sfs_pile(entry["dataset"]),
        fig=fig,
        ax_1=axes[ii],
        hat_as_true=True,
        no_cbar=True,
        return_cbar=True,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    axes[ii].set_title(entry["label"])
    axes[ii].set_xticks(np.arange(0, 1.2, 0.2))
    if ii % n_cols != 0:
        axes[ii].set_ylabel("")

for ii in range(n_traits, len(axes)):
    fig.delaxes(axes[ii])

cax = fig.add_axes([0.36, 0.04, 0.3, 0.025])
cc = sns.color_palette("Spectral", as_cmap=True)
cc = cc.reversed()
norm = plt.Normalize(-1, 2, clip=True)
sm = plt.cm.ScalarMappable(cmap=cc)
sm.set_array([])
sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
cbar = cax.figure.colorbar(sm, orientation="horizontal", ticks=norm(sel_ticks), cax=cax)
cbar.ax.set_xlabel(r"$S_{ud}$", fontsize=26)
cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))

fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
fig.savefig("full_smiles_selected_all_datasets.pdf", bbox_inches="tight")
