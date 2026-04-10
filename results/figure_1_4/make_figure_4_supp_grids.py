import os
import sys
import pickle
import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import smilenfer.plotting as splot
import smilenfer.posterior as spost
import smilenfer.statistics as sstats
import smilenfer.simulation as sim


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIT_DIR = os.path.join(RESULTS_DIR, "all_opt_fits")
UKBB_FM_DIR = os.path.join(FIT_DIR, "ukbb_finemapping")
UKBB_DATA_DIR = os.path.join(DATA_DIR, "final", "UKBB_susiex")
MVP_DATA_DIR = os.path.join(DATA_DIR, "final", "mvp_finemapping")

if UKBB_FM_DIR not in sys.path:
    sys.path.append(UKBB_FM_DIR)

from sample_finemapped import sample_finemap


splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

MIN_X = 0.01
ORIGINAL_P_THRESH = 5e-8
BBJ_P_THRESH = 5e-8
UKBB_P_THRESH = 5e-8
MVP_P_THRESH = 4.6e-11
N_E = 10000
N_COLS = 4
MODELS = ["plei", "stab", "full"]
SOURCES = ["original", "bbj", "ukbb_susiex", "mvp"]

_SFS_PILES = {}


def get_sfs_pile(dataset):
    if dataset == "bbj":
        pile_path = os.path.join(DATA_DIR, "SFS_pile", "joug_jpt_pile.pkl")
    else:
        pile_path = os.path.join(DATA_DIR, "SFS_pile", "tenn_eur_pile.pkl")

    if pile_path not in _SFS_PILES:
        with open(pile_path, "rb") as f:
            _SFS_PILES[pile_path] = sim.truncate_pile(pickle.load(f), 1e-8)
    return _SFS_PILES[pile_path]


def original_entries():
    traits, labels, _ = spost.original_trait_files()
    return [{"trait": trait, "label": label} for trait, label in zip(traits, labels)]


def bbj_entries():
    traits, labels, _ = spost.bbj_trait_files()
    return [{"trait": trait, "label": label} for trait, label in zip(traits, labels)]


def ukbb_susiex_entries():
    entries = []
    for path in sorted(os.listdir(UKBB_DATA_DIR)):
        if not path.startswith("susiex_cs_table_") or not path.endswith(".csv"):
            continue
        trait = path.replace("susiex_cs_table_", "").replace(".csv", "")
        label = spost.original_trait_names.get(trait, trait.replace("_", " "))
        entries.append({"trait": trait, "label": label})
    return entries


def mvp_entries():
    entries = []
    for path in sorted(os.listdir(MVP_DATA_DIR)):
        if not path.endswith("_mvp_eur_finemapping.tsv"):
            continue
        full_path = os.path.join(MVP_DATA_DIR, path)
        trait_df = pd.read_csv(full_path, sep="\t", usecols=["Description", "Category"])
        trait_df = trait_df[trait_df["Category"] == "PheCodes"]
        if trait_df.empty:
            continue
        label = trait_df["Description"].iloc[0]
        entries.append({"trait": label, "label": label})
    return entries


SOURCE_ENTRIES = {
    "original": original_entries(),
    "bbj": bbj_entries(),
    "ukbb_susiex": ukbb_susiex_entries(),
    "mvp": mvp_entries(),
}


def load_original_trait(trait):
    trait_path = os.path.join(DATA_DIR, "final", "original_traits", f"processed.{trait}.snps_low_r2.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df["median_n_eff"] = np.nanmedian(trait_df["n_eff"])
    trait_df["pval"] = 10 ** (-trait_df["neglog10p"])
    v_cut = stats.chi2.isf(ORIGINAL_P_THRESH, df=1) / trait_df["median_n_eff"].iloc[0]
    trait_df = trait_df[trait_df["var_exp"] > v_cut].copy()
    return trait_df, v_cut, trait_df["PosteriorMean"].to_numpy()


def load_bbj_trait(trait):
    trait_path = os.path.join(DATA_DIR, "final", "bbj_traits", f"processed.{trait}.max_r2.bbj.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df["median_n_eff"] = np.nanmedian(trait_df["n_eff"])
    trait_df["pval"] = 10 ** (-trait_df["neglog10p"])
    v_cut = stats.chi2.isf(BBJ_P_THRESH, df=1) / trait_df["median_n_eff"].iloc[0]
    trait_df = sstats.high_clump_trait_data(trait_df, dist=500000)
    return trait_df, v_cut, trait_df["PosteriorMean"].to_numpy()


def load_ukbb_susiex_trait(trait):
    susiex_path = os.path.join(UKBB_DATA_DIR, f"susiex_cs_table_{trait}.csv")
    original_path = os.path.join(DATA_DIR, "final", "original_traits", f"processed.{trait}.snps_low_r2.tsv")

    trait_df = pd.read_csv(susiex_path)
    original_df = pd.read_csv(original_path, sep="\t")

    n_eff_median = np.nanmedian(original_df["n_eff"])
    v_cut = stats.chi2.isf(UKBB_P_THRESH, df=1) / n_eff_median

    sampled_df = sample_finemap(trait_df)
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    sampled_df = sampled_df[sampled_df["var_exp"] > v_cut].copy()
    sampled_df = sampled_df[sampled_df["raf"].between(MIN_X, 1 - MIN_X)].copy()
    return sampled_df, v_cut, None


def sample_mvp_from_cs(trait_df, trait):
    sampled_rows = []
    for locus_cs in trait_df["Locus_CS"].dropna().unique():
        subset = trait_df[trait_df["Locus_CS"] == locus_cs].copy()
        sampled_rows.append(subset.sample(n=1, replace=True, weights=subset["CS-Level Pip"]))
    if not sampled_rows:
        return pd.DataFrame()
    return pd.concat(sampled_rows, ignore_index=True)


def to_risk(eaf, beta):
    return np.where(beta > 0, eaf, 1 - eaf), np.abs(beta)


def load_mvp_trait(trait):
    trait_path = os.path.join(MVP_DATA_DIR, f"{trait.replace(' ', '_')}_mvp_eur_finemapping.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df = trait_df[(trait_df["Description"] == trait) & (trait_df["Category"] == "PheCodes")].copy()
    trait_df["n_eff"] = 1 / (
        2 * trait_df["SE Population"] ** 2 * trait_df["EAF Population"] * (1 - trait_df["EAF Population"])
    )
    v_cut = stats.chi2.isf(MVP_P_THRESH, df=1) / np.nanmedian(trait_df["n_eff"])

    sampled_df = sample_mvp_from_cs(trait_df, trait)
    sampled_df = sampled_df[sampled_df["EAF Population"].between(MIN_X, 1 - MIN_X)].copy()

    eaf = sampled_df["EAF Population"].to_numpy()
    beta = sampled_df["Beta Population"].to_numpy()
    raf, rbeta = to_risk(eaf, beta)
    sampled_df["raf"] = raf
    sampled_df["rbeta"] = rbeta
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    sampled_df = sampled_df[sampled_df["var_exp"] > v_cut].copy()
    return sampled_df, v_cut, None


def get_fit_row_original(trait):
    fit_path = os.path.join(FIT_DIR, "original_traits", "opt_results_original_traits_eur_post.csv")
    fit_df = pd.read_csv(fit_path)
    return fit_df.loc[fit_df["trait"] == trait].iloc[0]


def get_fit_row_bbj(trait):
    fit_path = os.path.join(FIT_DIR, "bbj", "opt_results_high_bbj.csv")
    fit_df = pd.read_csv(fit_path)
    return fit_df.loc[fit_df["trait"] == trait].iloc[0]


def get_fit_row_ukbb_susiex(trait):
    fit_path = os.path.join(FIT_DIR, "ukbb_finemapping", "opt_results_ukbb_susiex.csv")
    fit_df = pd.read_csv(fit_path)
    fit_df = fit_df.loc[fit_df["trait"] == trait].copy()
    fit_row = fit_df.mean(numeric_only=True)
    fit_row["trait"] = trait
    return fit_row


def get_fit_row_mvp(trait):
    fit_path = os.path.join(FIT_DIR, "mvp", "opt_results_mvp_finemapping_eur.csv")
    fit_df = pd.read_csv(fit_path)
    fit_df = fit_df.loc[fit_df["trait"] == trait].copy()
    fit_row = fit_df.mean(numeric_only=True)
    fit_row["trait"] = trait
    return fit_row


def extract_params_for_plot(model, fit_row):
    if model == "plei":
        return {"Ne": N_E, "Ip": float(fit_row["Ip_plei"])}
    if model == "stab":
        return {"Ne": N_E, "I2": float(fit_row["I2_stab"])}
    if model == "full":
        return {"Ne": N_E, "I1": float(fit_row["I1_full"]), "I2": float(fit_row["I2_full"])}
    raise ValueError(f"Unsupported model: {model}")


def load_trait_and_fit(source, trait):
    if source == "original":
        trait_df, v_cut, beta_post = load_original_trait(trait)
        fit_row = get_fit_row_original(trait)
    elif source == "bbj":
        trait_df, v_cut, beta_post = load_bbj_trait(trait)
        fit_row = get_fit_row_bbj(trait)
    elif source == "ukbb_susiex":
        trait_df, v_cut, beta_post = load_ukbb_susiex_trait(trait)
        fit_row = get_fit_row_ukbb_susiex(trait)
    elif source == "mvp":
        trait_df, v_cut, beta_post = load_mvp_trait(trait)
        fit_row = get_fit_row_mvp(trait)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if trait_df.empty:
        raise ValueError(f"No loci remain after filtering for {source}:{trait}")

    return trait_df, v_cut, fit_row, beta_post


def ylabel_for_source(source):
    if source in ["original", "mvp"]:
        return r"Effect size ($\mathrm{OR}-1$)"
    return r"Effect size ($\beta$)"


def xlabel_for_source(source):
    if source in ["original", "mvp"]:
        return "Risk allele frequency"
    return "Trait-increasing allele frequency"


def add_shared_colorbar(fig, model):
    cax = fig.add_axes([0.36, 0.04, 0.3, 0.025])
    cc = sns.color_palette("Spectral", as_cmap=True)
    cc = cc.reversed()
    norm = plt.Normalize(-1, 2, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cc)
    sm.set_array([])
    sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
    cbar = cax.figure.colorbar(sm, orientation="horizontal", ticks=norm(sel_ticks), cax=cax)
    if model == "dir":
        cbar.ax.set_xlabel(r"$S_{dir}$", fontsize=26)
    elif model == "plei":
        cbar.ax.set_xlabel(r"Median $S_{ud}$", fontsize=26)
    else:
        cbar.ax.set_xlabel(r"$S_{ud}$", fontsize=26)
    cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))


def plot_source_model_grid(source, model):
    entries = SOURCE_ENTRIES[source]
    n_traits = len(entries)
    n_rows = math.ceil(n_traits / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(30, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ii, entry in enumerate(entries):
        print("plotting", source, model, entry["trait"])
        trait_df, v_cut, fit_row, beta_post = load_trait_and_fit(source, entry["trait"])
        params = extract_params_for_plot(model, fit_row)
        splot.plot_smile_fit(
            raf=trait_df["raf"].to_numpy(),
            beta_hat=trait_df["rbeta"].to_numpy(),
            beta_post=beta_post,
            v_cut=v_cut,
            model=model,
            params=params,
            WF_pile=get_sfs_pile(source),
            fig=fig,
            ax_1=axes[ii],
            hat_as_true=True,
            no_cbar=True,
            return_cbar=True,
            ylabel=ylabel_for_source(source),
            xlabel=xlabel_for_source(source),
        )
        axes[ii].set_title(entry["label"])
        axes[ii].set_xticks(np.arange(0, 1.2, 0.2))
        if ii % N_COLS != 0:
            axes[ii].set_ylabel("")

    for ii in range(n_traits, len(axes)):
        fig.delaxes(axes[ii])

    add_shared_colorbar(fig, model)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    output_path = os.path.join(SCRIPT_DIR, f"supp_smiles_{source}_{model}.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


for source in SOURCES:
    for model in MODELS:
        plot_source_model_grid(source, model)
