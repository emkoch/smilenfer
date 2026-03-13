import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import smilenfer.plotting as splot
import smilenfer.statistics as sstats
import smilenfer.simulation as sim


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FIT_DIR = os.path.join(SCRIPT_DIR, "..", "all_opt_fits")
UKBB_FM_DIR = os.path.join(FIT_DIR, "ukbb_finemapping")

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

# If changing PANEL_SPECS manually, these are the current trait options by dataset.
# Keep the trait string exactly as it appears here.
#
# original traits:
# arthrosis = Arthrosis
# asthma = Asthma
# bc = BC
# bmi = BMI
# cad = CAD
# dbp = DBP
# diverticulitis = Diverticulitis
# fvc = FVC
# gallstones = Gallstones
# glaucoma = Glaucoma
# grip_strength = Grip strength
# hdl = HDL
# height = Height
# hypothyroidism = Hypothyroidism
# ibd = IBD
# ldl = LDL
# malignant_neoplasms = Malignant neoplasms
# pulse_rate = Pulse rate
# rbc = RBC
# sbp = SBP
# scz = SCZ
# t2d = T2D
# triglycerides = Triglycerides
# urate = Urate
# uterine_fibroids = Uterine fibroids
# varicose_veins = Varicose veins
# wbc = WBC
#
# bbj traits:
# asthma = Asthma
# bc = BC
# bmi = BMI
# cad = CAD
# dbp = DBP
# gallstones = Gallstones
# hdl = HDL
# height = Height
# ldl = LDL
# rbc = RBC
# sbp = SBP
# t2d = T2D
# triglycerides = Triglycerides
# uterine_fibroids = Uterine fibroids
#
# ukbb_susiex traits:
# bmi = BMI
# dbp = DBP
# hdl = HDL
# height = Height
# ldl = LDL
# sbp = SBP
# triglycerides = Triglycerides
# wbc = WBC
#
# mvp traits:
# Atrial fibrillation = AF
# Basal cell carcinoma = Basal cell carcinoma
# Cancer of prostate = Prostate cancer
# Coronary atherosclerosis = CAD
# Diverticulosis and diverticulitis = Diverticulitis
# Glaucoma = Glaucoma
# Gout = Gout
# Hyperlipidemia = Hyperlipidemia
# Hypertension = Hypertension
# Hypothyroidism = Hypothyroidism
# Type 2 diabetes = T2D
#

PANEL_SPECS = [
    {
        "dataset": "original",
        "trait": "cad",
        "label": "CAD",
        "fit_type": "post",
        "model": "plei",
        "xlabel": "Risk allele frequency",
        "ylabel": r"Effect size ($\mathrm{OR}-1$)",
    },
    {
        "dataset": "original",
        "trait": "scz",
        "label": "SCZ",
        "fit_type": "post",
        "model": "plei",
        "xlabel": "Risk allele frequency",
        "ylabel": r"Effect size ($\mathrm{OR}-1$)",
    },
    {
        "dataset": "mvp",
        "trait": "Type 2 diabetes",
        "label": "MVP T2D",
        "fit_type": "mean",
        "model": "plei",
        "xlabel": "Risk allele frequency",
        "ylabel": r"Effect size ($\mathrm{OR}-1$)",
    },
    {
        "dataset": "mvp",
        "trait": "Cancer of prostate",
        "label": "MVP Prostate cancer",
        "fit_type": "mean",
        "model": "plei",
        "xlabel": "Risk allele frequency",
        "ylabel": r"Effect size ($\mathrm{OR}-1$)",
    },
    {
        "dataset": "original",
        "trait": "bmi",
        "label": "BMI",
        "fit_type": "post",
        "model": "plei",
        "xlabel": "Trait-increasing allele frequency",
        "ylabel": r"Effect size ($\beta$)",
    },
    {
        "dataset": "bbj",
        "trait": "height",
        "label": "BBJ height",
        "fit_type": "high",
        "model": "plei",
        "xlabel": "Trait-increasing allele frequency",
        "ylabel": r"Effect size ($\beta$)",
    },
    {
        "dataset": "original",
        "trait": "sbp",
        "label": "Systolic BP",
        "fit_type": "mean",
        "model": "plei",
        "xlabel": "Trait-increasing allele frequency",
        "ylabel": r"Effect size ($\beta$)",
    },
    {
        "dataset": "ukbb_susiex",
        "trait": "ldl",
        "label": "UKBB SuSiE-X LDL",
        "fit_type": "mean",
        "model": "plei",
        "xlabel": "Trait-increasing allele frequency",
        "ylabel": r"Effect size ($\beta$)",
    },
    
]

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


def load_original_trait(trait):
    trait_path = os.path.join(DATA_DIR, "final", "original_traits", f"processed.{trait}.snps_low_r2.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df["median_n_eff"] = np.nanmedian(trait_df["n_eff"])
    trait_df["pval"] = 10 ** (-trait_df["neglog10p"])
    v_cut = stats.chi2.isf(ORIGINAL_P_THRESH, df=1) / trait_df["median_n_eff"].iloc[0]
    trait_df = trait_df[trait_df["var_exp"] > v_cut].copy()
    return trait_df, v_cut


def load_bbj_trait(trait, fit_type):
    trait_path = os.path.join(DATA_DIR, "final", "bbj_traits", f"processed.{trait}.max_r2.bbj.tsv")
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df["median_n_eff"] = np.nanmedian(trait_df["n_eff"])
    trait_df["pval"] = 10 ** (-trait_df["neglog10p"])
    v_cut = stats.chi2.isf(BBJ_P_THRESH, df=1) / trait_df["median_n_eff"].iloc[0]

    if fit_type == "high":
        trait_df = sstats.high_clump_trait_data(trait_df, dist=500000)
    elif fit_type == "pval":
        trait_df = sstats.pval_clump_trait_data(trait_df, dist=500000)
    else:
        raise ValueError(f"Unsupported BBJ fit type: {fit_type}")

    return trait_df, v_cut


def load_ukbb_susiex_trait(trait):
    susiex_path = os.path.join(DATA_DIR, "final", "UKBB_susiex", f"susiex_cs_table_{trait}.csv")
    original_path = os.path.join(DATA_DIR, "final", "original_traits", f"processed.{trait}.snps_low_r2.tsv")

    trait_df = pd.read_csv(susiex_path)
    original_df = pd.read_csv(original_path, sep="\t")

    n_eff_median = np.nanmedian(original_df["n_eff"])
    v_cut = stats.chi2.isf(UKBB_P_THRESH, df=1) / n_eff_median

    sampled_df = sample_finemap(trait_df)
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    sampled_df = sampled_df[sampled_df["var_exp"] > v_cut].copy()
    sampled_df = sampled_df[sampled_df["raf"].between(MIN_X, 1 - MIN_X)].copy()
    return sampled_df, v_cut


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
    trait_path = os.path.join(
        DATA_DIR,
        "final",
        "mvp_finemapping",
        f"{trait.replace(' ', '_')}_mvp_eur_finemapping.tsv",
    )
    trait_df = pd.read_csv(trait_path, sep="\t")
    trait_df["n_eff"] = 1 / (
        2 * trait_df["SE Population"] ** 2 * trait_df["EAF Population"] * (1 - trait_df["EAF Population"])
    )
    v_cut = stats.chi2.isf(MVP_P_THRESH, df=1) / np.nanmedian(trait_df["n_eff"])
    trait_df = trait_df[(trait_df["Description"] == trait) & (trait_df["Category"] == "PheCodes")].copy()

    sampled_df = sample_mvp_from_cs(trait_df, trait)
    sampled_df = sampled_df[sampled_df["EAF Population"].between(MIN_X, 1 - MIN_X)].copy()

    eaf = sampled_df["EAF Population"].to_numpy()
    beta = sampled_df["Beta Population"].to_numpy()
    raf, rbeta = to_risk(eaf, beta)
    sampled_df["raf"] = raf
    sampled_df["rbeta"] = rbeta
    sampled_df["var_exp"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
    sampled_df = sampled_df[sampled_df["var_exp"] > v_cut].copy()
    return sampled_df, v_cut


def get_fit_row_original(trait):
    # fit_path = os.path.join(FIT_DIR, "original_traits", "opt_results_original_traits_eur_post.csv")
    fit_path = os.path.join(SCRIPT_DIR, "..", "first_mode_fits", "original_traits", "opt_results_original_traits_eur_post.csv")
    fit_df = pd.read_csv(fit_path)
    return fit_df.loc[fit_df["trait"] == trait].iloc[0]


def get_fit_row_bbj(trait, fit_type):
    if fit_type == "high":
        # fit_path = os.path.join(FIT_DIR, "bbj", "opt_results_high_bbj.csv")
        fit_path = os.path.join(SCRIPT_DIR, "..", "first_mode_fits", "bbj", "opt_results_high_bbj.csv")
    elif fit_type == "pval":
        # fit_path = os.path.join(FIT_DIR, "bbj", "opt_results_pval_bbj.csv")
        fit_path = os.path.join(SCRIPT_DIR, "..", "first_mode_fits", "bbj", "opt_results_pval_bbj.csv")
    else:
        raise ValueError(f"Unsupported BBJ fit type: {fit_type}")
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
    if model == "dir":
        return {"Ne": N_E, "I1": float(fit_row["I1_dir"])}
    if model == "full":
        return {"Ne": N_E, "I1": float(fit_row["I1_full"]), "I2": float(fit_row["I2_full"])}
    raise ValueError(f"Unsupported model: {model}")


def load_trait_and_fit(panel_spec):
    dataset = panel_spec["dataset"]
    trait = panel_spec["trait"]
    fit_type = panel_spec["fit_type"]

    if dataset == "original":
        trait_df, v_cut = load_original_trait(trait)
        fit_row = get_fit_row_original(trait)
        beta_post = trait_df["PosteriorMean"].to_numpy()
    elif dataset == "bbj":
        trait_df, v_cut = load_bbj_trait(trait, fit_type)
        fit_row = get_fit_row_bbj(trait, fit_type)
        beta_post = trait_df["PosteriorMean"].to_numpy()
    elif dataset == "ukbb_susiex":
        trait_df, v_cut = load_ukbb_susiex_trait(trait)
        fit_row = get_fit_row_ukbb_susiex(trait)
        beta_post = None
    elif dataset == "mvp":
        trait_df, v_cut = load_mvp_trait(trait)
        fit_row = get_fit_row_mvp(trait)
        beta_post = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if trait_df.empty:
        raise ValueError(f"No loci remain after filtering for {dataset}:{trait}")

    return trait_df, v_cut, fit_row, beta_post


def plot_trait_panel(fig, ax, panel_spec, index):
    trait_df, v_cut, fit_row, beta_post = load_trait_and_fit(panel_spec)
    params = extract_params_for_plot(panel_spec["model"], fit_row)
    sfs_pile = get_sfs_pile(panel_spec["dataset"])

    _, _, _, _ = splot.plot_smile_fit(
        raf=trait_df["raf"].to_numpy(),
        beta_hat=trait_df["rbeta"].to_numpy(),
        beta_post=beta_post,
        v_cut=v_cut,
        model=panel_spec["model"],
        params=params,
        WF_pile=sfs_pile,
        hat_as_true=True,
        return_cbar=True,
        no_cbar=True,
        fig=fig,
        ax_1=ax,
        ylabel=panel_spec["ylabel"],
        xlabel=panel_spec["xlabel"],
    )

    ax.text(
        0.10,
        0.975,
        panel_spec["label"],
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    if index % 4 != 0:
        ax.set_ylabel("")


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
        cbar.ax.set_xlabel(r"median $_{ud}$", fontsize=26) 
    else:
        cbar.ax.set_xlabel(r"$S_{ud}$", fontsize=26)
    cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))


fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for ii, panel_spec in enumerate(PANEL_SPECS):
    print("trait:", panel_spec["dataset"], panel_spec["trait"])
    plot_trait_panel(fig, axes[ii], panel_spec, ii)

for jj in range(len(PANEL_SPECS), len(axes)):
    axes[jj].axis("off")

add_shared_colorbar(fig, PANEL_SPECS[0]["model"])
fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
fig.savefig(os.path.join(SCRIPT_DIR, "figure_4.pdf"), bbox_inches="tight")
