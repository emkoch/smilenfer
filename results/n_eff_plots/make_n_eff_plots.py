import os
import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

import smilenfer.posterior as post
import smilenfer.plotting as splot
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data"
min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08
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

# traits_update = ["BMI", "BC", "HDL", "GRIP_STRENGTH", "FVC", "DBP", "CAD", 
#           "SBP", "RBC", "PULSE_RATE", "LDL", "IBD", "HEIGHT", "WBC", "URATE", 
#           "TRIGLYCERIDES", "T2D", "SCZ"]
# diseases_update =  ["ARTHROSIS", "ASTHMA", "DIVERTICULITIS", "GALLSTONES", "GLAUCOMA", "HYPOTHYROIDISM", 
#                     "MALIGNANT_NEOPLASMS", "UTERINE_FIBROIDS", "VARICOSE_VEINS"]
# # make all these names lowercase
# traits_update = [trait.lower() for trait in traits_update]
# diseases_update = [disease.lower() for disease in diseases_update]

# traits_update_labels = ["BMI", "Breast cancer", "HDL levels", "Grip strength", 
#                         "FVC", "Diastolic BP", "CAD", 
#                         "Systolic BP", "RBC", "Pulse rate", "LDL levels", "IBD", 
#                         "Standing height", "WBC", "Urate", 
#                         "Triglycerides", "Type 2 Diabetes", "SCZ"]

# diseases_update_labels = ["Arthrosis", "Asthma", "Diverticulitis", "Gallstones",
#                             "Glaucoma", "Hypothyroidism", "Malignant neoplasms", "Uterine fibroids",
#                             "Varicose veins"]

# # Sort the labels 
# trait_update_order = np.argsort(traits_update_labels)
# disease_update_order = np.argsort(diseases_update_labels)

# # Reorder the traits/diseases and then the labels
# traits_update = np.array(traits_update)[trait_update_order]
# traits_update_labels = np.array(traits_update_labels)[trait_update_order]
# diseases_update = np.array(diseases_update)[disease_update_order]
# diseases_update_labels = np.array(diseases_update_labels)[disease_update_order]

# # Create a merged list of traits and diseases and sort that as well
# all_traits = np.concatenate([traits_update, diseases_update])
# all_labels = np.concatenate([traits_update_labels, diseases_update_labels])
# all_order = np.argsort(all_labels)
# all_traits = all_traits[all_order]
# all_labels = all_labels[all_order]

# fname_trait = "clumped.{trait}.maf.5e-05.tsv.gz"
# fname_disease = "ash.{trait}.normal.block_mhc.finngen.tsv.gz"

# # Read in data for each trait
# data_traits_update = {trait: post.read_and_process_trait_data(os.path.join(data_dir, "clumped_ash", fname_trait.format(trait=trait))) 
#                       for trait in traits_update}
# data_diseases_update = {disease: post.read_and_process_trait_data(os.path.join(data_dir, "clumped_ash", 
#                                                                                fname_disease.format(trait=disease))) 
#                         for disease in diseases_update}

def calc_mlogp(beta, se):
    chi2 = (beta / se) ** 2
    return -stats.chi2.logsf(chi2, df=1) / np.log(10)


def load_ukbb_susiex_trait_data():
    data_traits = {}
    for trait in ukbb_susiex_traits:
        trait_data = pd.read_csv(os.path.join(ukbb_data_dir, f"susiex_cs_table_{trait}.csv"))
        trait_data["var_exp"] = 2 * trait_data.raf * (1 - trait_data.raf) * trait_data.rbeta ** 2
        trait_data["n_eff"] = (trait_data.rbeta / trait_data.se) ** 2 / trait_data["var_exp"]
        trait_data["mlogp"] = calc_mlogp(trait_data.rbeta, trait_data.se)
        data_traits[trait] = trait_data
    return data_traits


def load_mvp_trait_data():
    data_traits = {}
    for trait in mvp_traits:
        trait_data = pd.read_csv(
            os.path.join(mvp_data_dir, f"{trait.replace(' ', '_')}_mvp_eur_finemapping.tsv"),
            sep="\t",
        )
        trait_data = trait_data[(trait_data["Description"] == trait) & (trait_data["Category"] == "PheCodes")].copy()
        trait_data["var_exp"] = (
            2
            * trait_data["EAF Population"]
            * (1 - trait_data["EAF Population"])
            * trait_data["Beta Population"] ** 2
        )
        trait_data["n_eff"] = (trait_data["Beta Population"] / trait_data["SE Population"]) ** 2 / trait_data["var_exp"]
        trait_data["mlogp"] = calc_mlogp(trait_data["Beta Population"], trait_data["SE Population"])
        data_traits[trait] = trait_data
    return data_traits


def plot_neff_grid(all_traits, all_labels, data_traits_all, plot_name):
    n_traits = len(all_traits)
    n_rows = math.ceil(n_traits/4)
    fig, ax = plt.subplots(n_rows, 4, figsize=(30, 4.5*n_rows))
    ax = ax.flatten()
    for i, trait in enumerate(all_traits):
        data = data_traits_all[trait]
        mlogp = data.neglog10p if "neglog10p" in data.columns else data.mlogp
        splot.plot_local_neff(mlogp, data.n_eff, ax[i], trait_name = all_labels[i])
        ax[i].lines[0].set_color("darkslategrey")
        ax[i].lines[0].set_linewidth(2.5)
        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_alpha(0.9)
        ax[i].lines[0].set_zorder(5)
        ax[i].xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax[i].xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        ax[i].xaxis.set_minor_locator(mticker.NullLocator())
        ax[i].tick_params(axis="x", labelsize=14)
        finite_neff = np.asarray(data.n_eff)
        finite_neff = finite_neff[np.isfinite(finite_neff) & (finite_neff > 0)]
        gw_sig = np.asarray(mlogp) > -np.log10(5e-08)
        median_n_eff = np.median(np.asarray(data.n_eff)[gw_sig])
        y_min = min(np.min(finite_neff), median_n_eff)
        y_max = max(np.max(finite_neff), median_n_eff)
        y_pad = (y_max / y_min) ** 0.08
        ax[i].set_ylim(y_min / y_pad, y_max * y_pad)

    # remove empty axes
    for i in range(n_traits, len(ax)):
        fig.delaxes(ax[i])

    fig.tight_layout()
    fig.savefig(plot_name, bbox_inches='tight')


all_traits, all_labels, data_traits_all = post.original_trait_files()
plot_neff_grid(all_traits, all_labels, data_traits_all, "all_traits_neff.pdf")

bbj_traits, bbj_labels, bbj_data_traits = post.bbj_trait_files()
plot_neff_grid(bbj_traits, bbj_labels, bbj_data_traits, "bbj_traits_neff.pdf")

ukbb_susiex_data_traits = load_ukbb_susiex_trait_data()
plot_neff_grid(ukbb_susiex_traits, ukbb_susiex_labels, ukbb_susiex_data_traits, "ukbb_susiex_traits_neff.pdf")

mvp_data_traits = load_mvp_trait_data()
plot_neff_grid(mvp_traits, mvp_labels, mvp_data_traits, "mvp_traits_neff.pdf")
