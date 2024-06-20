## Code for Figure 2
## 
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import chi2

import smilenfer.plotting as splot
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

def log10_t(x): return x / np.log(10)

# A modified version of the function from smilenfer.plotting for making publication-level figures
def plot_ML_table(ML_table, ML_table_lax, trait_groups, trait_group_labels, 
                  ss=100, fit="post", fit_lax=None, logy=False, pval=False, kill_offset=False):
    if fit_lax is None:
        fit_lax = fit
    ML_tmp = ML_table.copy()
    ML_tmp_lax = ML_table_lax.copy() if ML_table_lax is not None else ML_table.copy()
    group_sizes = [len(group) for group in trait_groups.values()]
    trait_cats = list(trait_groups.keys())
    n_traits = sum(group_sizes)
    trait_mult = 1.5

    fig, axes = plt.subplots(nrows=1, ncols=len(trait_groups), sharey=True,
                             figsize=(max(n_traits*trait_mult, 6), 6),
                             gridspec_kw={'width_ratios': group_sizes})

    if kill_offset:
        offset_small = 0
        offset_large = 0
    else:
        offset_small = 0.2/3
        offset_large = 0.2

    marker_dir = ">"
    marker_stab = "s"
    marker_full = "D"
    marker_plei = "o"

    c_dir = "#a65628"
    c_stab = "#984ea3"
    c_full = "#999999"
    c_plei = "#377eb8"

    if len(trait_groups) == 1:
        axes = [axes]

    # Adjust likelihoods
    for model in ["neut", "dir", "stab", "full", "plei"]:
        if model == "neut":
            adjust = 0
        elif model == "full":
            adjust = 2
        else:
            adjust = 1
        if pval:
            adjust = 0
        ML_tmp["ll_" + model] = -(2*adjust - 2*ML_tmp["ll_" + model].to_numpy())
        ML_tmp_lax["ll_" + model] = -(2*adjust  - 2*ML_tmp_lax["ll_" + model].to_numpy())

    # Compute model comparison statistics
    for model in ["dir", "stab", "full", "plei"]:
        if not pval:
            ML_tmp["stat_" + model] = ML_tmp["ll_" + model] - ML_tmp["ll_neut"]
            ML_tmp_lax["stat_" + model] = ML_tmp_lax["ll_" + model] - ML_tmp_lax["ll_neut"]
            # replace negative values with 0
            ML_tmp["stat_" + model] = ML_tmp["stat_" + model].clip(lower=0)
            ML_tmp_lax["stat_" + model] = ML_tmp_lax["stat_" + model].clip(lower=0)
        if pval:
            if model == "full":
                ML_tmp["stat_" + model] = chi2.logsf(ML_tmp["ll_" + model] - ML_tmp["ll_neut"], 2)
                ML_tmp_lax["stat_" + model] = chi2.logsf(ML_tmp_lax["ll_" + model] - ML_tmp_lax["ll_neut"], 2)
            else: 
                ML_tmp["stat_" + model] = chi2.logsf(ML_tmp["ll_" + model] - ML_tmp["ll_neut"], 1)
                ML_tmp_lax["stat_" + model] = chi2.logsf(ML_tmp_lax["ll_" + model] - ML_tmp_lax["ll_neut"], 1)
                # compute -log10 p-values in numerically stable way

            # Take -log10 of p-values
            ML_tmp["stat_" + model] = ML_tmp["stat_" + model] / -np.log(10)
            ML_tmp_lax["stat_" + model] = ML_tmp_lax["stat_" + model] / -np.log(10)

    all_ML = np.concatenate([ML_tmp["ll_" + model] - ML_tmp["ll_neut"] for model in ["dir", "stab", "full", "plei"]] + 
                            [ML_tmp_lax["ll_" + model] - ML_tmp_lax["ll_neut"] for model in ["dir", "stab", "full", "plei"]])
    y_max = np.max(all_ML)

    for ii, ax in enumerate(axes):
        ax.set_xlim([-0.5, group_sizes[ii]-0.5])
        ax.set_xticks(range(group_sizes[ii]))
        xticklabels = trait_groups[trait_cats[ii]]
        xticklabels = [xticklabel.replace("_", " ") for xticklabel in xticklabels]
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='x', labelrotation = 80)
        ax.set_title(trait_group_labels[ii])
        for jj, trait in enumerate(trait_groups[trait_cats[ii]]):
            ML_trait = ML_tmp.loc[np.logical_and(ML_tmp.trait==trait, ML_tmp.beta==fit)]
            ML_trait_lax = ML_tmp_lax.loc[np.logical_and(ML_tmp_lax.trait==trait, ML_tmp_lax.beta==fit_lax)]
            
            plot_both = (ML_trait.ll_full > ML_trait.ll_stab).any() or (ML_trait_lax.ll_full > ML_trait_lax.ll_stab).any()
            ax.scatter(jj-offset_large, ML_trait.stat_dir,
                       marker=marker_dir, c=c_dir, s=ss, label="directional")
            ax.scatter(jj-offset_small, ML_trait.stat_stab,
                       marker=marker_stab, c=c_stab, s=ss, label="1D Stabilizing")
            if plot_both:
                ax.scatter(jj+offset_small, ML_trait.stat_full,
                           marker=marker_full, c=c_full, s=ss, label="dir. + 1-D stab.")
            ax.scatter(jj+offset_large, ML_trait.stat_plei,
                       marker=marker_plei, c=c_plei, s=ss, label="hD pleiotropic")
            
            if ML_table_lax is not None:
                ax.scatter(jj-0.2, ML_trait_lax.stat_dir,
                            marker=marker_dir, c=c_dir, s=ss, alpha=0.35)
                ax.scatter(jj-0.2/3, ML_trait_lax.stat_stab,
                                marker=marker_stab, c=c_stab, s=ss, alpha=0.35)
                if plot_both:
                    ax.scatter(jj+0.2/3, ML_trait_lax.stat_full,
                            marker=marker_full, c=c_full, s=ss, alpha=0.35)
                ax.scatter(jj+0.2, ML_trait_lax.stat_plei,
                                marker=marker_plei, c=c_plei, s=ss, alpha=0.35)
    
    if not pval:
        if logy:
            ax.set_ylim(-0.2, y_max*1.2)
            ax.set_yscale("symlog", linthresh=4)
            ax.set_yticks([0, 2.] + list(5*2**np.arange(0, np.log2(y_max/5))))
            ax.set_yticklabels(["<0", 2.] + list(5*2**np.arange(0, np.log2(y_max/5))))
        else:
            y1, y2 = axes[0].get_ylim()
            if y1 > 0:
                y1 = -1/np.log(10)
            axes[0].set_ylim(y1, y2)
        axes[0].set_ylabel(r"$-\Delta \mathrm{AIC}_{\mathrm{model} - \mathrm{neut}}$")
    else:
        ax.set_yscale("symlog", linthresh=10)
        ax.set_yticks([0, -np.log10(0.05), 2.] + list(5*2**np.arange(0, np.log2(20/5))))
        ax.set_yticklabels([0, np.round(-np.log10(0.05), 2), 2.] + list(5*2**np.arange(0, np.log2(20/5))))

        axes[0].set_ylabel(r"$-\log_{10} p\mathrm{-value}$")


    axes[-1].legend(labels=["Directional", "1D Stabilizing", "Dir. + Stab.", "hD Stabilizing"],
                    loc="upper left", bbox_to_anchor=(1.0, 0.9), ncol=1)
    return fig, axes

# A modified version of the function from smilenfer.plotting for making publication-level figures
def plot_ML_table_2(ML_table, trait_groups, trait_group_labels, trait_names,
                    ss=100, fit="post", fit_lax=None, logy=False, pval=False, kill_offset=False, kill_full=False):
    matplotlib.rcParams.update({'figure.facecolor': 'white'})
    matplotlib.rcParams.update({'axes.facecolor': 'white'})
    
    if fit_lax is None:
        fit_lax = fit
    ML_tmp = ML_table.copy()
    group_sizes = [len(group) for group in trait_groups.values()]
    trait_cats = list(trait_groups.keys())
    n_traits = sum(group_sizes)
    trait_mult = 1.5

    # One axes for each trait and then two we'll keep blank
    fig, axes = plt.subplots(nrows=1, ncols=len(trait_groups) - 1 + n_traits, sharey=True,
                             figsize=(max(n_traits*trait_mult, 6), 6))

    if kill_offset:
        offset_small = 0
        offset_large = 0
    else:
        offset_small = 0.2/3
        offset_large = 0.2

    marker_dir = ">"
    marker_stab = "s"
    marker_full = "D"
    marker_plei = "o"

    c_dir = "#FFB000"  # Coral
    c_stab = "#DC267F"  # Lime Green
    c_full = "#785EF0"  # Sky Blue
    c_plei = "#FE6100"  # Mauve

    # Make the colors darker
    c_dir_dark = matplotlib.colors.to_rgb(c_dir)
    c_dir_dark = tuple([x*0.8 for x in c_dir_dark])
    c_stab_dark = matplotlib.colors.to_rgb(c_stab)
    c_stab_dark = tuple([x*0.8 for x in c_stab_dark])
    c_full_dark = matplotlib.colors.to_rgb(c_full)
    c_full_dark = tuple([x*0.8 for x in c_full_dark])
    c_plei_dark = matplotlib.colors.to_rgb(c_plei)
    c_plei_dark = tuple([x*0.8 for x in c_plei_dark])


    if len(trait_groups) == 1:
        print("Only one trait group")
        #axes = [axes]
        #print(axes, axes[0], axes[0][0])

    # Adjust likelihoods
    for model in ["neut", "dir", "stab", "full", "plei"]:
        if model == "neut":
            adjust = 0
        elif model == "full":
            adjust = 2
        else:
            adjust = 1
        if pval:
            adjust = 0
        ML_tmp["ll_" + model] = -(2*adjust - 2*ML_tmp["ll_" + model].to_numpy())

    # Compute model comparison statistics
    for model in ["dir", "stab", "full", "plei"]:
        if not pval:
            ML_tmp["stat_" + model] = ML_tmp["ll_" + model] - ML_tmp["ll_neut"]
            # replace negative values with 0
            ML_tmp["stat_" + model] = ML_tmp["stat_" + model].clip(lower=0)
        if pval:
            if model == "full":
                ML_tmp["stat_" + model] = chi2.logsf(ML_tmp["ll_" + model] - ML_tmp["ll_neut"], 2)
            else: 
                ML_tmp["stat_" + model] = chi2.logsf(ML_tmp["ll_" + model] - ML_tmp["ll_neut"], 1)
                # compute -log10 p-values in numerically stable way

            # Take -log10 of p-values
            ML_tmp["stat_" + model] = ML_tmp["stat_" + model] / -np.log(10)

    all_ML = np.concatenate([ML_tmp["ll_" + model] - ML_tmp["ll_neut"] for model in ["dir", "stab", "full", "plei"]])
    y_max = np.max(all_ML)

    if not pval:
        if logy:
            axes[0].set_ylim(-0.2, y_max*1.2)
            axes[0].set_yscale("symlog", linthresh=4)
            axes[0].set_yticks([0, 2.] + list(5*2**np.arange(0, np.log2(y_max/5))))
            axes[0].set_yticklabels(["<0", 2.] + list(5*2**np.arange(0, np.log2(y_max/5))))
        else:
            y1, y2 = axes[0].get_ylim()
            if y1 > 0:
                y1 = -1/np.log(10)
            axes[0].set_ylim(y1, y2)
        axes[0].set_ylabel(r"$-\Delta \mathrm{AIC}_{\mathrm{model} - \mathrm{neut}}$")
    else:
        axes[0].set_yscale("symlog", linthresh=10)
        axes[0].set_yticks([ 2.] + list(5*2**np.arange(0, np.log2(20/5))))
        axes[0].set_yticklabels([ 2.] + list(5*2**np.arange(0, np.log2(20/5))))

        axes[0].set_ylabel(r"$-\log_{10} \mathrm{p-value}$")


    ii = 0 # index for axes, there are n_traits + len(trait_groups) - 1 axes in total
    for trait_cat in trait_cats:
        trait_group = trait_groups[trait_cat]
        for jj, trait in enumerate(trait_group):
            ax = axes[ii]
            ax.set_xlim([-0.5, 0.5])
            ax.set_xticks([0])
            ax.set_xticklabels([trait_names[trait]])
            ax.tick_params(axis='x', labelrotation = 80)

            # Add zero and nominal p-value lines
            ax.axhline(y=0, linestyle="--", linewidth=1.5, color="slategrey", alpha=0.8, label="zero")
            ax.axhline(y=-np.log10(0.05), linestyle="--", linewidth=1.5, color="palevioletred", alpha=0.8, label="Nominal Sig.")
            ax.axhline(y=-np.log10(0.05 / len(trait_names)), linestyle="--", linewidth=1.5, color="darkorchid", alpha=0.8, label="Adjusted Sig.")
            ML_trait = ML_tmp.loc[np.logical_and(ML_tmp.trait==trait, ML_tmp.beta==fit)]
            dir_h = ax.scatter(offset_large, ML_trait.stat_dir,
                       marker=marker_dir, c=c_dir, s=ss, label="directional", 
                       edgecolors=c_dir_dark, linewidths=1.5)
            dir_h.set_facecolor(matplotlib.colors.to_rgb(c_dir) + (0.5,))
            stab_h = ax.scatter(offset_small, ML_trait.stat_stab,
                       marker=marker_stab, c=c_stab, s=ss, label="1D Stabilizing",
                       edgecolors=c_stab_dark, linewidths=1.5)
            stab_h.set_facecolor(matplotlib.colors.to_rgb(c_stab) + (0.5,))
            if not kill_full:
                full_h = ax.scatter(offset_small, ML_trait.stat_full,
                        marker=marker_full, c=c_full, s=ss, label="dir. + 1-D stab.", 
                        edgecolors=c_full_dark, linewidths=1.5)
                full_h.set_facecolor(matplotlib.colors.to_rgb(c_full) + (0.5,))
            plei_h = ax.scatter(offset_large, ML_trait.stat_plei,
                       marker=marker_plei, c=c_plei, s=ss, label="hD pleiotropic",
                       edgecolors=c_plei_dark, linewidths=1.5)
            plei_h.set_facecolor(matplotlib.colors.to_rgb(c_plei) + (0.5,))

            # remove vertical grid lines entirely
            ax.grid(axis="x", which="both", linestyle="--", linewidth=1, color="black", alpha=0.0)           
            ii += 1
        # skip a column
        if ii < len(axes):
            # delete the axis
            fig.delaxes(axes[ii])
            ii += 1

    # Add titles over the group of axes for each trait group (e.g. Anthropometric)
    # Make these such that they are centered over the group of axes
    kk = 0
    for ii, trait_cat in enumerate(trait_cats):
        trait_group = trait_groups[trait_cat]
        # Get the first and last axes for this trait group
        first_ax = axes[kk]
        last_ax = axes[kk + len(trait_group)-1]
        kk = kk + len(trait_group) + 1
        # Get the x positions of these axes
        first_x = first_ax.get_position().x0
        last_x = last_ax.get_position().x1
        # Get the y position of these axes
        y = first_ax.get_position().y1
        # Compute the x position of the title
        x = (first_x + last_x) / 2
        # Add the title
        fig.text(x, y, trait_group_labels[ii], ha="center", va="bottom", fontsize=22)

    if not kill_full:
        legend1 = axes[-1].legend(labels=["Directional", "Single-trait stabilizing", "Dir. + 1-T Stab.", "Pleiotropic stabilizing"],
                                handles=[dir_h, stab_h, full_h, plei_h],
                                loc="upper left", bbox_to_anchor=(1.0, 0.9), ncol=1, title="Selection model")
    else:
        legend1 = axes[-1].legend(labels=["Directional", "Single-trait Stabilizing", "Pleiotropic Stabilizing"],
                                handles=[dir_h, stab_h, plei_h],
                                loc="upper left", bbox_to_anchor=(1.0, 0.9), ncol=1, title="Selection model")
    _ = axes[-1].legend(labels=[r"Neutral model", r"Nominal significance", r"Adjusted significance"], 
                        loc="lower left",  bbox_to_anchor=(1, 0.1), ncol=1)
    axes[-1].add_artist(legend1)
    legend1.set_title("Selection model", prop = {'weight': 'bold'})
    return fig, axes

trait_names = {"BMI":"BMI", "BC":"Breast cancer", "HDL":"HDL levels", "GRIP_STRENGTH":"Grip strength", 
               "FVC":"FVC", "DBP":"Diastolic BP", "CAD":"CAD", 
          "SBP":"Systolic BP", "RBC":"RBC", "PULSE_RATE":"Pulse rate", "LDL":"LDL levels", "IBD":"IBD", 
          "HEIGHT":"Standing height", "WBC":"WBC", "URATE":"Urate", 
          "TRIGLYCERIDES":"Triglycerides", "SCZ":"SCZ", "T2D": "T2D"}
trait_groups = {"Quantitative": ["HEIGHT", "BMI", "LDL", "HDL", "DBP", "SBP", "TRIGLYCERIDES", "URATE", "RBC", 
                                 "WBC", "FVC", "GRIP_STRENGTH", "PULSE_RATE"],
                "Disease": ["BC", "CAD", "IBD", "SCZ", "T2D"]}
trait_group_labels = ["Quantitative", "Disease"]

DATA_DIR = "../data/ML"
ml_pcut_table_new = pd.read_csv(os.path.join(DATA_DIR, "SIMPLE_ALL_TRAITS_NOFILTER_GENOMEWIDE/ML_all_flat_5e-08_new.csv"), sep=",")

diseases = ["ARTHROSIS", "ASTHMA", "DIVERTICULITIS", "GALLSTONES", "GLAUCOMA", "HYPOTHYROIDISM", "MALIGNANT_NEOPLASMS", 
            "UTERINE_FIBROIDS", "VARICOSE_VEINS"]
disease_names = {"ARTHROSIS":"Arthrosis", "ASTHMA":"Asthma", "DIVERTICULITIS":"Diverticulitis", "GALLSTONES":"Gallstones", 
                 "GLAUCOMA":"Glaucoma", "HYPOTHYROIDISM":"Hypothyroidism", "MALIGNANT_NEOPLASMS":"Malignant neoplasms", 
                 "UTERINE_FIBROIDS":"Uterine fibroids", "VARICOSE_VEINS":"Varicose veins"}
disease_groups = {"Disease": diseases}
disease_group_labels = ["Disease"]

ml_pcut_table_diseases = pd.read_csv(os.path.join(DATA_DIR, "new_diseases_simple_no_filter/ML_all_flat_5e-08_new.csv"), sep=",")


# merge the ml_ tables because they have the same columns
ml_all = ml_pcut_table_new.copy()# pd.concat([ml_pcut_table_new, ml_pcut_table_diseases])

UKB_quant_trait_names = {"GRIP_STRENGTH":"Grip strength", "FVC":"FVC", "PULSE_RATE":"Pulse rate"}
all_new_trait_names = {**UKB_quant_trait_names, **disease_names}
all_new_trait_groups = {"Quantitative": list(UKB_quant_trait_names.keys()), "Disease": diseases}

all_new_traits = [key.upper() for key in all_new_trait_names.keys()]

ml_new_traits = ml_all[ml_all.trait.isin(all_new_traits)]
fig, axes = plot_ML_table_2(ml_new_traits, all_new_trait_groups, ["Quantitative", "Disease"], all_new_trait_names,
                            ss=100, fit="ash", logy=True, pval=True, kill_offset=True, kill_full=False)

fig.savefig("figure_2_follow_full.pdf", bbox_inches="tight")

# Now let's make the figure with the main traits and diseases only by removing the other traits from the dictionary
# remove UKB traits from trait_names
for key in UKB_quant_trait_names.keys():
    trait_names.pop(key)

trait_groups = {"Quantitative": ["HEIGHT", "BMI", "LDL", "HDL", "DBP", "SBP", "TRIGLYCERIDES", "URATE", "RBC", "WBC"],
                "Disease": ["BC", "CAD", "IBD", "SCZ", "T2D"]}
trait_group_labels = ["Quantitative", "Disease"]

fig, axes = plot_ML_table_2(ml_pcut_table_new, trait_groups, trait_group_labels, trait_names,
                            ss=100, fit="ash", logy=True, pval=True, kill_offset=True, kill_full=False)
fig.savefig("figure_2_initial_full.pdf", bbox_inches="tight")