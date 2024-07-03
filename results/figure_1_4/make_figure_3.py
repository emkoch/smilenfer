## Code for Figure 3
## Examle fits for disease traits
import os
import pickle   
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import smilenfer.plotting as splot
import smilenfer.prior as prior
import smilenfer.posterior as post

splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data"
fname_trait = "clumped.{trait}.maf.5e-05.tsv.gz"
fname_disease = "ash.{trait}.normal.block_mhc.finngen.tsv.gz"

all_traits_update = ["BMI", "BC", "HDL", "GRIP_STRENGTH", "FVC", "DBP", "CAD", 
          "SBP", "RBC", "PULSE_RATE", "LDL", "IBD", "HEIGHT", "WBC", "URATE", 
          "TRIGLYCERIDES", "SCZ"]
all_diseases_update =  ["ARTHROSIS", "ASTHMA", "DIVERTICULITIS", "GALLSTONES", "GLAUCOMA", "HYPOTHYROIDISM", 
                    "MALIGNANT_NEOPLASMS", "UTERINE_FIBROIDS", "VARICOSE_VEINS"]
# make all these names lowercase
all_traits_update = [trait.lower() for trait in all_traits_update]
all_diseases_update = [disease.lower() for disease in all_diseases_update]

traits = ["cad", "scz", "t2d", "arthrosis"]
trait_labels = ["CAD", "SCZ", "T2D", "Arthrosis"]
min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08

with open("../data/WF_pile/WF_pile_trunc.pkl", "rb") as f:
    WF_pile = pickle.load(f)

# Read in data for each trait
_, _, _, _, _, _, all_traits, all_labels, data = splot.read_trait_files(os.path.join(data_dir, "clumped_ash"), fname="clumped.{ash_type}.{trait}.max_r2.tsv")

# # Read in parameter estimates for each trait
ML = pd.read_csv(os.path.join(data_dir, "ML", 
                                "SIMPLE_ALL_TRAITS_NOFILTER_GENOMEWIDE",
                                "ML_all_flat_5e-08_new.csv"), sep=",")


# Filter to traits in trait_labels and beta=="ash"
ML = ML[ML["beta"]=="ash"]
sel_params = [float(ML.loc[ML.trait==trait.upper(),"I2_stab"]) for trait in traits]

fig, axes = plt.subplots(2, 4, figsize=(24, 9))
# Loop over traits
for i, trait in enumerate(traits):
    print("trait: ", trait)
    Ne = 10000
    trait_df = data[trait].copy()
    v_cut = stats.chi2.ppf(q=1-p_thresh, df=1)/trait_df.median_n_eff[0]
    cut_rows = np.array(trait_df.var_exp > v_cut) & np.array(trait_df.maf >= min_x)
    cut_rows = cut_rows & np.array(trait_df.pval <= p_cutoff)

    # filter out the rows that don't meet cutoffs
    trait_df = trait_df[cut_rows]

    _, _, cbar, _ = splot.plot_smile_fit(raf=trait_df.raf.to_numpy(), 
                               beta_hat=trait_df.rbeta.to_numpy(),
                                      beta_post=trait_df.PosteriorMean.to_numpy(),
                               v_cut=v_cut,
                               model="stab",
                               params={"Ne":Ne, "I2":float(sel_params[i])}, 
                               WF_pile=WF_pile,
                               ylabel=r"Effect size ($\mathrm{OR}-1$)",
                               hat_as_true=True,
                               return_cbar=True,
                               no_cbar=True,
                               fig=fig, ax_1=axes[0,i])
    
    axes[0,i].text(0.1, 0.975, trait_labels[i].replace("_", " "), 
                 transform=axes[0,i].transAxes, fontsize=18, fontweight='bold', va='top')
    if i > 0:
        axes[0,i].set_ylabel("")
    axes[0,i].set_xlabel(r"Risk allele frequency")
    axes[0,i].set_yticklabels(["{:.3f}".format(float(t.get_text())) for t in axes[0,i].get_yticklabels()])
    axes[0,i].set_xticks(np.arange(0, 1.2, 0.2))

splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

# Now make a similar figure for quantitative traits
traits = ["bmi", "height", "sbp", "ldl"]
trait_labels = ["BMI", "Standing height", "Systolic_BP", "LDL levels"]

sel_params = [float(ML.loc[ML.trait==trait.upper(),"I2_stab"]) for trait in traits]

for i, trait in enumerate(traits):
    print("trait: ", trait)
    Ne = 10000
    trait_df = data[trait].copy()
    v_cut = stats.chi2.ppf(q=1-p_thresh, df=1)/trait_df.median_n_eff[0]
    cut_rows = np.array(trait_df.var_exp > v_cut) & np.array(trait_df.maf >= min_x)
    cut_rows = cut_rows & np.array(trait_df.pval <= p_cutoff)

    # filter out the rows that don't meet cutoffs
    trait_df = trait_df[cut_rows]

    _, _, cbar, _ = splot.plot_smile_fit(raf=trait_df.raf.to_numpy(), 
                               beta_hat=trait_df.rbeta.to_numpy(),
                                      beta_post=trait_df.PosteriorMean.to_numpy(),
                               v_cut=v_cut,
                               model="stab",
                               params={"Ne":Ne, "I2":float(sel_params[i])}, 
                               WF_pile=WF_pile,
                               hat_as_true=True,
                               return_cbar=True,
                               no_cbar=True,
                               fig=fig, ax_1=axes[1,i])
    
    axes[1,i].text(0.1, 0.975, trait_labels[i].replace("_", " "), 
                 transform=axes[1,i].transAxes, fontsize=18, fontweight='bold', va='top')
    for j in range(len(axes[1,i].collections)):
         axes[1,i].collections[j].set_alpha(0.8)
    if i > 0:
        axes[1,i].set_ylabel("")
    axes[1,i].set_xlabel(r"Trait-increasing allele frequency")
    axes[1,i].set_yticklabels(["{:.3f}".format(float(t.get_text())) for t in axes[1,i].get_yticklabels()])
    axes[1,i].set_xticks(np.arange(0, 1.2, 0.2))

# Set up colorbar
cax = fig.add_axes([0.36, -0.05, 0.3, 0.025])
cc = sns.color_palette("Spectral", as_cmap=True)
cc = cc.reversed()
norm = plt.Normalize(-1, 2, clip=True)
sm = plt.cm.ScalarMappable(cmap=cc)
sm.set_array([])
sel_ticks = np.log10([0.1, 1.0, 10.0, 100.0])
cbar = cax.figure.colorbar(sm, orientation="horizontal",
                            ticks=norm(sel_ticks), cax=cax)
cbar.ax.set_xlabel(r"$S_{ud}$", fontsize=26)
cbar.ax.set_xticklabels(np.round(10**sel_ticks, 1))


plt.tight_layout()
# Save figure as a pdf
fig.savefig("figure_3.pdf", bbox_inches="tight")
# ALso save as a png
fig.savefig("figure_3.png", bbox_inches="tight", dpi=300)
