import pandas as pd
import smilenfer.plotting as splot
import numpy as np
splot._plot_params()
import matplotlib.pyplot as plt

TRAITS = [
    "arthrosis",
    "asthma",
    "bc",
    "bmi",
    "cad",
    "dbp",
    "diverticulitis",
    "fvc",
    "gallstones",
    "glaucoma",
    "grip_strength",
    "hdl",
    "height",
    "hypothyroidism",
    "ibd",
    "ldl",
    "malignant_neoplasms",
    "pulse_rate",
    "rbc",
    "sbp",
    "scz",
    "t2d",
    "triglycerides",
    "urate",
    "uterine_fibroids",
    "varicose_veins",
    "wbc",  
]

TRAIT_NAMES = {
    "arthrosis": "Arthrosis",
    "asthma": "Asthma",
    "bc": "BC",
    "bmi": "BMI",
    "cad": "CAD",
    "dbp": "DBP",
    "diverticulitis": "Diverticulitis",
    "fvc": "FVC",
    "gallstones": "Gallstones",
    "glaucoma": "Glaucoma",
    "grip_strength": "Grip Strength",
    "hdl": "HDL",
    "height": "Height",
    "hypothyroidism": "Hypothyroidism",
    "ibd": "IBD",
    "ldl": "LDL",
    "malignant_neoplasms": "Malignant Neoplasms",
    "pulse_rate": "Pulse Rate",
    "rbc": "RBC",
    "sbp": "SBP",
    "scz": "SCZ",
    "t2d": "T2D",
    "triglycerides": "Triglycerides",
    "urate": "Urate",
    "uterine_fibroids": "Uterine Fibroids",
    "varicose_veins": "Varicose Veins",
    "wbc": "WBC",
}

df = pd.read_csv("stab_ud_std_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

axes[0].scatter(df["ll_I2_ud"] - df["ll_neut"], df["ll_I2_ud"] - df["ll_I2_std"])
for _, row in df.iterrows():
    axes[0].text(row["ll_I2_ud"] - row["ll_neut"],
                 row["ll_I2_ud"] - row["ll_I2_std"],
                 TRAIT_NAMES.get(row["Trait"], row["Trait"]),
                 fontsize=8, ha='right', va='bottom')
axes[0].axhline(0, ls="--", lw=1, color="black")
axes[0].axvline(0, ls="--", lw=1, color="black")
axes[0].set_xlabel(r"Support for selection: $\Delta$ log-likelihood (stab. (ud) $–$ neutral)", 
                   fontsize=10)
axes[0].set_ylabel(r"Support for underdominance:" + "\n" + r"$\Delta$ log-likelihood (stab. (ud) $–$ stab. (std))",
                    fontsize=10)
axes[0].set_xscale("symlog", linthresh=5)
axes[0].set_yscale("symlog", linthresh=5)
axes[0].set_title("1-Trait stabilizing", fontsize=10)
axes[0].tick_params(axis='both', which='both', labelsize=10)

axes[1].scatter(df["ll_Ip_ud"] - df["ll_neut"], df["ll_Ip_ud"] - df["ll_Ip_std"])
for _, row in df.iterrows():
    axes[1].text(row["ll_Ip_ud"] - row["ll_neut"],
                 row["ll_Ip_ud"] - row["ll_Ip_std"],
                 TRAIT_NAMES.get(row["Trait"], row["Trait"]),
                 fontsize=8, ha='right', va='bottom')
axes[1].axhline(0, ls="--", lw=1, color="black")
axes[1].axvline(0, ls="--", lw=1, color="black")
axes[1].set_xlabel(r"Support for selection: $\Delta$ log-likelihood (stab. (ud) $–$ neutral)", fontsize=10)
axes[1].set_xscale("symlog", linthresh=5)
axes[1].set_yscale("symlog", linthresh=5)
axes[1].set_title("Pleiotropic stabilizing", fontsize=10)
axes[1].tick_params(axis='both', which='both', labelsize=10)

# expand limits a bit
for ax in axes:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0]*1.1 if xlim[0]<0 else xlim[0]*0.9,
                xlim[1]*1.1 if xlim[1]>0 else xlim[1]*0.9)
    ax.set_ylim(ylim[0]*1.1 if ylim[0]<0 else ylim[0]*0.9,
                ylim[1]*1.1 if ylim[1]>0 else ylim[1]*0.9)

for ax in axes:
    # hide 10^0 labels (±1) so they don't crowd the 0
    for lbl, tick in zip(ax.get_xmajorticklabels(), ax.get_xticks()):
        if tick in (-1, 1):
            lbl.set_visible(False)
    for lbl, tick in zip(ax.get_ymajorticklabels(), ax.get_yticks()):
        if tick in (-1, 1):
            lbl.set_visible(False)

plt.tight_layout()
plt.savefig("stab_ud_std_vs_neut_panels.pdf", bbox_inches='tight')
