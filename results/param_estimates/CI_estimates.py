import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smilenfer.plotting as splot
splot._plot_params()

fit_dir = "../all_opt_fits/original_traits/"
fits_hat   = pd.read_csv(os.path.join(fit_dir, "opt_results_original_traits_eur_raw.csv"))
fits_post  = pd.read_csv(os.path.join(fit_dir, "opt_results_original_traits_eur_post.csv"))
df_ci      = pd.read_csv(os.path.join(fit_dir, "opt_fits_original_traits_eur_raw_ci.csv"))
df_ci_post = pd.read_csv(os.path.join(fit_dir, "opt_fits_original_traits_eur_post_ci.csv"))

trait_groups = {
    "Quantitative": ["height","bmi","ldl","hdl","dbp","sbp","triglycerides","urate",
                     "rbc","wbc","grip_strength","fvc","pulse_rate"],
    "Disease": ["bc","cad","ibd","scz","t2d","arthrosis","asthma","diverticulitis",
                "gallstones","glaucoma","hypothyroidism","malignant_neoplasms",
                "uterine_fibroids","varicose_veins"]
}
trait_names = {
    "height":"Height","bmi":"Body Mass Index","ldl":"LDL Cholesterol","hdl":"HDL Cholesterol",
    "dbp":"Diastolic Blood Pressure","sbp":"Systolic Blood Pressure","triglycerides":"Triglycerides",
    "urate":"Serum Urate","rbc":"Red Blood Cell Count","wbc":"White Blood Cell Count",
    "grip_strength":"Grip Strength","fvc":"Forced Vital Capacity","pulse_rate":"Pulse Rate",
    "bc":"Breast Cancer","cad":"Coronary Artery Disease","ibd":"Inflammatory Bowel Disease",
    "scz":"Schizophrenia","t2d":"Type 2 Diabetes","arthrosis":"Arthrosis","asthma":"Asthma",
    "diverticulitis":"Diverticulitis","gallstones":"Gallstones","glaucoma":"Glaucoma",
    "hypothyroidism":"Hypothyroidism","malignant_neoplasms":"Malignant Neoplasms",
    "uterine_fibroids":"Uterine Fibroids","varicose_veins":"Varicose Veins"
}

quant_traits = trait_groups["Quantitative"]
disease_traits = trait_groups["Disease"]

wanted_order = quant_traits + disease_traits
fits_post = fits_post.loc[fits_post.trait.isin(wanted_order)].copy()
fits_post["__order"] = fits_post.trait.map({t:i for i,t in enumerate(wanted_order)})
fits_post = fits_post.sort_values("__order")

boot_cis = {}
# Read bootstrap files
for trait in wanted_order:
    fname = os.path.join(fit_dir, "bootstrap", f"{trait}_standard_fits_raw_bootstrap.pkl")
    with open(fname, "rb") as f:
        boot_data = pickle.load(f)
        boot_lower_Ip = np.nanpercentile(boot_data["Ip_ests_raw"], 2.5)
        boot_upper_Ip = np.nanpercentile(boot_data["Ip_ests_raw"], 97.5)
        boot_lower_I2 = np.nanpercentile(boot_data["I2_ests_raw"], 2.5)
        boot_upper_I2 = np.nanpercentile(boot_data["I2_ests_raw"], 97.5)
        boot_cis[trait] = (boot_lower_Ip, boot_upper_Ip, boot_lower_I2, boot_upper_I2)

fig, axes = plt.subplots(2, 2, figsize=(max(8, 0.5*len(wanted_order)), 5*2), sharey=False)

ax_q1, ax_d1, ax_q, ax_d = axes.flatten()


def plot_group(ax, traits, param="Ip"):
    for ii, trait in enumerate(traits):
        rec = df_ci.loc[df_ci.trait == trait]
        if rec.empty:
            continue
        lower = rec[f"{param}_ci_lower"].iloc[0]
        upper = rec[f"{param}_ci_upper"].iloc[0]
        est   = rec[f"{param}_ci"].iloc[0]
        ax.fill_between([ii-0.4, ii+0.4], lower, upper, color="gray", alpha=0.5, linewidth=0, 
                        label="Curvature CI" if ii == 0 else "")
        ax.plot(ii, est, "o", color="red", ms=5, label="MLE" if ii == 0 else "")
        if param == "Ip":
            ax.plot(ii, boot_cis[trait][0], "o", color="blue", ms=5)
            ax.plot(ii, boot_cis[trait][1], "o", color="blue", ms=5, label="Bootstrap CI" if ii == 0 else "")
        elif param == "I2":
            ax.plot(ii, boot_cis[trait][2], "o", color="blue", ms=5)
            ax.plot(ii, boot_cis[trait][3], "o", color="blue", ms=5, label="Bootstrap CI" if ii == 0 else "")
    if param == "Ip":
        ax.set_xticks(range(len(traits)))
        ax.set_xticklabels([trait_names[t] for t in traits], rotation=60, ha="right", fontsize=8)
    elif param == "I2":
        # remove ticks
        ax.set_xticks([])

plot_group(ax_q, quant_traits)
plot_group(ax_d, disease_traits)
plot_group(ax_q1, quant_traits, param="I2")
plot_group(ax_d1, disease_traits, param="I2")

for ax, title in [(ax_q,"Quantitative"), (ax_d,"Disease")]:
    ax.set_xlabel("")
    ax.set_title(title, fontsize=11)

ax_q.set_ylabel(r"$I_p$")
ax_q.set_yscale("log")
ax_d.set_yscale("log")
ax_q1.set_ylabel(r"$I_2$")
ax_q1.set_yscale("log")
ax_d1.set_yscale("log")

ax_d1.legend()

plt.tight_layout()
plt.savefig("opt_fits_original_traits_eur_raw_ci_split.pdf", bbox_inches="tight")