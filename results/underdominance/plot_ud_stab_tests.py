import pandas as pd
import smilenfer.plotting as splot
import numpy as np
splot._plot_params()
import matplotlib.pyplot as plt

df = pd.read_csv("stab_ud_std_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Panel for I2
axes[0].scatter(df["ll_I2_ud"] - df["ll_neut"], df["ll_I2_ud"] - df["ll_I2_std"])
for _, row in df.iterrows():
    axes[0].text(row["ll_I2_ud"] - row["ll_neut"],
                 row["ll_I2_ud"] - row["ll_I2_std"],
                 row["Trait"],
                 fontsize=8, ha='right', va='bottom')
axes[0].axhline(0, ls="--", lw=1, color="black")
axes[0].axvline(0, ls="--", lw=1, color="black")
axes[0].set_xlabel("Δ log-likelihood (I2_ud – neut)", fontsize=10)
axes[0].set_ylabel("Δ log-likelihood (I2_ud – I2_std)", fontsize=10)
axes[0].set_xscale("symlog", linthresh=5)
axes[0].set_yscale("symlog", linthresh=5)
axes[0].set_title("1-D stabilizing", fontsize=10)
axes[0].tick_params(axis='both', which='both', labelsize=10)

# Panel for Ip
axes[1].scatter(df["ll_Ip_ud"] - df["ll_neut"], df["ll_Ip_ud"] - df["ll_Ip_std"])
for _, row in df.iterrows():
    axes[1].text(row["ll_Ip_ud"] - row["ll_neut"],
                 row["ll_Ip_ud"] - row["ll_Ip_std"],
                 row["Trait"],
                 fontsize=8, ha='right', va='bottom')
axes[1].axhline(0, ls="--", lw=1, color="black")
axes[1].axvline(0, ls="--", lw=1, color="black")
axes[1].set_xlabel("Δ log-likelihood (Ip_ud – neut)", fontsize=10)
axes[1].set_xscale("symlog", linthresh=5)
axes[1].set_yscale("symlog", linthresh=5)
axes[1].set_title("Pleiotropic stabilizing", fontsize=10)
axes[1].tick_params(axis='both', which='both', labelsize=10)

output_file = "stab_ud_std_vs_neut_panels_updated.pdf"
plt.tight_layout()
plt.savefig(output_file, bbox_inches='tight')
print(f"Plot saved to {output_file}")
