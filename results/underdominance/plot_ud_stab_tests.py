import matplotlib.pyplot as plt
import pandas as pd

import smilenfer.plotting as splot


splot._plot_params()

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


def plot_panel(df, x_col, y_col, title, output, ylabel=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    xx = df[x_col] - df["ll_neut"]
    ud_col = y_col.replace("_std", "_ud")
    yy = df[ud_col] - df[y_col]

    ax.scatter(xx, yy)
    for _, row in df.iterrows():
        ax.text(
            row[x_col] - row["ll_neut"],
            row[ud_col] - row[y_col],
            TRAIT_NAMES.get(row["Trait"], row["Trait"]),
            fontsize=8,
            ha="right",
            va="bottom",
        )

    ax.axhline(0, ls="--", lw=1, color="black")
    ax.axvline(0, ls="--", lw=1, color="black")
    ax.set_xlabel(
        "Support for selection:\n $\Delta$ log-likelihood (stab. (ud) $–$ neutral)",
        fontsize=10,
    )
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xscale("symlog", linthresh=5)
    ax.set_yscale("symlog", linthresh=5)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="both", which="both", labelsize=10)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(
        xlim[0] * 1.1 if xlim[0] < 0 else xlim[0] * 0.9,
        xlim[1] * 1.1 if xlim[1] > 0 else xlim[1] * 0.9,
    )
    ax.set_ylim(
        ylim[0] * 1.1 if ylim[0] < 0 else ylim[0] * 0.9,
        ylim[1] * 1.1 if ylim[1] > 0 else ylim[1] * 0.9,
    )

    for lbl, tick in zip(ax.get_xmajorticklabels(), ax.get_xticks()):
        if tick in (-1, 1):
            lbl.set_visible(False)
    for lbl, tick in zip(ax.get_ymajorticklabels(), ax.get_yticks()):
        if tick in (-1, 1):
            lbl.set_visible(False)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


df = pd.read_csv("stab_ud_std_results.csv")

plot_panel(
    df,
    "ll_I2_ud",
    "ll_I2_std",
    "1-Trait stabilizing",
    "stab_ud_std_vs_neut_1T.pdf",
    ylabel=r"Support for underdominance:" + "\n" + r"$\Delta$ log-likelihood (stab. (ud) $–$ stab. (std))",
)

plot_panel(
    df,
    "ll_Ip_ud",
    "ll_Ip_std",
    "Pleiotropic stabilizing",
    "stab_ud_std_vs_neut_plei.pdf",
)
