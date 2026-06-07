import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAB_CSV = os.path.join(SCRIPT_DIR, "graphld_ldl_neutral_and_replicates_wc_postmean_stab.csv")
PLEI_CSV = os.path.join(SCRIPT_DIR, "graphld_ldl_neutral_and_replicates_wc_postmean_plei.csv")
OUT_PDF = os.path.join(SCRIPT_DIR, "graphld_ldl_stab_vs_plei_scatter.pdf")

DATASET_ORDER = ["causal", "beta_p", "beta_p_wc", "beta_p_post"]
DATASET_LABELS = {
    "causal": "Causal",
    "beta_p": "Lead",
    "beta_p_wc": "WC",
    "beta_p_post": "Post.",
}
CONDITION_COLORS = {
    "neutral": "#4C78A8",
    "selected": "#E45756",
}

matplotlib.rcParams.update({"font.size": 10})
matplotlib.rcParams["figure.facecolor"] = "#ffffff"
matplotlib.rcParams["axes.facecolor"] = "#ffffff"
matplotlib.rcParams["savefig.facecolor"] = "#ffffff"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.style.use("bmh")
matplotlib.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})


def main():
    stab_df = pd.read_csv(STAB_CSV)
    plei_df = pd.read_csv(PLEI_CSV)

    stab_df = stab_df.loc[:, ["condition", "replicate", "dataset", "stat_stab"]].copy()
    plei_df = plei_df.loc[:, ["condition", "replicate", "dataset", "stat_plei"]].copy()
    fit_df = stab_df.merge(plei_df, on=["condition", "replicate", "dataset"], how="inner")

    lim_max = max(float(fit_df["stat_stab"].max()), float(fit_df["stat_plei"].max())) * 1.05

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.8), sharex=True, sharey=True)

    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = fit_df.loc[fit_df["dataset"] == dataset].copy()
        for condition in ["neutral", "selected"]:
            condition_df = sub.loc[sub["condition"] == condition]
            ax.scatter(
                condition_df["stat_stab"],
                condition_df["stat_plei"],
                s=34,
                alpha=0.75,
                color=CONDITION_COLORS[condition],
                edgecolors="black",
                linewidth=0.35,
                label=condition.capitalize(),
            )
        ax.plot([0, lim_max], [0, lim_max], color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlim(-0.1, lim_max)
        ax.set_ylim(-0.1, lim_max)
        ax.set_title(DATASET_LABELS[dataset])
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"Plei. $-\log_{10} \mathrm{p-value}$")
    for ax in axes:
        ax.set_xlabel(r"Stab. $-\log_{10} \mathrm{p-value}$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    print(" -", OUT_PDF)


if __name__ == "__main__":
    main()
