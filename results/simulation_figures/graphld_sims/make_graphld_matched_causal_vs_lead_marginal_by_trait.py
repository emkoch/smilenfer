import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import smilenfer.plotting as splot


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(SCRIPT_DIR, "graphld_locus_truth_vs_lead_by_trait.tsv")
OUT_PDF = os.path.join(SCRIPT_DIR, "graphld_matched_causal_vs_lead_marginal_by_trait.pdf")

splot._plot_params()
matplotlib.rcParams.update({"font.size": 18, "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    df = pd.read_csv(IN_PATH, sep="\t").sort_values(["trait", "locus_id"]).reset_index(drop=True)
    traits = df["trait"].drop_duplicates().tolist()

    n_cols = 4
    n_rows = int(np.ceil(len(traits) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    xvals = df["paired_causal_beta_marginal_abs"].to_numpy()
    yvals = df["lead_beta_adjusted"].to_numpy()
    xmin = float(np.nanmin(xvals[xvals > 0])) * 0.8
    xmax = float(np.nanmax(xvals)) * 1.15
    ymin = float(np.nanmin(yvals[yvals > 0])) * 0.8
    ymax = float(np.nanmax(yvals)) * 1.15
    ref_x = np.logspace(np.log10(xmin), np.log10(xmax), 200)

    for ii, trait in enumerate(traits):
        ax = axes[ii]
        trait_df = df.loc[df["trait"] == trait].copy()
        causal_lead_df = trait_df.loc[trait_df["lead_causative"]].copy()
        noncausal_lead_df = trait_df.loc[~trait_df["lead_causative"]].copy()

        ax.scatter(
            noncausal_lead_df["paired_causal_beta_marginal_abs"].to_numpy(),
            noncausal_lead_df["lead_beta_adjusted"].to_numpy(),
            s=34,
            alpha=0.22,
            color="#4C78A8",
            edgecolors="black",
            linewidths=0.2,
            rasterized=True,
        )
        ax.scatter(
            causal_lead_df["paired_causal_beta_marginal_abs"].to_numpy(),
            causal_lead_df["lead_beta_adjusted"].to_numpy(),
            s=42,
            alpha=0.85,
            color="#F58518",
            edgecolors="black",
            linewidths=0.25,
            rasterized=True,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(ref_x, ref_x, linestyle="dashed", color="black", linewidth=1.8)
        ax.plot(ref_x, 1.5 * ref_x, linestyle="dotted", color="black", linewidth=1.4)
        ax.plot(ref_x, 2.0 * ref_x, linestyle="dashdot", color="black", linewidth=1.2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2, which="both")
        ax.tick_params(axis="both", which="major", labelsize=16)

        ratio_med = np.median(trait_df["lead_beta_adjusted"] / trait_df["paired_causal_beta_marginal_abs"])
        corr = trait_df[["paired_causal_beta_marginal_abs", "lead_beta_adjusted"]].corr().iloc[0, 1]
        ax.text(
            0.05,
            0.95,
            trait_df["label"].iloc[0] + "\n" + str(trait_df.shape[0]) + " loci\nmedian y/x=" + f"{ratio_med:.2f}" + "\nr=" + f"{corr:.2f}",
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    for ii in range(len(traits), len(axes)):
        axes[ii].axis("off")

    fig.text(0.54, 0.02, "Matched causal SNP marginal effect size", ha="center", va="center", fontsize=24)
    fig.text(0.02, 0.54, "Adjusted lead SNP effect size", ha="center", va="center", rotation="vertical", fontsize=24)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#4C78A8",
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=8,
            alpha=1.0,
            label="noncausal lead SNP",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#F58518",
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=8,
            alpha=1.0,
            label="causal lead SNP",
        ),
        Line2D([0], [0], linestyle="dashed", color="black", linewidth=1.8, label="y = x"),
        Line2D([0], [0], linestyle="dotted", color="black", linewidth=1.4, label="y = 1.5x"),
        Line2D([0], [0], linestyle="dashdot", color="black", linewidth=1.2, label="y = 2x"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", OUT_PDF)


if __name__ == "__main__":
    main()
