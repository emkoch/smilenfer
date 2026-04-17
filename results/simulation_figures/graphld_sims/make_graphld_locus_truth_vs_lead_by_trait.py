import glob
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

import smilenfer.plotting as splot


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "sims", "graphld_sims"))
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "graphld_adjusted_lead_input_summary.tsv")
OUT_TSV = os.path.join(SCRIPT_DIR, "graphld_locus_truth_vs_lead_by_trait.tsv")
OUT_PDF = os.path.join(SCRIPT_DIR, "graphld_locus_truth_vs_lead_by_trait.pdf")

P_THRESH = 5e-8

splot._plot_params()
matplotlib.rcParams.update({"font.size": 18, "pdf.fonttype": 42, "ps.fonttype": 42})


def trait_from_path(path):
    trait_name = os.path.basename(path).replace(".tsv.gz", "").replace(".tsv", "")
    return trait_name.split("_seed_")[0].replace("simulated_", "")


def load_trait_locus_table(path, n_eff_fit, label):
    df = pd.read_csv(path, sep="\t")
    df["lead"] = df["lead"].astype(bool)
    df["causative"] = df["causative"].astype(bool)
    df["gws"] = df["p"] < P_THRESH
    df["trait"] = trait_from_path(path)
    df["locus_id"] = df["chr"].astype(str) + ":" + df["clump"].astype(str)

    lead_df = df.loc[df["lead"] & df["gws"]].copy().reset_index(drop=True)
    chi2_stat = stats.chi2.isf(lead_df["p"].to_numpy(), df=1)
    lead_df["lead_beta_adjusted"] = np.sqrt(
        chi2_stat / (2.0 * n_eff_fit * lead_df["raf"].to_numpy() * (1.0 - lead_df["raf"].to_numpy()))
    )

    locus_rows = []
    for _, lead_row in lead_df.iterrows():
        locus_df = df.loc[df["locus_id"] == lead_row["locus_id"]].copy()
        causal_df = locus_df.loc[locus_df["causative"]].copy()
        if causal_df.empty:
            continue

        beta_scale = max(float(lead_row["beta_marginal_abs"]), 1e-12)
        raf_scale = max(min(float(lead_row["raf"]), 1.0 - float(lead_row["raf"])), 0.01)
        causal_df["match_score"] = np.sqrt(
            ((causal_df["beta_marginal_abs"].to_numpy() - float(lead_row["beta_marginal_abs"])) / beta_scale) ** 2
            + ((causal_df["raf"].to_numpy() - float(lead_row["raf"])) / raf_scale) ** 2
        )

        paired_row = causal_df.loc[causal_df["match_score"].idxmin()]
        nearest_row = causal_df.loc[np.abs(causal_df["dist_to_lead"]).idxmin()]
        top_p_row = causal_df.loc[causal_df["p"].idxmin()]
        max_beta_row = causal_df.loc[causal_df["beta_abs"].idxmax()]

        locus_rows.append(
            {
                "trait": lead_row["trait"],
                "label": label,
                "locus_id": lead_row["locus_id"],
                "chr": int(lead_row["chr"]),
                "clump": int(lead_row["clump"]),
                "lead_snp": lead_row["snp"],
                "lead_causative": bool(lead_row["causative"]),
                "lead_beta_abs_true": float(lead_row["beta_abs"]),
                "lead_beta_marginal_abs": float(lead_row["beta_marginal_abs"]),
                "lead_beta_adjusted": float(lead_row["lead_beta_adjusted"]),
                "lead_raf": float(lead_row["raf"]),
                "lead_p": float(lead_row["p"]),
                "n_causal_in_locus": int(causal_df.shape[0]),
                "paired_causal_beta_abs": float(paired_row["beta_abs"]),
                "paired_causal_beta_marginal_abs": float(paired_row["beta_marginal_abs"]),
                "paired_causal_raf": float(paired_row["raf"]),
                "paired_causal_snp": paired_row["snp"],
                "paired_causal_dist_to_lead": float(paired_row["dist_to_lead"]),
                "paired_causal_p": float(paired_row["p"]),
                "paired_match_score": float(paired_row["match_score"]),
                "nearest_causal_beta_abs": float(nearest_row["beta_abs"]),
                "nearest_causal_beta_marginal_abs": float(nearest_row["beta_marginal_abs"]),
                "nearest_causal_raf": float(nearest_row["raf"]),
                "nearest_causal_snp": nearest_row["snp"],
                "nearest_causal_dist_to_lead": float(nearest_row["dist_to_lead"]),
                "top_p_causal_beta_abs": float(top_p_row["beta_abs"]),
                "top_p_causal_beta_marginal_abs": float(top_p_row["beta_marginal_abs"]),
                "top_p_causal_raf": float(top_p_row["raf"]),
                "top_p_causal_snp": top_p_row["snp"],
                "top_p_causal_p": float(top_p_row["p"]),
                "max_causal_beta_abs": float(max_beta_row["beta_abs"]),
                "max_causal_beta_marginal_abs": float(max_beta_row["beta_marginal_abs"]),
                "max_causal_raf": float(max_beta_row["raf"]),
                "max_causal_snp": max_beta_row["snp"],
            }
        )

    return pd.DataFrame(locus_rows)


def main():
    summary_df = pd.read_csv(SUMMARY_PATH, sep="\t").sort_values("trait").reset_index(drop=True)
    n_eff_map = dict(zip(summary_df["trait"], summary_df["n_eff_fit"]))
    label_map = dict(zip(summary_df["trait"], summary_df["label"]))

    locus_tables = []
    trait_paths = sorted(glob.glob(os.path.join(DATA_DIR, "simulated_*_loci.tsv.gz")))

    for trait_path in trait_paths:
        trait = trait_from_path(trait_path)
        locus_tables.append(load_trait_locus_table(trait_path, n_eff_map[trait], label_map[trait]))

    locus_df = pd.concat(locus_tables, ignore_index=True)
    locus_df.to_csv(OUT_TSV, sep="\t", index=False)

    traits = summary_df["trait"].tolist()
    n_cols = 4
    n_rows = int(np.ceil(len(traits) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    xvals = locus_df["paired_causal_beta_abs"].to_numpy()
    yvals = locus_df["lead_beta_adjusted"].to_numpy()
    xmin = float(np.nanmin(xvals[xvals > 0])) * 0.8
    xmax = float(np.nanmax(xvals)) * 1.15
    ymin = float(np.nanmin(yvals[yvals > 0])) * 0.8
    ymax = float(np.nanmax(yvals)) * 1.15
    ref_x = np.logspace(np.log10(xmin), np.log10(xmax), 200)

    for ii, trait in enumerate(traits):
        ax = axes[ii]
        trait_df = locus_df.loc[locus_df["trait"] == trait].copy()
        causal_lead_df = trait_df.loc[trait_df["lead_causative"]].copy()
        noncausal_lead_df = trait_df.loc[~trait_df["lead_causative"]].copy()

        ax.scatter(
            noncausal_lead_df["paired_causal_beta_abs"].to_numpy(),
            noncausal_lead_df["lead_beta_adjusted"].to_numpy(),
            s=34,
            alpha=0.22,
            color="#4C78A8",
            edgecolors="black",
            linewidths=0.2,
            rasterized=True,
        )
        ax.scatter(
            causal_lead_df["paired_causal_beta_abs"].to_numpy(),
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
        ax.plot(ref_x, 10 * ref_x, linestyle="dotted", color="black", linewidth=1.4)
        ax.plot(ref_x, 100 * ref_x, linestyle="dashdot", color="black", linewidth=1.2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2, which="both")
        ax.tick_params(axis="both", which="major", labelsize=16)
        ratio_med = np.median(trait_df["lead_beta_adjusted"] / trait_df["paired_causal_beta_abs"])
        ax.text(
            0.05,
            0.95,
            label_map[trait] + "\n" + str(trait_df.shape[0]) + " loci\nmedian y/x=" + f"{ratio_med:.1f}",
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    for ii in range(len(traits), len(axes)):
        axes[ii].axis("off")

    fig.text(0.54, 0.02, "Paired causal SNP true effect size", ha="center", va="center", fontsize=24)
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
        Line2D([0], [0], linestyle="dotted", color="black", linewidth=1.4, label="y = 10x"),
        Line2D([0], [0], linestyle="dashdot", color="black", linewidth=1.2, label="y = 100x"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", OUT_TSV)
    print("Wrote:", OUT_PDF)


if __name__ == "__main__":
    main()
