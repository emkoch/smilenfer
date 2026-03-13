import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

import smilenfer.plotting as splot


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
ORIGINAL_DIR = os.path.join(DATA_DIR, "final", "original_traits")
SUSIEX_DIR = os.path.join(DATA_DIR, "final", "UKBB_susiex")

P_THRESH = 5e-8
TRAITS = [
    "bmi",
    "dbp",
    "hdl",
    "height",
    "ldl",
    "sbp",
    "triglycerides",
    "wbc",
]
TRAIT_LABELS = {
    "bmi": "BMI",
    "dbp": "DBP",
    "hdl": "HDL",
    "height": "Height",
    "ldl": "LDL",
    "sbp": "SBP",
    "triglycerides": "Triglycerides",
    "wbc": "WBC",
}
BIN_QUANTILES = np.array([0.0, 0.12, 0.24, 0.38, 0.52, 0.66, 0.80, 0.90, 1.0])


splot._plot_params()
matplotlib.rcParams.update({"font.size": 16})


def wilson_bounds(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - radius, center + radius


def load_trait_data(trait):
    original_path = os.path.join(ORIGINAL_DIR, f"processed.{trait}.snps_low_r2.tsv")
    susiex_path = os.path.join(SUSIEX_DIR, f"susiex_cs_table_{trait}.csv")

    original_df = pd.read_csv(original_path, sep="\t")
    original_df["orig_locus"] = original_df["chr"].astype(int).astype(str) + ":" + original_df["pos"].astype(int).astype(str)
    susiex_df = pd.read_csv(susiex_path)

    v_cut = stats.chi2.isf(P_THRESH, df=1) / np.nanmedian(original_df["n_eff"])
    mapped_loci = set(susiex_df.loc[susiex_df["locus"].notna(), "orig_locus"])

    original_df["has_cs_mapping"] = original_df["orig_locus"].isin(mapped_loci)
    original_df["power_distance_log10"] = np.log10(original_df["var_exp"] / v_cut)
    return original_df


def summarize_trait(trait_df):
    dd = trait_df.copy()
    bin_edges = np.unique(np.nanquantile(dd["power_distance_log10"], BIN_QUANTILES))
    if bin_edges.size < 2:
        bin_edges = np.array([np.nanmin(dd["power_distance_log10"]), np.nanmax(dd["power_distance_log10"]) + 1e-6])
    if bin_edges[0] == bin_edges[-1]:
        bin_edges[-1] = bin_edges[-1] + 1e-6

    dd["distance_bin"] = pd.cut(dd["power_distance_log10"], bins=bin_edges, right=True, include_lowest=True)

    rows = []
    for distance_bin, bin_df in dd.groupby("distance_bin", observed=False):
        n_loci = int(len(bin_df))
        if n_loci == 0:
            continue
        n_mapped = int(bin_df["has_cs_mapping"].sum())
        mapped_rate = n_mapped / n_loci
        lo, hi = wilson_bounds(n_mapped, n_loci)
        rows.append(
            {
                "distance_mid": float(np.nanmedian(bin_df["power_distance_log10"])),
                "mapped_rate": mapped_rate,
                "mapped_rate_ci_lo": lo,
                "mapped_rate_ci_hi": hi,
                "n_loci": n_loci,
            }
        )

    summary_df = pd.DataFrame(rows)
    negative_n = int((dd["power_distance_log10"] < 0).sum())
    return summary_df, negative_n


def plot_trait_panel(ax, trait, x_left, x_right):
    trait_df = load_trait_data(trait)
    summary_df, negative_n = summarize_trait(trait_df)

    x = summary_df["distance_mid"].to_numpy()
    y = summary_df["mapped_rate"].to_numpy()
    ylo = summary_df["mapped_rate_ci_lo"].to_numpy()
    yhi = summary_df["mapped_rate_ci_hi"].to_numpy()

    ax.plot(x, y, color="#F58518", marker="o", linewidth=2)
    ax.fill_between(x, ylo, yhi, color="#F58518", alpha=0.2)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(TRAIT_LABELS[trait])
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\log_{10}(\hat{v}/v^{*})$")
    ax.text(
        0.02,
        1.02,
        f"n<0: {negative_n}",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.grid(alpha=0.2)


all_distance = []
for trait in TRAITS:
    dd = load_trait_data(trait)
    all_distance.append(dd["power_distance_log10"].to_numpy())
all_distance = np.concatenate(all_distance)
all_distance = all_distance[np.isfinite(all_distance)]
x_left = min(-0.05, float(np.nanmin(all_distance)) - 0.02)
x_right = float(np.nanquantile(all_distance, 0.99)) + 0.08

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
axes = axes.flatten()

for ii, trait in enumerate(TRAITS):
    plot_trait_panel(axes[ii], trait, x_left, x_right)
    if ii % 4 == 0:
        axes[ii].set_ylabel("Mapped rate")

fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "ukbb_vs_susiex_power_distance_mapping.pdf"), bbox_inches="tight")
