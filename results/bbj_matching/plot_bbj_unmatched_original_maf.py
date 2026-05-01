import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bbj_matching_common import DATA_DIR, SCRIPT_DIR
from bbj_plotting import set_publication_style


OUT_FIG = SCRIPT_DIR / "bbj_unmatched_original_maf.pdf"
OUT_BINS = DATA_DIR / "bbj_unmatched_original_maf_bins.tsv"


def set_unmatched_maf_style():
    set_publication_style()
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
        }
    )


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return values, np.arange(1, len(values) + 1) / len(values)


def plot_figure(status, by_bin):
    matched = status.loc[status["matched_bbj"]]
    unmatched = status.loc[~status["matched_bbj"]]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.45),
        gridspec_kw={"width_ratios": [1.05, 1.05]},
    )

    ax = axes[0]
    for df, label, color in [
        (matched, "Matched", "#0072B2"),
        (unmatched, "Unmatched", "#D55E00"),
    ]:
        x, y = ecdf(df["maf_orig"])
        ax.step(
            x,
            y,
            where="post",
            color=color,
            linewidth=1.8,
            label=f"{label} (n={len(df)})",
        )
    ax.set_xscale("log")
    ax.set_xlim(0.01, 0.52)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Original MAF")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Original-fit loci")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.22, linewidth=0.6)

    ax = axes[1]
    x = np.arange(len(by_bin))
    ax.bar(
        x,
        by_bin["frac_unmatched"],
        color="#D55E00",
        alpha=0.82,
        edgecolor="black",
        linewidth=0.35,
    )
    for idx, row in by_bin.iterrows():
        ax.text(
            idx,
            row["frac_unmatched"] + 0.025,
            f"n={int(row['n_loci'])}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_ylim(0, min(1.0, max(0.35, by_bin["frac_unmatched"].max() * 1.25)))
    ax.set_xticks(x)
    ax.set_xticklabels(by_bin["maf_bin"], rotation=35, ha="right")
    ax.set_xlabel("Original MAF bin")
    ax.set_ylabel("Fraction unmatched")
    ax.set_title("Drop-out by original MAF")
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)


def main():
    set_unmatched_maf_style()
    status = pd.read_csv(DATA_DIR / "bbj_match_status.tsv", sep="\t")
    by_bin = pd.read_csv(OUT_BINS, sep="\t")
    plot_figure(status, by_bin)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
