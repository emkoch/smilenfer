import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bbj_matching_common import (
    DATA_DIR,
    FIT_COLORS,
    FIT_LABELS,
    FIT_OFFSETS,
    FIT_ORDER,
    SCRIPT_DIR,
    TRAIT_LABELS,
    trait_order,
)
from bbj_plotting import set_publication_style


IN_TABLE = DATA_DIR / "bbj_beta_replacement_raw_combined_audit.tsv"
OUT_FIG = SCRIPT_DIR / "bbj_beta_replacement_raw_combined_aic.pdf"


def plot_aic(table):
    traits = trait_order(table["trait"].unique())
    x = np.arange(len(traits))
    fig, ax = plt.subplots(figsize=(11.5, 4.2))

    for fit_name in FIT_ORDER:
        sub = table.loc[table["fit_name"] == fit_name].set_index("trait").reindex(traits)
        ax.scatter(
            x + FIT_OFFSETS[fit_name],
            sub["plei_aic_gain"],
            s=36 if fit_name == "full_original" else 30,
            marker="D" if fit_name == "full_original" else "o",
            color=FIT_COLORS[fit_name],
            edgecolor="white",
            linewidth=0.35,
            label=FIT_LABELS[fit_name],
            zorder=3,
        )

    ax.axhline(0, color="#7A7A7A", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    ax.axhline(
        chi2.isf(0.05, 1) - 2,
        color="#C44E52",
        linestyle=(0, (4, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax.axhline(
        chi2.isf(0.05 / len(traits), 1) - 2,
        color="#6F3E8B",
        linestyle=(0, (4, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax.text(
        len(traits) - 0.45,
        chi2.isf(0.05, 1) - 2,
        r"$P=0.05$",
        ha="right",
        va="bottom",
        color="#C44E52",
        fontsize=8.5,
    )
    ax.text(
        len(traits) - 0.45,
        chi2.isf(0.05 / len(traits), 1) - 2,
        r"$P=0.05/n$",
        ha="right",
        va="bottom",
        color="#6F3E8B",
        fontsize=8.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([TRAIT_LABELS.get(t, t) for t in traits], rotation=45, ha="right")
    ax.set_ylabel(r"$-\Delta \mathrm{AIC}_{\mathrm{plei-neut}}$")
    ax.set_xlim(-0.6, len(traits) - 0.4)
    ax.set_ylim(
        min(-2.5, np.nanmin(table["plei_aic_gain"]) * 1.08),
        np.nanmax(table["plei_aic_gain"]) * 1.55,
    )
    ax.set_yscale("symlog", linthresh=2.0, linscale=0.8)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)


def main():
    set_publication_style()
    table = pd.read_csv(IN_TABLE, sep="\t")
    plot_aic(table)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
