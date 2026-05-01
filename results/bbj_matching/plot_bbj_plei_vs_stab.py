import matplotlib
import numpy as np
import pandas as pd


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
OUT_TABLE = DATA_DIR / "bbj_plei_vs_stab_conditions.tsv"
OUT_FIG = SCRIPT_DIR / "bbj_plei_vs_stab_conditions_aic.pdf"


def make_table(audit):
    table = audit.loc[
        :,
        ["trait", "fit_name", "n_loci", "n_replaced", "plei_ll_gain", "stab_ll_gain"],
    ].copy()
    table["plei_vs_stab_ll_gain"] = table["plei_ll_gain"] - table["stab_ll_gain"]
    table["plei_vs_stab_aic_gain"] = 2 * table["plei_vs_stab_ll_gain"]
    table["trait_label"] = table["trait"].map(TRAIT_LABELS).fillna(table["trait"])
    table.to_csv(OUT_TABLE, sep="\t", index=False)
    return table


def plot_table(table):
    traits = trait_order(table["trait"].unique())
    x = np.arange(len(traits))
    fig, ax = plt.subplots(figsize=(11.5, 4.2))

    for fit_name in FIT_ORDER:
        sub = table.loc[table["fit_name"] == fit_name].set_index("trait").reindex(traits)
        ax.scatter(
            x + FIT_OFFSETS[fit_name],
            sub["plei_vs_stab_aic_gain"],
            s=36 if fit_name == "full_original" else 30,
            marker="D" if fit_name == "full_original" else "o",
            color=FIT_COLORS[fit_name],
            edgecolor="white",
            linewidth=0.35,
            label=FIT_LABELS[fit_name],
            zorder=3,
        )

    yvals = table["plei_vs_stab_aic_gain"].to_numpy(dtype=float)
    yvals = yvals[np.isfinite(yvals)]
    ax.axhline(0, color="#7A7A7A", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    ax.set_ylim(min(-5.0, np.nanmin(yvals) * 1.12), max(5.0, np.nanmax(yvals) * 1.55))
    ax.set_yscale("symlog", linthresh=2.0, linscale=0.8)
    ax.set_xlim(-0.6, len(traits) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([TRAIT_LABELS.get(t, t) for t in traits], rotation=45, ha="right")
    ax.set_ylabel(r"$-\Delta \mathrm{AIC}_{\mathrm{plei-stab}}$")
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)


def main():
    set_publication_style()
    audit = pd.read_csv(IN_TABLE, sep="\t")
    table = make_table(audit)
    plot_table(table)
    print(f"Wrote {OUT_TABLE}")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
