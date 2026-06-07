import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAB_CSV = os.path.join(SCRIPT_DIR, "graphld_ldl_neutral_and_replicates_wc_postmean_stab.csv")
PLEI_CSV = os.path.join(SCRIPT_DIR, "graphld_ldl_neutral_and_replicates_wc_postmean_plei.csv")
OUT_PDF = os.path.join(SCRIPT_DIR, "graphld_ldl_replicate_stab_plei_pvals.pdf")

ORDER = [
    ("neutral", "causal"),
    ("neutral", "beta_p"),
    ("neutral", "beta_p_wc"),
    ("neutral", "beta_p_post"),
    ("selected", "causal"),
    ("selected", "beta_p"),
    ("selected", "beta_p_wc"),
    ("selected", "beta_p_post"),
]

PVAL_LABELS = [
    "Neutral\ncausal",
    "Neutral\nlead",
    "Neutral\nWC",
    "Neutral\npost.",
    "Selected\ncausal",
    "Selected\nlead",
    "Selected\nWC",
    "Selected\npost.",
]

X_POS = {key: idx for idx, key in enumerate(ORDER)}

COLORS = {
    "causal": "#648FFF",
    "beta_p": "#DC267F",
    "beta_p_wc": "#FE6100",
    "beta_p_post": "#785EF0",
}

plt.style.use("bmh")
matplotlib.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10


def add_panel(ax, fit_df, stat_col):
    rank_maps = {}
    for condition in ["neutral", "selected"]:
        causal_condition = fit_df.loc[
            (fit_df["condition"] == condition) & (fit_df["dataset"] == "causal"),
            ["replicate", stat_col],
        ].copy()
        causal_condition = causal_condition.sort_values(stat_col, ascending=False).reset_index(drop=True)
        rank_maps[condition] = {rep: ii for ii, rep in enumerate(causal_condition["replicate"].tolist())}

    for condition, dataset in ORDER:
        x0 = X_POS[(condition, dataset)]
        color = COLORS[dataset]
        sub = fit_df.loc[(fit_df["condition"] == condition) & (fit_df["dataset"] == dataset)].copy()
        sub["rep_rank"] = sub["replicate"].map(rank_maps[condition])
        sub = sub.sort_values("rep_rank").reset_index(drop=True)
        vals = sub[stat_col].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), x0) + jitter,
            vals,
            s=28,
            alpha=0.45,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

    ax.axhline(-np.log10(0.05), color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


stab_df = pd.read_csv(STAB_CSV)
plei_df = pd.read_csv(PLEI_CSV)

fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True)

add_panel(axes[0], stab_df, "stat_stab")
add_panel(axes[1], plei_df, "stat_plei")

y_max = max(float(stab_df["stat_stab"].max()), float(plei_df["stat_plei"].max()))
y_top = max(4.0, 0.35 * y_max) + y_max

for ax in axes:
    ax.set_ylim(-0.1, y_top)
    ax.set_ylabel(r"$-\log_{10} \mathrm{p-value}$")

axes[0].text(0.01, 0.98, "Single-trait stabilizing", transform=axes[0].transAxes, ha="left", va="top")
axes[1].text(0.01, 0.98, "Pleiotropic stabilizing", transform=axes[1].transAxes, ha="left", va="top")

axes[1].set_xticks(np.arange(len(ORDER)))
axes[1].set_xticklabels(PVAL_LABELS)
axes[0].tick_params(axis="x", labelbottom=False)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)
