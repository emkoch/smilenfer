import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import smilenfer.posterior as spost

matplotlib.rcParams.update(
    {
        "font.size": 18,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIT_FILE = os.path.join(SCRIPT_DIR, "ir", "posterior_mean", "results", "ir_estimates_all.csv")
OUT_FILE = os.path.join(SCRIPT_DIR, "pleiotropy_ranking_stability.pdf")
LABEL_OFFSETS = {
    "diverticulitis": (10, -10),
    "hdl": (10, -2),
    "ldl": (10, -14),
    "rbc": (10, 10),
    "triglycerides": (10, -8),
    "urate": (14, -2),
}

fit_df = pd.read_csv(FIT_FILE)
fit_df = fit_df[fit_df.drop_count == 0].copy()
fit_df["before"] = 2 * (fit_df.Ip_LL - fit_df.I2_LL)
fit_df["after"] = 2 + 2 * (fit_df.Ip_LL - fit_df.Ir_LL)
fit_df["meaningful_1t"] = fit_df["after"] < -2
fit_df = fit_df.sort_values("before", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(1, 1, figsize=(8.6, 9.2))

x_before = 0.0
x_after = 1.0

for _, row in fit_df.iterrows():
    color = "#d55e00" if row["meaningful_1t"] else "0.68"
    zorder = 3 if row["meaningful_1t"] else 2
    ax.plot(
        [x_before, x_after],
        [row["before"], row["after"]],
        color=color,
        linewidth=1.5 if row["meaningful_1t"] else 1.0,
        alpha=0.95 if row["meaningful_1t"] else 0.8,
        zorder=zorder,
    )
    ax.scatter(
        [x_before, x_after],
        [row["before"], row["after"]],
        s=60 if row["meaningful_1t"] else 48,
        color=color,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.95 if row["meaningful_1t"] else 0.8,
        zorder=zorder + 0.5,
    )

for _, row in fit_df.loc[fit_df["meaningful_1t"]].iterrows():
    dx, dy = LABEL_OFFSETS.get(row.trait, (10, 0))
    ax.annotate(
        spost.original_trait_names.get(row.trait, row.trait),
        (x_after, row["after"]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.7", alpha=0.95),
        zorder=5,
    )

all_vals = fit_df[["before", "after"]].to_numpy().ravel()
all_vals = all_vals[np.isfinite(all_vals)]
axis_max = np.max(np.abs(all_vals)) * 1.15

ax.axhline(0, color="0.3", linewidth=1.0, zorder=1)
ax.axhline(2, color="0.72", linewidth=0.9, linestyle="--", zorder=1)
ax.axhline(-2, color="0.72", linewidth=0.9, linestyle="--", zorder=1)
ax.axvline(0.5, color="0.88", linewidth=1.0, zorder=1)

ax.set_yscale("symlog", linthresh=2)
ax.set_ylim(-axis_max, axis_max)
ax.set_xlim(-0.14, 1.38)
ax.set_xticks([x_before, x_after])
ax.set_xticklabels([r"Single-trait: $r=2$", r"Single-trait: free $r$"])
ax.tick_params(axis="x", labelsize=15)

ax.set_ylabel(
    r"Evidence for pleiotropy: $\Delta \mathrm{AIC}_{\mathrm{1T}-\mathrm{PLEI}}$",
    fontweight="bold",
)

fig.tight_layout()
fig.savefig(OUT_FILE, bbox_inches="tight")
