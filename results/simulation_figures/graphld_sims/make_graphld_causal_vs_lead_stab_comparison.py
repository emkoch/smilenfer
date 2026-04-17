import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import smilenfer.plotting as splot


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIT_PATH = os.path.join(SCRIPT_DIR, "opt_results_graphld_causal_vs_lead_matched.csv")
COUNT_PATH = os.path.join(SCRIPT_DIR, "graphld_causal_vs_lead_matched_counts.tsv")
BAR_PDF = os.path.join(SCRIPT_DIR, "graphld_causal_vs_lead_stab_diff_bars.pdf")

TRAIT_ORDER = ["height", "ldl", "dbp", "fvc", "grip_strength", "asthma", "arthrosis"]
TRAIT_GROUP_BREAKS = [5]
MODEL_INFO = {
    "stab": {"label": "Single-trait stabilizing", "color": "#DC267F"},
    "plei": {"label": "Pleiotropic stabilizing", "color": "#FE6100"},
}

splot._plot_params()
matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11.5,
        "axes.titlesize": 13.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.0,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def pair_values(causal_vals, lead_vals):
    causal_vals = np.asarray(causal_vals, dtype=float)
    lead_vals = np.asarray(lead_vals, dtype=float)
    if len(causal_vals) == 1 and len(lead_vals) == 1:
        return causal_vals.copy(), lead_vals.copy()
    if len(causal_vals) == 1:
        return np.full(len(lead_vals), causal_vals[0]), lead_vals.copy()
    if len(lead_vals) == 1:
        return causal_vals.copy(), np.full(len(causal_vals), lead_vals[0])
    n_pairs = min(len(causal_vals), len(lead_vals))
    return np.sort(causal_vals)[:n_pairs], np.sort(lead_vals)[:n_pairs]


def summarize_pairs(fit_df):
    summary_rows = []
    for trait in TRAIT_ORDER:
        trait_df = fit_df.loc[fit_df["trait"] == trait].copy()
        for model in MODEL_INFO:
            causal_vals = trait_df.loc[trait_df["dataset"] == "causal", "stat_" + model].to_numpy()
            lead_vals = trait_df.loc[trait_df["dataset"] == "lead", "stat_" + model].to_numpy()
            xs, ys = pair_values(causal_vals, lead_vals)
            diffs = ys - xs
            summary_rows.append(
                {
                    "trait": trait,
                    "model": model,
                    "causal_med": float(np.median(causal_vals)),
                    "lead_med": float(np.median(lead_vals)),
                    "diff_med": float(np.median(diffs)),
                    "diff_q10": float(np.quantile(diffs, 0.1)),
                    "diff_q90": float(np.quantile(diffs, 0.9)),
                    "n_pairs": int(len(diffs)),
                    "causal_n": int(len(causal_vals)),
                    "lead_n": int(len(lead_vals)),
                }
            )
    return pd.DataFrame(summary_rows)


def make_bars(summary_df, count_df):
    label_map = dict(zip(count_df["trait"], count_df["label"]))
    xticks = np.arange(len(TRAIT_ORDER), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)

    for ax, model in zip(axes, ["stab", "plei"]):
        model_df = summary_df.loc[summary_df["model"] == model].set_index("trait").loc[TRAIT_ORDER].reset_index()
        diff_med = model_df["diff_med"].to_numpy()
        diff_low = diff_med - model_df["diff_q10"].to_numpy()
        diff_high = model_df["diff_q90"].to_numpy() - diff_med

        ax.bar(
            xticks,
            diff_med,
            width=0.7,
            color=MODEL_INFO[model]["color"],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.errorbar(
            xticks,
            diff_med,
            yerr=np.vstack([diff_low, diff_high]),
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
        ax.axhline(0, color="black", linestyle="dashed", linewidth=1.1, zorder=1)
        for break_idx in TRAIT_GROUP_BREAKS:
            ax.axvline(break_idx - 0.5, color="#BBBBBB", linewidth=0.8, zorder=1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([label_map[trait] for trait in TRAIT_ORDER], rotation=55, ha="right", rotation_mode="anchor")
        ax.set_title(MODEL_INFO[model]["label"])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Lead - causal evidence")
    fig.suptitle("Lead Minus Causal Evidence For Stabilizing Models", y=0.98, fontsize=15)
    fig.tight_layout()
    fig.savefig(BAR_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    fit_df = pd.read_csv(FIT_PATH)
    count_df = pd.read_csv(COUNT_PATH, sep="\t")
    summary_df = summarize_pairs(fit_df)
    make_bars(summary_df, count_df)
    print("Wrote:", BAR_PDF)


if __name__ == "__main__":
    main()
