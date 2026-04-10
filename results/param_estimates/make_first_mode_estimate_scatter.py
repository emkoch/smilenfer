import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.rcParams.update({"font.size": 15})
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"
plt.style.use("bmh")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_FIT_DIR = os.path.join(SCRIPT_DIR, "..", "all_opt_fits")
PRE_FIT_DIR = os.path.join(SCRIPT_DIR, "pre_first_mode")
OUTPUT_PDF = os.path.join(SCRIPT_DIR, "first_mode_vs_original_estimate_scatter.pdf")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "first_mode_vs_original_estimate_scatter.png")

HIGHLIGHT_COLOR = "#f58518"
LOG10_SHIFT_TO_LABEL = 0.5

MODEL_INFO = [
    ("stab", "I2_stab", r"$I_2$", "Single-trait stabilizing"),
    ("plei", "Ip_plei", r"$I_p$", "Pleiotropic stabilizing"),
]

FIT_SPECS = [
    ("original_traits_eur_raw", "original_traits/opt_results_original_traits_eur_raw.csv"),
    ("original_traits_eur_post", "original_traits/opt_results_original_traits_eur_post.csv"),
    ("bbj_high", "bbj/opt_results_high_bbj.csv"),
    ("bbj_pval", "bbj/opt_results_pval_bbj.csv"),
    ("ukbb_susiex", "ukbb_finemapping/opt_results_ukbb_susiex.csv"),
    ("mvp_finemapping_eur", "mvp/opt_results_mvp_finemapping_eur.csv"),
]

LABEL_OFFSETS = {
    ("stab", "bbj_high", "sbp"): (12, 12),
    ("stab", "bbj_pval", "sbp"): (10, -18),
    ("stab", "bbj_pval", "hdl"): (10, 10),
    ("plei", "original_traits_eur_post", "scz"): (12, 10),
    ("plei", "bbj_high", "uterine_fibroids"): (12, -18),
    ("plei", "bbj_pval", "uterine_fibroids"): (12, 10),
}


def make_point_label(row):
    source = row["source_name"]
    if source.startswith("original_traits"):
        suffix = "orig"
    elif source == "bbj_high":
        suffix = "bbj-high"
    elif source == "bbj_pval":
        suffix = "bbj-pval"
    elif source == "ukbb_susiex":
        suffix = "susiex"
    elif source == "mvp_finemapping_eur":
        suffix = "mvp"
    else:
        suffix = source
    label = f"{row['trait'].upper()} ({suffix})"
    if "sample" in row and not pd.isna(row["sample"]) and int(row["sample"]) != 0:
        label = f"{label} s{int(row['sample'])}"
    return label


def load_comparison_rows():
    rows = []
    for source_name, rel_path in FIT_SPECS:
        cur_path = os.path.join(CURRENT_FIT_DIR, rel_path)
        pre_path = os.path.join(PRE_FIT_DIR, rel_path)

        cur_df = pd.read_csv(cur_path)
        pre_df = pd.read_csv(pre_path)

        merge_cols = ["trait"]
        if "sample" in cur_df.columns and "sample" in pre_df.columns:
            merge_cols.append("sample")

        comp_df = pre_df.merge(cur_df, on=merge_cols, suffixes=("_pre", "_cur"))
        comp_df["source_name"] = source_name
        rows.append(comp_df)
    return pd.concat(rows, ignore_index=True)


comp_df = load_comparison_rows()

fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))

for ax, (model_key, col_name, symbol, title) in zip(axes, MODEL_INFO):
    x_col = f"{col_name}_pre"
    y_col = f"{col_name}_cur"

    model_df = comp_df.loc[:, ["trait", "source_name", x_col, y_col] + ([ "sample"] if "sample" in comp_df.columns else [])].copy()
    model_df = model_df.rename(columns={x_col: "original_I", y_col: "first_mode_I"})
    model_df = model_df[
        model_df["original_I"].notna()
        & model_df["first_mode_I"].notna()
        & (model_df["original_I"] > 0)
        & (model_df["first_mode_I"] > 0)
    ].copy()
    model_df["model"] = model_key

    label_df = model_df[
        (model_df["first_mode_I"].apply(math.log10) - model_df["original_I"].apply(math.log10)).abs()
        >= LOG10_SHIFT_TO_LABEL
    ].copy()

    finite_vals = pd.concat([model_df["original_I"], model_df["first_mode_I"]], ignore_index=True)
    vmin = finite_vals.min()
    vmax = finite_vals.max()

    ax.scatter(
        model_df["original_I"],
        model_df["first_mode_I"],
        s=34,
        color="0.45",
        alpha=0.6,
        linewidth=0,
        zorder=2,
    )

    ax.scatter(
        label_df["original_I"],
        label_df["first_mode_I"],
        s=74,
        color=HIGHLIGHT_COLOR,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    for _, row in label_df.iterrows():
        offset = LABEL_OFFSETS.get((model_key, row["source_name"], row["trait"]), (5, 5))
        ax.annotate(
            make_point_label(row),
            (row["original_I"], row["first_mode_I"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color=HIGHLIGHT_COLOR,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.2},
        )

    ax.plot([vmin, vmax], [vmin, vmax], ls="--", lw=1.2, color="0.25", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(vmin / 1.6, vmax * 3.2)
    ax.set_ylim(vmin / 1.6, vmax * 2.2)
    ax.set_xlabel(f"Pre-first-mode estimate ({symbol})", fontweight="bold")
    ax.set_ylabel(f"Current canonical estimate ({symbol})", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.text(
        0.03,
        0.97,
        f"{len(label_df)} labeled / {len(model_df)} total",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.95, "pad": 3.0},
    )

fig.tight_layout()
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

print(f"Wrote {OUTPUT_PDF}")
print(f"Wrote {OUTPUT_PNG}")
