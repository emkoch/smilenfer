import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import plot_ir_dropcount as ir_plot


matplotlib.rcParams.update(
    {
        "font.size": 16,
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
FIGURE_INPUTS = [
    (
        os.path.join(SCRIPT_DIR, "underdominance", "posterior_mean", "stab_ud_std_results.csv"),
        os.path.join(SCRIPT_DIR, "ir", "posterior_mean", "results", "ir_estimates_all.csv"),
        os.path.join(SCRIPT_DIR, "figure_4_with_ud_panel_posterior_mean.pdf"),
    ),
    (
        os.path.join(SCRIPT_DIR, "underdominance", "ml_shrink", "stab_ud_std_results.csv"),
        os.path.join(SCRIPT_DIR, "ir", "ml_shrink", "results", "ir_estimates_all.csv"),
        os.path.join(SCRIPT_DIR, "figure_4_with_ud_panel_ml_shrink.pdf"),
    ),
]
SINGLE_TRAIT_IR_INPUT = os.path.join(SCRIPT_DIR, "ir", "posterior_mean", "results", "ir_estimates_all.csv")
SINGLE_TRAIT_IR_OUTPUT = os.path.join(SCRIPT_DIR, "single_trait_r_scaling_pleiotropic_outliers.pdf")
RR0_IR_INPUT = os.path.join(SCRIPT_DIR, "results_rr0", "ir_rr0_estimates_all.csv")
RR0_IR_OUTPUT = os.path.join(SCRIPT_DIR, "figure_4_rr0.pdf")


def make_figure(ud_input, ir_input, output):
    ud_df = pd.read_csv(ud_input)
    ud_df = ud_df.rename(columns={"Trait": "trait"}).copy()
    ud_df["x_ud"] = ud_df["ll_Ip_ud"] - ud_df["ll_neut"]
    ud_df["y_ud"] = ud_df["ll_Ip_ud"] - ud_df["ll_Ip_std"]

    ir_fits = pd.read_csv(ir_input)
    ir_fits = ir_fits[ir_fits["drop_count"].isin(ir_plot.DROP_COUNTS)].copy()
    ir_fits["x_1d"] = ir_fits["Ir_LL"] - ir_fits["I2_LL"]
    ir_fits["x_pleio"] = ir_fits["Ipr_LL"] - ir_fits["Ip_LL"]
    ir_panel = ir_fits[ir_fits["drop_count"].isin([0, 5])].copy()

    traits = sorted(ir_fits["trait"].unique())
    trait_codes = {trait: ir_plot.make_trait_code(trait) for trait in traits}
    trait_colors = ir_plot.make_trait_color_map(traits)

    ir_xmin, ir_xmax, ir_ymax = ir_plot.get_axis_limits(ir_panel)
    fig = plt.figure(figsize=(15.8, 5.4))
    outer = fig.add_gridspec(1, 3, width_ratios=[2.9, 3.6, 3.3], wspace=0.28)

    ax_ud = fig.add_subplot(outer[0, 0])
    ax_scatter = fig.add_subplot(outer[0, 1])
    right_gs = outer[0, 2].subgridspec(3, 1, height_ratios=[0.78, 0.12, 1.10])
    ax_legend = fig.add_subplot(right_gs[0, 0])
    ax_hist_pad = fig.add_subplot(right_gs[1, 0])
    ax_hist = fig.add_subplot(right_gs[2, 0])

    for _, row in ud_df.iterrows():
        ax_ud.scatter(
            row["x_ud"],
            row["y_ud"],
            s=140,
            color=trait_colors[row["trait"]],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax_ud.text(
            row["x_ud"],
            row["y_ud"],
            trait_codes[row["trait"]],
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="white",
            zorder=4,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="black")],
        )
    ax_ud.set_xscale("symlog", linthresh=5)
    ax_ud.set_yscale("symlog", linthresh=5)
    ax_ud.set_xticks([0, 10, 100, 1000])
    ax_ud.set_yticks([-10, -1, 0, 1, 10])
    ax_ud.axvline(0, color="0.2", linewidth=1.0, linestyle="--", zorder=1)
    ax_ud.axhline(0, color="0.2", linewidth=1.0, linestyle="--", zorder=1)
    ax_ud.tick_params(axis="both", labelsize=12)
    ax_ud.set_xlabel("Log-likelihood difference\n(ud model − neutral)", fontsize=11)
    ax_ud.set_ylabel("Log-likelihood difference\n(ud model − additive stabilizing)", fontsize=11)
    ax_ud.set_title("Underdominance test (pleiotropic)", fontsize=12, pad=8)

    xlim = ax_ud.get_xlim()
    ax_ud.set_xlim(
        xlim[0] * 1.12 if xlim[0] < 0 else xlim[0] * 0.9,
        xlim[1] * 1.18 if xlim[1] > 0 else xlim[1] * 0.9,
    )
    offsets = []
    for coll in ax_ud.collections:
        coll_offsets = getattr(coll, "get_offsets", lambda: None)()
        if coll_offsets is not None and len(coll_offsets):
            offsets.append(np.asarray(coll_offsets))
    if offsets:
        yvals = np.concatenate(offsets, axis=0)[:, 1]
    else:
        yvals = np.array([0.0])
    ymax = np.nanmax(np.abs(yvals))
    ymax = 1.18 * ymax if np.isfinite(ymax) and ymax > 0 else 1.0
    ax_ud.set_ylim(-ymax, ymax)

    ax_ud.set_xscale("symlog", linthresh=5)
    ax_ud.set_yscale("symlog", linthresh=5)
    ax_ud.set_xticks([0, 10, 100, 1000])
    ax_ud.set_yticks([-10, -1, 0, 1, 10])
    ax_ud.axvline(0, color="0.2", linewidth=1.0, linestyle="--", zorder=1)
    ax_ud.axhline(0, color="0.2", linewidth=1.0, linestyle="--", zorder=1)
    ax_ud.tick_params(axis="both", labelsize=12)
    ax_ud.set_xlabel("Log-likelihood difference\n(ud model − neutral)", fontsize=11)
    ax_ud.set_ylabel("Log-likelihood difference\n(ud model − additive stabilizing)", fontsize=11)
    ax_ud.set_title("Underdominance test (pleiotropic)", fontsize=12, pad=8)

    for trait in sorted(ir_panel["trait"].unique()):
        trait_rows = ir_panel[ir_panel["trait"] == trait].set_index("drop_count")
        ordered = trait_rows.reindex([0, 5]).dropna(subset=["x_pleio", "Ipr_r"])
        if len(ordered) < 2:
            continue
        ax_scatter.plot(
            ordered["x_pleio"].values,
            ordered["Ipr_r"].values,
            color=trait_colors[trait],
            alpha=0.5,
            linewidth=0.8,
            zorder=2,
        )
        ax_scatter.annotate(
            "",
            xy=(ordered.loc[5, "x_pleio"], ordered.loc[5, "Ipr_r"]),
            xytext=(ordered.loc[0, "x_pleio"], ordered.loc[0, "Ipr_r"]),
            arrowprops=dict(arrowstyle="->", lw=1.0, color=trait_colors[trait], alpha=0.7),
            zorder=2.5,
        )
        for drop_count, row in ordered.iterrows():
            marker = "o" if drop_count == 0 else "D"
            marker_size = 120 if drop_count == 0 else 92
            alpha_value = 0.98 if drop_count == 0 else 0.82
            ax_scatter.scatter(
                row["x_pleio"],
                row["Ipr_r"],
                s=marker_size,
                marker=marker,
                color=trait_colors[trait],
                alpha=alpha_value,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )

    for _, row in ir_panel[ir_panel["drop_count"] == 0].iterrows():
        ax_scatter.text(
            row["x_pleio"],
            row["Ipr_r"],
            trait_codes[row["trait"]],
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="white",
            zorder=4.2,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="black")],
        )

    ir_plot.format_main_axis(
        ax_scatter,
        ir_xmin,
        ir_xmax,
        ir_ymax,
        ylabel=r"$\hat{r}$ (effect size versus $s$ scaling: $|\beta|^r$)",
        title="Effect-size scaling test (pleiotropic)",
    )
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5.5, markerfacecolor="0.7", markeredgecolor="black", label="0 loci"),
        Line2D([], [], marker="D", linestyle="", markersize=5.0, markerfacecolor="0.7", markeredgecolor="black", label="5 loci"),
    ]
    legend = ax_scatter.legend(
        handles=handles,
        title="Outliers dropped",
        loc="upper right",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax_scatter.add_artist(legend)

    ax_legend.axis("off")
    handles = []
    for trait in traits:
        trait_name = ir_plot.PROFESSIONAL_TRAIT_NAMES.get(trait, trait)
        if output.endswith("posterior_mean.pdf") and trait == "dbp":
            trait_name = "DBP"
        handles.append(
            Line2D(
                [],
                [],
                color=trait_colors[trait],
                linestyle="-",
                linewidth=2.0,
                marker="s",
                markersize=6,
                markerfacecolor=trait_colors[trait],
                markeredgecolor="none",
                label=f"{trait_name} ({trait_codes[trait]})",
            )
        )
    ax_legend.legend(
        handles=handles,
        title="Traits",
        loc="upper left",
        bbox_to_anchor=(-0.1, 1.2),
        frameon=False,
        fontsize=7,
        title_fontsize=9,
        ncol=3 if len(traits) > 20 else 2,
        labelspacing=0.7,
    )

    ax_hist_pad.axis("off")
    ir_plot.plot_r_histogram(ax_hist, ir_fits, "Ipr_r")

    ax_ud.text(-0.12, 1.02, "A", transform=ax_ud.transAxes, fontsize=16, fontweight="bold", ha="right", va="bottom")
    ax_scatter.text(-0.08, 1.02, "B", transform=ax_scatter.transAxes, fontsize=16, fontweight="bold", ha="right", va="bottom")
    ax_hist.text(-0.12, 1.02, "C", transform=ax_hist.transAxes, fontsize=16, fontweight="bold", ha="right", va="bottom")
    fig.tight_layout(rect=[0.01, 0.05, 0.98, 0.965])
    for ax in (ax_hist_pad, ax_hist):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.02, pos.width, pos.height])

    pos = ax_scatter.get_position()
    fig.text(
        pos.x0 + pos.width / 2,
        max(0.01, pos.y0 - 0.08),
        r"Log-likelihood difference ($|\beta|^r$ model − standard $|\beta|^2$)",
        ha="center",
        fontsize=12,
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_ir_figure(ir_input, x_col, y_col, hist_col, title, ylabel, output):
    ir_fits = pd.read_csv(ir_input)
    ir_fits = ir_fits[ir_fits["drop_count"].isin(ir_plot.DROP_COUNTS)].copy()
    ir_fits["x_1d"] = ir_fits["Ir_LL"] - ir_fits["I2_LL"]
    ir_fits["x_pleio"] = ir_fits["Ipr_LL"] - ir_fits["Ip_LL"]
    ir_panel = ir_fits[ir_fits["drop_count"].isin([0, 5])].copy()

    traits = sorted(ir_fits["trait"].unique())
    trait_codes = {trait: ir_plot.make_trait_code(trait) for trait in traits}
    trait_colors = ir_plot.make_trait_color_map(traits)
    ir_xmin, ir_xmax, ir_ymax = ir_plot.get_axis_limits(ir_panel)

    fig = plt.figure(figsize=(10.8, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.6, 3.4])
    ax_scatter = fig.add_subplot(outer[0, 0])
    right_gs = outer[0, 1].subgridspec(3, 1, height_ratios=[0.78, 0.12, 1.10])
    ax_legend = fig.add_subplot(right_gs[0, 0])
    ax_hist_pad = fig.add_subplot(right_gs[1, 0])
    ax_hist = fig.add_subplot(right_gs[2, 0])

    for trait in sorted(ir_panel["trait"].unique()):
        trait_rows = ir_panel[ir_panel["trait"] == trait].set_index("drop_count")
        ordered = trait_rows.reindex([0, 5]).dropna(subset=[x_col, y_col])
        if len(ordered) < 2:
            continue
        ax_scatter.plot(
            ordered[x_col].values,
            ordered[y_col].values,
            color=trait_colors[trait],
            alpha=0.5,
            linewidth=0.8,
            zorder=2,
        )
        ax_scatter.annotate(
            "",
            xy=(ordered.loc[5, x_col], ordered.loc[5, y_col]),
            xytext=(ordered.loc[0, x_col], ordered.loc[0, y_col]),
            arrowprops=dict(arrowstyle="->", lw=1.0, color=trait_colors[trait], alpha=0.7),
            zorder=2.5,
        )
        for drop_count, row in ordered.iterrows():
            marker = "o" if drop_count == 0 else "D"
            marker_size = 120 if drop_count == 0 else 92
            alpha_value = 0.98 if drop_count == 0 else 0.82
            ax_scatter.scatter(
                row[x_col],
                row[y_col],
                s=marker_size,
                marker=marker,
                color=trait_colors[trait],
                alpha=alpha_value,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )

    for _, row in ir_panel[ir_panel["drop_count"] == 0].iterrows():
        ax_scatter.text(
            row[x_col],
            row[y_col],
            trait_codes[row["trait"]],
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="white",
            zorder=4.2,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="black")],
        )

    ir_plot.format_main_axis(
        ax_scatter,
        ir_xmin,
        ir_xmax,
        ir_ymax,
        ylabel=ylabel,
        title=title,
    )
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5.5, markerfacecolor="0.7", markeredgecolor="black", label="0 loci"),
        Line2D([], [], marker="D", linestyle="", markersize=5.0, markerfacecolor="0.7", markeredgecolor="black", label="5 loci"),
    ]
    legend = ax_scatter.legend(
        handles=handles,
        title="Outliers dropped",
        loc="upper right",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax_scatter.add_artist(legend)

    ax_legend.axis("off")
    handles = []
    for trait in traits:
        trait_name = ir_plot.PROFESSIONAL_TRAIT_NAMES.get(trait, trait)
        handles.append(
            Line2D(
                [],
                [],
                color=trait_colors[trait],
                linestyle="-",
                linewidth=2.0,
                marker="s",
                markersize=6,
                markerfacecolor=trait_colors[trait],
                markeredgecolor="none",
                label=f"{trait_name} ({trait_codes[trait]})",
            )
        )
    ax_legend.legend(
        handles=handles,
        title="Traits",
        loc="upper left",
        bbox_to_anchor=(-0.1, 1.2),
        frameon=False,
        fontsize=7,
        title_fontsize=9,
        ncol=3 if len(traits) > 20 else 2,
        labelspacing=0.7,
    )

    ax_hist_pad.axis("off")
    ir_plot.plot_r_histogram(ax_hist, ir_fits, hist_col)

    ax_scatter.text(-0.08, 1.02, "A", transform=ax_scatter.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")
    ax_hist.text(-0.12, 1.02, "B", transform=ax_hist.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")
    fig.tight_layout(rect=[0.01, 0.05, 0.98, 1])
    for ax in (ax_hist_pad, ax_hist):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.02, pos.width, pos.height])

    pos = ax_scatter.get_position()
    fig.text(
        pos.x0 + pos.width / 2,
        max(0.01, pos.y0 - 0.08),
        r"Log-likelihood difference ($|\beta|^r$ model − standard $|\beta|^2$)",
        ha="center",
        fontsize=12,
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


for ud_input, ir_input, output in FIGURE_INPUTS:
    make_figure(ud_input, ir_input, output)

make_ir_figure(
    SINGLE_TRAIT_IR_INPUT,
    "x_1d",
    "Ir_r",
    "Ir_r",
    "Effect-size scaling test (single-trait)",
    r"$\hat{r}$ (effect size versus $s$ scaling: $|\beta|^r$)",
    SINGLE_TRAIT_IR_OUTPUT,
)
make_ir_figure(
    RR0_IR_INPUT,
    "x_pleio",
    "Ipr_r",
    "Ipr_r",
    "Effect-size scaling test (pleiotropic)",
    r"$\hat{r}$ (effect size versus $s$ scaling: $|\beta|^r$)",
    RR0_IR_OUTPUT,
)
