import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.stats import chi2

import smilenfer.plotting as splot
import smilenfer.posterior as spost

splot._plot_params()
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

RESULTS_DIR = "results"
RESULTS_FILE = "ir_estimates_all.csv"
DROP_COUNTS = [0, 1, 2, 5]

MARKERS_BY_DROP = {0: "o", 1: "s", 2: "^", 5: "D"}
SIZES_BY_DROP = {0: 70, 1: 50, 2: 50, 5: 80}
ALPHAS_BY_DROP = {0: 0.95, 1: 0.7, 2: 0.7, 5: 0.95}

DOF_EXTRA_R = 1
NOMINAL_LL_THRESHOLD = chi2.ppf(0.95, DOF_EXTRA_R) / 2
R_REFERENCE_LINE = 2.0

TRAIT_GROUPS = {
    # Edit manually if a grouped color scheme is needed in comparison plots.
    # "bmi": "anthropometric",
    # "height": "anthropometric",
    # "hdl": "lipids",
    # "ldl": "lipids",
}

GROUP_COLORS = {
    "anthropometric": "tab:blue",
    "lipids": "tab:orange",
    "blood": "tab:green",
    "autoimmune": "tab:red",
    "other": "0.5",
}

PROFESSIONAL_TRAIT_NAMES = spost.original_trait_names
TRAIT_CODES = {
    "arthrosis": "AR",
    "asthma": "AS",
    "bc": "BC",
    "bmi": "BM",
    "cad": "CD",
    "dbp": "DB",
    "diverticulitis": "DV",
    "fvc": "FV",
    "gallstones": "GA",
    "glaucoma": "GL",
    "grip_strength": "GS",
    "hdl": "HD",
    "height": "HT",
    "hypothyroidism": "HY",
    "ibd": "IB",
    "ldl": "LD",
    "malignant_neoplasms": "MN",
    "pulse_rate": "PR",
    "rbc": "RC",
    "sbp": "SB",
    "scz": "SZ",
    "t2d": "T2",
    "triglycerides": "TG",
    "urate": "UR",
    "uterine_fibroids": "UF",
    "varicose_veins": "VV",
    "wbc": "WC",
}

def make_trait_code(trait):
    if trait in TRAIT_CODES:
        return TRAIT_CODES[trait]
    cleaned = "".join(ch for ch in trait if ch.isalnum()).upper()
    if len(cleaned) >= 2:
        return cleaned[:2]
    if len(cleaned) == 1:
        return cleaned * 2
    return "??"


def make_trait_color_map(traits):
    traits = sorted(traits)
    candidate_colors = []
    for cmap_name in ["tab10", "tab20", "tab20b", "tab20c"]:
        cmap = plt.cm.get_cmap(cmap_name)
        for ii in range(cmap.N):
            candidate_colors.append(cmap(ii))

    filtered_colors = []
    seen = set()
    for rgba in candidate_colors:
        rr, gg, bb, aa = rgba
        luminance = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
        if luminance < 0.18 or luminance > 0.9:
            continue
        color_key = (round(rr, 6), round(gg, 6), round(bb, 6), round(aa, 6))
        if color_key in seen:
            continue
        seen.add(color_key)
        filtered_colors.append(rgba)

    chosen_colors = []
    if filtered_colors:
        chosen_colors.append(filtered_colors[0])
        while len(chosen_colors) < len(traits) and len(chosen_colors) < len(filtered_colors):
            best_color = None
            best_dist = -1
            for color in filtered_colors:
                if color in chosen_colors:
                    continue
                rr, gg, bb, _ = color
                min_dist = min(
                    (rr - rr0) ** 2 + (gg - gg0) ** 2 + (bb - bb0) ** 2
                    for rr0, gg0, bb0, _ in chosen_colors
                )
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_color = color
            if best_color is None:
                break
            chosen_colors.append(best_color)

    if len(chosen_colors) < len(traits):
        cmap = plt.cm.get_cmap("tab20")
        grid = np.linspace(0, 1, len(traits), endpoint=False)
        chosen_colors = [cmap(x) for x in grid]

    return {trait: chosen_colors[ii % len(chosen_colors)] for ii, trait in enumerate(traits)}


def marker_code_fontsize(marker_size):
    return max(5, min(7, (marker_size ** 0.5) * 0.25))


def get_trait_group_color(trait):
    trait_group = TRAIT_GROUPS.get(trait, "other")
    return GROUP_COLORS.get(trait_group, "0.5")


def load_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, RESULTS_DIR, RESULTS_FILE)
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Missing results: {results_path}")

    ir_fits = pd.read_csv(results_path)
    ir_fits = ir_fits[ir_fits["drop_count"].isin(DROP_COUNTS)].copy()
    ir_fits["x_1d"] = ir_fits["Ir_LL"] - ir_fits["I2_LL"]
    ir_fits["x_pleio"] = ir_fits["Ipr_LL"] - ir_fits["Ip_LL"]
    return ir_fits


def get_axis_limits(ir_fits):
    x_values = ir_fits[["x_1d", "x_pleio"]].to_numpy().ravel()
    y_values = ir_fits[["Ir_r", "Ipr_r"]].to_numpy().ravel()
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]

    if x_values.size == 0 or y_values.size == 0:
        raise ValueError("No finite values found for plotting.")

    xmin = min(-0.5, x_values.min())
    xmax = x_values.max() * 1.3
    ymax = y_values.max() * 1.2

    if xmin == xmax:
        xpad = 1.0 if xmin == 0 else abs(xmin) * 0.5
        xmin -= xpad
        xmax += xpad

    if not np.isfinite(ymax) or ymax <= 0:
        ymax = max(1.0, np.nanmax(y_values)) * 1.2

    return xmin, xmax, ymax


def format_main_axis(ax, xmin, xmax, ymax, ylabel=None, title=None):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.5, 1, 2, 4])
    ax.set_yticklabels([0, 0.5, 1, 2, 4])
    ax.axvline(1, color="0.6", linestyle="--", linewidth=0.6, zorder=0)
    ax.axhline(R_REFERENCE_LINE, color="0.35", linewidth=1.1, linestyle="-", zorder=1.5)
    ax.axvline(NOMINAL_LL_THRESHOLD, color="0.35", linewidth=1.1, linestyle="-", zorder=1.5)
    ax.text(
        0.14,
        0.02,
        "no evidence\nfor alt. scaling",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.35",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6),
    )
    ax.text(
        0.86,
        0.02,
        "evidence\nfor alt. scaling",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.35",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6),
    )
    ax.text(
        0.98,
        R_REFERENCE_LINE + 0.02,
        "stabilizing scaling",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    ax.tick_params(axis="both", labelsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)


def add_trait_paths(ax, ir_fits, x_col, y_col, trait_colors, trait_codes, label_drop_five):
    for trait in sorted(ir_fits["trait"].unique()):
        trait_rows = ir_fits[ir_fits["trait"] == trait].set_index("drop_count")
        ordered = trait_rows.reindex(DROP_COUNTS).dropna(subset=[x_col, y_col])
        if len(ordered) < 2:
            continue

        ax.plot(
            ordered[x_col].values,
            ordered[y_col].values,
            color=trait_colors[trait],
            alpha=0.5,
            linewidth=0.8,
            zorder=2,
        )

        if 0 in ordered.index and 5 in ordered.index:
            ax.annotate(
                "",
                xy=(ordered.loc[5, x_col], ordered.loc[5, y_col]),
                xytext=(ordered.loc[0, x_col], ordered.loc[0, y_col]),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.0,
                    color=trait_colors[trait],
                    alpha=0.7,
                ),
                zorder=2.5,
            )

        for drop_count, row in ordered.iterrows():
            marker = MARKERS_BY_DROP.get(drop_count, "o")
            marker_size = SIZES_BY_DROP.get(drop_count, 50)
            alpha_value = ALPHAS_BY_DROP.get(drop_count, 0.8)
            edgecolor = "black" if drop_count in (0, 5) else "none"
            linewidth = 0.4 if edgecolor != "none" else 0.0

            ax.scatter(
                row[x_col],
                row[y_col],
                s=marker_size,
                marker=marker,
                color=trait_colors[trait],
                alpha=alpha_value,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=3,
            )

            if label_drop_five and drop_count == 5:
                ax.text(
                    row[x_col],
                    row[y_col],
                    trait_codes[trait],
                    ha="center",
                    va="center",
                    fontsize=marker_code_fontsize(marker_size),
                    fontweight="bold",
                    color="white",
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.1, foreground="black")],
                )


def add_drop_count_legend(ax, location):
    handles = []
    for drop_count in DROP_COUNTS:
        marker = MARKERS_BY_DROP.get(drop_count, "o")
        marker_size = SIZES_BY_DROP.get(drop_count, 50)
        edgecolor = "black" if drop_count in (0, 5) else "none"
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="",
                markersize=max(4, (marker_size ** 0.5) / 2.0),
                markerfacecolor="0.7",
                markeredgecolor=edgecolor,
                label=f"{drop_count} loci",
            )
        )
    legend = ax.legend(
        handles=handles,
        title="Outliers dropped",
        loc=location,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(legend)


def add_trait_legend(ax, traits, trait_colors):
    ax.axis("off")
    ncol = 3 if len(traits) > 20 else 2
    handles = [
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
            label=f"{spost.original_trait_names.get(trait, trait)} ({make_trait_code(trait)})",
        )
        for trait in traits
    ]
    ax.legend(
        handles=handles,
        title="Traits",
        loc="upper left",
        bbox_to_anchor=(-0.1, 1.2),
        frameon=False,
        fontsize=7,
        title_fontsize=9,
        ncol=ncol,
        labelspacing=0.7,
    )


def plot_r_histogram(ax, ir_fits, r_col):
    r_0 = ir_fits.loc[ir_fits["drop_count"] == 0, r_col].to_numpy()
    r_5 = ir_fits.loc[ir_fits["drop_count"] == 5, r_col].to_numpy()
    all_r = np.concatenate([r_0, r_5])
    all_r = all_r[np.isfinite(all_r)]

    r_min = 0.0
    r_max = np.nanmax(all_r) if all_r.size else 1.0
    if not np.isfinite(r_max) or r_max <= 0:
        r_max = 1.0
    bins_r = np.linspace(r_min, r_max, 9)

    n_5, _, _ = ax.hist(
        r_5,
        bins=bins_r,
        color="tab:blue",
        alpha=0.4,
        edgecolor="darkblue",
        linewidth=1.6,
        label="5 dropped",
    )
    n_0, _, _ = ax.hist(
        r_0,
        bins=bins_r,
        color="tab:orange",
        alpha=0.4,
        edgecolor="chocolate",
        linewidth=1.6,
        label="0 dropped",
    )

    xticks = np.linspace(r_min, r_max, 4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick:.2g}" for tick in xticks], fontsize=9)
    ax.set_xlabel(r"$\hat{r}$", fontsize=12)
    ax.set_ylabel("Trait count", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_count = max(np.max(n_0) if len(n_0) else 0, np.max(n_5) if len(n_5) else 0)
    ymax = max_count * 1.2 if max_count > 0 else 1.0
    ax.set_ylim(0, ymax)


def plot_paths_traitcolors_hists(ir_fits, xmin, xmax, ymax, out_path):
    traits = sorted(ir_fits["trait"].unique())
    trait_codes = {trait: make_trait_code(trait) for trait in traits}
    trait_colors = make_trait_color_map(traits)

    fig = plt.figure(figsize=(10.8, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.6, 3.4])
    ax_scatter = fig.add_subplot(outer[0, 0])
    right_gs = outer[0, 1].subgridspec(3, 1, height_ratios=[0.8, 0.15, 1.15])
    ax_legend = fig.add_subplot(right_gs[0, 0])
    ax_hist0 = fig.add_subplot(right_gs[1, 0])
    ax_hist5 = fig.add_subplot(right_gs[2, 0])

    add_trait_paths(ax_scatter, ir_fits, "x_1d", "Ir_r", trait_colors, trait_codes, label_drop_five=True)
    format_main_axis(
        ax_scatter,
        xmin,
        xmax,
        ymax,
        ylabel=r"$\hat{r}$ (effect size / selection scaling: $s=|\beta|^r$)",
        title="Single-trait stabilizing",
    )

    add_drop_count_legend(ax_scatter, "upper right")
    add_trait_legend(ax_legend, traits, trait_colors)

    ax_hist0.axis("off")
    plot_r_histogram(ax_hist5, ir_fits, "Ir_r")

    ax_scatter.text(-0.08, 1.02, "A", transform=ax_scatter.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")
    ax_hist5.text(-0.12, 1.02, "B", transform=ax_hist5.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")

    fig.tight_layout(rect=[0.01, 0.05, 0.98, 1])
    for ax in (ax_hist0, ax_hist5):
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
    fig.savefig(out_path, bbox_inches="tight")


def plot_paths_traitcolors_hists_pleioonly(ir_fits, xmin, xmax, ymax, out_path):
    traits = sorted(ir_fits["trait"].unique())
    trait_codes = {trait: make_trait_code(trait) for trait in traits}
    trait_colors = make_trait_color_map(traits)

    fig = plt.figure(figsize=(10.8, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.6, 3.4])
    ax_scatter = fig.add_subplot(outer[0, 0])
    right_gs = outer[0, 1].subgridspec(3, 1, height_ratios=[0.8, 0.15, 1.15])
    ax_legend = fig.add_subplot(right_gs[0, 0])
    ax_hist0 = fig.add_subplot(right_gs[1, 0])
    ax_hist5 = fig.add_subplot(right_gs[2, 0])

    add_trait_paths(ax_scatter, ir_fits, "x_pleio", "Ipr_r", trait_colors, trait_codes, label_drop_five=True)
    format_main_axis(
        ax_scatter,
        xmin,
        xmax,
        ymax,
        ylabel=r"$\hat{r}$ (effect size versus $s$ scaling: $|\beta|^r$)",
        title="Pleiotropic stabilizing",
    )

    add_drop_count_legend(ax_scatter, "upper right")
    add_trait_legend(ax_legend, traits, trait_colors)

    ax_hist0.axis("off")
    plot_r_histogram(ax_hist5, ir_fits, "Ipr_r")

    ax_scatter.text(-0.08, 1.02, "A", transform=ax_scatter.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")
    ax_hist5.text(-0.12, 1.02, "B", transform=ax_hist5.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")

    fig.tight_layout(rect=[0.01, 0.05, 0.98, 1])
    for ax in (ax_hist0, ax_hist5):
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
    fig.savefig(out_path, bbox_inches="tight")


def main():
    ir_fits = load_results()
    xmin, xmax, ymax = get_axis_limits(ir_fits)

    plot_paths_traitcolors_hists(
        ir_fits,
        xmin,
        xmax,
        ymax,
        "single_trait_r_scaling_pleiotropic_outliers.pdf",
    )
    plot_paths_traitcolors_hists_pleioonly(
        ir_fits,
        xmin,
        xmax,
        ymax,
        "figure_5.pdf",
    )


if __name__ == "__main__":
    main()
