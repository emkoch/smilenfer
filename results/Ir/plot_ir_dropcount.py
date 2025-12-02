import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None
import smilenfer.plotting as splot

RESULTS_DIR = "results"          # matched to Snakefile_Ir
DROP_COUNTS = [0, 1, 2, 5]       # matched to Snakefile_Ir

DOF_EXTRA_R = 1  # r-model adds one parameter (r) relative to the standard model
if chi2 is not None:
    NOMINAL_LL_THRESHOLD = 0.5 * chi2.ppf(0.95, DOF_EXTRA_R)
else:
    NOMINAL_LL_THRESHOLD = 0.5 * 3.841458820694124  # chi2(df=1, 0.95)
R_REFERENCE_LINE = 2.0

TRAIT_GROUPS = {
    # Example only; edit manually as needed
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

# Matches the original_trait_names mapping in smilenfer/posterior.py (short professional labels).
PROFESSIONAL_TRAIT_NAMES = {
    # Quantitative traits
    "height": "Standing height",
    "bmi": "BMI",
    "ldl": "LDL levels",
    "hdl": "HDL levels",
    "dbp": "Diastolic BP",
    "sbp": "Systolic BP",
    "triglycerides": "Triglycerides",
    "urate": "Urate",
    "rbc": "RBC",
    "wbc": "WBC",
    "grip_strength": "Grip strength",
    "fvc": "FVC",
    "pulse_rate": "Pulse rate",

    # Disease traits
    "bc": "Breast cancer",
    "cad": "CAD",
    "ibd": "IBD",
    "scz": "SCZ",
    "t2d": "T2D",
    "arthrosis": "Arthrosis",
    "asthma": "Asthma",
    "diverticulitis": "Diverticulitis",
    "gallstones": "Gallstones",
    "glaucoma": "Glaucoma",
    "hypothyroidism": "Hypothyroidism",
    "malignant_neoplasms": "Malignant neoplasms",
    "uterine_fibroids": "Uterine fibroids",
    "varicose_veins": "Varicose veins",
}

def make_trait_code(trait):
    """Return a two-character code for the trait."""
    cleaned = "".join(ch for ch in trait if ch.isalnum()).upper()
    if len(cleaned) >= 2:
        return cleaned[:2]
    if len(cleaned) == 1:
        return cleaned * 2
    return "??"


def make_trait_color_map(traits):
    """Return a stable, well-separated color map for the given traits."""
    traits = sorted(traits)
    palettes = [
        plt.cm.get_cmap("tab10"),
        plt.cm.get_cmap("tab20"),
        plt.cm.get_cmap("tab20b"),
        plt.cm.get_cmap("tab20c"),
    ]
    candidate_colors = []
    for cmap in palettes:
        for i in range(cmap.N):
            candidate_colors.append(cmap(i))

    # Filter by luminance and remove duplicates
    filtered = []
    seen = set()
    for rgba in candidate_colors:
        r, g, b, a = rgba
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum < 0.18 or lum > 0.9:
            continue
        key = (round(r, 6), round(g, 6), round(b, 6), round(a, 6))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(rgba)

    def min_sq_dist(color, chosen):
        r, g, b, _ = color
        dists = [(r - rc) ** 2 + (g - gc) ** 2 + (b - bc) ** 2 for rc, gc, bc, _ in chosen]
        return min(dists) if dists else float("inf")

    chosen = []
    if filtered:
        chosen.append(filtered[0])
        while len(chosen) < len(traits) and len(chosen) < len(filtered):
            best = None
            best_dist = -1
            for cand in filtered:
                if cand in chosen:
                    continue
                d = min_sq_dist(cand, chosen)
                if d > best_dist:
                    best_dist = d
                    best = cand
            if best is None:
                break
            chosen.append(best)

    if len(chosen) < len(traits):
        cmap = plt.cm.get_cmap("tab20")
        positions = np.linspace(0, 1, len(traits), endpoint=False)
        chosen = [cmap(p) for p in positions]

    return {trait: chosen[i % len(chosen)] for i, trait in enumerate(traits)}


def marker_code_fontsize(marker_size):
    """Choose a small font size that fits within a diamond marker."""
    # Marker size is in points^2; scale sublinearly and clamp
    return max(5, min(7, (marker_size ** 0.5) * 0.25))


def get_trait_group_color(trait):
    group = TRAIT_GROUPS.get(trait, "other")
    return GROUP_COLORS.get(group, "0.5")

LABELLED_TRAITS = None

def load_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, RESULTS_DIR, "ir_estimates_all.csv")
    if not os.path.isfile(results_file):
        raise FileNotFoundError(f"Missing results: {results_file}")

    df = pd.read_csv(results_file)
    df = df[df["drop_count"].isin(DROP_COUNTS)].copy()
    df["x_1d"] = df["Ir_LL"] - df["I2_LL"]
    df["x_pleio"] = df["Ipr_LL"] - df["Ip_LL"]
    return df, results_file


def get_axis_limits(df):
    x_vals = df[["x_1d", "x_pleio"]].to_numpy().ravel()
    y_vals = df[["Ir_r", "Ipr_r"]].to_numpy().ravel()

    x_vals = x_vals[np.isfinite(x_vals)]
    y_vals = y_vals[np.isfinite(y_vals)]

    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("No finite values found for plotting.")

    xmin = min(-0.5, x_vals.min())
    xmax = x_vals.max() * 1.3
    ymax = y_vals.max() * 1.2

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("Invalid x limits computed for plotting.")
    if xmin == xmax:
        pad = 1.0 if xmin == 0 else abs(xmin) * 0.5
        xmin -= pad
        xmax += pad

    if not np.isfinite(ymax) or ymax <= 0:
        ymax = max(1.0, np.nanmax(y_vals)) * 1.2

    return xmin, xmax, ymax


def format_axes(ax, xmin, xmax, ymax):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=2)
    ax.axvline(1, color="0.6", linestyle="--", linewidth=0.6, zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.1, ymax)
    ax.set_yticks([0, 0.5, 1, 2, 4])
    ax.set_yticklabels([0, 0.5, 1, 2, 4])


"""Unused plotting utilities (kept for reference, not executed).

def plot_by_dropcount(ir_fits, drop_counts, xmin, xmax, ymax, out_dir):
    ncols = 2 if len(drop_counts) > 1 else 1
    nrows = math.ceil(len(drop_counts) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 5.5 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, dc in enumerate(drop_counts):
        ax = axes[idx]
        sub = ir_fits[ir_fits.drop_count == dc]

        ax.scatter(
            sub["x_1d"],
            sub["Ir_r"],
            color="blue",
            alpha=0.5,
            s=40,
            label="1-dimensional" if idx == 0 else None,
        )
        ax.scatter(
            sub["x_pleio"],
            sub["Ipr_r"],
            color="red",
            alpha=0.5,
            s=40,
            label="pleiotropic" if idx == 0 else None,
        )

        for _, row in sub.iterrows():
            ax.annotate(row.trait, (row["x_1d"], row["Ir_r"]), fontsize=9)
            ax.annotate(row.trait, (row["x_pleio"], row["Ipr_r"]), fontsize=9)

        format_axes(ax, xmin, xmax, ymax)
        if idx < len(drop_counts) - ncols:
            ax.set_xticklabels([])
        ax.set_title(f"drop_count = {dc}", pad=4, fontsize=11)

    for ax in axes[len(drop_counts):]:
        ax.set_visible(False)

    fig.supxlabel(r"LLhood difference $I_r\beta^r$ model − standard", y=0.04)
    fig.supylabel(r"$\hat{r}$ (exponent in $I_r\times\beta^r$)", x=0.04)
    fig.legend(loc="upper right", frameon=False)
    fig.tight_layout(rect=[0.05, 0.05, 0.94, 0.97])
    fig.savefig(os.path.join(out_dir, "ir_vs_irpleio_facet_by_dropcount.pdf"), bbox_inches="tight")


def draw_path(ax, df, x_col, y_col, cmap, norm, marker, annotate_drop):
    start_dc = DROP_COUNTS[0]
    end_dc = DROP_COUNTS[-1]

    # Draw arrows and points trait by trait
    for trait, group in df.groupby("trait"):
        ordered = (
            group.set_index("drop_count")
            .reindex(DROP_COUNTS)
            .dropna(subset=[x_col, y_col])
        )
        if len(ordered) < 2:
            continue

        line_color = get_trait_group_color(trait)

        # Polyline through available drop_counts
        ax.plot(
            ordered[x_col].values,
            ordered[y_col].values,
            color=line_color,
            alpha=0.5,
            linewidth=0.8,
            zorder=2,
        )

        # Single arrow from start -> end if both present
        if start_dc in ordered.index and end_dc in ordered.index:
            x_start, y_start = ordered.loc[start_dc, [x_col, y_col]]
            x_end, y_end = ordered.loc[end_dc, [x_col, y_col]]
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.1,
                    color=line_color,
                    alpha=0.8,
                ),
                zorder=2.5,
            )

        # Scatter points for each drop_count of this trait
        for dc, row in ordered.iterrows():
            color = cmap(norm(dc))
            if dc == start_dc or dc == end_dc:
                size = 80 if annotate_drop else 60
                edgecolor = "black"
                lw = 0.5
                alpha = 0.9
            else:
                size = 35
                edgecolor = "none"
                lw = 0.0
                alpha = 0.6

            ax.scatter(
                row[x_col],
                row[y_col],
                s=size,
                marker=marker,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=lw,
                zorder=3,
            )

    # Annotate trait names at baseline (drop_count = start_dc)
    base = df[df.drop_count == start_dc].reset_index(drop=True)

    fig = ax.figure
    dpi = fig.dpi
    points_to_pixels = dpi / 72.0  # 72 points per inch
    fontsize = 11

    def estimate_label_box(ax, x_data, y_data, text, dx_pts, dy_pts):
        # convert data point to display coords (pixels)
        x0, y0 = ax.transData.transform((x_data, y_data))
        # convert offset in points to pixels
        x_anchor = x0 + dx_pts * points_to_pixels
        y_anchor = y0 + dy_pts * points_to_pixels
        # approximate text width/height in pixels based on length and fontsize
        # width: about 0.6 * fontsize * len(text) in points, then to pixels
        width_px = 0.6 * fontsize * max(len(text), 1) * points_to_pixels
        # height: about 1.2 * fontsize in points, then to pixels
        height_px = 1.2 * fontsize * points_to_pixels
        # treat anchor as bottom-left corner of the bbox
        left = x_anchor
        right = x_anchor + width_px
        bottom = y_anchor
        top = y_anchor + height_px
        return left, right, bottom, top

    def boxes_overlap(box1, box2):
        l1, r1, b1, t1 = box1
        l2, r2, b2, t2 = box2
        if r1 <= l2 or r2 <= l1:
            return False
        if t1 <= b2 or t2 <= b1:
            return False
        return True

    placed_boxes = []

    # optional ordering: sort by y to stabilize layout
    base_sorted = base.sort_values(by=y_col).reset_index(drop=True)

    for _, row in base_sorted.iterrows():
        trait = row.trait
        if LABELLED_TRAITS is not None and trait not in LABELLED_TRAITS:
            continue
        x_data = row[x_col]
        y_data = row[y_col]
        text = trait
        best_dx = 0.0
        best_dy = 0.0
        found = False

        # bias angles to the left when near x=0
        if x_data <= 1.5:
            angle_iter = list(range(180, 360, 15)) + list(range(0, 180, 15))
        else:
            angle_iter = list(range(0, 360, 15))

        # search offsets in increasing radius around the point
        # radii in points, from very close to farther out
        for radius in range(4, 80, 4):  # 4,8,12,... points
            if found:
                break
            # angles in degrees, to spread labels in different directions
            for angle_deg in angle_iter:
                theta = np.deg2rad(angle_deg)
                dx_pts = radius * np.cos(theta)
                dy_pts = radius * np.sin(theta)

                candidate_box = estimate_label_box(ax, x_data, y_data, text, dx_pts, dy_pts)

                # check overlap with all already placed labels
                if all(not boxes_overlap(candidate_box, b) for b in placed_boxes):
                    best_dx = dx_pts
                    best_dy = dy_pts
                    placed_boxes.append(candidate_box)
                    found = True
                    break

        # if we somehow did not find a non-overlapping spot (extremely unlikely), fall back to a small fixed offset
        if not found:
            best_dx = -10.0 if x_data <= 1.5 else 10.0
            best_dy = 6.0
            fallback_box = estimate_label_box(ax, x_data, y_data, text, best_dx, best_dy)
            placed_boxes.append(fallback_box)

        ax.annotate(
            text,
            (x_data, y_data),
            fontsize=fontsize,
            xytext=(best_dx, best_dy),
            textcoords="offset points",
            bbox=dict(facecolor="#fff7cc", alpha=0.9, edgecolor="none", pad=1),
            color=get_trait_group_color(trait),
            arrowprops=dict(
                arrowstyle="-",
                color="#d4b200",
                lw=0.8,
                alpha=0.9,
            ),
            zorder=6,
        )


def plot_paths(ir_fits, xmin, xmax, ymax, out_path, annotate_drop=False):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 8), sharex=True, sharey=True
    )

    # Shared colormap for drop_count across both panels (discrete, high contrast)
    dc_values = sorted(ir_fits["drop_count"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.99, len(dc_values)))
    cmap = mcolors.ListedColormap(colors)
    # build boundaries centered between drop counts
    bounds = []
    for i, dc in enumerate(dc_values):
        if i == 0:
            prev_mid = dc - 0.5
        else:
            prev_mid = (dc_values[i - 1] + dc) / 2
        if i == len(dc_values) - 1:
            next_mid = dc + 0.5
        else:
            next_mid = (dc + dc_values[i + 1]) / 2
        bounds.append(prev_mid)
        if i == len(dc_values) - 1:
            bounds.append(next_mid)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 1D model: circles; pleiotropic model: circles (panel encodes model)
    draw_path(
        ax1,
        ir_fits,
        "x_1d",
        "Ir_r",
        cmap,
        norm,
        marker="o",
        annotate_drop=annotate_drop,
    )
    draw_path(
        ax2,
        ir_fits,
        "x_pleio",
        "Ipr_r",
        cmap,
        norm,
        marker="o",
        annotate_drop=annotate_drop,
    )

    for ax in (ax1, ax2):
        format_axes(ax, xmin, xmax, ymax)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="both", labelsize=14)
        ax.tick_params(axis="both", labelsize=9)
    ax1.set_ylabel(r"$r$ (exponent in $I_r\times \beta^r$)",
                   fontsize=16)

    ax1.set_title("1-dimensional model", fontsize=12)
    ax2.set_title("Pleiotropic model", fontsize=12)

    # Shared legend mapping color -> drop_count (discrete)
    start_dc = DROP_COUNTS[0]
    end_dc = DROP_COUNTS[-1]
    drop_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=col,
            markeredgecolor="black" if dc in (start_dc, end_dc) else "none",
            label=f"{int(dc)} outliers",
        )
        for dc, col in zip(dc_values, colors)
    ]
    drop_legend = ax2.legend(
        handles=drop_handles,
        title="Outliers dropped",
        loc="upper right",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    ax2.add_artist(drop_legend)

    fig.supxlabel(
        r"Log-likelihood difference ($I_r\beta^r$ model − standard)",
        y=0.04,
        x=0.38,
    )

    # Panel labels
    ax1.text(-0.08, 1.02, "A", transform=ax1.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")
    ax2.text(-0.08, 1.02, "B", transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")

    fig.tight_layout(rect=[0, 0, 0.97, 1])
    fig.savefig(out_path, bbox_inches="tight")


def plot_paths_traitcolors(ir_fits, xmin, xmax, ymax, out_path, annotate_drop=False):
    traits = sorted(ir_fits["trait"].unique())
    n_traits = len(traits)
    if n_traits <= 10:
        cmap_traits = plt.cm.tab10
    elif n_traits <= 20:
        cmap_traits = plt.cm.tab20
    else:
        cmap_traits = plt.cm.get_cmap("tab20", n_traits)
    color_vals = np.linspace(0, 1, n_traits, endpoint=False)
    trait_colors = {t: cmap_traits(color_vals[i]) for i, t in enumerate(traits)}

    markers_by_dc = {0: "o", 1: "s", 2: "^", 5: "D"}
    sizes_by_dc = {0: 70, 1: 50, 2: 50, 5: 80}
    alpha_by_dc = {0: 0.9, 1: 0.7, 2: 0.7, 5: 0.9}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    for trait in traits:
        sub = ir_fits[ir_fits["trait"] == trait].set_index("drop_count")
        for ax, x_col, y_col in (
            (ax1, "x_1d", "Ir_r"),
            (ax2, "x_pleio", "Ipr_r"),
        ):
            ordered = sub.reindex(DROP_COUNTS).dropna(subset=[x_col, y_col])
            if len(ordered) < 2:
                continue
            ax.plot(
                ordered[x_col].values,
                ordered[y_col].values,
                color=trait_colors[trait],
                alpha=0.6,
                linewidth=1.0,
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
            for dc, row in ordered.iterrows():
                marker = markers_by_dc.get(dc, "o")
                size = sizes_by_dc.get(dc, 50)
                alpha = alpha_by_dc.get(dc, 0.8)
                edgecolor = "black" if dc in (0, 5) else "none"
                ax.scatter(
                    row[x_col],
                    row[y_col],
                    s=size,
                    marker=marker,
                    color=trait_colors[trait],
                    alpha=alpha,
                    edgecolor=edgecolor,
                    linewidth=0.4,
                    zorder=3,
                )

    for ax in (ax1, ax2):
        format_axes(ax, xmin, xmax, ymax)
        ax.set_ylim(0, ymax)
    ax1.set_ylabel(r"$r$ (exponent in $I_r\times \beta^r$)")
    ax1.set_title("1-dimensional model", fontsize=12)
    ax2.set_title("Pleiotropic model", fontsize=12)

    # Legend for drop_count encoding
    dc_handles = []
    for dc in DROP_COUNTS:
        marker = markers_by_dc.get(dc, "o")
        size = sizes_by_dc.get(dc, 50)
        edgecolor = "black" if dc in (0, 5) else "none"
        dc_handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="",
                markersize=max(4, (size ** 0.5) / 2),
                markerfacecolor="0.7",
                markeredgecolor=edgecolor,
                label=f"drop_count = {dc}",
            )
        )
    dc_legend = ax1.legend(
        handles=dc_handles,
        title="Outlier level",
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    ax1.add_artist(dc_legend)

    # Trait color legend (figure-level)
    trait_handles = [
        Line2D(
            [],
            [],
            color=trait_colors[t],
            linestyle="-",
            linewidth=2.0,
            label=PROFESSIONAL_TRAIT_NAMES.get(t, t),
        )
        for t in traits
    ]
    fig.legend(
        handles=trait_handles,
        title="Traits",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
    )

    # place x-label centered under the two scatter panels only
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    center_x = (pos1.x0 + pos2.x1) / 2
    min_y = min(pos1.y0, pos2.y0)
    y_pos = max(0.01, min_y - 0.08)
    fig.text(
        center_x,
        y_pos,
        r"Log-likelihood difference ($I_r\beta^r$ model − standard)",
        ha="center",
    )

    ax1.text(-0.08, 1.02, "A", transform=ax1.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")
    ax2.text(-0.08, 1.02, "B", transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_path, bbox_inches="tight")
"""


def plot_paths_traitcolors_hists(ir_fits, xmin, xmax, ymax, out_path):
    traits = sorted(ir_fits["trait"].unique())
    trait_codes = {t: make_trait_code(t) for t in traits}
    trait_colors = make_trait_color_map(traits)
    n_traits = len(traits)

    markers_by_dc = {0: "o", 1: "s", 2: "^", 5: "D"}
    sizes_by_dc = {0: 70, 1: 50, 2: 50, 5: 80}
    alpha_by_dc = {0: 0.95, 1: 0.7, 2: 0.7, 5: 0.95}

    fig = plt.figure(figsize=(13, 5))
    outer = fig.add_gridspec(1, 3, width_ratios=[3.0, 3.0, 3.4])
    ax1 = fig.add_subplot(outer[0, 0])
    ax2 = fig.add_subplot(outer[0, 1])
    right_gs = outer[0, 2].subgridspec(3, 1, height_ratios=[0.8, 0.15, 1.15])
    ax_legend = fig.add_subplot(right_gs[0, 0])
    ax_hist0 = fig.add_subplot(right_gs[1, 0])
    ax_hist5 = fig.add_subplot(right_gs[2, 0])

    for trait in traits:
        sub = ir_fits[ir_fits["trait"] == trait].set_index("drop_count")
        for ax, x_col, y_col in (
            (ax1, "x_1d", "Ir_r"),
            (ax2, "x_pleio", "Ipr_r"),
        ):
            ordered = sub.reindex(DROP_COUNTS).dropna(subset=[x_col, y_col])
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
            for dc, row in ordered.iterrows():
                marker = markers_by_dc.get(dc, "o")
                size = sizes_by_dc.get(dc, 50)
                alpha_pt = alpha_by_dc.get(dc, 0.8)
                edge = "black" if dc in (0, 5) else "none"
                lw_pt = 0.4 if edge != "none" else 0.0
                ax.scatter(
                    row[x_col],
                    row[y_col],
                    s=size,
                    marker=marker,
                    color=trait_colors[trait],
                    alpha=alpha_pt,
                    edgecolor=edge,
                    linewidth=lw_pt,
                    zorder=3,
                )
                if dc == 5:
                    ax.text(
                        row[x_col],
                        row[y_col],
                        trait_codes[trait],
                        ha="center",
                        va="center",
                        fontsize=marker_code_fontsize(size),
                        fontweight="bold",
                        color="white",
                        zorder=4,
                    )

    for ax in (ax1, ax2):
        format_axes(ax, xmin, xmax, ymax)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="both", labelsize=14)
        ax.axhline(R_REFERENCE_LINE, color="0.35", linewidth=1.1, linestyle="-", zorder=1.5)
        ax.axvline(NOMINAL_LL_THRESHOLD, color="0.35", linewidth=1.1, linestyle="-", zorder=1.5)
    ax1.set_ylabel(r"$\hat{r}$ (effect size versus $s$ scaling: $|\beta|^r$)",
                   fontsize=16)
    ax1.set_title("Single-trait stabilizing", fontsize=12)
    ax2.set_title("Pleiotropic stabilizing", fontsize=12)

    # drop_count legend
    dc_handles = []
    for dc in DROP_COUNTS:
        marker = markers_by_dc.get(dc, "o")
        size = sizes_by_dc.get(dc, 50)
        edge = "black" if dc in (0, 5) else "none"
        dc_handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="",
                markersize=max(4, (size ** 0.5) / 2.0),
                markerfacecolor="0.7",
                markeredgecolor=edge,
                label=f"{dc} loci",
            )
        )
    dc_legend = ax1.legend(
        handles=dc_handles,
        title="Outliers dropped",
        loc="lower left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax1.add_artist(dc_legend)

    # Trait legend in its own axis
    ax_legend.axis("off")
    ncol = 3 if n_traits > 20 else 2
    trait_handles = [
        Line2D(
            [],
            [],
            color=trait_colors[t],
            linestyle="-",
            linewidth=2.0,
            marker="s",
            markersize=6,
            markerfacecolor=trait_colors[t],
            markeredgecolor="none",
            label=PROFESSIONAL_TRAIT_NAMES.get(t, t),
        )
        for t in traits
    ]
    ax_legend.legend(
        handles=trait_handles,
        title="Traits",
        loc="upper left",
        bbox_to_anchor=(-0.1, 1.2),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        ncol=ncol,
        labelspacing=0.8,
    )

    # Histograms: only r (pleio) for drop=0 vs drop=5
    ax_hist0.axis("off")

    r_pleio_0 = ir_fits.loc[ir_fits.drop_count == 0, "Ipr_r"].to_numpy()
    r_pleio_5 = ir_fits.loc[ir_fits.drop_count == 5, "Ipr_r"].to_numpy()
    all_r = np.concatenate([r_pleio_0, r_pleio_5])
    all_r = all_r[np.isfinite(all_r)]
    r_min = 0.0
    r_max = np.nanmax(all_r) if all_r.size else 1.0
    if not np.isfinite(r_max) or r_max <= 0:
        r_max = 1.0
    bins_r = np.linspace(r_min, r_max, 9)

    
    n_rpleio5, _, _ = ax_hist5.hist(
        r_pleio_5,
        bins=bins_r,
        color="tab:blue",
        alpha=0.4,
        edgecolor="darkblue",
        linewidth=1.6,
        label="5 dropped",
    )
    n_rpleio0, _, _ = ax_hist5.hist(
        r_pleio_0,
        bins=bins_r,
        color="tab:orange",
        alpha=0.4,
        edgecolor="chocolate",
        linewidth=1.6,
        label="0 dropped",
    )

    xticks_r = np.linspace(r_min, r_max, 4)
    ax_hist5.set_xticks(xticks_r)
    ax_hist5.set_xticklabels([f"{t:.2g}" for t in xticks_r], fontsize=7)
    ax_hist5.set_xlabel(r"$\hat{r}$", fontsize=14)
    ax_hist5.set_ylabel("Trait count", fontsize=14)
    ax_hist5.tick_params(axis="both", labelsize=13)
    ax_hist5.legend(loc="upper right", fontsize=7, frameon=False)
    ax_hist5.set_title("Pleiotropic stabilizing", fontsize=11, pad=4)

    max_r = max(np.max(n_rpleio0) if len(n_rpleio0) else 0, np.max(n_rpleio5) if len(n_rpleio5) else 0)
    ymax_hist = max_r * 1.2 if max_r > 0 else 1.0
    ax_hist5.set_ylim(0, ymax_hist)

    ax_hist5.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax_hist5.grid(False, axis="x")
    ax_hist5.spines["top"].set_visible(False)
    ax_hist5.spines["right"].set_visible(False)

    # Panel labels
    ax1.text(-0.08, 1.02, "A", transform=ax1.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")
    ax2.text(-0.08, 1.02, "B", transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")
    ax_hist5.text(-0.12, 1.02, "C", transform=ax_hist5.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")

    fig.tight_layout(rect=[0, 0.05, 0.95, 1])

    # Nudge histogram column down slightly for more breathing room beneath the legend
    hist_shift = -0.02
    for ax in (ax_hist0, ax_hist5):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + hist_shift, pos.width, pos.height])

    # Place x-label centered under the two scatter panels (not the histogram column)
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    center_x = (pos1.x0 + pos2.x1) / 2
    min_y = min(pos1.y0, pos2.y0)
    y_pos = max(0.01, min_y - 0.12)
    fig.text(
        center_x,
        y_pos,
        r"Log-likelihood difference ($|\beta|^r$ model − standard $|\beta|^2$)",
        ha="center",
        fontsize=16,
    )
    fig.savefig(out_path, bbox_inches="tight")


def plot_paths_traitcolors_hists_pleioonly(ir_fits, xmin, xmax, ymax, out_path):
    """Pleio-only version of plot_paths_traitcolors_hists with expanded scatter panel."""
    traits = sorted(ir_fits["trait"].unique())
    n_traits = len(traits)
    trait_codes = {t: make_trait_code(t) for t in traits}
    trait_colors = make_trait_color_map(traits)

    markers_by_dc = {0: "o", 1: "s", 2: "^", 5: "D"}
    sizes_by_dc = {0: 70, 1: 50, 2: 50, 5: 80}
    alpha_by_dc = {0: 0.95, 1: 0.7, 2: 0.7, 5: 0.95}

    fig = plt.figure(figsize=(10.5, 5))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.6, 3.4])
    ax_scatter = fig.add_subplot(outer[0, 0])
    right_gs = outer[0, 1].subgridspec(3, 1, height_ratios=[0.8, 0.15, 1.15])
    ax_legend = fig.add_subplot(right_gs[0, 0])
    ax_hist0 = fig.add_subplot(right_gs[1, 0])
    ax_hist5 = fig.add_subplot(right_gs[2, 0])

    for trait in traits:
        sub = ir_fits[ir_fits["trait"] == trait].set_index("drop_count")
        ordered = sub.reindex(DROP_COUNTS).dropna(subset=["x_pleio", "Ipr_r"])
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
        if 0 in ordered.index and 5 in ordered.index:
            ax_scatter.annotate(
                "",
                xy=(ordered.loc[5, "x_pleio"], ordered.loc[5, "Ipr_r"]),
                xytext=(ordered.loc[0, "x_pleio"], ordered.loc[0, "Ipr_r"]),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.0,
                    color=trait_colors[trait],
                    alpha=0.7,
                ),
                zorder=2.5,
            )
        for dc, row in ordered.iterrows():
            marker = markers_by_dc.get(dc, "o")
            size = sizes_by_dc.get(dc, 50)
            alpha_pt = alpha_by_dc.get(dc, 0.8)
            edge = "black" if dc in (0, 5) else "none"
            lw_pt = 0.4 if edge != "none" else 0.0
            ax_scatter.scatter(
                row["x_pleio"],
                row["Ipr_r"],
                s=size,
                marker=marker,
                color=trait_colors[trait],
                alpha=alpha_pt,
                edgecolor=edge,
                linewidth=lw_pt,
                zorder=3,
            )
            if dc == 5:
                ax_scatter.text(
                    row["x_pleio"],
                    row["Ipr_r"],
                    trait_codes[trait],
                    ha="center",
                    va="center",
                    fontsize=marker_code_fontsize(size),
                    fontweight="bold",
                    color="white",
                    zorder=4,
                )

    format_axes(ax_scatter, xmin, xmax, ymax)
    ax_scatter.set_ylim(0, ymax)
    ax_scatter.tick_params(axis="both", labelsize=14)
    ax_scatter.axhline(R_REFERENCE_LINE, color="0.35", linewidth=1.1, linestyle="-", zorder=1.5)
    ax_scatter.axvline(NOMINAL_LL_THRESHOLD, color="0.35", linewidth=1.1, linestyle="-", zorder=1.5)
    ax_scatter.set_ylabel(r"$\hat{r}$ (effect size versus $s$ scaling: $|\beta|^r$)",
                          fontsize=16)
    ax_scatter.set_title("Pleiotropic stabilizing", fontsize=12)

    # drop_count legend
    dc_handles = []
    for dc in DROP_COUNTS:
        marker = markers_by_dc.get(dc, "o")
        size = sizes_by_dc.get(dc, 50)
        edge = "black" if dc in (0, 5) else "none"
        dc_handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="",
                markersize=max(4, (size ** 0.5) / 2.0),
                markerfacecolor="0.7",
                markeredgecolor=edge,
                label=f"{dc} loci",
            )
        )
    dc_legend = ax_scatter.legend(
        handles=dc_handles,
        title="Outliers dropped",
        loc="upper right",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax_scatter.add_artist(dc_legend)

    # Trait legend in its own axis
    ax_legend.axis("off")
    ncol = 3 if n_traits > 20 else 2
    trait_handles = [
        Line2D(
            [],
            [],
            color=trait_colors[t],
            linestyle="-",
            linewidth=2.0,
            marker="s",
            markersize=6,
            markerfacecolor=trait_colors[t],
            markeredgecolor="none",
            label=PROFESSIONAL_TRAIT_NAMES.get(t, t),
        )
        for t in traits
    ]
    ax_legend.legend(
        handles=trait_handles,
        title="Traits",
        loc="upper left",
        bbox_to_anchor=(-0.1, 1.2),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        ncol=ncol,
        labelspacing=0.8,
    )

    # Histograms: only r (pleio) for drop=0 vs drop=5
    ax_hist0.axis("off")

    r_pleio_0 = ir_fits.loc[ir_fits.drop_count == 0, "Ipr_r"].to_numpy()
    r_pleio_5 = ir_fits.loc[ir_fits.drop_count == 5, "Ipr_r"].to_numpy()
    all_r = np.concatenate([r_pleio_0, r_pleio_5])
    all_r = all_r[np.isfinite(all_r)]
    r_min = 0.0
    r_max = np.nanmax(all_r) if all_r.size else 1.0
    if not np.isfinite(r_max) or r_max <= 0:
        r_max = 1.0
    bins_r = np.linspace(r_min, r_max, 9)

    n_rpleio5, _, _ = ax_hist5.hist(
        r_pleio_5,
        bins=bins_r,
        color="tab:blue",
        alpha=0.4,
        edgecolor="darkblue",
        linewidth=1.6,
        label="5 dropped",
    )
    n_rpleio0, _, _ = ax_hist5.hist(
        r_pleio_0,
        bins=bins_r,
        color="tab:orange",
        alpha=0.4,
        edgecolor="chocolate",
        linewidth=1.6,
        label="0 dropped",
    )

    xticks_r = np.linspace(r_min, r_max, 4)
    ax_hist5.set_xticks(xticks_r)
    ax_hist5.set_xticklabels([f"{t:.2g}" for t in xticks_r], fontsize=7)
    ax_hist5.set_xlabel(r"$\hat{r}$", fontsize=14)
    ax_hist5.set_ylabel("Trait count", fontsize=14)
    ax_hist5.tick_params(axis="both", labelsize=13)
    ax_hist5.legend(loc="upper right", fontsize=7, frameon=False)
    ax_hist5.set_title("Pleiotropic stabilizing", fontsize=11, pad=4)

    max_r = max(np.max(n_rpleio0) if len(n_rpleio0) else 0, np.max(n_rpleio5) if len(n_rpleio5) else 0)
    ymax_hist = max_r * 1.2 if max_r > 0 else 1.0
    ax_hist5.set_ylim(0, ymax_hist)

    ax_hist5.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax_hist5.grid(False, axis="x")
    ax_hist5.spines["top"].set_visible(False)
    ax_hist5.spines["right"].set_visible(False)

    # Panel label
    ax_scatter.text(-0.08, 1.02, "A", transform=ax_scatter.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")
    ax_hist5.text(-0.12, 1.02, "B", transform=ax_hist5.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")

    fig.tight_layout(rect=[0, 0.05, 0.95, 1])

    # Nudge histogram column down slightly for breathing room beneath the legend
    hist_shift = -0.02
    for ax in (ax_hist0, ax_hist5):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + hist_shift, pos.width, pos.height])

    # Place x-label centered under scatter panel only
    pos_scatter = ax_scatter.get_position()
    fig.text(
        pos_scatter.x0 + pos_scatter.width / 2,
        max(0.01, pos_scatter.y0 - 0.12),
        r"Log-likelihood difference ($|\beta|^r$ model − standard $|\beta|^2$)",
        ha="center",
        fontsize=16,
    )

    fig.savefig(out_path, bbox_inches="tight")
def main():
    splot._plot_params()
    ir_fits, results_file = load_results()
    drop_counts = [dc for dc in DROP_COUNTS if dc in ir_fits.drop_count.unique()]
    xmin, xmax, ymax = get_axis_limits(ir_fits)
    plot_dir = "."
    # plot_by_dropcount(ir_fits, drop_counts, xmin, xmax, ymax, plot_dir)
    # plot_paths(ir_fits, xmin, xmax, ymax, os.path.join(plot_dir, "ir_vs_irpleio_faceted_dc_path.pdf"))
    # plot_paths(
    #     ir_fits,
    #     xmin,
    #     xmax,
    #     ymax,
    #     os.path.join(plot_dir, "ir_vs_irpleio_faceted_dc_marked_2.pdf"),
    #     annotate_drop=True,
    # )
    # plot_paths_traitcolors(
    #     ir_fits,
    #     xmin,
    #     xmax,
    #     ymax,
    #     os.path.join(plot_dir, "ir_vs_irpleio_traitcolors.pdf"),
    #     annotate_drop=False,
    # )
    plot_paths_traitcolors_hists(
        ir_fits,
        xmin,
        xmax,
        ymax,
        os.path.join(plot_dir, "ir_vs_irpleio_traitcolors_hists.pdf"),
    )
    plot_paths_traitcolors_hists_pleioonly(
        ir_fits,
        xmin,
        xmax,
        ymax,
        os.path.join(plot_dir, "ir_vs_irpleio_traitcolors_hists_pleioonly.pdf"),
    )


if __name__ == "__main__":
    main()
