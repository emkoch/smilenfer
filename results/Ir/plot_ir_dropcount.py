import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

DROP_COUNTS = [0, 1, 2, 5]

NOMINAL_LL_THRESHOLD = chi2.ppf(0.95, 1) / 2
R_REFERENCE_LINE = 2.0

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
