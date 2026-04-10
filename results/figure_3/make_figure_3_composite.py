from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.special import gammaln
from scipy.stats import chi2


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_STYLES = {
    "dir": {
        "label": "Directional",
        "marker": ">",
        "color": "#FFB000",
    },
    "stab": {
        "label": "Single-trait stabilizing",
        "marker": "s",
        "color": "#DC267F",
    },
    "full": {
        "label": "Directional + stabilizing",
        "marker": "D",
        "color": "#785EF0",
    },
    "plei": {
        "label": "Pleiotropic stabilizing",
        "marker": "o",
        "color": "#FE6100",
    },
}
MODEL_ORDER = ["dir", "stab", "full", "plei"]
MODEL_OFFSETS = {"dir": -0.30, "stab": -0.10, "full": 0.10, "plei": 0.30}
GROUP_GAP_WIDTH = 0.50
PANEL_GAP_UNITS = 0.75
MIN_TRAIT_COUNT = 40


@dataclass
class PanelSpec:
    panel_id: str
    title: str
    primary: pd.DataFrame
    trait_groups: dict
    trait_group_labels: list
    trait_names: dict
    trait_count_map: dict
    adjusted_n: int | None = None
    secondary: pd.DataFrame | None = None
    samples: pd.DataFrame | None = None


def set_publication_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.8,
            "axes.titlesize": 9.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "legend.fontsize": 10.2,
            "legend.title_fontsize": 10.8,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 8.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def darken_color(color, factor=0.75):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def stable_chi2_log10sf(x, df):
    x = np.asarray(x, dtype=float)
    ln_sf = chi2.logsf(x, df)
    out = np.empty_like(ln_sf, dtype=float)
    finite = np.isfinite(ln_sf)
    out[finite] = ln_sf[finite] / np.log(10)
    if np.any(~finite):
        a = df / 2.0
        z = x[~finite] / 2.0
        asym_ln = (a - 1.0) * np.log(z) - z - gammaln(a)
        corr = 1.0 + (a - 1.0) / z
        out[~finite] = asym_ln / np.log(10) + np.log10(corr)
    return out


def prepare_table(df, pval):
    table = df.copy()
    for model in ["neut"] + MODEL_ORDER:
        adjust = 0 if model == "neut" else (2 if model == "full" else 1)
        if pval:
            adjust = 0
        table[f"ll_{model}"] = -(2 * adjust - 2 * table[f"ll_{model}"].to_numpy())

    for model in MODEL_ORDER:
        if pval:
            dfree = 2 if model == "full" else 1
            delta = table[f"ll_{model}"] - table["ll_neut"]
            table[f"stat_{model}"] = -stable_chi2_log10sf(delta, dfree)
        else:
            table[f"stat_{model}"] = table[f"ll_{model}"] - table["ll_neut"]
    return table


def nice_upper_bound(raw_max, pval):
    if pval:
        if raw_max <= 5:
            return 5
        if raw_max <= 10:
            return 10
        return int(np.ceil(raw_max / 5.0) * 5)
    if raw_max <= 20:
        return 20
    if raw_max <= 50:
        return 50
    if raw_max <= 100:
        return 100
    if raw_max <= 200:
        return 200
    if raw_max <= 500:
        return 500
    if raw_max <= 1000:
        return 1000
    return int(np.ceil(raw_max / 250.0) * 250)


def get_panel_bounds(processed_tables, pval):
    all_values = []
    for table in processed_tables:
        if table is None:
            continue
        for model in MODEL_ORDER:
            values = pd.to_numeric(table[f"stat_{model}"], errors="coerce").to_numpy(dtype=float)
            all_values.append(values[np.isfinite(values)])

    if not all_values:
        return (-0.5, 5.0) if pval else (-2.5, 20.0)

    values = np.concatenate(all_values)
    if pval:
        upper = nice_upper_bound(float(np.nanmax(values)) * 1.28, pval=True)
        return (-0.4, upper)

    lower = min(-2.5, float(np.nanmin(values)) * 1.1)
    upper = nice_upper_bound(float(np.nanmax(values)) * 1.18, pval=False)
    return (lower, upper)


def get_stat_mode_config(stat_mode):
    if stat_mode == "aic":
        return {
            "pval": False,
            "ylabel": r"$-\Delta \mathrm{AIC}_{\mathrm{model-neut}}$",
            "suffix": "aic",
        }
    if stat_mode == "pval":
        return {
            "pval": True,
            "ylabel": r"$-\log_{10}(P)$",
            "suffix": "pval",
        }
    raise ValueError(f"Unknown stat mode: {stat_mode}")


def make_group_widths(trait_groups):
    widths = []
    gap_indices = []
    n_groups = len(trait_groups)
    for group_index, traits in enumerate(trait_groups.values()):
        widths.extend([1.0] * len(traits))
        if group_index < n_groups - 1:
            gap_indices.append(len(widths))
            widths.append(GROUP_GAP_WIDTH)
    return widths, set(gap_indices)


def panel_width_units(spec):
    n_traits = sum(len(group) for group in spec.trait_groups.values())
    n_group_gaps = max(len(spec.trait_groups) - 1, 0)
    return n_traits + GROUP_GAP_WIDTH * n_group_gaps


def draw_reference_lines(ax, n_traits):
    ax.axhline(0, color="#7A7A7A", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    ax.axhline(-np.log10(0.05), color="#C44E52", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    ax.axhline(
        -np.log10(0.05 / n_traits),
        color="#6F3E8B",
        linestyle=(0, (4, 2)),
        linewidth=0.8,
        zorder=1,
    )


def style_axis(ax, label, show_ylabel, ylabel):
    ax.set_xticks([0])
    ax.set_xticklabels([label], rotation=55, ha="right", rotation_mode="anchor", fontsize=11.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65)
    ax.grid(axis="x", visible=False)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="x", pad=1.5)


def load_trait_count_table(base_dir):
    count_path = base_dir / "figure_3_trait_counts.csv"
    if not count_path.exists():
        return {}
    count_df = pd.read_csv(count_path)
    return {
        (row.panel_id, row.trait): f"{int(row.n)}"
        for row in count_df.itertuples(index=False)
    }


def filter_panel_spec(spec, min_count):
    trait_groups = {}
    trait_group_labels = []
    for group_label, (group_name, traits) in zip(spec.trait_group_labels, spec.trait_groups.items()):
        kept = [trait for trait in traits if int(spec.trait_count_map.get(trait, 0)) >= min_count]
        if kept:
            trait_groups[group_name] = kept
            trait_group_labels.append(group_label)

    keep_traits = {trait for traits in trait_groups.values() for trait in traits}
    primary = spec.primary[spec.primary["trait"].isin(keep_traits)].copy()
    secondary = None if spec.secondary is None else spec.secondary[spec.secondary["trait"].isin(keep_traits)].copy()
    samples = None if spec.samples is None else spec.samples[spec.samples["trait"].isin(keep_traits)].copy()

    return PanelSpec(
        panel_id=spec.panel_id,
        title=spec.title,
        primary=primary,
        trait_groups=trait_groups,
        trait_group_labels=trait_group_labels,
        trait_names={trait: spec.trait_names[trait] for trait in keep_traits},
        trait_count_map={trait: spec.trait_count_map[trait] for trait in keep_traits},
        adjusted_n=spec.adjusted_n,
        secondary=secondary,
        samples=samples,
    )


def draw_points(ax, row, face_alpha, edge_alpha, size, linewidth, zorder, fill=True):
    for model in MODEL_ORDER:
        value = row.get(f"stat_{model}")
        if pd.isna(value):
            continue
        style = MODEL_STYLES[model]
        edge = darken_color(style["color"])
        face = matplotlib.colors.to_rgba(style["color"], face_alpha) if fill else "none"
        ax.scatter(
            MODEL_OFFSETS[model],
            value,
            marker=style["marker"],
            s=size,
            facecolors=face,
            edgecolors=matplotlib.colors.to_rgba(edge, edge_alpha),
            linewidths=linewidth,
            zorder=zorder,
        )


def render_panel(fig, subplot_spec, spec, y_limits, pval, ylabel, panel_show_ylabel):
    widths, gap_indices = make_group_widths(spec.trait_groups)
    panel_grid = GridSpecFromSubplotSpec(
        2,
        len(widths),
        subplot_spec=subplot_spec,
        width_ratios=widths,
        height_ratios=[0.09, 0.91],
        hspace=0.0,
        wspace=0.08,
    )

    primary = prepare_table(spec.primary, pval)
    secondary = prepare_table(spec.secondary, pval) if spec.secondary is not None else None
    samples = prepare_table(spec.samples, pval) if spec.samples is not None else None
    if samples is None and "sample" in primary.columns and primary["sample"].max() > 0:
        samples = primary.loc[primary["sample"] > 0].copy()
        main_rows = primary.loc[primary["sample"] == 0].copy()
        if not main_rows.empty:
            primary = main_rows

    axes = []
    count_axes = []
    group_bounds = []
    share_ax = None
    col = 0
    n_panel_traits = sum(len(v) for v in spec.trait_groups.values())
    adjusted_n = spec.adjusted_n if spec.adjusted_n is not None else n_panel_traits

    for group_label, group_traits in zip(spec.trait_group_labels, spec.trait_groups.values()):
        first_ax = None
        last_ax = None
        for trait_index, trait in enumerate(group_traits):
            count_ax = fig.add_subplot(panel_grid[0, col])
            count_ax.axis("off")
            count_axes.append(count_ax)

            ax = fig.add_subplot(panel_grid[1, col], sharey=share_ax)
            share_ax = ax if share_ax is None else share_ax
            if first_ax is None:
                first_ax = ax
            last_ax = ax
            axes.append(ax)
            col += 1

            label = spec.trait_names[trait]
            style_axis(
                ax=ax,
                label=label,
                show_ylabel=(panel_show_ylabel and trait_index == 0 and group_label == spec.trait_group_labels[0]),
                ylabel=ylabel,
            )
            ax.set_xlim(-0.48, 0.48)
            ax.set_ylim(*y_limits)

            if pval:
                draw_reference_lines(ax, adjusted_n)
                ax.set_yscale("symlog", linthresh=10)
                upper = y_limits[1]
                tick_candidates = [2, 5, 10, 20, 50, 100, 200, 500]
                ax.set_yticks([tick for tick in tick_candidates if tick <= upper])
            else:
                ax.axhline(0, color="#7A7A7A", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
                ax.set_yscale("symlog", linthresh=10)
                ax.set_yticks([0, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

            row = primary.loc[primary["trait"] == trait]
            if row.empty:
                raise ValueError(f"Missing primary results for trait '{trait}' in panel {spec.panel_id}.")
            draw_points(ax, row.iloc[0], face_alpha=0.75, edge_alpha=1.0, size=50, linewidth=1.0, zorder=4, fill=True)

            if samples is not None:
                sample_rows = samples.loc[samples["trait"] == trait]
                for _, sample_row in sample_rows.iterrows():
                    draw_points(
                        ax,
                        sample_row,
                        face_alpha=0.0,
                        edge_alpha=0.20,
                        size=19,
                        linewidth=0.55,
                        zorder=2,
                        fill=False,
                    )

            if secondary is not None:
                secondary_row = secondary.loc[secondary["trait"] == trait]
                if not secondary_row.empty:
                    draw_points(
                        ax,
                        secondary_row.iloc[0],
                        face_alpha=0.0,
                        edge_alpha=0.95,
                        size=32,
                        linewidth=1.0,
                        zorder=5,
                        fill=False,
                    )

            count_label = spec.trait_count_map.get(trait)
            if count_label is not None:
                count_ax.text(
                    0.5,
                    0.30,
                    count_label,
                    transform=count_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.9,
                    color="#333333",
                )

        if first_ax is not None and last_ax is not None:
            group_bounds.append((group_label, first_ax, last_ax))
        if col < len(widths) and col in gap_indices:
            count_gap_ax = fig.add_subplot(panel_grid[0, col])
            count_gap_ax.axis("off")
            gap_ax = fig.add_subplot(panel_grid[1, col])
            gap_ax.axis("off")
            col += 1

    return {
        "axes": axes,
        "count_axes": count_axes,
        "bounds_axes": axes + count_axes,
        "group_bounds": group_bounds,
        "primary": primary,
        "secondary": secondary,
        "samples": samples,
    }


def annotate_panel(fig, rendered_panel, panel_id, title, title_offset=0.022, group_offset=0.008, center_title=False):
    bounds_axes = rendered_panel.get("bounds_axes", rendered_panel["axes"])
    count_axes = rendered_panel.get("count_axes", [])
    x0 = min(ax.get_position().x0 for ax in bounds_axes)
    x1 = max(ax.get_position().x1 for ax in bounds_axes)
    y1 = max(ax.get_position().y1 for ax in bounds_axes)
    if count_axes:
        count_y0 = min(ax.get_position().y0 for ax in count_axes)
        count_y1 = max(ax.get_position().y1 for ax in count_axes)
        group_y = count_y0 + 0.76 * (count_y1 - count_y0)
    else:
        group_y = y1 + group_offset

    fig.text(x0, y1 + title_offset, panel_id, ha="left", va="bottom", fontsize=18.0, fontweight="bold")
    if center_title:
        fig.text(0.5 * (x0 + x1), y1 + title_offset, title, ha="center", va="bottom", fontsize=14.0)
    else:
        fig.text(x0 + 0.040, y1 + title_offset, title, ha="left", va="bottom", fontsize=14.0)

    for group_label, first_ax, last_ax in rendered_panel["group_bounds"]:
        x_center = 0.5 * (first_ax.get_position().x0 + last_ax.get_position().x1)
        fig.text(
            x_center,
            group_y,
            group_label,
            ha="center",
            va="bottom",
            fontsize=13.2,
            color="#333333",
        )


def add_row_count_prefix(fig, rendered_panel, text="n ="):
    count_axes = rendered_panel.get("count_axes", [])
    if not count_axes:
        return
    first_count_ax = count_axes[0]
    pos = first_count_ax.get_position()
    y = pos.y0 + 0.30 * pos.height
    x = pos.x0 - 0.010
    fig.text(x, y, text, ha="right", va="center", fontsize=11.6, color="#333333")


def add_global_legends(legend_ax, pval):
    legend_ax.axis("off")

    model_handles = []
    for model in MODEL_ORDER:
        style = MODEL_STYLES[model]
        model_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                markerfacecolor=matplotlib.colors.to_rgba(style["color"], 0.75),
                markeredgecolor=darken_color(style["color"]),
                markeredgewidth=0.9,
                markersize=8.2,
                label=style["label"],
            )
        )

    threshold_handles = [
        Line2D([0], [0], color="#7A7A7A", linestyle=(0, (4, 2)), linewidth=0.9, label="Neutral baseline"),
    ]
    if pval:
        threshold_handles.extend(
            [
                Line2D([0], [0], color="#C44E52", linestyle=(0, (4, 2)), linewidth=0.9, label=r"Nominal threshold ($P=0.05$)"),
                Line2D([0], [0], color="#6F3E8B", linestyle=(0, (4, 2)), linewidth=0.9, label="Bonferroni threshold"),
            ]
        )

    legend_models = legend_ax.legend(
        handles=model_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        frameon=False,
        title="Selection model",
        handletextpad=0.5,
        borderaxespad=0.0,
        labelspacing=0.80,
        fontsize=11.8,
        title_fontsize=12.2,
    )
    legend_models._legend_box.align = "left"

    legend_thresholds = legend_ax.legend(
        handles=threshold_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.50 if pval else 0.44),
        frameon=False,
        title="Reference lines",
        handlelength=2.2,
        borderaxespad=0.0,
        labelspacing=0.80,
        fontsize=10.6,
        title_fontsize=11.0,
    )
    legend_thresholds._legend_box.align = "left"

    legend_ax.add_artist(legend_models)


def build_three_row_layout(fig, panels, shared_limits, pval, ylabel):
    width_a = panel_width_units(panels[0])
    width_b = panel_width_units(panels[1]) * 1.15
    width_c = panel_width_units(panels[2])
    width_d = panel_width_units(panels[3])
    width_e = panel_width_units(panels[4])
    legend_width = 4.3
    row1_total = width_b
    row2_total = width_a + PANEL_GAP_UNITS + width_c + PANEL_GAP_UNITS + legend_width
    row3_total = width_d + PANEL_GAP_UNITS + width_e
    full_units = max(row1_total, row2_total, row3_total)

    def row_ratios(left_content, right_content=None):
        if right_content is None:
            content = left_content
        else:
            content = left_content + PANEL_GAP_UNITS + right_content
        spare = max(full_units - content, 0.0)
        left_pad = spare / 2.0
        right_pad = spare - left_pad
        ratios = []
        if left_pad > 1e-6:
            ratios.append(left_pad)
        ratios.append(left_content)
        if right_content is not None:
            ratios.append(PANEL_GAP_UNITS)
            ratios.append(right_content)
        if right_pad > 1e-6:
            ratios.append(right_pad)
        return ratios

    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.46)

    row1 = GridSpecFromSubplotSpec(
        1,
        len(row_ratios(width_b, None)),
        subplot_spec=outer[0],
        width_ratios=row_ratios(width_b, None),
        wspace=0.0,
    )
    row2 = GridSpecFromSubplotSpec(
        1,
        len(row_ratios(width_a + PANEL_GAP_UNITS + width_c, legend_width)),
        subplot_spec=outer[1],
        width_ratios=row_ratios(width_a + PANEL_GAP_UNITS + width_c, legend_width),
        wspace=0.0,
    )
    row3 = GridSpecFromSubplotSpec(
        1,
        len(row_ratios(width_d, width_e)),
        subplot_spec=outer[2],
        width_ratios=row_ratios(width_d, width_e),
        wspace=0.0,
    )

    def content_indices(ratios, has_right):
        if has_right:
            if len(ratios) == 3:
                return 0, 2
            return 1, 3
        if len(ratios) == 1:
            return 0, None
        return 1, None

    row1_first, _ = content_indices(row_ratios(width_b, None), False)
    row2_first, row2_second = content_indices(row_ratios(width_a + PANEL_GAP_UNITS + width_c, legend_width), True)
    row3_first, row3_second = content_indices(row_ratios(width_d, width_e), True)

    rendered = [None] * len(panels)
    rendered[1] = render_panel(fig, row1[row1_first], panels[1], shared_limits, pval, ylabel, True)

    row2_panels = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=row2[row2_first],
        width_ratios=[width_a, PANEL_GAP_UNITS, width_c],
        wspace=0.0,
    )
    rendered[0] = render_panel(fig, row2_panels[0], panels[0], shared_limits, pval, ylabel, True)
    rendered[2] = render_panel(fig, row2_panels[2], panels[2], shared_limits, pval, ylabel, False)

    rendered[3] = render_panel(fig, row3[row3_first], panels[3], shared_limits, pval, ylabel, True)
    rendered[4] = render_panel(fig, row3[row3_second], panels[4], shared_limits, pval, ylabel, False)

    legend_ax = fig.add_subplot(row2[row2_second])
    return rendered, legend_ax


def load_panel_specs(base_dir):
    fit_root = base_dir.parent / "all_opt_fits"
    trait_counts = load_trait_count_table(base_dir)

    individual_gwas_groups = {
        "Individual disease GWAS": ["bc", "cad", "ibd", "scz", "t2d"],
    }
    ukbb_finngen_groups = {
        "Quantitative": [
            "bmi",
            "dbp",
            "fvc",
            "grip_strength",
            "hdl",
            "height",
            "ldl",
            "pulse_rate",
            "rbc",
            "sbp",
            "triglycerides",
            "urate",
            "wbc",
        ],
        "Disease": [
            "arthrosis",
            "asthma",
            "diverticulitis",
            "gallstones",
            "glaucoma",
            "hypothyroidism",
            "malignant_neoplasms",
            "uterine_fibroids",
            "varicose_veins",
        ],
    }
    original_names = {
        "height": "Height",
        "bmi": "BMI",
        "ldl": "LDL",
        "hdl": "HDL",
        "dbp": "DBP",
        "sbp": "SBP",
        "triglycerides": "Triglycerides",
        "urate": "Urate",
        "rbc": "RBC",
        "wbc": "WBC",
        "grip_strength": "Grip\nstrength",
        "fvc": "FVC",
        "pulse_rate": "Pulse\nrate",
        "bc": "Breast\ncancer",
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
        "malignant_neoplasms": "Malignant\nneoplasms",
        "uterine_fibroids": "Uterine\nfibroids",
        "varicose_veins": "Varicose\nveins",
    }

    bbj_groups = {
        "Quantitative": ["bmi", "dbp", "hdl", "height", "ldl", "rbc", "sbp", "triglycerides"],
        "Disease": ["bc", "cad", "gallstones", "t2d", "uterine_fibroids"],
    }
    bbj_names = {
        "bmi": "BMI",
        "dbp": "DBP",
        "hdl": "HDL",
        "height": "Height",
        "ldl": "LDL",
        "rbc": "RBC",
        "sbp": "SBP",
        "triglycerides": "Triglycerides",
        "bc": "Breast\ncancer",
        "cad": "CAD",
        "gallstones": "Gallstones",
        "t2d": "T2D",
        "uterine_fibroids": "Uterine\nfibroids",
    }

    ukbb_groups = {
        "Quantitative": ["bmi", "dbp", "hdl", "height", "ldl", "sbp", "triglycerides", "wbc"],
    }
    ukbb_names = {
        "bmi": "BMI",
        "dbp": "DBP",
        "hdl": "HDL",
        "height": "Height",
        "ldl": "LDL",
        "sbp": "SBP",
        "triglycerides": "Triglycerides",
        "wbc": "WBC",
    }

    mvp_groups = {
        "Disease": [
            "Atrial fibrillation",
            "Basal cell carcinoma",
            "Cancer of prostate",
            "Coronary atherosclerosis",
            "Diverticulosis and diverticulitis",
            "Glaucoma",
            "Gout",
            "Hyperlipidemia",
            "Hypertension",
            "Hypothyroidism",
            "Type 2 diabetes",
        ],
    }
    mvp_names = {
        "Atrial fibrillation": "Atrial\nfibrillation",
        "Basal cell carcinoma": "Basal cell\ncarcinoma",
        "Cancer of prostate": "Prostate\ncancer",
        "Coronary atherosclerosis": "Coronary\natherosclerosis",
        "Diverticulosis and diverticulitis": "Diverticulosis and\ndiverticulitis",
        "Glaucoma": "Glaucoma",
        "Gout": "Gout",
        "Hyperlipidemia": "Hyperlipidemia",
        "Hypertension": "Hypertension",
        "Hypothyroidism": "Hypothyroidism",
        "Type 2 diabetes": "T2D",
    }

    shared_ab_n = sum(len(group) for group in individual_gwas_groups.values()) + sum(
        len(group) for group in ukbb_finngen_groups.values()
    )

    panels = [
        PanelSpec(
            panel_id="A",
            title="Disease GWAS",
            primary=pd.read_csv(fit_root / "original_traits" / "opt_results_original_traits_eur_post.csv"),
            trait_groups=individual_gwas_groups,
            trait_group_labels=["Disease"],
            trait_names=original_names,
            trait_count_map={trait: trait_counts[("A", trait)] for trait in original_names if ("A", trait) in trait_counts},
            adjusted_n=shared_ab_n,
        ),
        PanelSpec(
            panel_id="B",
            title="UKBB/FinnGen",
            primary=pd.read_csv(fit_root / "original_traits" / "opt_results_original_traits_eur_post.csv"),
            trait_groups=ukbb_finngen_groups,
            trait_group_labels=["Quantitative", "Disease"],
            trait_names=original_names,
            trait_count_map={trait: trait_counts[("B", trait)] for trait in original_names if ("B", trait) in trait_counts},
            adjusted_n=shared_ab_n,
        ),
        PanelSpec(
            panel_id="C",
            title="Biobank Japan",
            primary=pd.read_csv(fit_root / "bbj" / "opt_results_high_bbj.csv"),
            samples=pd.read_csv(fit_root / "bbj" / "opt_results_random_bbj.csv"),
            trait_groups=bbj_groups,
            trait_group_labels=["Quantitative", "Disease"],
            trait_names=bbj_names,
            trait_count_map={trait: trait_counts[("C", trait)] for trait in bbj_names if ("C", trait) in trait_counts},
        ),
        PanelSpec(
            panel_id="D",
            title="UK Biobank SuSiE-X",
            primary=pd.read_csv(fit_root / "ukbb_finemapping" / "opt_results_ukbb_susiex.csv"),
            trait_groups=ukbb_groups,
            trait_group_labels=["Quantitative"],
            trait_names=ukbb_names,
            trait_count_map={trait: trait_counts[("D", trait)] for trait in ukbb_names if ("D", trait) in trait_counts},
        ),
        PanelSpec(
            panel_id="E",
            title="Million Veteran Program",
            primary=pd.read_csv(fit_root / "mvp" / "opt_results_mvp_finemapping_eur.csv"),
            trait_groups=mvp_groups,
            trait_group_labels=["Disease"],
            trait_names=mvp_names,
            trait_count_map={trait: trait_counts[("E", trait)] for trait in mvp_names if ("E", trait) in trait_counts},
        ),
    ]

    return [filter_panel_spec(panel, MIN_TRAIT_COUNT) for panel in panels]


def build_figure(stat_mode):
    set_publication_style()
    base_dir = Path(__file__).resolve().parent
    panels = load_panel_specs(base_dir)
    config = get_stat_mode_config(stat_mode)
    pval = config["pval"]

    processed_tables = []
    for panel in panels:
        processed_tables.append(prepare_table(panel.primary, pval))
        if panel.secondary is not None:
            processed_tables.append(prepare_table(panel.secondary, pval))
        if panel.samples is not None:
            processed_tables.append(prepare_table(panel.samples, pval))
    shared_limits = get_panel_bounds(processed_tables, pval=pval)

    fig = plt.figure(figsize=(15.9, 14.6))
    rendered, legend_ax = build_three_row_layout(fig, panels, shared_limits, pval, config["ylabel"])
    fig.subplots_adjust(left=0.045, right=0.995, top=0.982, bottom=0.050)
    fig.canvas.draw()
    display_ids = ["B", "A", "C", "D", "E"]
    for panel_spec, panel_render, display_id in zip(panels, rendered, display_ids):
        annotate_panel(
            fig,
            panel_render,
            display_id,
            panel_spec.title,
            title_offset=0.020,
            group_offset=0.0075,
            center_title=True,
        )

    add_row_count_prefix(fig, rendered[1])
    add_row_count_prefix(fig, rendered[0])
    add_row_count_prefix(fig, rendered[3])
    add_global_legends(legend_ax, pval)
    return fig


def save_figure(stat_mode, output_path):
    fig = build_figure(stat_mode)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    save_figure("pval", base_dir / "figure_3_composite_publication_three_row_pval.pdf")
    save_figure("aic", base_dir / "figure_3_composite_publication_three_row_aic.pdf")


if __name__ == "__main__":
    main()
