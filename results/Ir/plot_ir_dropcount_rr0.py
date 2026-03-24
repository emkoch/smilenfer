import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import smilenfer.plotting as splot
import plot_ir_dropcount as base_plot

RESULTS_DIR_BASE = "results"
RESULTS_DIR_RR0 = "results_rr0"
DROP_COUNTS = [0, 1, 2, 5]


def load_results_generic(results_dir, summary_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, results_dir, summary_name)
    if not os.path.isfile(results_file):
        raise FileNotFoundError(f"Missing results: {results_file}")
    df = pd.read_csv(results_file)
    df = df[df["drop_count"].isin(DROP_COUNTS)].copy()
    df["x_1d"] = df["Ir_LL"] - df["I2_LL"]
    df["x_pleio"] = df["Ipr_LL"] - df["Ip_LL"]
    return df, results_file


def load_all():
    base_df, _ = load_results_generic(RESULTS_DIR_BASE, "ir_estimates_all.csv")
    rr0_df, _ = load_results_generic(RESULTS_DIR_RR0, "ir_rr0_estimates_all.csv")
    merged = pd.merge(
        base_df,
        rr0_df,
        on=["trait", "drop_count"],
        suffixes=("_base", "_rr0"),
        how="inner",
    )
    return rr0_df, merged


def plot_rr0_paths(rr0_df, out_path):
    xmin, xmax, ymax = base_plot.get_axis_limits(rr0_df)
    base_plot.plot_paths_traitcolors_hists(
        rr0_df,
        xmin,
        xmax,
        ymax,
        out_path,
    )


def friendly_name(trait):
    return base_plot.PROFESSIONAL_TRAIT_NAMES.get(trait, trait)


def choose_variant_id(df):
    for col in ("snpid", "variant", "rsid", "id"):
        if col in df.columns:
            return col
    return None


def load_outlier_tables(per_trait_dir, suffix):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, per_trait_dir)
    tables = {}
    if not os.path.isdir(out_dir):
        return tables
    for fname in os.listdir(out_dir):
        if not fname.endswith(f"{suffix}.tsv"):
            continue
        trait = fname.replace(f"{suffix}.tsv", "")
        path = os.path.join(out_dir, fname)
        try:
            tables[trait] = pd.read_csv(path, sep="\t")
        except Exception:
            continue
    return tables


def overlap_summary(base_tables, rr0_tables, top_k):
    rows = []
    for trait, df_base in base_tables.items():
        if trait not in rr0_tables:
            continue
        df_rr0 = rr0_tables[trait]
        key_base = choose_variant_id(df_base)
        key_rr0 = choose_variant_id(df_rr0)
        if key_base is None or key_rr0 is None:
            # fall back to positional index
            df_base = df_base.reset_index().rename(columns={"index": "row_id"})
            df_rr0 = df_rr0.reset_index().rename(columns={"index": "row_id"})
            key_base = key_rr0 = "row_id"
        top_base = df_base.sort_values("deviation", ascending=False).head(top_k)[key_base]
        top_rr0 = df_rr0.sort_values("deviation", ascending=False).head(top_k)[key_rr0]
        set_base = set(top_base)
        set_rr0 = set(top_rr0)
        overlap = len(set_base & set_rr0)
        union = len(set_base | set_rr0)
        frac = overlap / max(1, len(set_base)) if len(set_base) else np.nan
        jaccard = overlap / max(1, union)
        rows.append(
            {
                "trait": trait,
                "top_k": top_k,
                "overlap": overlap,
                "base_count": len(set_base),
                "rr0_count": len(set_rr0),
                "overlap_frac_base": frac,
                "jaccard": jaccard,
            }
        )
    return pd.DataFrame(rows)


def plot_overlap_bars(overlap_df, out_path, title):
    if overlap_df.empty:
        print(f"No overlap data to plot for {title}")
        return
    overlap_df = overlap_df.sort_values("overlap_frac_base", ascending=False)
    trait_labels = [
        base_plot.PROFESSIONAL_TRAIT_NAMES.get(t, t) for t in overlap_df["trait"]
    ]
    y_pos = np.arange(len(overlap_df))
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(overlap_df) + 2))
    ax.barh(
        y_pos,
        overlap_df["overlap_frac_base"],
        color="tab:blue",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(trait_labels, fontsize=10)
    ax.set_xlabel(f"Fraction of top-{int(overlap_df.top_k.iloc[0])} outliers shared", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_r_change_correlations(merged, out_path):
    needed = merged[merged.drop_count.isin([0, 5])].copy()
    if needed.empty:
        print("Missing drop_count 0/5 data; skipping r-change correlation plot.")
        return
    base_pivot = needed.pivot(index="trait", columns="drop_count", values="Ir_r_base")
    rr0_pivot = needed.pivot(index="trait", columns="drop_count", values="Ir_r_rr0")
    base_pivot_p = needed.pivot(index="trait", columns="drop_count", values="Ipr_r_base")
    rr0_pivot_p = needed.pivot(index="trait", columns="drop_count", values="Ipr_r_rr0")
    # keep traits with both drop counts present
    common_traits = base_pivot.dropna(axis=0, how="any").index
    common_traits = common_traits.intersection(rr0_pivot.dropna(axis=0, how="any").index)
    common_traits = common_traits.intersection(base_pivot_p.dropna(axis=0, how="any").index)
    common_traits = common_traits.intersection(rr0_pivot_p.dropna(axis=0, how="any").index)
    if len(common_traits) == 0:
        print("No traits with both drop_count 0 and 5 for baseline and rr0.")
        return

    base_delta = (base_pivot.loc[common_traits, 5] - base_pivot.loc[common_traits, 0]).rename("delta_r_base")
    rr0_delta = (rr0_pivot.loc[common_traits, 5] - rr0_pivot.loc[common_traits, 0]).rename("delta_r_rr0")
    base_delta_p = (base_pivot_p.loc[common_traits, 5] - base_pivot_p.loc[common_traits, 0]).rename("delta_r_base_p")
    rr0_delta_p = (rr0_pivot_p.loc[common_traits, 5] - rr0_pivot_p.loc[common_traits, 0]).rename("delta_r_rr0_p")

    df = pd.concat([base_delta, rr0_delta, base_delta_p, rr0_delta_p], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=False, sharey=False)
    panels = [
        ("delta_r_base", "delta_r_rr0", "Single-trait stabilizing"),
        ("delta_r_base_p", "delta_r_rr0_p", "Pleiotropic stabilizing"),
    ]
    for ax, (xcol, ycol, title) in zip(axes, panels):
        x = df[xcol]
        y = df[ycol]
        finite = x.notna() & y.notna()
        x = x[finite]
        y = y[finite]
        traits = df.index[finite]
        if len(x) == 0:
            ax.set_visible(False)
            continue
        max_abs = np.nanmax(np.abs(np.concatenate([x.to_numpy(), y.to_numpy()])))
        if not np.isfinite(max_abs):
            max_abs = 1.0
        lim = max_abs * 1.1 if max_abs > 0 else 0.5
        ax.plot([-lim, lim], [-lim, lim], color="0.55", linestyle="--", linewidth=1.0, zorder=0)
        ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)

        colors = [base_plot.get_trait_group_color(t) for t in traits]
        ax.scatter(x, y, color=colors, alpha=0.8, s=55, edgecolor="white", linewidth=0.6, zorder=2)
        for xi, yi, trait in zip(x, y, traits):
            ax.annotate(
                friendly_name(trait),
                (xi, yi),
                fontsize=9,
                color=base_plot.get_trait_group_color(trait),
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("$\\Delta \\hat{r}$ (drop5 − drop0)\nPleiotropic stabilizing outliers dropped", fontsize=12)
        ax.set_ylabel("$\\Delta \\hat{r}$ (drop5 − drop0)\nr=0 outliers dropped", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="normal")
        ax.tick_params(labelsize=10)
        rho = x.corr(y, method="spearman")
        if np.isfinite(rho):
            ax.text(
                0.02,
                0.95,
                f"Spearman ρ = {rho:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8", boxstyle="round,pad=0.25"),
            )
        ax.grid(True, alpha=0.25, linewidth=0.7, linestyle=":")
    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)


def main():
    splot._plot_params()
    rr0_df, merged = load_all()
    base_outliers = load_outlier_tables("per_trait", "_outliers")
    rr0_outliers = load_outlier_tables("per_trait_rr0", "_outliers_rr0")
    plot_dir = "."

    plot_rr0_paths(
        rr0_df,
        os.path.join(plot_dir, "single_trait_r_scaling_r0_outliers.pdf"),
    )

    for top_k in (20,):
        overlap_df = overlap_summary(base_outliers, rr0_outliers, top_k=top_k)
        if overlap_df.empty:
            continue
        plot_overlap_bars(
            overlap_df,
            os.path.join(plot_dir, f"outlier_overlap_top{top_k}.pdf"),
            title=f"Pleiotropic stabilizing outliers vs r=0 outliers (top {top_k})",
        )

    plot_r_change_correlations(
        merged,
        out_path=os.path.join(plot_dir, "r_change_correlation_drop0_vs_drop5.pdf"),
    )


if __name__ == "__main__":
    main()
