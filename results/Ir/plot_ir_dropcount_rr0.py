import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import smilenfer.plotting as splot
import plot_ir_dropcount as base_plot  # reuse helpers, color mapping, etc.

RESULTS_DIR_BASE = "results"          # original Snakefile_Ir outputs
RESULTS_DIR_RR0  = "results_rr0"      # rr=0 outlier pipeline outputs
DROP_COUNTS      = [0, 1, 2, 5]


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
    base_df, base_path = load_results_generic(RESULTS_DIR_BASE, "ir_estimates_all.csv")
    rr0_df, rr0_path = load_results_generic(RESULTS_DIR_RR0, "ir_rr0_estimates_all.csv")
    merged = pd.merge(
        base_df,
        rr0_df,
        on=["trait", "drop_count"],
        suffixes=("_base", "_rr0"),
        how="inner",
    )
    return base_df, rr0_df, merged, base_path, rr0_path


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


def scatter_compare(ax, sub, base_col, rr0_col, label):
    ax.scatter(
        sub[base_col],
        sub[rr0_col],
        color="tab:blue",
        alpha=0.65,
        s=40,
        edgecolor="none",
    )
    for _, row in sub.iterrows():
        ax.annotate(
            base_plot.PROFESSIONAL_TRAIT_NAMES.get(row.trait, row.trait),
            (row[base_col], row[rr0_col]),
            fontsize=8,
            color=base_plot.get_trait_group_color(row.trait),
        )
    lims = np.array([
        np.nanmin([sub[base_col].min(), sub[rr0_col].min()]),
        np.nanmax([sub[base_col].max(), sub[rr0_col].max()]),
    ])
    if not np.isfinite(lims).all():
        lims = np.array([0, 1])
    span = lims[1] - lims[0]
    pad = 0.05 * span if span > 0 else 1.0
    ax.plot(lims, lims, color="0.4", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xlim(lims[0] - pad, lims[1] + pad)
    ax.set_ylim(lims[0] - pad, lims[1] + pad)
    ax.set_title(label, fontsize=11)


def plot_pipeline_alignment(merged, drop_count, out_path):
    sub = merged[merged.drop_count == drop_count].copy()
    if sub.empty:
        print(f"No records for drop_count={drop_count}; skipping {out_path}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    compare_specs = [
        ("x_1d_base", "x_1d_rr0", r"$\Delta$LL (Pleiotropic stabilizing outliers vs r=0 outliers)$"),
        ("x_pleio_base", "x_pleio_rr0", r"$\Delta$LL (Pleiotropic stabilizing outliers vs r=0 outliers)$"),
        ("Ir_r_base", "Ir_r_rr0", r"$\hat{r}$ (Pleiotropic stabilizing outliers vs r=0 outliers)$"),
        ("Ipr_r_base", "Ipr_r_rr0", r"$\hat{r}$ (Pleiotropic stabilizing outliers vs r=0 outliers)$"),
    ]
    for ax, (bcol, rcol, title) in zip(axes, compare_specs):
        scatter_compare(ax, sub, bcol, rcol, title)
        ax.set_xlabel("Pleiotropic stabilizing outliers", fontsize=10)
        ax.set_ylabel("r=0 outliers", fontsize=10)
        ax.tick_params(labelsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")


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


def merge_deviation_tables(base_tables, rr0_tables):
    merged = {}
    for trait, df_base in base_tables.items():
        if trait not in rr0_tables or "deviation" not in df_base.columns:
            continue
        df_rr0 = rr0_tables[trait]
        if "deviation" not in df_rr0.columns:
            continue
        key_base = choose_variant_id(df_base)
        key_rr0 = choose_variant_id(df_rr0)
        if key_base is None or key_rr0 is None:
            df_base = df_base.reset_index().rename(columns={"index": "row_id"})
            df_rr0 = df_rr0.reset_index().rename(columns={"index": "row_id"})
            key_base = key_rr0 = "row_id"
        keep_cols_base = [key_base, "deviation"]
        keep_cols_rr0 = [key_rr0, "deviation"]
        shared = pd.merge(
            df_base[keep_cols_base],
            df_rr0[keep_cols_rr0],
            left_on=key_base,
            right_on=key_rr0,
            suffixes=("_base", "_rr0"),
            how="inner",
        )
        if shared.empty:
            continue
        shared["trait"] = trait
        merged[trait] = shared
    return merged


def deviation_corr_summary(merged_dev):
    rows = []
    for trait, df in merged_dev.items():
        if df.empty:
            continue
        rho = df[["deviation_base", "deviation_rr0"]].corr(method="spearman").iloc[0, 1]
        rho = np.abs(rho) if np.isfinite(rho) else np.nan
        rows.append({"trait": trait, "n_shared": len(df), "spearman_rho": rho})
    return pd.DataFrame(rows)


def plot_deviation_corr_bar(dev_corr, out_path):
    if dev_corr.empty:
        print("No deviation correlation data to plot.")
        return
    dev_corr = dev_corr.sort_values("spearman_rho", ascending=False)
    trait_labels = [friendly_name(t) for t in dev_corr["trait"]]
    y_pos = np.arange(len(dev_corr))
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(dev_corr) + 2))
    ax.barh(
        y_pos,
        dev_corr["spearman_rho"],
        color="tab:green",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(trait_labels, fontsize=9)
    ax.set_xlabel("|Spearman| of outlier deviations (baseline vs rr=0)", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_inference_pairs_single(merged, drop_count, out_path):
    sub = merged[merged.drop_count == drop_count].copy()
    if sub.empty:
        print(f"No inference records for drop_count={drop_count}")
        return
    metrics = [
        ("x_1d_base", "x_1d_rr0", r"$\Delta$LL (1D)"),
        ("x_pleio_base", "x_pleio_rr0", r"$\Delta$LL (pleio)"),
        ("Ir_r_base", "Ir_r_rr0", r"$\hat{r}$ (1D)"),
        ("Ipr_r_base", "Ipr_r_rr0", r"$\hat{r}$ (pleio)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, (bcol, rcol, title) in zip(axes, metrics):
        sub = sub.copy()
        sub["abs_diff"] = (sub[rcol] - sub[bcol]).abs()
        ordered = sub.sort_values("abs_diff", ascending=False)
        y_pos = np.arange(len(ordered))
        ax.hlines(y_pos, ordered[bcol], ordered[rcol], color="0.7", linewidth=1.4, zorder=1)
        ax.scatter(ordered[bcol], y_pos, color="0.3", s=36, label="baseline", zorder=2)
        ax.scatter(ordered[rcol], y_pos, color="tab:blue", s=48, label="rr=0", zorder=3)
        for yy, (_, row) in enumerate(ordered.iterrows()):
            ax.text(
                max(row[bcol], row[rcol]) + 0.01,
                yy,
                friendly_name(row.trait),
                va="center",
                fontsize=8,
                color=base_plot.get_trait_group_color(row.trait),
            )
        xmin = np.nanmin([ordered[bcol].min(), ordered[rcol].min()])
        xmax = np.nanmax([ordered[bcol].max(), ordered[rcol].max()])
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            xmin, xmax = -1, 1
        pad = 0.08 * (xmax - xmin) if xmax != xmin else 0.5
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.2, linewidth=0.6)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Baseline vs rr=0 inference (drop_count=0 only)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")


def plot_shift_lollipop(merged, drop_counts, out_path, top_n=25):
    sub = merged[merged.drop_count.isin(drop_counts)].copy()
    if sub.empty:
        print("No inference records for requested drop_counts")
        return
    metrics = [
        ("Ir_r", "r_diff_1d", r"$\Delta \hat{r}$ (1D)"),
        ("Ipr_r", "r_diff_pleio", r"$\Delta \hat{r}$ (pleio)"),
        ("x_1d", "ll_diff_1d", r"$\Delta$LL (1D)"),
        ("x_pleio", "ll_diff_pleio", r"$\Delta$LL (pleio)"),
    ]
    sub["r_diff_1d"] = sub["Ir_r_rr0"] - sub["Ir_r_base"]
    sub["r_diff_pleio"] = sub["Ipr_r_rr0"] - sub["Ipr_r_base"]
    sub["ll_diff_1d"] = sub["x_1d_rr0"] - sub["x_1d_base"]
    sub["ll_diff_pleio"] = sub["x_pleio_rr0"] - sub["x_pleio_base"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (_, col, title) in zip(axes, metrics):
        agg = (
            sub.groupby("trait")[col]
            .apply(lambda s: s.abs().max())
            .sort_values(ascending=False)
            .head(top_n)
        )
        y_pos = np.arange(len(agg))
        colors = [base_plot.get_trait_group_color(t) for t in agg.index]
        ax.hlines(y_pos, 0, agg.values, color="0.8", linewidth=1.2, zorder=1)
        ax.scatter(agg.values, y_pos, color=colors, s=50, zorder=2, edgecolor="none")
        for yy, (trait, val) in enumerate(agg.items()):
            ax.text(
                val + max(agg.values) * 0.01,
                yy,
                friendly_name(trait),
                va="center",
                fontsize=8,
                color=base_plot.get_trait_group_color(trait),
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("max |rr=0 − baseline| across drop_counts", fontsize=9)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.2, linewidth=0.6)
    fig.suptitle(f"Largest inference shifts across drop_counts {drop_counts}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")


def plot_overlap_vs_inference_shift(overlap_df, merged_infer, drop_count, out_path):
    if overlap_df.empty:
        print("No overlap data to plot inference shift.")
        return
    sub_infer = merged_infer[merged_infer.drop_count == drop_count].copy()
    if sub_infer.empty:
        print(f"No inference records for drop_count={drop_count}")
        return
    overlap_df = overlap_df.rename(columns={"overlap_frac_base": "overlap_frac"})
    comb = pd.merge(
        overlap_df[["trait", "overlap_frac"]],
        sub_infer[
            ["trait", "x_1d_base", "x_1d_rr0", "x_pleio_base", "x_pleio_rr0", "Ir_r_base", "Ir_r_rr0", "Ipr_r_base", "Ipr_r_rr0"]
        ],
        on="trait",
        how="inner",
    )
    if comb.empty:
        print("No matching traits between overlap and inference tables.")
        return
    comb["r_diff_1d"] = (comb["Ir_r_rr0"] - comb["Ir_r_base"]).abs()
    comb["r_diff_pleio"] = (comb["Ipr_r_rr0"] - comb["Ipr_r_base"]).abs()
    comb["ll_diff_1d"] = (comb["x_1d_rr0"] - comb["x_1d_base"]).abs()
    comb["ll_diff_pleio"] = (comb["x_pleio_rr0"] - comb["x_pleio_base"]).abs()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        ("overlap_frac", "r_diff_1d", r"|Δ\hat{r}| (1D)"),
        ("overlap_frac", "r_diff_pleio", r"|Δ\hat{r}| (pleio)"),
        ("overlap_frac", "ll_diff_1d", r"|ΔLL| (1D)"),
        ("overlap_frac", "ll_diff_pleio", r"|ΔLL| (pleio)"),
    ]
    for ax, (x_col, y_col, y_label) in zip(axes.flatten(), panels):
        ax.scatter(
            comb[x_col],
            comb[y_col],
            color="tab:blue",
            alpha=0.7,
            s=50,
            edgecolor="none",
        )
        for _, row in comb.iterrows():
            ax.annotate(
                friendly_name(row.trait),
                (row[x_col], row[y_col]),
                fontsize=8,
                color=base_plot.get_trait_group_color(row.trait),
            )
        ax.set_xlabel("Fraction of top-10 outliers shared", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.6)
    fig.suptitle(f"Outlier overlap vs inference shifts (drop_count={drop_count})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")


def plot_r_change_correlations(merged, out_path):
    # Compute r change between drop_count 0 and 5 for baseline and rr0
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
        # symmetric limits around zero for comparability
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
        # Spearman rho annotation
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
    base_df, rr0_df, merged, base_path, rr0_path = load_all()
    base_outliers = load_outlier_tables("per_trait", "_outliers")
    rr0_outliers = load_outlier_tables("per_trait_rr0", "_outliers_rr0")
    plot_dir = "."

    # rr0-only plots (mirroring base visuals, with distinct filenames)
    plot_rr0_paths(
        rr0_df,
        os.path.join(plot_dir, "ir_rr0_vs_irpleio_traitcolors_hists.pdf"),
    )

    # Skip inference shift visuals (not informative)

    # Outlier-set stability: fraction of top-k loci shared between pipelines
    overlap_df_top10 = None
    for top_k in (20,):
        overlap_df = overlap_summary(base_outliers, rr0_outliers, top_k=top_k)
        if overlap_df.empty:
            continue
        if top_k == 10:
            overlap_df_top10 = overlap_df.copy()
        plot_overlap_bars(
            overlap_df,
            os.path.join(plot_dir, f"outlier_overlap_top{top_k}.pdf"),
            title=f"Pleiotropic stabilizing outliers vs r=0 outliers (top {top_k})",
        )

    # Deviation agreement
    merged_dev = merge_deviation_tables(base_outliers, rr0_outliers)
    dev_corr = deviation_corr_summary(merged_dev)
    # intentionally skip plotting deviation rank correlations

    # Correlation of r changes between pipelines (drop0 vs drop5)
    plot_r_change_correlations(
        merged,
        out_path=os.path.join(plot_dir, "r_change_correlation_drop0_vs_drop5.pdf"),
    )


if __name__ == "__main__":
    main()
