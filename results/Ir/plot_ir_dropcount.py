import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import smilenfer.plotting as splot

RESULTS_DIR = "results"          # matched to Snakefile_Ir
DROP_COUNTS = [0, 1, 2, 5]       # matched to Snakefile_Ir


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
    xmin = min(-0.5, df[["x_1d", "x_pleio"]].min().min())
    xmax = df[["x_1d", "x_pleio"]].max().max() * 1.3
    ymax = df[["Ir_r", "Ipr_r"]].max().max() * 1.2
    return xmin, xmax, ymax


def format_axes(ax, xmin, xmax, ymax):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=2)
    ax.axvline(1, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.1, ymax)
    ax.set_yticks([0, 0.5, 1, 2, 4])
    ax.set_yticklabels([0, 0.5, 1, 2, 4])


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
    fig.supylabel(r"$r$ (exponent in $I_r\times\beta^r$)", x=0.04)
    fig.legend(loc="upper right", frameon=False)
    fig.tight_layout(rect=[0.05, 0.05, 0.94, 0.97])
    fig.savefig(os.path.join(out_dir, "ir_vs_irpleio_facet_by_dropcount.pdf"), bbox_inches="tight")


def draw_path(ax, df, x_col, y_col, colour, annotate_drop):
    start_dc = 0
    end_dc = DROP_COUNTS[-1]

    for _, row in df.iterrows():
        size = 150 if row["drop_count"] in (start_dc, end_dc) else 70
        ax.scatter(
            row[x_col],
            row[y_col],
            s=size if annotate_drop else 60,
            color=colour,
            alpha=0.45 if annotate_drop else 0.5,
            edgecolor="k" if annotate_drop else None,
            linewidth=0.25 if annotate_drop else 0,
        )
        if annotate_drop:
            ax.text(
                row[x_col],
                row[y_col],
                str(int(row["drop_count"])),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

    for trait, group in df.groupby("trait"):
        ordered = (
            group.set_index("drop_count")
            .reindex(DROP_COUNTS)
            .dropna(subset=[x_col])
        )
        if len(ordered) > 1:
            ax.plot(ordered[x_col], ordered[y_col], color=colour, alpha=0.25, linewidth=0.8)

    base = df[df.drop_count == 0]
    for _, row in base.iterrows():
        ax.annotate(
            row.trait,
            (row[x_col], row[y_col]),
            fontsize=10,
            xytext=(4, 4),
            textcoords="offset points",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        )


def plot_paths(ir_fits, xmin, xmax, ymax, out_path, annotate_drop=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    draw_path(ax1, ir_fits, "x_1d", "Ir_r", "blue", annotate_drop)
    draw_path(ax2, ir_fits, "x_pleio", "Ipr_r", "red", annotate_drop)

    for ax in (ax1, ax2):
        format_axes(ax, xmin, xmax, ymax)
        ax.set_xlabel(r"LLhood difference $I_r\beta^r$ model − standard")
    ax1.set_ylabel(r"$r$ (exponent in $I_r\times \beta^r$)")

    ax1.set_title("1-dimensional model")
    ax2.set_title("Pleiotropic model")
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig(out_path, bbox_inches="tight")


def main():
    splot._plot_params()
    ir_fits, results_file = load_results()
    drop_counts = [dc for dc in DROP_COUNTS if dc in ir_fits.drop_count.unique()]
    xmin, xmax, ymax = get_axis_limits(ir_fits)
    plot_dir = os.path.dirname(results_file)

    plot_by_dropcount(ir_fits, drop_counts, xmin, xmax, ymax, plot_dir)
    plot_paths(ir_fits, xmin, xmax, ymax, os.path.join(plot_dir, "ir_vs_irpleio_faceted_dc_path.pdf"))
    plot_paths(
        ir_fits,
        xmin,
        xmax,
        ymax,
        os.path.join(plot_dir, "ir_vs_irpleio_faceted_dc_marked_2.pdf"),
        annotate_drop=True,
    )


if __name__ == "__main__":
    main()
