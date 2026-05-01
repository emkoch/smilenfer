import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bbj_matching_common import DATA_DIR, SCRIPT_DIR, TRAIT_LABELS, trait_order


OUT_PART1 = SCRIPT_DIR / "bbj_raw_replacement_smile_conditions_long_part1.pdf"
OUT_PART2 = SCRIPT_DIR / "bbj_raw_replacement_smile_conditions_long_part2.pdf"

CONDITIONS = [
    ("full_original", "Full original\n(no matching)"),
    ("matched_original", "Matched original"),
    ("raw_all", "Raw BBJ\nreplace all"),
    ("raw_z2", r"Raw BBJ replace" + "\n" + r"$Z_{\mathrm{exp}}\geq2$"),
    ("raw_z5", r"Raw BBJ replace" + "\n" + r"$Z_{\mathrm{exp}}\geq5$"),
]


def set_smile_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def condition_data(trait, original, matched):
    full = original.loc[original["trait"] == trait].copy()
    sub = matched.loc[matched["trait"] == trait].copy()

    out = {
        "full_original": {
            "raf": full["raf"].to_numpy(),
            "beta": np.abs(full["PosteriorMean"].to_numpy()),
            "v_cut": full["v_cut"].iloc[0],
            "n_replaced": 0,
            "n_total": len(full),
            "n_full": len(full),
            "n_lost": 0,
            "replaced": np.zeros(len(full), dtype=bool),
        }
    }

    matched_beta = np.abs(sub["PosteriorMean"].to_numpy())
    out["matched_original"] = {
        "raf": sub["raf"].to_numpy(),
        "beta": matched_beta,
        "v_cut": sub["v_cut"].iloc[0],
        "n_replaced": 0,
        "n_total": len(sub),
        "n_full": len(full),
        "n_lost": len(full) - len(sub),
        "replaced": np.zeros(len(sub), dtype=bool),
    }

    for name, cutoff in [("raw_all", 0), ("raw_z2", 2), ("raw_z5", 5)]:
        replaced = sub["expected_z_bbj"].to_numpy() >= cutoff
        beta = matched_beta.copy()
        beta[replaced] = np.abs(sub.loc[replaced, "beta_bbj"].to_numpy())
        out[name] = {
            "raf": sub["raf"].to_numpy(),
            "beta": beta,
            "v_cut": sub["v_cut"].iloc[0],
            "n_replaced": int(np.sum(replaced)),
            "n_total": len(sub),
            "n_full": len(full),
            "n_lost": len(full) - len(sub),
            "replaced": replaced,
        }
    return out


def draw_panel(ax, data, title, show_ylabel=False):
    raf = data["raf"]
    beta = data["beta"]
    replaced = data["replaced"]
    v_cut = data["v_cut"]
    x_set = np.arange(0.01, 1, 0.01)
    boundary = np.sqrt(v_cut / (2 * x_set * (1 - x_set)))
    upper = max(np.nanmax(beta) * 1.35, np.nanmax(boundary) * 0.85)

    ax.set_xlim(-0.02, 1.02)
    ax.set_yscale("log")
    ax.set_ylim(max(np.nanmin(beta[beta > 0]) * 0.75, 1e-5), upper)
    if upper > np.nanmax(boundary):
        ax.plot(
            np.concatenate(([0.01], x_set, [0.99])),
            np.concatenate(([upper], boundary, [upper])),
            color="darkslategrey",
            linestyle="dashed",
            linewidth=1.3,
        )
    else:
        ax.plot(x_set, boundary, color="darkslategrey", linestyle="dashed", linewidth=1.3)

    ax.scatter(raf[~replaced], beta[~replaced], color="#BBBBBB", edgecolors="black", s=16, alpha=0.48, linewidths=0.25)
    if np.any(replaced):
        ax.scatter(raf[replaced], beta[replaced], color="#0072B2", edgecolors="black", s=18, alpha=0.78, linewidths=0.25)

    if data["n_total"] == data["n_full"] and data["n_lost"] == 0 and data["n_replaced"] == 0:
        count_text = f"{data['n_full']} original-fit loci"
    elif data["n_lost"] > 0 and data["n_replaced"] == 0:
        count_text = f"{data['n_total']}/{data['n_full']} original-fit loci matched; {data['n_lost']} unmatched"
    else:
        count_text = f"{data['n_replaced']}/{data['n_total']} matched loci replaced"

    ax.set_title(f"{title}\n{count_text}")
    ax.set_xlabel("RAF")
    ax.set_ylabel("Effect used in fit" if show_ylabel else "")
    ax.grid(alpha=0.18, linewidth=0.5)


def write_page(traits, original, matched, out_path):
    fig, axes = plt.subplots(len(traits), len(CONDITIONS), figsize=(14.5, 2.55 * len(traits)), sharex=True)
    for row_idx, trait in enumerate(traits):
        data = condition_data(trait, original, matched)
        for col_idx, (condition, title) in enumerate(CONDITIONS):
            ax = axes[row_idx, col_idx]
            draw_panel(ax, data[condition], title if row_idx == 0 else "", show_ylabel=col_idx == 0)
            ax.set_box_aspect(0.55)
            ax.title.set_fontsize(9.4)
            ax.tick_params(axis="both", which="major", pad=1.5)
            ax.xaxis.labelpad = 1.5
            ax.yaxis.labelpad = 1.5
            if col_idx == 0:
                ax.text(
                    -0.38,
                    0.5,
                    TRAIT_LABELS.get(trait, trait),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    fig.set_size_inches(14.5, 1.75 * len(traits))
    fig.tight_layout(h_pad=0.18, w_pad=0.65)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    set_smile_style()
    original = pd.read_csv(DATA_DIR / "bbj_original_fit_loci.tsv", sep="\t")
    matched = pd.read_csv(DATA_DIR / "bbj_matched_loci.tsv", sep="\t")
    traits = trait_order(matched["trait"].unique())
    split = int(np.ceil(len(traits) / 2))
    write_page(traits[:split], original, matched, OUT_PART1)
    write_page(traits[split:], original, matched, OUT_PART2)
    print(f"Wrote {OUT_PART1}")
    print(f"Wrote {OUT_PART2}")


if __name__ == "__main__":
    main()
