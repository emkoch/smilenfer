import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

import smilenfer.simulation as sim
import smilenfer.statistics as sstats
import smilenfer.posterior as spost


matplotlib.use("Agg")
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
fit_dir = os.path.join(script_dir, "..", "all_opt_fits", "original_traits")
trait_dir = os.path.join(data_dir, "final", "original_traits")
fit_file = os.path.join(fit_dir, "opt_results_original_traits_eur_post.csv")
sfs_pile_file = os.path.join(data_dir, "SFS_pile", "tenn_eur_pile.pkl")
plot_name = os.path.join(script_dir, "pleiotropy_locus_contribution_diagnostic.pdf")
trait_names = dict(spost.original_trait_names)
trait_names.update({"height": "Height", "hdl": "HDL", "ldl": "LDL"})

p_thresh = 5e-08
min_x = 0.01
ne = 10000
n_points = 1000
n_x = 1000

plot_traits = [
    "height",
    "bmi",
    "hdl",
    "rbc",
    "bc",
    "cad",
    "scz",
    "t2d",
]
disease_traits = [
    "arthrosis",
    "asthma",
    "bc",
    "cad",
    "diverticulitis",
    "gallstones",
    "glaucoma",
    "hypothyroidism",
    "ibd",
    "malignant_neoplasms",
    "scz",
    "t2d",
    "uterine_fibroids",
    "varicose_veins",
]

matplotlib.rcParams.update({"figure.facecolor": "white"})
matplotlib.rcParams.update({"axes.facecolor": "white"})
matplotlib.rcParams.update({"savefig.facecolor": "white"})
matplotlib.rcParams.update({"axes.edgecolor": "#222222"})
matplotlib.rcParams.update({"axes.linewidth": 0.8})
matplotlib.rcParams.update({"font.family": "sans-serif"})
matplotlib.rcParams.update({"font.sans-serif": ["DejaVu Sans"]})
matplotlib.rcParams.update({"font.size": 8.5})
matplotlib.rcParams.update({"axes.labelsize": 9.6})
matplotlib.rcParams.update({"axes.titlesize": 9.4})
matplotlib.rcParams.update({"xtick.labelsize": 8.2})
matplotlib.rcParams.update({"ytick.labelsize": 8.2})
matplotlib.rcParams.update({"legend.fontsize": 8.4})
matplotlib.rcParams.update({"pdf.fonttype": 42})
matplotlib.rcParams.update({"ps.fonttype": 42})


def load_trait_data(trait):
    path = os.path.join(trait_dir, f"processed.{trait}.snps_low_r2.tsv")
    trait_df = pd.read_csv(path, sep="\t")
    n_eff_median = np.nanmedian(trait_df["n_eff"])
    v_cut = stats.chi2.isf(p_thresh, df=1) / n_eff_median

    raf = trait_df["raf"].to_numpy()
    beta_obs = trait_df["rbeta"].to_numpy()
    beta = trait_df["PosteriorMean"].to_numpy()
    v_obs = 2 * raf * (1 - raf) * beta_obs**2
    keep = v_obs > v_cut

    out = trait_df.loc[keep].copy()
    out["trait"] = trait
    out["trait_label"] = trait_names.get(trait, trait)
    out["beta_fit"] = beta[keep]
    out["beta_obs"] = beta_obs[keep]
    out["v_obs"] = v_obs[keep]
    out["v_cut"] = v_cut
    out["v_ratio"] = out["v_obs"] / v_cut
    out["maf"] = np.minimum(out["raf"], 1 - out["raf"])
    return out


def pointwise_ll(grid_x, model_grid, selection_grid, raf, beta, beta_obs, v_cut, intensity):
    d_x = np.maximum(sstats.discov_x(beta_obs, v_cut), min_x)
    scaled_selection = 2 * ne * beta**2 * intensity
    ll = np.zeros(len(raf))
    for i in range(len(raf)):
        ll[i] = sstats.variant_ll(
            grid_x,
            model_grid,
            selection_grid,
            scaled_selection[i],
            raf[i],
            beta[i],
            v_cut,
            min_x,
            n_x,
            d_x=d_x[i],
        )
    return ll


def build_locus_table():
    fits = pd.read_csv(fit_file)
    with open(sfs_pile_file, "rb") as handle:
        sfs_pile = sim.truncate_pile(pickle.load(handle), 1e-8)

    grid_x_stab, selection_stab, tau_stab = sstats.build_simple_grid(
        sfs_pile,
        min_x=min_x,
        n_points=n_points,
    )
    grid_x_plei, _, _, selection_plei, grid_plei = sstats.build_integration_grid(
        sfs_pile,
        min_x=min_x,
        n_points=n_points,
    )

    rows = []
    for fit in fits.itertuples(index=False):
        trait_df = load_trait_data(fit.trait)
        raf = trait_df["raf"].to_numpy()
        beta = trait_df["beta_fit"].to_numpy()
        beta_obs = trait_df["beta_obs"].to_numpy()
        v_cut = float(trait_df["v_cut"].iloc[0])

        ll_stab = pointwise_ll(
            grid_x_stab,
            tau_stab.T,
            selection_stab,
            raf,
            beta,
            beta_obs,
            v_cut,
            fit.I2_stab,
        )
        ll_plei = pointwise_ll(
            grid_x_plei,
            grid_plei,
            selection_plei,
            raf,
            beta,
            beta_obs,
            v_cut,
            fit.Ip_plei,
        )

        trait_df["ll_stab_locus"] = ll_stab
        trait_df["ll_plei_locus"] = ll_plei
        trait_df["delta_ll_plei_stab"] = ll_plei - ll_stab
        rows.append(
            trait_df.loc[:,
                [
                    "trait",
                    "trait_label",
                    "snp",
                    "raf",
                    "maf",
                    "beta_obs",
                    "beta_fit",
                    "v_obs",
                    "v_cut",
                    "v_ratio",
                    "ll_stab_locus",
                    "ll_plei_locus",
                    "delta_ll_plei_stab",
                ],
            ]
        )

    return pd.concat(rows, ignore_index=True)


def plot_examples(loci):
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.6))
    axes = axes.flatten()

    cmap = plt.get_cmap("coolwarm")
    for idx, trait in enumerate(plot_traits):
        ax = axes[idx]
        trait_loci = loci.loc[loci["trait"] == trait]
        x_label = "RAF" if trait in disease_traits else "Trait-increasing AF"
        vmax = np.nanquantile(np.abs(trait_loci["delta_ll_plei_stab"]), 0.97)
        vmax = max(vmax, 0.08)
        colors = trait_loci["delta_ll_plei_stab"].clip(-vmax, vmax)
        sc = ax.scatter(
            trait_loci["raf"],
            np.log10(trait_loci["v_ratio"]),
            c=colors,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            s=15 if len(trait_loci) < 300 else 10,
            linewidth=0.15,
            edgecolor="none",
            alpha=0.92,
        )
        ax.axhline(0, color="#444444", linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, max(1.0, np.nanmax(np.log10(trait_loci["v_ratio"])) * 1.08))
        ax.set_title(trait_names.get(trait, trait))
        ax.set_xlabel(x_label)
        if idx in {0, 4}:
            ax.set_ylabel(r"$\log_{10}(\hat v/v^*)$")
        ax.grid(alpha=0.2, linewidth=0.5)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.018)
        cbar.set_ticks([-vmax, 0, vmax])
        cbar.ax.set_yticklabels([f"{-vmax:.2g}", "0", f"{vmax:.2g}"])
        cbar.ax.tick_params(labelsize=6.7, length=2)

    fig.text(
        0.985,
        0.5,
        r"$\Delta \ell_i$ pleiotropic - 1D",
        rotation=90,
        va="center",
        ha="center",
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.11,
        top=0.92,
        wspace=0.46,
        hspace=0.34,
    )
    fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)

loci = build_locus_table()
plot_examples(loci)
