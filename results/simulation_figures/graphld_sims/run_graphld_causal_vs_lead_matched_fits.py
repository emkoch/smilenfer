import argparse
import glob
import os
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import special, stats

import smilenfer.plotting as splot
import smilenfer.posterior as spost
import smilenfer.simulation as sim
import smilenfer.statistics as sstats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "sims", "graphld_sims"))
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "graphld_adjusted_lead_input_summary.tsv")
SFS_PILE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "SFS_pile", "tenn_eur_pile.pkl"))
COUNT_PATH = os.path.join(SCRIPT_DIR, "graphld_causal_vs_lead_matched_counts.tsv")
FIT_PKL_PATH = os.path.join(SCRIPT_DIR, "opt_fits_graphld_causal_vs_lead_matched.pkl")
FIT_CSV_PATH = os.path.join(SCRIPT_DIR, "opt_results_graphld_causal_vs_lead_matched.csv")
PVAL_PDF_PATH = os.path.join(SCRIPT_DIR, "graphld_causal_vs_lead_matched_pval.pdf")
PVAL_SYMLOG_PDF_PATH = os.path.join(SCRIPT_DIR, "graphld_causal_vs_lead_matched_pval_symlog.pdf")

MIN_X = 0.01
P_THRESH = 5e-8
P_CUTOFF = 5e-8
N_E = 10000
DEFAULT_N_SUBSAMPLES = 20
RNG_SEED = 20260416

TRAIT_ORDER = ["height", "ldl", "dbp", "fvc", "grip_strength", "asthma", "arthrosis"]
TRAIT_GROUP_BREAKS = [5]

MODEL_STYLES = {
    "dir": {"label": "Directional", "marker": ">", "color": "#FFB000"},
    "stab": {"label": "Single-trait stabilizing", "marker": "s", "color": "#DC267F"},
    "full": {"label": "Directional + stabilizing", "marker": "D", "color": "#785EF0"},
    "plei": {"label": "Pleiotropic stabilizing", "marker": "o", "color": "#FE6100"},
}
MODEL_ORDER = ["dir", "stab", "full", "plei"]
MODEL_OFFSETS = {"dir": -0.30, "stab": -0.10, "full": 0.10, "plei": 0.30}

splot._plot_params()
matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11.0,
        "axes.titlesize": 13.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.0,
        "legend.fontsize": 10.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 9.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def stable_chi2_log10sf(x, df):
    x = np.asarray(x, dtype=float)
    ln_sf = stats.chi2.logsf(x, df)
    out = np.empty_like(ln_sf, dtype=float)
    finite = np.isfinite(ln_sf)
    out[finite] = -ln_sf[finite] / np.log(10)
    if np.any(~finite):
        a = df / 2.0
        z = x[~finite] / 2.0
        asym_ln = (a - 1.0) * np.log(z) - z - special.gammaln(a)
        corr = 1.0 + (a - 1.0) / z
        out[~finite] = -(asym_ln / np.log(10) + np.log10(corr))
    return out


def trait_from_path(path):
    trait_name = os.path.basename(path).replace(".tsv.gz", "").replace(".tsv", "")
    return trait_name.split("_seed_")[0].replace("simulated_", "")


def get_input_paths():
    trait_paths = sorted(glob.glob(os.path.join(DATA_DIR, "simulated_*_loci.tsv.gz")))
    return trait_paths


def load_all_adjusted(raw_path, n_eff_fit):
    df = pd.read_csv(raw_path, sep="\t")
    df["lead"] = df["lead"].astype(bool)
    df["causative"] = df["causative"].astype(bool)
    df["pval"] = df["p"].astype(float)
    chi2_stat = stats.chi2.isf(df["pval"].to_numpy(), df=1)
    df["rbeta"] = np.sqrt(chi2_stat / (2.0 * n_eff_fit * df["raf"].to_numpy() * (1.0 - df["raf"].to_numpy())))
    df["maf"] = np.minimum(df["raf"], 1.0 - df["raf"])
    df["var_exp"] = 2.0 * df["raf"] * (1.0 - df["raf"]) * df["rbeta"] ** 2
    return df


def fit_subset(sfs_pile_eur, raf, beta, v_cut):
    result = sstats.infer_all_standard(
        sfs_pile_eur,
        N_E,
        raf,
        beta,
        v_cut,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
    )
    result = sstats.correct_all_standard_first_mode(
        result,
        sfs_pile_eur,
        N_E,
        raf,
        beta,
        v_cut,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
    )
    return result


def prepare_fit_table(df):
    fit_df = df.copy()
    for model in ["neut"] + MODEL_ORDER:
        n_par = 0 if model == "neut" else (2 if model == "full" else 1)
        fit_df["ll_" + model] = -(2 * n_par - 2 * fit_df["ll_" + model].to_numpy())
    for model in MODEL_ORDER:
        dfree = 2 if model == "full" else 1
        delta = fit_df["ll_" + model] - fit_df["ll_neut"]
        fit_df["stat_" + model] = stable_chi2_log10sf(delta, dfree)
    return fit_df


def build_trait_sets(summary_df):
    n_eff_map = dict(zip(summary_df["trait"], summary_df["n_eff_fit"]))
    label_map = dict(zip(summary_df["trait"], summary_df["label"]))
    trait_sets = {}
    count_rows = []

    for trait_path in get_input_paths():
        trait = trait_from_path(trait_path)
        n_eff_fit = float(n_eff_map[trait])
        v_cut = stats.chi2.isf(P_THRESH, df=1) / n_eff_fit
        all_df = load_all_adjusted(trait_path, n_eff_fit)
        keep = (
            (all_df["var_exp"].to_numpy() > v_cut)
            & (all_df["maf"].to_numpy() >= MIN_X)
            & (all_df["pval"].to_numpy() <= P_CUTOFF)
        )
        causal_df = all_df.loc[keep & all_df["causative"].to_numpy()].copy().reset_index(drop=True)
        lead_df = all_df.loc[keep & all_df["lead"].to_numpy()].copy().reset_index(drop=True)
        n_target = int(min(causal_df.shape[0], lead_df.shape[0]))

        trait_sets[trait] = {
            "label": label_map[trait],
            "n_eff_fit": n_eff_fit,
            "v_cut": v_cut,
            "n_target": n_target,
            "causal": causal_df,
            "lead": lead_df,
        }
        count_rows.append(
            {
                "trait": trait,
                "label": label_map[trait],
                "n_causal_keep": int(causal_df.shape[0]),
                "n_lead_keep": int(lead_df.shape[0]),
                "n_target": n_target,
                "causal_subsampled": int(causal_df.shape[0] > n_target),
                "lead_subsampled": int(lead_df.shape[0] > n_target),
                "n_eff_fit": float(n_eff_fit),
                "v_cut": float(v_cut),
            }
        )

    return trait_sets, pd.DataFrame(count_rows)


def run_fits(trait_sets, n_subsamples):
    with open(SFS_PILE, "rb") as f:
        sfs_pile_eur = sim.truncate_pile(pickle.load(f), 1e-8)
    rng = np.random.default_rng(RNG_SEED)

    all_results = {"causal": {}, "lead": {}}
    meta_rows = []

    for trait in TRAIT_ORDER:
        trait_spec = trait_sets[trait]
        for dataset in ["causal", "lead"]:
            data_df = trait_spec[dataset]
            n_available = int(data_df.shape[0])
            n_target = int(trait_spec["n_target"])
            if n_target == 0:
                continue

            result_list = []
            n_draws = 1 if n_available == n_target else n_subsamples
            print(
                "Fitting",
                trait,
                dataset + ":",
                "n_available=" + str(n_available) + ",",
                "n_target=" + str(n_target) + ",",
                "n_draws=" + str(n_draws),
                flush=True,
            )

            for sample_idx in range(n_draws):
                if n_available == n_target:
                    sub_df = data_df
                else:
                    take = rng.choice(n_available, size=n_target, replace=False)
                    sub_df = data_df.iloc[np.sort(take)].copy().reset_index(drop=True)

                result = fit_subset(
                    sfs_pile_eur,
                    sub_df["raf"].to_numpy(),
                    sub_df["rbeta"].to_numpy(),
                    float(trait_spec["v_cut"]),
                )
                result["trait"] = trait
                result["sample"] = sample_idx
                result_list.append(result)
                meta_rows.append(
                    {
                        "trait": trait,
                        "dataset": dataset,
                        "sample": sample_idx,
                        "n_available": n_available,
                        "n_target": n_target,
                        "subsampled": int(n_available > n_target),
                    }
                )

            all_results[dataset][trait] = result_list

    return all_results, pd.DataFrame(meta_rows)


def flatten_results(all_results, meta_df):
    fit_frames = []
    for dataset, result_dict in all_results.items():
        fit_df = spost.prepare_data_from_opt_results(result_dict)
        fit_df["dataset"] = dataset
        fit_frames.append(fit_df)
    fit_df = pd.concat(fit_frames, ignore_index=True)
    fit_df = fit_df.merge(meta_df, on=["trait", "dataset", "sample"], how="left")
    return prepare_fit_table(fit_df)


def plot_comparison(fit_df, count_df, out_pdf, symlog=False):
    label_map = dict(zip(count_df["trait"], count_df["label"]))
    count_map = count_df.set_index("trait")
    xticks = np.arange(len(TRAIT_ORDER), dtype=float)
    xticklabels = [label_map[trait] + "\n(" + str(int(count_map.loc[trait, "n_target"])) + ")" for trait in TRAIT_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8), sharey=True)
    dataset_titles = {"causal": "Causal smiles", "lead": "Lead smiles"}

    ymax_data = np.ceil(max(float(np.nanmax(fit_df["stat_" + model].to_numpy())) for model in MODEL_ORDER) * 1.12)
    ymax = max(20.0 if symlog else 6.0, ymax_data)

    for ax, dataset in zip(axes, ["causal", "lead"]):
        dataset_df = fit_df.loc[fit_df["dataset"] == dataset].copy()
        for model in MODEL_ORDER:
            model_style = MODEL_STYLES[model]
            for ii, trait in enumerate(TRAIT_ORDER):
                trait_df = dataset_df.loc[dataset_df["trait"] == trait].copy()
                if trait_df.empty:
                    continue

                x0 = xticks[ii] + MODEL_OFFSETS[model]
                vals = trait_df["stat_" + model].to_numpy()
                if len(vals) > 1:
                    jitter = np.linspace(-0.025, 0.025, len(vals))
                    ax.scatter(
                        x0 + jitter,
                        vals,
                        marker=model_style["marker"],
                        s=48,
                        facecolors="none",
                        edgecolors=model_style["color"],
                        linewidths=0.8,
                        alpha=0.35,
                        zorder=2,
                    )
                    center_val = float(np.median(vals))
                else:
                    center_val = float(vals[0])

                ax.scatter(
                    [x0],
                    [center_val],
                    marker=model_style["marker"],
                    s=92,
                    facecolor=model_style["color"],
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=3,
                )

        ax.axhline(0, color="#7A7A7A", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
        ax.axhline(-np.log10(0.05), color="#C44E52", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
        ax.axhline(-np.log10(0.05 / len(TRAIT_ORDER)), color="#6F3E8B", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
        for break_idx in TRAIT_GROUP_BREAKS:
            ax.axvline(break_idx - 0.5, color="#BBBBBB", linewidth=0.8, zorder=1)

        ax.set_xlim(-0.65, len(TRAIT_ORDER) - 0.35)
        ax.set_ylim(-0.4, ymax)
        if symlog:
            ax.set_yscale("symlog", linthresh=10)
            yticks = [2.0, 5.0, 10.0, 20.0]
            ax.set_yticks(yticks)
            ax.set_yticklabels(["2", "5", "10", "20"])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_title(dataset_titles[dataset])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$-\log_{10} \mathrm{p-value}$")

    model_handles = []
    for model in MODEL_ORDER:
        model_handles.append(
            Line2D(
                [0],
                [0],
                marker=MODEL_STYLES[model]["marker"],
                linestyle="None",
                markerfacecolor=MODEL_STYLES[model]["color"],
                markeredgecolor="black",
                markeredgewidth=0.45,
                markersize=8,
                label=MODEL_STYLES[model]["label"],
            )
        )

    sample_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=7,
            label="single full fit",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="subsample fits",
        ),
    ]
    ref_handles = [
        Line2D([0], [0], color="#7A7A7A", linestyle=(0, (4, 2)), linewidth=0.8, label="neutral"),
        Line2D([0], [0], color="#C44E52", linestyle=(0, (4, 2)), linewidth=0.8, label="nominal"),
        Line2D([0], [0], color="#6F3E8B", linestyle=(0, (4, 2)), linewidth=0.8, label="adjusted"),
    ]

    axes[1].legend(handles=model_handles, loc="upper left", bbox_to_anchor=(1.01, 1.01), frameon=False, title="Selection model")
    fig.legend(handles=sample_handles + ref_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subsamples", type=int, default=DEFAULT_N_SUBSAMPLES)
    args = parser.parse_args()

    summary_df = pd.read_csv(SUMMARY_PATH, sep="\t")
    summary_df = summary_df.loc[summary_df["trait"].isin(TRAIT_ORDER)].copy().reset_index(drop=True)
    trait_sets, count_df = build_trait_sets(summary_df)
    count_df.to_csv(COUNT_PATH, sep="\t", index=False)

    all_results, meta_df = run_fits(trait_sets, args.n_subsamples)
    with open(FIT_PKL_PATH, "wb") as f:
        pickle.dump(all_results, f)

    fit_df = flatten_results(all_results, meta_df)
    fit_df.to_csv(FIT_CSV_PATH, index=False)

    plot_comparison(fit_df, count_df, PVAL_PDF_PATH, symlog=False)
    plot_comparison(fit_df, count_df, PVAL_SYMLOG_PDF_PATH, symlog=True)

    print("Wrote:")
    print(" -", COUNT_PATH)
    print(" -", FIT_PKL_PATH)
    print(" -", FIT_CSV_PATH)
    print(" -", PVAL_PDF_PATH)
    print(" -", PVAL_SYMLOG_PDF_PATH)


if __name__ == "__main__":
    main()
