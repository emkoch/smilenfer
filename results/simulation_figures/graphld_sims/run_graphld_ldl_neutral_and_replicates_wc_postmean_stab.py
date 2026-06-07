import os
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import smilenfer.simulation as sim
import smilenfer.statistics as sstats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "data", "sims", "graphld_sims")
)
REPLICATE_DIR = os.path.join(DATA_DIR, "replicate")
SFS_PILE = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "data", "SFS_pile", "tenn_eur_pile.pkl")
)

OUT_CSV = os.path.join(SCRIPT_DIR, "graphld_ldl_neutral_and_replicates_wc_postmean_stab.csv")
OUT_PDF = os.path.join(SCRIPT_DIR, "graphld_ldl_neutral_and_replicates_wc_postmean_stab_pvals.pdf")

MIN_X = 0.01
P_THRESH = 5e-8
N_E = 10000
SIGN_SEED = 20260504 + sum((ii + 1) * ord(char) for ii, char in enumerate("ldl"))

ORDER = [
    ("neutral", "causal"),
    ("neutral", "beta_p"),
    ("neutral", "beta_p_wc"),
    ("neutral", "beta_p_post"),
    ("selected", "causal"),
    ("selected", "beta_p"),
    ("selected", "beta_p_wc"),
    ("selected", "beta_p_post"),
]
LABELS = [
    "Neutral\ncausal",
    "Neutral\nlead",
    "Neutral\nWC",
    "Neutral\npost.",
    "Selected\ncausal",
    "Selected\nlead",
    "Selected\nWC",
    "Selected\npost.",
]
COLORS = {
    "causal": "#648FFF",
    "beta_p": "#DC267F",
    "beta_p_wc": "#FE6100",
    "beta_p_post": "#785EF0",
}

matplotlib.rcParams.update({"font.size": 10})
matplotlib.rcParams["figure.facecolor"] = "#ffffff"
matplotlib.rcParams["axes.facecolor"] = "#ffffff"
matplotlib.rcParams["savefig.facecolor"] = "#ffffff"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.style.use("bmh")
matplotlib.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})


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


def wc_mle_effect(beta_hat, se, threshold):
    if not np.isfinite(beta_hat) or not np.isfinite(se) or not np.isfinite(threshold):
        return np.nan
    if beta_hat <= 0 or se <= 0:
        return 0.0

    def neg_loglik(beta):
        z = (beta_hat - beta) / se
        t = (threshold - beta) / se
        return 0.5 * z * z + stats.norm.logsf(t)

    upper = max(beta_hat + 8 * se, threshold + 8 * se, 1.0e-12)
    res = optimize.minimize_scalar(neg_loglik, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-13})
    if not res.success:
        return np.nan
    return max(float(res.x), 0.0)


def assign_neutral_raf(df):
    rng = np.random.default_rng(SIGN_SEED)
    signs = rng.choice([-1, 1], size=len(df))
    return np.where(signs > 0, df["maf"].to_numpy(), 1.0 - df["maf"].to_numpy())


def estimate_postmean_prior_var(beta_hat, se):
    z_thresh = stats.norm.isf(P_THRESH / 2.0)

    def neg_loglik(log_sigma2):
        sigma2 = np.exp(log_sigma2)
        total_var = sigma2 + se**2
        ll_obs = stats.norm.logpdf(beta_hat, loc=0.0, scale=np.sqrt(total_var))
        thresh_prob = 2.0 * stats.norm.sf(z_thresh * se / np.sqrt(total_var))
        if np.any(thresh_prob <= 0) or np.any(~np.isfinite(thresh_prob)):
            return np.inf
        return -np.sum(ll_obs - np.log(thresh_prob))

    init_sigma2 = max(float(np.mean(beta_hat**2 - se**2)), 1e-12)
    result = optimize.minimize_scalar(
        neg_loglik,
        bounds=(np.log(1e-12), np.log(max(1.0, init_sigma2 * 1000.0))),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        return init_sigma2
    return float(np.exp(result.x))


def load_table(path):
    df = pd.read_csv(path, sep="\t")
    for col in ["causative", "lead", "secondary_causative"]:
        df[col] = df[col].astype(bool)
    df["pval"] = df["p"].astype(float)
    if "raf" in df.columns:
        df["raf"] = df["raf"].astype(float)
    else:
        df["raf"] = assign_neutral_raf(df)
    if "maf" in df.columns:
        df["maf"] = df["maf"].astype(float)
    else:
        df["maf"] = np.minimum(df["raf"], 1.0 - df["raf"])
    df["maf"] = np.minimum(df["raf"], 1.0 - df["raf"])
    df["beta_abs"] = np.abs(df["beta_abs"].astype(float))

    n_eff = float(df["n_eff"].median())
    v_cut = float(stats.chi2.isf(P_THRESH, df=1) / n_eff)
    freq_term = df["raf"].to_numpy() * (1.0 - df["raf"].to_numpy())
    chi2_stat = stats.chi2.isf(df["pval"].to_numpy(), df=1)
    se = 1.0 / np.sqrt(2.0 * n_eff * freq_term)

    df["beta_p"] = np.sqrt(chi2_stat / (2.0 * n_eff * freq_term))
    df["beta_true_smiles"] = df["beta_abs"] / np.sqrt(freq_term)
    df["v_beta_p"] = 2.0 * freq_term * df["beta_p"].to_numpy() ** 2
    df["v_beta_true_smiles"] = 2.0 * freq_term * df["beta_true_smiles"].to_numpy() ** 2
    df["beta_p_wc"] = np.maximum(
        np.array(
            [
                wc_mle_effect(bb, ss, aa)
                for bb, ss, aa in zip(
                    df["beta_p"].to_numpy(),
                    se,
                    np.sqrt(v_cut / (2.0 * freq_term)),
                )
            ]
        ),
        1e-12,
    )
    lead_mask = (
        df["lead"].to_numpy()
        & (df["maf"].to_numpy() >= MIN_X)
        & (df["pval"].to_numpy() <= P_THRESH)
        & np.isfinite(df["beta_p"].to_numpy())
        & (df["v_beta_p"].to_numpy() > v_cut)
        & np.isfinite(se)
        & (se > 0)
    )
    prior_var = estimate_postmean_prior_var(df.loc[lead_mask, "beta_p"].to_numpy(), se[lead_mask])
    shrink = prior_var / (prior_var + se**2)
    df["beta_p_post"] = np.maximum(df["beta_p"].to_numpy() * shrink, 1e-12)
    return df, n_eff, v_cut


def build_specs(path):
    df, n_eff, v_cut = load_table(path)
    causal = df.loc[
        df["causative"]
        & (df["maf"] >= MIN_X)
        & np.isfinite(df["beta_true_smiles"])
        & (df["v_beta_true_smiles"] > v_cut)
    ].copy()
    causal = causal.reset_index(drop=True)
    lead = df.loc[
        df["lead"]
        & (df["maf"] >= MIN_X)
        & (df["pval"] <= P_THRESH)
        & np.isfinite(df["beta_p"])
        & (df["v_beta_p"] > v_cut)
    ].copy()
    lead = lead.reset_index(drop=True)
    return {
        "causal": {"data": causal, "beta_col": "beta_true_smiles", "n_eff": n_eff, "v_cut": v_cut},
        "beta_p": {"data": lead, "beta_col": "beta_p", "n_eff": n_eff, "v_cut": v_cut},
        "beta_p_wc": {"data": lead, "beta_col": "beta_p_wc", "n_eff": n_eff, "v_cut": v_cut},
        "beta_p_post": {"data": lead, "beta_col": "beta_p_post", "n_eff": n_eff, "v_cut": v_cut},
    }


def fit_stab(sfs_pile, spec):
    df = spec["data"]
    beta = df[spec["beta_col"]].to_numpy()
    raf = df["raf"].to_numpy()
    v_cut = spec["v_cut"]
    ll_neut = sstats.llhood_s(
        sfs_pile,
        N_E,
        raf,
        beta,
        v_cut,
        1e-9,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
    )
    fit = sstats.infer_I2(
        sfs_pile,
        N_E,
        raf,
        beta,
        v_cut,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
    )
    ll_stab = -fit.fun
    lrt_stab = 2 * (ll_stab - ll_neut)
    return {
        "n": int(len(df)),
        "n_eff": float(spec["n_eff"]),
        "v_cut": float(v_cut),
        "ll_neut": float(ll_neut),
        "ll_stab": float(ll_stab),
        "I2_stab": float(10 ** fit.x[0]),
        "dLL_stab": float(ll_stab - ll_neut),
        "dLL_per_locus_stab": float((ll_stab - ll_neut) / len(df)),
        "stat_stab": float(stable_chi2_log10sf(lrt_stab, 1)),
    }


def plot_pvals(fit_df):
    rank_maps = {}
    for condition in ["neutral", "selected"]:
        causal_condition = fit_df.loc[
            (fit_df["condition"] == condition) & (fit_df["dataset"] == "causal"),
            ["replicate", "stat_stab"],
        ].copy()
        causal_condition = (
            causal_condition.sort_values("stat_stab", ascending=False)
            .reset_index(drop=True)
        )
        rank_maps[condition] = {rep: ii for ii, rep in enumerate(causal_condition["replicate"].tolist())}

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(ORDER))
    for ii, (condition, dataset) in enumerate(ORDER):
        color = COLORS[dataset]
        sub = fit_df.loc[(fit_df["condition"] == condition) & (fit_df["dataset"] == dataset)].copy()
        sub["rep_rank"] = sub["replicate"].map(rank_maps[condition])
        sub = sub.sort_values("rep_rank").reset_index(drop=True)
        vals = sub["stat_stab"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), x[ii]) + jitter,
            vals,
            s=28,
            alpha=0.45,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
    ax.axhline(-np.log10(0.05), color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel(r"$-\log_{10} \mathrm{p-value}$")
    ax.set_title("Trait simulation comparison")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-0.1, max(4.0, 0.35 * fit_df["stat_stab"].max()) + fit_df["stat_stab"].max())
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    with open(SFS_PILE, "rb") as f:
        sfs_pile = sim.truncate_pile(pickle.load(f), 1e-8)

    rows = []

    neutral_paths = sorted(
        os.path.join(REPLICATE_DIR, fname)
        for fname in os.listdir(REPLICATE_DIR)
        if fname.startswith("simulated_neutral_ldl_replicate_seed_") and fname.endswith(".tsv")
    )
    total_neutral = len(neutral_paths)
    for rep_index, path in enumerate(neutral_paths, start=1):
        rep_name = os.path.basename(path).replace("_loci.tsv", "")
        print("[neutral {}/{}] Loading {}".format(rep_index, total_neutral, rep_name), flush=True)
        specs = build_specs(path)
        for dataset in ["causal", "beta_p", "beta_p_wc", "beta_p_post"]:
            print("  fitting neutral", dataset, "n=" + str(len(specs[dataset]["data"])), flush=True)
            row = fit_stab(sfs_pile, specs[dataset])
            row["condition"] = "neutral"
            row["replicate"] = rep_name
            row["dataset"] = dataset
            row["beta_col"] = specs[dataset]["beta_col"]
            rows.append(row)

    paths = sorted(
        os.path.join(REPLICATE_DIR, fname)
        for fname in os.listdir(REPLICATE_DIR)
        if fname.startswith("simulated_ldl_replicate_seed_") and fname.endswith(".tsv")
    )
    total = len(paths)
    for rep_index, path in enumerate(paths, start=1):
        rep_name = os.path.basename(path).replace("_loci.tsv", "")
        print("[{}/{}] Loading {}".format(rep_index, total, rep_name), flush=True)
        specs = build_specs(path)
        for dataset in ["causal", "beta_p", "beta_p_wc", "beta_p_post"]:
            print("  fitting selected", dataset, "n=" + str(len(specs[dataset]["data"])), flush=True)
            row = fit_stab(sfs_pile, specs[dataset])
            row["condition"] = "selected"
            row["replicate"] = rep_name
            row["dataset"] = dataset
            row["beta_col"] = specs[dataset]["beta_col"]
            rows.append(row)

    fit_df = pd.DataFrame(rows)
    fit_df.to_csv(OUT_CSV, index=False)
    plot_pvals(fit_df)
    print("Wrote:")
    print(" -", OUT_CSV)
    print(" -", OUT_PDF)
    print(
        fit_df.groupby(["condition", "dataset"])[["n", "dLL_stab", "dLL_per_locus_stab", "stat_stab", "I2_stab"]]
        .median()
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
