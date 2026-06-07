import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPLICATE_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "data", "sims", "graphld_sims", "replicate")
)
SELECTED_DATA_PATH = os.path.join(REPLICATE_DIR, "simulated_ldl_replicate_seed_20260521_loci.tsv")
NEUTRAL_DATA_PATH = os.path.join(REPLICATE_DIR, "simulated_neutral_ldl_replicate_seed_30260521_loci.tsv")

OUT_PDF = os.path.join(SCRIPT_DIR, "graphld_ldl_replicate_example_seed_20260521_smiles.pdf")
OUT_TSV = os.path.join(SCRIPT_DIR, "graphld_ldl_replicate_example_seed_20260521_smiles_counts.tsv")

MIN_X = 0.01
P_THRESH = 5e-8

ORDER = ["causal", "beta_p", "beta_p_wc", "beta_p_post"]
PANEL_TITLES = {
    "causal": "causal effects",
    "beta_p": "lead estimated effects",
    "beta_p_wc": "lead WC effects",
    "beta_p_post": "lead posterior-mean effects",
}

matplotlib.rcParams.update({"font.size": 10})
matplotlib.rcParams["figure.facecolor"] = "#ffffff"
matplotlib.rcParams["axes.facecolor"] = "#ffffff"
matplotlib.rcParams["savefig.facecolor"] = "#ffffff"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.style.use("bmh")
matplotlib.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})


def discovery_curve(ax, v_cut, y_max):
    x_set = np.arange(MIN_X, 1, MIN_X)
    discov_betas = np.sqrt(v_cut / (2 * x_set * (1 - x_set)))
    if y_max > np.max(discov_betas):
        ax.plot(
            np.concatenate(([MIN_X], x_set, [1 - MIN_X])),
            np.concatenate(([y_max], discov_betas, [y_max])),
            color="darkslategrey",
            linestyle="dashed",
            linewidth=1.2,
        )
    else:
        ax.plot(x_set, discov_betas, color="darkslategrey", linestyle="dashed", linewidth=1.2)


def wc_mle_effect(beta_hat, se, threshold):
    if not np.isfinite(beta_hat) or not np.isfinite(se) or not np.isfinite(threshold):
        return np.nan
    if beta_hat <= 0 or se <= 0:
        return 0.0

    def neg_loglik(beta):
        z = (beta_hat - beta) / se
        t = (threshold - beta) / se
        return 0.5 * z * z + stats.norm.logsf(t)

    upper = max(beta_hat + 8 * se, threshold + 8 * se, 1e-12)
    res = optimize.minimize_scalar(neg_loglik, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-13})
    return max(float(res.x), 0.0) if res.success else np.nan


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


def load_sets(data_path):
    df = pd.read_csv(data_path, sep="\t")
    for col in ["causative", "lead", "secondary_causative"]:
        df[col] = df[col].astype(bool)
    df["pval"] = df["p"].astype(float)
    df["raf"] = df["raf"].astype(float)
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

    thresholds = np.sqrt(v_cut / (2.0 * freq_term))
    df["beta_p_wc"] = np.maximum(
        np.array(
            [
                wc_mle_effect(bb, ss, aa)
                for bb, ss, aa in zip(df["beta_p"].to_numpy(), se, thresholds)
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
        "causal": (causal, "beta_true_smiles", v_cut),
        "beta_p": (lead, "beta_p", v_cut),
        "beta_p_wc": (lead, "beta_p_wc", v_cut),
        "beta_p_post": (lead, "beta_p_post", v_cut),
    }


def main():
    selected_panel_data = load_sets(SELECTED_DATA_PATH)
    neutral_panel_data = load_sets(NEUTRAL_DATA_PATH)
    count_rows = []
    for condition, panel_data in [("selected", selected_panel_data), ("neutral", neutral_panel_data)]:
        count_rows.extend({"condition": condition, "dataset": key, "n": len(panel_data[key][0])} for key in ORDER)
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(OUT_TSV, sep="\t", index=False)

    all_beta = []
    all_v_cut = []
    for panel_data in [selected_panel_data, neutral_panel_data]:
        for key in ORDER:
            plot_df, beta_col, v_cut = panel_data[key]
            vals = plot_df[beta_col].to_numpy()
            all_beta.extend(vals[np.isfinite(vals) & (vals > 0)])
            all_v_cut.append(v_cut)
    y_max = max(
        float(np.max(all_beta)),
        max(np.sqrt(v / (2.0 * MIN_X * (1.0 - MIN_X))) for v in all_v_cut),
    ) * 1.35
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.0), sharex=True, sharey=True)
    row_specs = [
        ("Selected example", selected_panel_data),
        ("Neutral example", neutral_panel_data),
    ]
    for row_idx, (row_label, panel_data) in enumerate(row_specs):
        for col_idx, key in enumerate(ORDER):
            ax = axes[row_idx, col_idx]
            plot_df, beta_col, v_cut = panel_data[key]
            discovery_curve(ax, v_cut, y_max)
            ax.scatter(
                plot_df["raf"],
                plot_df[beta_col],
                s=18,
                alpha=0.42,
                color="#4C78A8",
                edgecolors="none",
                rasterized=True,
            )
            ax.text(
                0.05,
                0.93,
                "n=" + str(plot_df.shape[0]),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.75, pad=0.25),
            )
            if row_idx == 0:
                ax.set_title(PANEL_TITLES[key])
            if col_idx == 0:
                ax.set_ylabel(row_label + "\n" + r"$|\beta|$")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, y_max)
            ax.grid(alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("RAF")

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    print(" -", OUT_PDF)
    print(" -", OUT_TSV)


if __name__ == "__main__":
    main()
