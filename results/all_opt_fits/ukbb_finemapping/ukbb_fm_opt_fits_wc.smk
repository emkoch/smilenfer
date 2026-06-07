import os

BASE_DIR = "../data"
FINAL_DIR = os.path.join(BASE_DIR, "final", "UKBB_susiex")
ORIGINAL_DIR = os.path.join(BASE_DIR, "final", "original_traits")
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
MIN_X = 0.01
P_THRESH = 5e-8
N_SAMPLES = 1
SAMPLES = list(range(N_SAMPLES))

TRAITS = [
    "bmi",
    "dbp",
    "hdl",
    "height",
    "ldl",
    "sbp",
    "triglycerides",
    "wbc",
]

rule ukbb_fit_one_sample_wc:
    input:
        trait_file=os.path.join(FINAL_DIR, "susiex_cs_table_{trait}.csv"),
        original_trait_file=os.path.join(ORIGINAL_DIR, "processed.{trait}.snps_low_r2.tsv"),
        sfs_pile=SFS_PILE
    output:
        table=os.path.join("ukbb_finemapping", "samples", "{trait}_sample_{sample}.wc.tsv"),
        pkl=os.path.join("ukbb_finemapping", "samples", "{trait}_sample_{sample}.wc.pkl")
    params:
        trait="{trait}",
        sample=lambda wc: int(wc.sample)
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import optimize
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim
        from sample_finemapped import sample_finemap

        def inv_mills(tt):
            return np.exp(stats.norm.logpdf(tt) - stats.norm.logsf(tt))

        def wc_mle_effect(beta_hat, se, threshold):
            if not np.all(np.isfinite([beta_hat, se, threshold])):
                return np.nan
            if beta_hat <= 0 or se <= 0:
                return 0.0

            def residual(beta):
                return beta + se * inv_mills((threshold - beta) / se) - beta_hat

            lo = 0.0
            hi = float(beta_hat)
            f_lo = residual(lo)
            f_hi = residual(hi)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo <= 0 <= f_hi:
                res = optimize.brentq(residual, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=100)
                return min(max(float(res), 0.0), beta_hat)
            if np.isfinite(f_lo) and f_lo > 0:
                return 0.0
            return max(float(beta_hat), 0.0)

        trait = params.trait
        sample = params.sample
        trait_data = pd.read_csv(input.trait_file)
        original_trait_data = pd.read_csv(input.original_trait_file, sep="\t")

        n_eff_median = float(np.nanmedian(original_trait_data["n_eff"]))
        v_cut = float(stats.chi2.isf(P_THRESH, df=1) / n_eff_median)

        sampled_df = sample_finemap(trait_data)
        sampled_df["var_exp_raw"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
        sampled_df = sampled_df[sampled_df["var_exp_raw"] > v_cut].copy()
        sampled_df = sampled_df[sampled_df["raf"].between(MIN_X, 1 - MIN_X)].copy()
        raf = sampled_df["raf"].to_numpy(dtype=float)
        raw_beta = np.abs(sampled_df["rbeta"].to_numpy(dtype=float))
        se = sampled_df["se"].to_numpy(dtype=float)
        z_gws = np.sqrt(stats.chi2.isf(P_THRESH, df=1))
        beta_v_threshold = np.sqrt(v_cut / (2 * raf * (1 - raf)))
        beta_p_threshold = z_gws * se
        threshold = np.maximum(beta_v_threshold, beta_p_threshold)
        beta_wc = np.array([
            wc_mle_effect(bb, ss, aa)
            for bb, ss, aa in zip(raw_beta, se, threshold)
        ])

        sampled_df["rbeta_raw_wc"] = np.maximum(beta_wc, 1e-12)
        sampled_df["raw_wc_threshold"] = threshold
        sampled_df["raw_wc_beta_v_threshold"] = beta_v_threshold
        sampled_df["raw_wc_beta_p_threshold"] = beta_p_threshold
        sampled_df["raw_wc_distance_raw_se"] = (raw_beta - threshold) / se
        sampled_df["var_exp_raw_wc"] = 2 * raf * (1 - raf) * sampled_df["rbeta_raw_wc"].to_numpy(dtype=float) ** 2
        sampled_df["n_eff_median"] = n_eff_median
        sampled_df["v_cut"] = v_cut

        raf = sampled_df["raf"].to_numpy(dtype=float)
        beta_obs = np.abs(sampled_df["rbeta"].to_numpy(dtype=float))
        beta_wc = np.abs(sampled_df["rbeta_raw_wc"].to_numpy(dtype=float))

        sfs_pile = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)
        opt_result = sstats.infer_all_standard(
            sfs_pile,
            10000,
            raf,
            beta_wc,
            v_cut,
            min_x=MIN_X,
            n_points=1000,
            n_x=1000,
            beta_obs=beta_obs,
        )
        opt_result = sstats.correct_all_standard_first_mode(
            opt_result,
            sfs_pile,
            10000,
            raf,
            beta_wc,
            v_cut,
            min_x=MIN_X,
            n_points=1000,
            n_x=1000,
            beta_obs=beta_obs,
        )
        opt_result["sample"] = sample
        opt_result["trait"] = trait

        sampled_df.to_csv(output.table, sep="\t", index=False)
        with open(output.pkl, "wb") as f:
            pickle.dump(opt_result, f)


rule ukbb_aggregate_wc:
    input:
        pkls=expand(os.path.join("ukbb_finemapping", "samples", "{trait}_sample_{sample}.wc.pkl"), trait=TRAITS, sample=SAMPLES),
        tables=expand(os.path.join("ukbb_finemapping", "samples", "{trait}_sample_{sample}.wc.tsv"), trait=TRAITS, sample=SAMPLES)
    output:
        pkl=os.path.join("ukbb_finemapping", "opt_fits_ukbb_susiex_wc.pkl"),
        csv=os.path.join("ukbb_finemapping", "opt_results_ukbb_susiex_wc.csv")
    run:
        import pickle
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl_file in input.pkls:
            with open(pkl_file, "rb") as f:
                res = pickle.load(f)
            trait = res["trait"]
            if trait not in opt_results:
                opt_results[trait] = []
            opt_results[trait].append(res)

        with open(output.pkl, "wb") as f:
            pickle.dump(opt_results, f)
        spost.prepare_data_from_opt_results(opt_results).to_csv(output.csv, index=False)
