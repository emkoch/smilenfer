import os

BASE_DIR = "../data"
FINAL_DIR = os.path.join(BASE_DIR, "final", "original_traits")
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
MIN_X = 0.01
P_THRESH = 5e-8

TRAITS = [
    "arthrosis",
    "asthma",
    "bc",
    "bmi",
    "cad",
    "dbp",
    "diverticulitis",
    "fvc",
    "gallstones",
    "glaucoma",
    "grip_strength",
    "hdl",
    "height",
    "hypothyroidism",
    "ibd",
    "ldl",
    "malignant_neoplasms",
    "pulse_rate",
    "rbc",
    "sbp",
    "scz",
    "t2d",
    "triglycerides",
    "urate",
    "uterine_fibroids",
    "varicose_veins",
    "wbc",
]


rule original_traits_build_wc_inputs:
    input:
        trait_file=os.path.join(FINAL_DIR, "processed.{trait}.snps_low_r2.tsv")
    output:
        os.path.join("original_traits", "raw_wc_inputs", "processed.{trait}.snps_low_r2.raw_wc.tsv")
    run:
        import numpy as np
        import pandas as pd
        from scipy import optimize
        from scipy import stats
        import smilenfer.statistics as sstats

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

        dd = pd.read_csv(input.trait_file, sep="\t")
        n_eff_median = float(np.nanmedian(dd["n_eff"]))
        v_cut = float(stats.chi2.isf(P_THRESH, df=1) / n_eff_median)
        raf = dd["raf"].to_numpy(dtype=float)
        raw_beta = np.abs(dd["rbeta"].to_numpy(dtype=float))
        se = dd["se"].to_numpy(dtype=float)
        z_gws = np.sqrt(stats.chi2.isf(P_THRESH, df=1))
        beta_v_threshold = np.sqrt(v_cut / (2 * raf * (1 - raf)))
        beta_p_threshold = z_gws * se
        threshold = np.maximum(beta_v_threshold, beta_p_threshold)
        beta_wc = np.array([
            wc_mle_effect(bb, ss, aa)
            for bb, ss, aa in zip(raw_beta, se, threshold)
        ])

        dd["rbeta_raw_wc"] = np.maximum(beta_wc, 1e-12)
        dd["raw_wc_threshold"] = threshold
        dd["raw_wc_beta_v_threshold"] = beta_v_threshold
        dd["raw_wc_beta_p_threshold"] = beta_p_threshold
        dd["raw_wc_distance_raw_se"] = (raw_beta - threshold) / se
        dd["var_exp_raw_wc"] = 2 * raf * (1 - raf) * dd["rbeta_raw_wc"].to_numpy(dtype=float) ** 2
        dd["var_exp_raw"] = 2 * dd["raf"] * (1 - dd["raf"]) * dd["rbeta"] ** 2
        dd = dd[dd["var_exp_raw"] > v_cut].copy()
        dd = dd[dd["raf"].between(MIN_X, 1 - MIN_X)].copy()
        dd["n_eff_median"] = n_eff_median
        dd["v_cut"] = v_cut
        dd.to_csv(output[0], sep="\t", index=False)


rule original_traits_fit_wc:
    input:
        trait_file=os.path.join("original_traits", "raw_wc_inputs", "processed.{trait}.snps_low_r2.raw_wc.tsv"),
        sfs_pile=SFS_PILE
    output:
        os.path.join("original_traits", "fits_raw_wc", "{trait}_standard_fits_raw_wc.pkl")
    params:
        trait="{trait}"
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        dd = pd.read_csv(input.trait_file, sep="\t")
        raf = dd["raf"].to_numpy(dtype=float)
        beta_obs = np.abs(dd["rbeta"].to_numpy(dtype=float))
        beta_wc = np.abs(dd["rbeta_raw_wc"].to_numpy(dtype=float))
        v_cut = float(np.nanmedian(dd["v_cut"]))

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
        opt_result["trait"] = params.trait

        with open(output[0], "wb") as handle:
            pickle.dump(opt_result, handle)


rule original_traits_aggregate_wc:
    input:
        pkls=expand(os.path.join("original_traits", "fits_raw_wc", "{trait}_standard_fits_raw_wc.pkl"), trait=TRAITS),
        tables=expand(os.path.join("original_traits", "raw_wc_inputs", "processed.{trait}.snps_low_r2.raw_wc.tsv"), trait=TRAITS)
    output:
        pkl=os.path.join("original_traits", "opt_fits_original_traits_eur_wc.pkl"),
        csv=os.path.join("original_traits", "opt_results_original_traits_eur_wc.csv")
    run:
        import pickle
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl_file in input.pkls:
            with open(pkl_file, "rb") as handle:
                res = pickle.load(handle)
            opt_results[res["trait"]] = res

        with open(output.pkl, "wb") as handle:
            pickle.dump(opt_results, handle)

        opt_df = spost.prepare_data_from_opt_results(opt_results)
        opt_df.to_csv(output.csv, index=False)
