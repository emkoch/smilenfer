import os

BASE_DIR = "../data"
FINAL_DIR = os.path.join(BASE_DIR, "final", "bbj_traits")
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "joug_jpt_pile.pkl")
MIN_X = 0.01
P_THRESH = 5e-8

TRAITS = [
    "asthma",
    "bc",
    "bmi",
    "cad",
    "dbp",
    "gallstones",
    "hdl",
    "height",
    "ldl",
    "rbc",
    "sbp",
    "t2d",
    "triglycerides",
    "uterine_fibroids",
]


rule bbj_fit_high_wc:
    input:
        trait_file=os.path.join(FINAL_DIR, "processed.{trait}.max_r2.bbj.tsv"),
        sfs_pile=SFS_PILE
    output:
        table=os.path.join("bbj", "high_clumps", "high_{trait}.wc.tsv"),
        pkl=os.path.join("bbj", "high_clumps", "high_{trait}.wc.pkl")
    params:
        trait="{trait}"
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import optimize
        from scipy import stats
        import smilenfer.simulation as sim
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

        trait = params.trait
        dd = pd.read_csv(input.trait_file, sep="\t")

        n_eff_median = float(np.nanmedian(dd["n_eff"]))
        v_cut = float(stats.chi2.isf(P_THRESH, df=1) / n_eff_median)

        clumped = sstats.high_clump_trait_data(dd, dist=500000).copy()
        raf = clumped["raf"].to_numpy(dtype=float)
        raw_beta = np.abs(clumped["rbeta"].to_numpy(dtype=float))
        se = clumped["se"].to_numpy(dtype=float)
        z_gws = np.sqrt(stats.chi2.isf(P_THRESH, df=1))
        beta_v_threshold = np.sqrt(v_cut / (2 * raf * (1 - raf)))
        beta_p_threshold = z_gws * se
        threshold = np.maximum(beta_v_threshold, beta_p_threshold)
        beta_wc = np.array([
            wc_mle_effect(bb, ss, aa)
            for bb, ss, aa in zip(raw_beta, se, threshold)
        ])

        clumped["rbeta_raw_wc"] = np.maximum(beta_wc, 1e-12)
        clumped["raw_wc_threshold"] = threshold
        clumped["raw_wc_beta_v_threshold"] = beta_v_threshold
        clumped["raw_wc_beta_p_threshold"] = beta_p_threshold
        clumped["raw_wc_distance_raw_se"] = (raw_beta - threshold) / se
        clumped["var_exp_raw_wc"] = 2 * raf * (1 - raf) * clumped["rbeta_raw_wc"].to_numpy(dtype=float) ** 2
        clumped["var_exp_raw"] = 2 * clumped["raf"] * (1 - clumped["raf"]) * clumped["rbeta"] ** 2
        clumped = clumped[clumped["var_exp_raw"] > v_cut].copy()
        clumped = clumped[clumped["raf"].between(MIN_X, 1 - MIN_X)].copy()
        clumped["n_eff_median"] = n_eff_median
        clumped["v_cut"] = v_cut

        raf = clumped["raf"].to_numpy(dtype=float)
        beta_obs = np.abs(clumped["rbeta"].to_numpy(dtype=float))
        beta_wc = np.abs(clumped["rbeta_raw_wc"].to_numpy(dtype=float))

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
        opt_result["trait"] = trait

        clumped.to_csv(output.table, sep="\t", index=False)
        with open(output.pkl, "wb") as f:
            pickle.dump(opt_result, f)


rule bbj_aggregate_wc:
    input:
        pkls=expand(os.path.join("bbj", "high_clumps", "high_{trait}.wc.pkl"), trait=TRAITS),
        tables=expand(os.path.join("bbj", "high_clumps", "high_{trait}.wc.tsv"), trait=TRAITS)
    output:
        pkl=os.path.join("bbj", "opt_fits_high_bbj_wc.pkl"),
        csv=os.path.join("bbj", "opt_results_high_bbj_wc.csv")
    run:
        import pickle
        import numpy as np
        import pandas as pd

        model_mapping = {
            "I1_effects": "dir",
            "I2_effects": "stab",
            "Ip_effects": "plei",
            "full_effects": "full",
        }

        opt_results = {}
        for pkl_file in input.pkls:
            with open(pkl_file, "rb") as f:
                res = pickle.load(f)
            trait = res["trait"]
            opt_results[trait] = res

        with open(output.pkl, "wb") as f:
            pickle.dump(opt_results, f)

        rows = []
        for trait in sorted(opt_results.keys()):
            result = opt_results[trait]
            if "sample" not in result:
                result["sample"] = 0

            row = {"trait": trait, "sample": result["sample"]}
            row["ll_neut"] = result.get("ll_neut", np.nan)

            for model_key, mapped_name in model_mapping.items():
                good_entry = model_key in result and result[model_key] is not None and hasattr(result[model_key], "fun")
                row[f"ll_{mapped_name}"] = -result[model_key].fun if good_entry else np.nan
                if model_key == "I1_effects":
                    row["I1_dir"] = result[model_key].x[0] if good_entry else np.nan
                elif model_key == "I2_effects":
                    row["I2_stab"] = 10**result[model_key].x[0] if good_entry else np.nan
                elif model_key == "Ip_effects":
                    row["Ip_plei"] = 10**result[model_key].x[0] if good_entry else np.nan
                elif model_key == "full_effects":
                    row["I1_full"] = result[model_key].x[0] if good_entry else np.nan
                    row["I2_full"] = 10**result[model_key].x[1] if good_entry else np.nan
            rows.append(row)

        pd.DataFrame(rows).sort_values(by=["trait", "sample"]).reset_index(drop=True).to_csv(output.csv, index=False)
