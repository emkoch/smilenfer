import os

BASE_DIR = "../../data"
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

rule all:
    input:
        "opt_fits_original_traits_eur_raw.pkl",
        "opt_results_original_traits_eur_raw.csv",
        "opt_fits_original_traits_eur_post.pkl",
        "opt_results_original_traits_eur_post.csv",
        "opt_fits_original_traits_eur_raw_ci.pkl",
        "opt_fits_original_traits_eur_raw_ci.csv",
        "opt_fits_original_traits_eur_post_ci.pkl",
        "opt_fits_original_traits_eur_post_ci.csv",
        expand("{trait}_standard_fits_raw_bootstrap.pkl", trait=TRAITS)

rule fit_one_sample:
    input:
        trait_file=lambda wc: os.path.join(
            FINAL_DIR,
            f"processed.{wc.trait}.snps_low_r2.tsv"
        ),
        sfs_pile=SFS_PILE
    output:
        "{trait}_standard_fits_raw.pkl",
        "{trait}_standard_fits_post.pkl"
    params:
        trait="{trait}"
    run:
        import pickle   
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        trait = params.trait

        trait_df = pd.read_csv(input.trait_file, sep="\t")

        n_eff_median = np.nanmedian(trait_df["n_eff"])
        
        v_cut = stats.chi2.isf(P_THRESH, df=1) / n_eff_median

        raf = trait_df.raf.to_numpy()
        rbeta = trait_df.rbeta.to_numpy()
        rbeta_post = trait_df.PosteriorMean.to_numpy()

        v_exp = 2 * raf * (1 - raf) * rbeta ** 2

        keep = v_exp > v_cut
        raf_keep = raf[keep]
        rbeta_keep = rbeta[keep]
        rbeta_post_keep = rbeta_post[keep]

        sfs_pile_eur = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)

        opt_result_raw = sstats.infer_all_standard(
            sfs_pile_eur,
            10000,
            raf_keep,
            rbeta_keep,
            v_cut,
            min_x=MIN_X,
            n_points=1000,
            n_x=1000,
            beta_obs=None,
        )
        opt_result_raw["trait"] = trait

        with open(output[0], "wb") as f:
            pickle.dump(opt_result_raw, f)

        opt_result_post = sstats.infer_all_standard(
            sfs_pile_eur,
            10000,
            raf_keep,
            rbeta_post_keep,
            v_cut,
            min_x=MIN_X,
            n_points=1000,
            n_x=1000,
            beta_obs=rbeta_keep,
        )

        opt_result_post["trait"] = trait

        with open(output[1], "wb") as f:
            pickle.dump(opt_result_post, f)

rule ci_one_sample:
    input:
        trait_file=lambda wc: os.path.join(
            FINAL_DIR,
            f"processed.{wc.trait}.snps_low_r2.tsv"
        ),
        sfs_pile=SFS_PILE,
        fits_raw="{trait}_standard_fits_raw.pkl",
        fits_post="{trait}_standard_fits_post.pkl"
    output:
        raw_ci="{trait}_standard_fits_raw_ci.pkl",
        post_ci="{trait}_standard_fits_post_ci.pkl"
    params:
        trait="{trait}"
    run:
        import os
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        trait = params.trait

        # --- Rebuild the same mask/inputs as in fit_one_sample ---
        trait_df = pd.read_csv(input.trait_file, sep="\t")
        n_eff_median = np.nanmedian(trait_df["n_eff"])
        v_cut = stats.chi2.isf(P_THRESH, df=1) / n_eff_median

        raf = trait_df.raf.to_numpy()
        rbeta = trait_df.rbeta.to_numpy()
        rbeta_post = trait_df.PosteriorMean.to_numpy()

        v_exp = 2 * raf * (1 - raf) * rbeta ** 2
        keep = v_exp > v_cut
        raf_keep = raf[keep]
        rbeta_keep = rbeta[keep]
        rbeta_post_keep = rbeta_post[keep]

        sfs_pile_eur = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)

        # --- Load optimize results ---
        opt_raw = pickle.load(open(input.fits_raw, "rb"))
        opt_post = pickle.load(open(input.fits_post, "rb"))

        def _get(res_dict, key):
            return res_dict[key] if key in res_dict and res_dict[key] is not None else None

        def _extract_mles(res_dict):
            out = {}
            rI1 = _get(res_dict, "I1_effects")
            rIp = _get(res_dict, "Ip_effects")
            rI2 = _get(res_dict, "I2_effects")
            if rI1 is not None: out["I1_mle"] = float(np.ravel(rI1.x)[0])
            if rIp is not None: out["Ip_mle"] = 10.0 ** float(np.ravel(rIp.x)[0])
            if rI2 is not None: out["I2_mle"] = 10.0 ** float(np.ravel(rI2.x)[0])
            return out

        def _compute_cis(mles, beta_vec, beta_obs):
            cis = {"trait": trait}
            Ne = 10000
            if "Ip_mle" in mles:
                cis["Ip_ci"] = sstats.wald_ci_Ip(
                    sfs_pile_eur, Ne, raf_keep, beta_vec, v_cut, mles["Ip_mle"],
                    min_x=MIN_X, n_points=1000, n_x=1000, beta_obs=beta_obs, spline=True, name=trait
                )
            if "I2_mle" in mles:
                hh = 1.0
                while True:
                    try:
                        cis["I2_ci"] = sstats.wald_ci_I2(
                            sfs_pile_eur, Ne, raf_keep, beta_vec, v_cut, mles["I2_mle"],
                            min_x=MIN_X, n_points=2000, n_x=2000, beta_obs=beta_obs,
                            spline=True, name=f"{trait}_i2_hh{hh:.3g}", hh=hh
                        )
                        break
                    except (ValueError, TypeError) as e:
                        hh *= 0.75
                        print(f"Retrying I2 CI with hh={hh:.3g} due to error: {e}")
            if "I1_mle" in mles:
                hh_I1 = 0.01
                cis["I1_ci"] = sstats.wald_ci_I1(
                    sfs_pile_eur, Ne, raf_keep, beta_vec, v_cut, mles["I1_mle"],
                    min_x=MIN_X, n_points=1000, n_x=1000, beta_obs=beta_obs,
                    hh=hh_I1, spline=True, name=trait
                )
            return cis

        mles_raw = _extract_mles(opt_raw)
        mles_post = _extract_mles(opt_post)

        cis_raw = _compute_cis(mles_raw, rbeta_keep, beta_obs=None)
        cis_post = _compute_cis(mles_post, rbeta_post_keep, beta_obs=rbeta_keep)

        with open(output.raw_ci, "wb") as f:
            pickle.dump(cis_raw, f)
        with open(output.post_ci, "wb") as f:
            pickle.dump(cis_post, f)

rule bootstrap_one_sample:
    input:
        trait_file=lambda wc: os.path.join(
                FINAL_DIR,
                f"processed.{wc.trait}.snps_low_r2.tsv"
            ),
            sfs_pile=SFS_PILE
    output:
        "{trait}_standard_fits_raw_bootstrap.pkl",
        "{trait}_standard_fits_post_bootstrap.pkl"
    params:
        n_bootstraps=1000,
        trait="{trait}"
    run:
        import pickle   
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        trait = params.trait

        trait_df = pd.read_csv(input.trait_file, sep="\t")

        n_eff_median = np.nanmedian(trait_df["n_eff"])
        
        v_cut = stats.chi2.isf(P_THRESH, df=1) / n_eff_median

        raf = trait_df.raf.to_numpy()
        rbeta = trait_df.rbeta.to_numpy()
        rbeta_post = trait_df.PosteriorMean.to_numpy()

        v_exp = 2 * raf * (1 - raf) * rbeta ** 2

        keep = v_exp > v_cut
        raf_keep = raf[keep]
        rbeta_keep = rbeta[keep]
        rbeta_post_keep = rbeta_post[keep]

        sfs_pile_eur = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)

        I2_bs_raw, I2_bs_ests_raw, _ = sstats.bootstrap_I2(sfs_pile_eur, 10000, raf_keep, rbeta_keep, v_cut, 
                                           n_boot=params.n_bootstraps)
        I2_bs_post, I2_bs_ests_post, _ = sstats.bootstrap_I2(sfs_pile_eur, 10000, raf_keep, rbeta_post_keep, v_cut, 
                                           n_boot=params.n_bootstraps, beta_obs=rbeta_keep)
        Ip_bs_raw, Ip_bs_ests_raw, _ = sstats.bootstrap_Ip(sfs_pile_eur, 10000, raf_keep, rbeta_keep, v_cut, 
                                           n_boot=params.n_bootstraps)
        Ip_bs_post, Ip_bs_ests_post, _ = sstats.bootstrap_Ip(sfs_pile_eur, 10000, raf_keep, rbeta_post_keep, v_cut, 
                                           n_boot=params.n_bootstraps, beta_obs=rbeta_keep)
        I1_bs_raw, I1_bs_ests_raw, _ = sstats.bootstrap_I1(sfs_pile_eur, 10000, raf_keep, rbeta_keep, v_cut, 
                                           n_boot=params.n_bootstraps)
        I1_bs_post, I1_bs_ests_post, _ = sstats.bootstrap_I1(sfs_pile_eur, 10000, raf_keep, rbeta_post_keep, v_cut, 
                                           n_boot=params.n_bootstraps, beta_obs=rbeta_keep)

        # Put the results in a dictionary
        bootstrap_results_raw = {
            "I2_raw": I2_bs_raw,
            "I2_ests_raw": I2_bs_ests_raw,
            "Ip_raw": Ip_bs_raw,
            "Ip_ests_raw": Ip_bs_ests_raw,
            "I1_raw": I1_bs_raw,
            "I1_ests_raw": I1_bs_ests_raw
        }
        bootstrap_results_post = {
            "I2_post": I2_bs_post,
            "I2_ests_post": I2_bs_ests_post,
            "Ip_post": Ip_bs_post,
            "Ip_ests_post": Ip_bs_ests_post,
            "I1_post": I1_bs_post,
            "I1_ests_post": I1_bs_ests_post
        }

        with open(output[0], "wb") as f:
            pickle.dump(bootstrap_results_raw, f)
        with open(output[1], "wb") as f:
            pickle.dump(bootstrap_results_post, f)

rule aggregate_results_ci_raw:
    input:
        expand("{trait}_standard_fits_raw_ci.pkl", trait=TRAITS)
    output:
        "opt_fits_original_traits_eur_raw_ci.pkl",
        "opt_fits_original_traits_eur_raw_ci.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        ci_results = {}

        for pkl_file in input:
            with open(pkl_file, "rb") as f:
                cis = pickle.load(f)
            if not isinstance(cis, dict):
                raise TypeError(f"Did not find a valid dictionary, got {type(cis)} from {pkl_file}")

            # Aggregate the CI results
            ci_results[cis["trait"]] = cis

        with open(output[0], "wb") as f:
            pickle.dump(ci_results, f)

        ci_df = spost.prepare_data_from_ci_results(ci_results)
        ci_df.to_csv(output[1], index=False)

rule aggregate_results_ci_post:
    input:
        expand("{trait}_standard_fits_post_ci.pkl", trait=TRAITS)
    output:
        "opt_fits_original_traits_eur_post_ci.pkl",
        "opt_fits_original_traits_eur_post_ci.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        ci_results = {}

        for pkl_file in input:
            with open(pkl_file, "rb") as f:
                cis = pickle.load(f)
            if not isinstance(cis, dict):
                raise TypeError(f"Did not find a valid dictionary, got {type(cis)} from {pkl_file}")

            # Aggregate the CI results
            ci_results[cis["trait"]] = cis

        with open(output[0], "wb") as f:
            pickle.dump(ci_results, f)

        ci_df = spost.prepare_data_from_ci_results(ci_results)
        ci_df.to_csv(output[1], index=False)

rule aggregate_results_raw:
    input:
        expand("{trait}_standard_fits_raw.pkl", trait=TRAITS)
    output:
        "opt_fits_original_traits_eur_raw.pkl",
        "opt_results_original_traits_eur_raw.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl_file in input:
            with open(pkl_file, "rb") as f:
                res = pickle.load(f)
            if not isinstance(res, dict):
                raise TypeError(f"Did not find a valid dictionary, got {type(res)} from {pkl_file}")
            opt_results[res["trait"]] = res

        with open(output[0], "wb") as f:
            pickle.dump(opt_results, f)

        opt_df = spost.prepare_data_from_opt_results(opt_results)
        opt_df.to_csv(output[1], index=False)

rule aggregate_results_post:
    input:
        expand("{trait}_standard_fits_post.pkl", trait=TRAITS)
    output:
        "opt_fits_original_traits_eur_post.pkl",
        "opt_results_original_traits_eur_post.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl_file in input:
            with open(pkl_file, "rb") as f:
                res = pickle.load(f)
            if not isinstance(res, dict):
                raise TypeError(f"Did not find a valid dictionary, got {type(res)} from {pkl_file}")
            opt_results[res["trait"]] = res

        with open(output[0], "wb") as f:
            pickle.dump(opt_results, f)

        opt_df = spost.prepare_data_from_opt_results(opt_results)
        opt_df.to_csv(output[1], index=False)
