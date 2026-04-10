# Snakefile

import os

#––– Configuration
BASE_DIR    = "../../../data"
FINAL_DIR   = os.path.join(BASE_DIR, "final", "original_traits_mvp_effects")
SFS_PILE    = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
MIN_X       = 0.01
P_THRESH    = 5e-8

TRAITS = [
    "bc", "bmi", "cad", "dbp", "hdl", "ibd", "ldl", "rbc",
    "sbp", "t2d", "wbc", "arthrosis", "asthma", "diverticulitis",
    "gallstones", "glaucoma", "height", "hypothyroidism",
    "triglycerides", "varicose"
]

POPS = ["eur", "amr", "afr", "meta"]


rule all:
    input:
        # one pickle+CSV per population
        expand("opt_fits_mvp_matching_{population}.pkl", population=POPS),
        expand("opt_results_mvp_matching_{population}.csv", population=POPS)


rule fit_trait_pop:
    """
    For each (trait, population), read the allele‐matched BBJ→MVP file,
    align risk alleles, filter by var_exp & RAF, and run infer_all_standard.
    """
    input:
        trait_file = os.path.join(
            FINAL_DIR,
            "mvp_{trait}_allele_matched_hg19snp_max_r2_filtered.tsv"
        ),
        sfs_pile    = SFS_PILE
    output:
        "mvp_trait_{trait}_pop_{population}.pkl"
    params:
        trait = "{trait}",
        population   = lambda wc: wc.population
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        # local copy of allele‐matching logic
        def process_match_df(df):
            populations = ["EUR","AMR","AFR","META"]
            for pp in populations:
                match_key = "beta_mvp_match_" + pp
                if match_key in df.columns:
                    signed = df[match_key].to_numpy(dtype=float)
                else:
                    ea_col = "ea_mvp_" + pp if "ea_mvp_" + pp in df.columns else "ea_mvp"
                    if ea_col not in df.columns:
                        continue
                    risk_match = df.rallele.astype(str).to_numpy() == df[ea_col].astype(str).to_numpy()
                    beta_key = "beta_mvp_" + pp
                    or_key = "or_mvp_" + pp
                    if beta_key in df.columns:
                        raw = df[beta_key].to_numpy(dtype=float)
                        signed = np.where(risk_match, raw, -raw)
                    elif or_key in df.columns:
                        raw = df[or_key].to_numpy(dtype=float)
                        signed = np.full(len(raw), np.nan)
                        valid = np.isfinite(raw) & (raw > 0)
                        signed[valid] = np.where(
                            risk_match[valid],
                            raw[valid] - 1.0,
                            1.0 / raw[valid] - 1.0,
                        )
                    else:
                        continue
                df["raf_" + pp] = np.where(signed < 0, 1 - df.raf, df.raf)
                df["rbeta_" + pp] = np.abs(signed)

        trait = params.trait
        population   = params.population.upper() # match column names

        df = pd.read_csv(input.trait_file, sep="\t")

        process_match_df(df)

        raf_col   = f"raf_{population}"
        rbeta_col = f"rbeta_{population}"
        if raf_col not in df.columns:
            # nothing to do for this pop

            results = {}
            results["ll_neut"]      = None
            results["I2_effects"]   = None
            results["Ip_effects"]   = None
            results["I1_effects"]   = None
            results["full_effects"] = None
            results["trait"]        = trait
            results["population"]   = params.population

            with open(output[0], "wb") as f:
                pickle.dump(results, f)
            return

        n_eff_med = np.nanmedian(df["median_n_eff"])
        v_cut     = stats.chi2.isf(P_THRESH, df=1) / n_eff_med

        mask = df["var_exp"].to_numpy() > v_cut
        mask &= df[raf_col].between(MIN_X, 1 - MIN_X).to_numpy()

        raf         = df.loc[mask, raf_col].to_numpy()
        rbeta_mvp   = df.loc[mask, rbeta_col].to_numpy()
        rbeta_orig  = df.loc[mask, "rbeta"].to_numpy()

        valid = ~np.isnan(rbeta_mvp)
        raf        = raf[valid]
        rbeta_mvp  = rbeta_mvp[valid]
        rbeta_orig = rbeta_orig[valid]

        sfs = sim.truncate_pile(
            pickle.load(open(input.sfs_pile, "rb")),
            1e-8
        )

        opt = sstats.infer_all_standard(
            sfs, 10_000,
            raf, rbeta_mvp,
            stats.chi2.isf(P_THRESH, df=1) / n_eff_med,
            min_x=MIN_X, n_points=1000, n_x=1000,
            beta_obs=rbeta_orig
        )
        opt = sstats.correct_all_standard_first_mode(
            opt, sfs, 10_000,
            raf, rbeta_mvp,
            stats.chi2.isf(P_THRESH, df=1) / n_eff_med,
            min_x=MIN_X, n_points=1000, n_x=1000,
            beta_obs=rbeta_orig
        )
        opt["trait"]      = trait
        opt["population"] = params.population

        # 8) write pickle
        with open(output[0], "wb") as f:
            pickle.dump(opt, f)


rule aggregate_results:
    """
    For each population, collect all per‐trait pickles into one dict,
    then dump both the raw pickle and a flattened CSV.
    """
    input:
        expand("mvp_trait_{trait}_pop_{population}.pkl", trait=TRAITS, population=POPS)
    output:
        "opt_fits_mvp_matching_{population}.pkl",
        "opt_results_mvp_matching_{population}.csv"
    params:
        population = lambda wc: wc.population
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        population = params.population
        results = {}

        # collect trait‐level dicts
        for trait in TRAITS:
            pkl = f"mvp_trait_{trait}_pop_{population}.pkl"
            with open(pkl, "rb") as f:
                res = pickle.load(f)
            if not isinstance(res, dict):
                raise ValueError(f"Expected dict in {pkl}, got {type(res)}")
            results[trait] = res

        # 1) dump raw dict
        with open(output[0], "wb") as f:
            pickle.dump(results, f)

        # 2) flatten and write CSV
        df = spost.prepare_data_from_opt_results(results)
        df.to_csv(output[1], index=False)
