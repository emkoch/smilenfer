import os

BASE_DIR = "../../data"
FINAL_DIR = os.path.join(BASE_DIR, "final", "UKBB_susiex")
ORIGINAL_DIR = os.path.join(BASE_DIR, "final", "original_traits")
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
MIN_X = 0.01
P_THRESH = 5e-8

TRAITS = [
    "bmi",
    "dbp",
    "hdl",
    "height",
    "ldl",
    "sbp",
    "triglycerides",
    "wbc"
]

N_SAMPLES = 20
SAMPLES = list(range(N_SAMPLES))

rule all:
    input:
        "opt_fits_ukbb_susiex.pkl",
        "opt_results_ukbb_susiex.csv"

rule fit_one_sample:
    input:
        trait_file=os.path.join(
            FINAL_DIR,
            "susiex_cs_table_{trait}.csv"
        ),
        original_trait_file=os.path.join(
            ORIGINAL_DIR,
            "processed.{trait}.snps_low_r2.tsv"
        ),
        sfs_pile=SFS_PILE
    output:
        "samples/{trait}_sample_{sample}.pkl"
    params:
        trait="{trait}",
        sample=lambda wc: int(wc.sample)
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        from sample_finemapped import sample_finemap

        # Load the trait data
        trait_file = input.trait_file
        trait_data = pd.read_csv(trait_file)

        trait = params.trait
        sample = params.sample

        # Load the original trait data
        original_trait_file = input.original_trait_file
        original_trait_data = pd.read_csv(original_trait_file, sep="\t")
        print(original_trait_data.head())

        n_eff_median = np.nanmedian(original_trait_data["n_eff"])
        v_cut = stats.chi2.isf(P_THRESH, df=1) / n_eff_median

        sampled_df = sample_finemap(trait_data)

        sampled_df["var_exp"] = (
            2
            * sampled_df["raf"]
            * (1 - sampled_df["raf"])
            * sampled_df["rbeta"] ** 2
        )

        sampled_df = sampled_df[sampled_df["var_exp"] > v_cut].copy()
        sampled_df = sampled_df[sampled_df["raf"].between(MIN_X, 1 - MIN_X)].copy()

        raf = sampled_df["raf"].to_numpy()
        rbeta = sampled_df["rbeta"].to_numpy()

        sfs_pile_eur = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)

        opt_result = sstats.infer_all_standard(
            sfs_pile_eur,
            10000,
            raf,
            rbeta,
            v_cut,
            min_x=MIN_X,
            n_points=1000,
            n_x=1000,
            beta_obs=None,
        )
        opt_result["sample"] = sample
        opt_result["trait"] = trait

        with open(output[0], "wb") as f:
            pickle.dump(opt_result, f)

rule aggregate_results:
    input:
        expand("samples/{trait}_sample_{sample}.pkl", trait=TRAITS, sample=SAMPLES)
    output:
        "opt_fits_ukbb_susiex.pkl",
        "opt_results_ukbb_susiex.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl_file in input:
            with open(pkl_file, "rb") as f:
                result = pickle.load(f)
            if not isinstance(result, dict):
                raise ValueError(f"Expected a dictionary in {pkl_file}, got {type(result)}")
            t = result["trait"]
            if t not in opt_results:
                opt_results[t] = []
            opt_results[t].append(result)

        with open(output[0], "wb") as f:
            pickle.dump(opt_results, f)

        opt_df = spost.prepare_data_from_opt_results(opt_results)
        opt_df.to_csv(output[1], index=False)
