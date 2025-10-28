import os

BASE_DIR = "../../data"
FINAL_DIR = os.path.join(BASE_DIR, "final", "mvp_finemapping")
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
MIN_X = 0.01
P_THRESH = 4.6e-11

TRAITS = [
    "Type 2 diabetes",
    "Cancer of prostate",
    "Atrial fibrillation",
    "Glaucoma",
    "Hypothyroidism",
    "Coronary atherosclerosis",
    "Hyperlipidemia",
    "Hypertension",
    "Basal cell carcinoma",
    "Gout",
    "Diverticulosis and diverticulitis",
]

N_SAMPLES = 20
SAMPLES = list(range(N_SAMPLES))

rule all:
    input:
        "opt_fits_mvp_finemapping_eur.pkl",
        "opt_results_mvp_finemapping_eur.csv"

rule fit_one_sample:
    input:
        trait_file=lambda wc: os.path.join(
            FINAL_DIR,
            f"{wc.trait.replace(' ', '_')}_mvp_eur_finemapping.tsv"
        ),
        sfs_pile=SFS_PILE
    output:
        "{trait}_sample_{sample}.pkl"
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

        def sample_from_CS(dd, trait, category="PheCodes"):
            sub = dd[(dd["Description"] == trait) & (dd["Category"] == category)].copy()
            loci = sub["Locus_CS"].unique()
            if len(loci) == 0:
                return None
            sub["sampled"] = 0
            for loc in loci:
                ss = sub[sub["Locus_CS"] == loc].copy()
                rows = ss.sample(n=1, replace=True, weights=ss["CS-Level Pip"])
                sub.loc[rows.index, "sampled"] += 1
            return sub[sub["sampled"] > 0].copy()

        def to_risk(eaf, beta):
            return np.where(beta > 0, eaf, 1 - eaf), np.abs(beta)

        trait = params.trait
        sample = params.sample

        all_trait = pd.read_csv(input.trait_file, sep="\t")
        if all_trait.empty:
            with open(output[0], "wb") as f:
                pickle.dump({}, f)
            return

        all_trait["var_exp"] = (
            2
            * all_trait["EAF Population"]
            * (1 - all_trait["EAF Population"])
            * all_trait["Beta Population"] ** 2
        )
        all_trait["n_eff"] = 1 / (
            2
            * all_trait["SE Population"] ** 2
            * all_trait["EAF Population"]
            * (1 - all_trait["EAF Population"])
        )

        n_eff_median = np.nanmedian(all_trait["n_eff"])
        v_cut = stats.chi2.isf(P_THRESH, df=1) / n_eff_median

        sampled_df = sample_from_CS(all_trait, trait)
        if sampled_df is None:
            with open(output[0], "wb") as f:
                pickle.dump({}, f)
            return

        sampled_df = sampled_df[sampled_df["EAF Population"].between(0.01, 0.99)]

        eaf = sampled_df["EAF Population"].to_numpy()
        beta = sampled_df["Beta Population"].to_numpy()
        raf, rbeta = to_risk(eaf, beta)
        v_exp = 2 * raf * (1 - raf) * rbeta ** 2

        keep = v_exp > v_cut
        if not keep.any():
            with open(output[0], "wb") as f:
                pickle.dump({}, f)
            return

        raf_keep = raf[keep]
        rbeta_keep = rbeta[keep]

        sfs_pile_eur = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)

        opt_result = sstats.infer_all_standard(
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
        opt_result["sample"] = sample
        opt_result["trait"] = trait

        with open(output[0], "wb") as f:
            pickle.dump(opt_result, f)

rule aggregate_results:
    input:
        expand("{trait}_sample_{sample}.pkl", trait=TRAITS, sample=SAMPLES)
    output:
        "opt_fits_mvp_finemapping_eur.pkl",
        "opt_results_mvp_finemapping_eur.csv"
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
            t = res["trait"]
            if t not in opt_results:
                opt_results[t] = []
            opt_results[t].append(res)

        with open(output[0], "wb") as f:
            pickle.dump(opt_results, f)

        opt_df = spost.prepare_data_from_opt_results(opt_results)
        opt_df.to_csv(output[1], index=False)
