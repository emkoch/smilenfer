import os

BASE_DIR = "../data"
FINAL_DIR = os.path.join(BASE_DIR, "final", "mvp_finemapping")
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
MIN_X = 0.01
P_THRESH = 4.6e-11
N_SAMPLES = 1
SAMPLES = list(range(N_SAMPLES))

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

rule mvp_fit_one_sample_wc:
    input:
        trait_file=lambda wc: os.path.join(
            FINAL_DIR,
            f"{wc.trait.replace(' ', '_')}_mvp_eur_finemapping.tsv"
        ),
        sfs_pile=SFS_PILE
    output:
        table=os.path.join("mvp", "samples", "{trait}_sample_{sample}.wc.tsv"),
        pkl=os.path.join("mvp", "samples", "{trait}_sample_{sample}.wc.pkl")
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

        def sample_from_CS(dd, trait, category="PheCodes"):
            sub = dd[(dd["Description"] == trait) & (dd["Category"] == category)].copy()
            loci = sub["Locus_CS"].unique()
            if len(loci) == 0:
                return None
            sub["sampled"] = 0
            for loc in loci:
                ss = sub[sub["Locus_CS"] == loc].copy()
                rows = ss.sample(n=1, replace=True, weights=ss["CS-Level Pip"], random_state=sample)
                sub.loc[rows.index, "sampled"] += 1
            return sub[sub["sampled"] > 0].copy()

        def to_risk(eaf, beta):
            return np.where(beta > 0, eaf, 1 - eaf), np.abs(beta)

        all_trait = pd.read_csv(input.trait_file, sep="\t")
        all_trait["var_exp"] = 2 * all_trait["EAF Population"] * (1 - all_trait["EAF Population"]) * all_trait["Beta Population"] ** 2
        all_trait["n_eff"] = 1 / (
            2
            * all_trait["SE Population"] ** 2
            * all_trait["EAF Population"]
            * (1 - all_trait["EAF Population"])
        )

        n_eff_median = float(np.nanmedian(all_trait["n_eff"]))
        v_cut = float(stats.chi2.isf(P_THRESH, df=1) / n_eff_median)

        sampled_df = sample_from_CS(all_trait, trait)
        if sampled_df is None:
            sampled_df = all_trait.iloc[0:0].copy()

        sampled_df = sampled_df[sampled_df["EAF Population"].between(MIN_X, 1 - MIN_X)].copy()

        eaf = sampled_df["EAF Population"].to_numpy(dtype=float)
        beta = sampled_df["Beta Population"].to_numpy(dtype=float)
        raf, rbeta = to_risk(eaf, beta)
        sampled_df["raf"] = raf
        sampled_df["rbeta"] = rbeta
        sampled_df["se"] = sampled_df["SE Population"].to_numpy(dtype=float)
        sampled_df["var_exp_raw"] = 2 * sampled_df["raf"] * (1 - sampled_df["raf"]) * sampled_df["rbeta"] ** 2
        sampled_df = sampled_df[sampled_df["var_exp_raw"] > v_cut].copy()
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


rule mvp_aggregate_wc:
    input:
        pkls=expand(os.path.join("mvp", "samples", "{trait}_sample_{sample}.wc.pkl"), trait=TRAITS, sample=SAMPLES),
        tables=expand(os.path.join("mvp", "samples", "{trait}_sample_{sample}.wc.tsv"), trait=TRAITS, sample=SAMPLES)
    output:
        pkl=os.path.join("mvp", "opt_fits_mvp_finemapping_eur_wc.pkl"),
        csv=os.path.join("mvp", "opt_results_mvp_finemapping_eur_wc.csv")
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
