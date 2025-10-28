# Snakefile

import os

#––– Configuration
BASE_DIR    = "../../data/"
BBJ_DIR     = os.path.join(BASE_DIR, "final", "bbj_traits")
SFS_PILE    = os.path.join(BASE_DIR, "SFS_pile", "joug_jpt_pile.pkl")
MIN_X       = 0.01
P_THRESH    = 5e-8

TRAITS      = [
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
N_SAMPLES   = 20
SAMPLES     = list(range(N_SAMPLES))

#––– The “all” target: make sure all per‐trait fits and the aggregate are built
rule all:
    input:
        # p‑value‐based clumps
        expand("clumps/pval_{trait}.pkl", trait=TRAITS),
        # high‑|β| clumps
        expand("clumps/high_{trait}.pkl", trait=TRAITS),
        # random clumps (20 replicates)
        expand("clumps/{trait}_random_{sample}.pkl", trait=TRAITS, sample=SAMPLES),
        # aggregate outputs
        "opt_fits_pval_bbj.pkl",
        "opt_results_pval_bbj.csv",
        "opt_fits_high_bbj.pkl",
        "opt_results_high_bbj.csv",
        "opt_fits_random_bbj.pkl",
        "opt_results_random_bbj.csv"

#––– Inference on p‑value‐clumped index SNPs
rule fit_pval:
    input:
        trait_file = os.path.join(BBJ_DIR, "processed.{trait}.max_r2.bbj.tsv"),
        sfs_pile    = SFS_PILE
    output:
        "clumps/pval_{trait}.pkl"
    params:
        trait = "{trait}"
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        trait = params.trait
        dd = pd.read_csv(input.trait_file, sep="\t")

        n_eff_med = np.nanmedian(dd["n_eff"])
        v_cut     = stats.chi2.isf(P_THRESH, df=1) / n_eff_med

        clumped = sstats.pval_clump_trait_data(dd, dist=500_000)
        raf     = clumped["raf"].to_numpy()
        rbeta   = clumped["rbeta"].to_numpy()

        # 4) load & truncate the SFS pile
        sfs = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)

        # 5) run the standard inference
        opt = sstats.infer_all_standard(
            sfs, 10_000, raf, rbeta,
            v_cut,
            min_x=MIN_X, n_points=1000, n_x=1000,
            beta_obs=None
        )
        opt["trait"]  = trait
        opt["method"] = "pval"

        # 6) write out a pickle
        with open(output[0], "wb") as f:
            pickle.dump(opt, f)


#––– Inference on high‐|β| clumps
rule fit_high:
    input:
        trait_file = os.path.join(BBJ_DIR, "processed.{trait}.max_r2.bbj.tsv"),
        sfs_pile    = SFS_PILE
    output:
        "clumps/high_{trait}.pkl"
    params:
        trait = "{trait}"
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        trait = params.trait
        dd    = pd.read_csv(input.trait_file, sep="\t")
        n_eff_med = np.nanmedian(dd["n_eff"])
        v_cut     = stats.chi2.isf(P_THRESH, df=1) / n_eff_med

        clumped = sstats.high_clump_trait_data(dd, dist=500000)
        raf     = clumped["raf"].to_numpy()
        rbeta   = clumped["rbeta"].to_numpy()

        sfs = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)
        opt = sstats.infer_all_standard(
            sfs, 10000, raf, rbeta,
            v_cut,
            min_x=MIN_X, n_points=1000, n_x=1000,
            beta_obs=None
        )
        opt["trait"]  = trait
        opt["method"] = "high"

        with open(output[0], "wb") as f:
            pickle.dump(opt, f)


#––– Inference on random clumps (20 replicates per trait)
rule fit_random:
    input:
        trait_file = os.path.join(BBJ_DIR, "processed.{trait}.max_r2.bbj.tsv"),
        sfs_pile    = SFS_PILE
    output:
        "clumps/{trait}_random_{sample}.pkl"
    params:
        trait  = "{trait}",
        sample = lambda wc: int(wc.sample)
    threads: 1
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from scipy import stats
        import smilenfer.statistics as sstats
        import smilenfer.simulation as sim

        trait  = params.trait
        sample = params.sample

        dd    = pd.read_csv(input.trait_file, sep="\t")
        n_eff_med = np.nanmedian(dd["n_eff"])
        v_cut     = stats.chi2.isf(P_THRESH, df=1) / n_eff_med

        clumped = sstats.random_clump_trait_data(dd, dist=500_000)
        raf     = clumped["raf"].to_numpy()
        rbeta   = clumped["rbeta"].to_numpy()

        sfs = sim.truncate_pile(pickle.load(open(input.sfs_pile, "rb")), 1e-8)
        opt = sstats.infer_all_standard(
            sfs, 10_000, raf, rbeta,
            v_cut,
            min_x=MIN_X, n_points=1000, n_x=1000,
            beta_obs=None
        )
        opt["trait"]  = trait
        opt["sample"] = sample
        opt["method"] = "random"

        with open(output[0], "wb") as f:
            pickle.dump(opt, f)

# Aggregate p-value‐based fits
rule aggregate_pval:
    input:
        expand("clumps/pval_{trait}.pkl", trait=TRAITS)
    output:
        "opt_fits_pval_bbj.pkl",
        "opt_results_pval_bbj.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl in input:
            with open(pkl, "rb") as f:
                res = pickle.load(f)
            opt_results[res["trait"]] = res

        # dump raw dict‐of‐lists
        with open(output[0], "wb") as f:
            pickle.dump(opt_results, f)

        # flatten and write CSV
        df = spost.prepare_data_from_opt_results(opt_results)
        df.to_csv(output[1], index=False)


# Aggregate high‐|β| fits
rule aggregate_high:
    input:
        expand("clumps/high_{trait}.pkl", trait=TRAITS)
    output:
        "opt_fits_high_bbj.pkl",
        "opt_results_high_bbj.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl in input:
            with open(pkl, "rb") as f:
                res = pickle.load(f)
            opt_results[res["trait"]] = res

        with open(output[0], "wb") as f:
            pickle.dump(opt_results, f)

        df = spost.prepare_data_from_opt_results(opt_results)
        df.to_csv(output[1], index=False)


# Aggregate random replicates
rule aggregate_random:
    input:
        expand("clumps/{trait}_random_{sample}.pkl", trait=TRAITS, sample=SAMPLES)
    output:
        "opt_fits_random_bbj.pkl",
        "opt_results_random_bbj.csv"
    run:
        import pickle
        import pandas as pd
        import smilenfer.posterior as spost

        opt_results = {}
        for pkl in input:
            with open(pkl, "rb") as f:
                res = pickle.load(f)
            t = res["trait"]
            opt_results.setdefault(t, []).append(res)

        with open(output[0], "wb") as f:
            pickle.dump(opt_results, f)

        df = spost.prepare_data_from_opt_results(opt_results)
        df.to_csv(output[1], index=False)

