import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats

import smilenfer.simulation as sim
import smilenfer.statistics as sstats

BASE_DIR = "../data"
FINAL_DIR = os.path.join(BASE_DIR, "final", "original_traits")
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
MIN_X = 0.01
P_THRESH = 5e-8

sfs_pile_eur = sim.truncate_pile(pickle.load(open(SFS_PILE, "rb")), 1e-8)

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

I2_ud = []
I2_std = []
Ip_ud = []
Ip_std = []
ll_I2_ud = []
ll_I2_std = []
ll_Ip_ud = []
ll_Ip_std = []
ll_neut = []

for trait in TRAITS:
    print(f"Processing trait: {trait}")
    ff = os.path.join(FINAL_DIR, f"processed.{trait}.snps_low_r2.tsv")
    trait_df = pd.read_csv(ff, sep="\t")
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

    I2_result_ud = sstats.infer_I2(
        sfs_pile_eur,
        10000,
        raf_keep,
        rbeta_keep,
        v_cut,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
        ud=True
    )
    I2_ud.append(I2_result_ud.x[0])
    ll_I2_ud.append(-I2_result_ud.fun)

    I2_result_std = sstats.infer_I2(
        sfs_pile_eur,
        10000,
        raf_keep,
        rbeta_keep,
        v_cut,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
        ud=False
    )
    I2_std.append(I2_result_std.x[0])
    ll_I2_std.append(-I2_result_std.fun)

    Ip_result_ud = sstats.infer_Ip(
        sfs_pile_eur,
        10000,
        raf_keep,
        rbeta_keep,
        v_cut,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
        ud=True
    )
    Ip_ud.append(Ip_result_ud.x[0])
    ll_Ip_ud.append(-Ip_result_ud.fun)

    Ip_result_std = sstats.infer_Ip(
        sfs_pile_eur,
        10000,
        raf_keep,
        rbeta_keep,
        v_cut,
        min_x=MIN_X,
        n_points=1000,
        n_x=1000,
        beta_obs=None,
        ud=False
    )
    Ip_std.append(Ip_result_std.x[0])
    ll_Ip_std.append(-Ip_result_std.fun)

    neut_ll = sstats.llhood_neut(sfs_pile_eur, rbeta_keep, raf_keep, v_cut, MIN_X, n_x=1000, n_points=1000)
    ll_neut.append(neut_ll)

# make a DataFrame
results_df = pd.DataFrame({
    "Trait": TRAITS,
    "I2_ud": I2_ud,
    "I2_std": I2_std,
    "Ip_ud": Ip_ud,
    "Ip_std": Ip_std,
    "ll_I2_ud": ll_I2_ud,
    "ll_I2_std": ll_I2_std,
    "ll_Ip_ud": ll_Ip_ud,
    "ll_Ip_std": ll_Ip_std,
    "ll_neut": ll_neut
})
results_df.to_csv("stab_ud_std_results.csv", index=False)