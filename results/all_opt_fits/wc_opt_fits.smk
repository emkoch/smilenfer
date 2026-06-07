include: "original_traits/orignal_opt_fits_wc.smk"
include: "bbj/bbj_opt_fits_wc.smk"
include: "ukbb_finemapping/ukbb_fm_opt_fits_wc.smk"
include: "mvp/mvp_opt_fits_wc.smk"


rule all:
    input:
        "original_traits/opt_fits_original_traits_eur_wc.pkl",
        "original_traits/opt_results_original_traits_eur_wc.csv",
        "bbj/opt_fits_high_bbj_wc.pkl",
        "bbj/opt_results_high_bbj_wc.csv",
        "ukbb_finemapping/opt_fits_ukbb_susiex_wc.pkl",
        "ukbb_finemapping/opt_results_ukbb_susiex_wc.csv",
        "mvp/opt_fits_mvp_finemapping_eur_wc.pkl",
        "mvp/opt_results_mvp_finemapping_eur_wc.csv"
