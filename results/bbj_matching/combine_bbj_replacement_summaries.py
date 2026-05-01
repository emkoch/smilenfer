import pandas as pd

from bbj_matching_common import DATA_DIR, ORIGINAL_FITS, TRAIT_LABELS, trait_order


RAW_SUMMARY = DATA_DIR / "bbj_beta_replacement_raw_production_summary.tsv"
RAW_ALL_SUMMARY = DATA_DIR / "bbj_beta_replacement_raw_all_production_summary.tsv"
OUT_SUMMARY = DATA_DIR / "bbj_beta_replacement_raw_combined_summary.tsv"
OUT_AUDIT = DATA_DIR / "bbj_beta_replacement_raw_combined_audit.tsv"


def read_replacement_summaries():
    raw = pd.read_csv(RAW_SUMMARY, sep="\t")
    raw_all = pd.read_csv(RAW_ALL_SUMMARY, sep="\t")
    summary = pd.concat([raw, raw_all], ignore_index=True)
    summary = summary.sort_values(["trait", "fit_name"]).reset_index(drop=True)
    summary.to_csv(OUT_SUMMARY, sep="\t", index=False)
    return summary


def add_full_original(summary):
    traits = trait_order(summary["trait"].unique())
    original = pd.read_csv(ORIGINAL_FITS)
    original = original.loc[original["trait"].isin(traits)].copy()
    original["fit_name"] = "full_original"
    original["n_loci"] = pd.NA
    original["n_replaced"] = 0
    original["plei_ll_gain"] = original["ll_plei"] - original["ll_neut"]
    original["stab_ll_gain"] = original["ll_stab"] - original["ll_neut"]
    original["dir_ll_gain"] = original["ll_dir"] - original["ll_neut"]
    original["full_ll_gain"] = original["ll_full"] - original["ll_neut"]
    original["plei_aic_gain"] = 2 * original["plei_ll_gain"] - 2
    original["stab_aic_gain"] = 2 * original["stab_ll_gain"] - 2
    original["dir_aic_gain"] = 2 * original["dir_ll_gain"] - 2
    original["full_aic_gain"] = 2 * original["full_ll_gain"] - 4

    cols = [
        "trait",
        "fit_name",
        "n_loci",
        "n_replaced",
        "plei_ll_gain",
        "stab_ll_gain",
        "dir_ll_gain",
        "full_ll_gain",
        "plei_aic_gain",
        "stab_aic_gain",
        "dir_aic_gain",
        "full_aic_gain",
    ]
    audit = pd.concat([original[cols], summary[cols]], ignore_index=True)
    audit["trait_label"] = audit["trait"].map(TRAIT_LABELS).fillna(audit["trait"])
    audit.to_csv(OUT_AUDIT, sep="\t", index=False)
    return audit


def main():
    summary = read_replacement_summaries()
    add_full_original(summary)
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_AUDIT}")


if __name__ == "__main__":
    main()
