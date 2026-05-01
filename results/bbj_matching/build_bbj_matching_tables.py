import os
from pathlib import Path

import numpy as np
import pandas as pd

from bbj_matching_common import (
    DATA_DIR,
    TRAIT_LABELS,
    load_original_fit_set,
    slope_through_origin,
)


BBJ_MATCH_DIR = Path(
    "/home/evan/Drive/Work/Sunyaev/POLYGENIC_SELECTION/PROGRAMS"
    "/smilenfer-dev/results/data/eur_plus_bbj"
)
MAF_BINS = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
MAF_BIN_LABELS = [
    r"$0.01$--$0.02$",
    r"$0.02$--$0.05$",
    r"$0.05$--$0.10$",
    r"$0.10$--$0.20$",
    r"$0.20$--$0.50$",
]


def read_bbj_match_table(path):
    df = pd.read_csv(path, sep="\t")
    needed = {"snp", "raf_bbj", "beta_bbj", "se_bbj", "pval_bbj"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df.loc[
        :,
        ["snp", "raf_bbj", "beta_bbj", "se_bbj", "pval_bbj"],
    ].drop_duplicates("snp")


def fraction_maf_below(values, cutoff):
    if len(values) == 0:
        return np.nan
    return np.mean(values < cutoff)


def summarize_match_status(status):
    rows = []
    for (trait, trait_label), trait_status in status.groupby(["trait", "trait_label"]):
        matched = trait_status.loc[trait_status["matched_bbj"]]
        unmatched = trait_status.loc[~trait_status["matched_bbj"]]
        rows.append(
            {
                "trait": trait,
                "trait_label": trait_label,
                "original_fit_loci": len(trait_status),
                "matched_loci": len(matched),
                "median_maf_matched": np.nanmedian(matched["maf_orig"]),
                "median_maf_unmatched": np.nanmedian(unmatched["maf_orig"]),
                "frac_unmatched": len(unmatched) / len(trait_status),
                "frac_matched_maf_lt_005": fraction_maf_below(matched["maf_orig"], 0.05),
                "frac_unmatched_maf_lt_005": fraction_maf_below(unmatched["maf_orig"], 0.05),
                "unmatched_loci": len(unmatched),
            }
        )
    return pd.DataFrame(rows).sort_values("trait_label")


def summarize_match_status_by_maf(status):
    status = status.copy()
    status["unmatched_bbj"] = ~status["matched_bbj"]
    status["maf_bin"] = pd.cut(
        status["maf_orig"],
        bins=MAF_BINS,
        labels=MAF_BIN_LABELS,
        include_lowest=True,
    )
    return (
        status.groupby("maf_bin", observed=True)
        .agg(
            n_loci=("snp", "size"),
            n_unmatched=("unmatched_bbj", "sum"),
            frac_unmatched=("unmatched_bbj", "mean"),
            median_maf=("maf_orig", "median"),
        )
        .reset_index()
    )


def load_all_matches():
    original_rows = []
    matched_rows = []
    status_rows = []
    summary_rows = []

    for fname in sorted(os.listdir(BBJ_MATCH_DIR)):
        if not (fname.startswith("clumped.") and fname.endswith(".ukbb_bbj.tsv")):
            continue
        trait = fname.replace("clumped.", "").replace(".ukbb_bbj.tsv", "")
        original = load_original_fit_set(trait)
        if original is None:
            continue

        bbj = read_bbj_match_table(BBJ_MATCH_DIR / fname)
        merged = original.merge(bbj, on="snp", how="left", validate="one_to_one")
        merged["matched_bbj"] = merged["raf_bbj"].notna() & merged["beta_bbj"].notna()
        merged["trait_label"] = merged["trait"].map(TRAIT_LABELS).fillna(merged["trait"])

        matched = merged.loc[merged["matched_bbj"]].copy()
        matched["maf_bbj"] = np.minimum(
            matched["raf_bbj"].to_numpy(),
            1 - matched["raf_bbj"].to_numpy(),
        )
        matched["bbj_lower_maf"] = matched["maf_bbj"] < matched["maf_orig"]
        matched["bbj_maf_lt_001"] = matched["maf_bbj"] < 0.01
        matched["maf_ratio_bbj_orig"] = matched["maf_bbj"] / matched["maf_orig"]

        scale_all = slope_through_origin(matched["rbeta"], matched["beta_bbj"])
        matched["beta_bbj_expected"] = matched["rbeta"] * scale_all
        matched["expected_z_bbj"] = matched["beta_bbj_expected"] / matched["se_bbj"]

        scale_subset = matched.loc[matched["expected_z_bbj"] >= 2]
        scale_z2 = slope_through_origin(scale_subset["rbeta"], scale_subset["beta_bbj"])
        if not np.isfinite(scale_z2):
            scale_z2 = scale_all
        matched["bbj_effect_scale_all"] = scale_all
        matched["bbj_effect_scale"] = scale_z2
        matched["beta_bbj_scaled"] = matched["beta_bbj"] / scale_z2
        matched["expected_z_bbj_unscaled"] = matched["rbeta"] / matched["se_bbj"]
        matched["observed_z_bbj"] = matched["beta_bbj"] / matched["se_bbj"]

        original_rows.append(original)
        matched_rows.append(matched)
        status_rows.append(
            merged.loc[
                :,
                [
                    "trait",
                    "trait_label",
                    "snp",
                    "raf",
                    "maf_orig",
                    "rbeta",
                    "PosteriorMean",
                    "matched_bbj",
                ],
            ]
        )
        summary_rows.append(
            {
                "trait": trait,
                "trait_label": TRAIT_LABELS.get(trait, trait),
                "original_fit_loci": len(merged),
                "matched_loci": int(merged["matched_bbj"].sum()),
                "unmatched_loci": int((~merged["matched_bbj"]).sum()),
            }
        )

    if not matched_rows:
        raise RuntimeError(f"No BBJ exact-match tables found in {BBJ_MATCH_DIR}")

    return (
        pd.concat(original_rows, ignore_index=True),
        pd.concat(matched_rows, ignore_index=True),
        pd.concat(status_rows, ignore_index=True),
        pd.DataFrame(summary_rows).sort_values("trait_label"),
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    original, matched, status, summary = load_all_matches()
    maf_summary = summarize_match_status(status)
    maf_bins = summarize_match_status_by_maf(status)

    original.to_csv(DATA_DIR / "bbj_original_fit_loci.tsv", sep="\t", index=False)
    matched.to_csv(DATA_DIR / "bbj_matched_loci.tsv", sep="\t", index=False)
    status.to_csv(DATA_DIR / "bbj_match_status.tsv", sep="\t", index=False)
    summary.to_csv(DATA_DIR / "bbj_match_counts.tsv", sep="\t", index=False)
    maf_summary.to_csv(
        DATA_DIR / "bbj_unmatched_original_maf_summary.tsv",
        sep="\t",
        index=False,
    )
    maf_bins.to_csv(DATA_DIR / "bbj_unmatched_original_maf_bins.tsv", sep="\t", index=False)

    print(f"Wrote {DATA_DIR / 'bbj_original_fit_loci.tsv'}")
    print(f"Wrote {DATA_DIR / 'bbj_matched_loci.tsv'}")
    print(f"Wrote {DATA_DIR / 'bbj_match_status.tsv'}")
    print(f"Wrote {DATA_DIR / 'bbj_match_counts.tsv'}")
    print(f"Wrote {DATA_DIR / 'bbj_unmatched_original_maf_summary.tsv'}")
    print(f"Wrote {DATA_DIR / 'bbj_unmatched_original_maf_bins.tsv'}")


if __name__ == "__main__":
    main()
