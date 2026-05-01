import argparse
import pickle

import numpy as np
import pandas as pd

import smilenfer.simulation as sim
import smilenfer.statistics as sstats

from bbj_matching_common import DATA_DIR, MIN_X, NE, SCRIPT_DIR, SFS_PILE


FIT_SPECS = [
    ("matched_original_post", None, "post"),
    ("raw_all", 0.0, "raw"),
    ("raw_z2", 2.0, "raw"),
    ("raw_z5", 5.0, "raw"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traits", nargs="*", default=None)
    parser.add_argument("--fits", nargs="*", default=None)
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--n-x", type=int, default=1000)
    parser.add_argument("--output-prefix", default="bbj_beta_replacement_raw")
    return parser.parse_args()


def make_beta_vector(df, z_cutoff, replacement):
    beta = df["PosteriorMean"].to_numpy().copy()
    use_bbj = np.zeros(len(df), dtype=bool)
    if z_cutoff is None:
        return np.abs(beta), use_bbj

    use_bbj = df["expected_z_bbj"].to_numpy() >= z_cutoff
    if replacement != "raw":
        raise ValueError("Committed BBJ replacement fits use raw BBJ effects only")

    beta[use_bbj] = df.loc[use_bbj, "beta_bbj"].to_numpy()
    return np.abs(beta), use_bbj


def fit_one_trait(sfs_pile, trait_df, fit_name, z_cutoff, replacement, n_points, n_x):
    raf = trait_df["raf"].to_numpy()
    beta_obs = trait_df["rbeta"].to_numpy()
    beta, use_bbj = make_beta_vector(trait_df, z_cutoff, replacement)
    v_cut = trait_df["v_cut"].iloc[0]

    keep = 2 * raf * (1 - raf) * beta_obs**2 > v_cut
    raf = raf[keep]
    beta = beta[keep]
    beta_obs = beta_obs[keep]
    use_bbj = use_bbj[keep]

    result = sstats.infer_all_standard(
        sfs_pile,
        NE,
        raf,
        beta,
        v_cut,
        min_x=MIN_X,
        n_points=n_points,
        n_x=n_x,
        beta_obs=beta_obs,
    )
    result = sstats.correct_all_standard_first_mode(
        result,
        sfs_pile,
        NE,
        raf,
        beta,
        v_cut,
        min_x=MIN_X,
        n_points=n_points,
        n_x=n_x,
        beta_obs=beta_obs,
    )
    result["fit_name"] = fit_name
    result["sample"] = 0
    result["n_loci"] = int(len(raf))
    result["n_replaced"] = int(np.sum(use_bbj))
    result["z_cutoff"] = np.nan if z_cutoff is None else z_cutoff
    result["replacement"] = replacement
    return result


def flatten_results(results):
    rows = []
    model_names = {
        "I1_effects": "dir",
        "I2_effects": "stab",
        "Ip_effects": "plei",
        "full_effects": "full",
    }
    for fit_name, fit_results in results.items():
        for trait, result in fit_results.items():
            row = {
                "trait": trait,
                "sample": result.get("sample", 0),
                "fit_name": fit_name,
                "ll_neut": result.get("ll_neut", np.nan),
                "n_loci": result.get("n_loci", np.nan),
                "n_replaced": result.get("n_replaced", np.nan),
                "z_cutoff": result.get("z_cutoff", np.nan),
                "replacement": result.get("replacement", ""),
            }
            for key, name in model_names.items():
                entry = result.get(key)
                ok = entry is not None and hasattr(entry, "fun")
                row[f"ll_{name}"] = -entry.fun if ok else np.nan
                if key == "I1_effects":
                    row["I1_dir"] = entry.x[0] if ok else np.nan
                elif key == "I2_effects":
                    row["I2_stab"] = 10 ** entry.x[0] if ok else np.nan
                elif key == "Ip_effects":
                    row["Ip_plei"] = 10 ** entry.x[0] if ok else np.nan
                elif key == "full_effects":
                    row["I1_full"] = entry.x[0] if ok else np.nan
                    row["I2_full"] = 10 ** entry.x[1] if ok else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["trait", "fit_name"]).reset_index(drop=True)


def summarize_fits(fits):
    rows = []
    for row in fits.itertuples(index=False):
        rows.append(
            {
                "trait": row.trait,
                "fit_name": row.fit_name,
                "n_loci": int(row.n_loci),
                "n_replaced": int(row.n_replaced),
                "stab_ll_gain": row.ll_stab - row.ll_neut,
                "plei_ll_gain": row.ll_plei - row.ll_neut,
                "dir_ll_gain": row.ll_dir - row.ll_neut,
                "full_ll_gain": row.ll_full - row.ll_neut,
                "stab_aic_gain": 2 * (row.ll_stab - row.ll_neut) - 2,
                "plei_aic_gain": 2 * (row.ll_plei - row.ll_neut) - 2,
                "dir_aic_gain": 2 * (row.ll_dir - row.ll_neut) - 2,
                "full_aic_gain": 2 * (row.ll_full - row.ll_neut) - 4,
                "plei_vs_stab_ll_gain": row.ll_plei - row.ll_stab,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    matched = pd.read_csv(DATA_DIR / "bbj_matched_loci.tsv", sep="\t")
    if args.traits is not None:
        matched = matched.loc[matched["trait"].isin(set(args.traits))].copy()

    fit_specs = [spec for spec in FIT_SPECS if args.fits is None or spec[0] in set(args.fits)]
    if matched.empty or not fit_specs:
        raise ValueError("No matched loci or fits selected")

    with open(SFS_PILE, "rb") as handle:
        sfs_pile = sim.truncate_pile(pickle.load(handle), 1e-8)

    results = {}
    for fit_name, z_cutoff, replacement in fit_specs:
        results[fit_name] = {}
        for trait in sorted(matched["trait"].unique()):
            trait_df = matched.loc[matched["trait"] == trait].copy()
            results[fit_name][trait] = fit_one_trait(
                sfs_pile, trait_df, fit_name, z_cutoff, replacement, args.n_points, args.n_x
            )

    out_pkl = SCRIPT_DIR / f"{args.output_prefix}_fits.pkl"
    out_csv = SCRIPT_DIR / f"{args.output_prefix}_fits.csv"
    out_summary = DATA_DIR / f"{args.output_prefix}_summary.tsv"
    with open(out_pkl, "wb") as handle:
        pickle.dump(results, handle)
    fits = flatten_results(results)
    fits.to_csv(out_csv, index=False)
    summarize_fits(fits).to_csv(out_summary, sep="\t", index=False)
    print(f"Wrote {out_pkl}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
