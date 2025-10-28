import pandas as pd
import numpy as np
import os

SUSIEX_LOCI_FILE = "../../data/fine_mapping/SuSiEx/susiex_loci_filtered.csv"
SUSIEX_PIP_FILE  = "../../data/fine_mapping/SuSiEx/susiex_pip_combined.csv"
ORIGINAL_FINAL_LOC = "../../data/final/original_traits/"


def get_susiex_locus(chrom, pos, loci_file):
    # verify that loci_file has the correct columns
    required_columns = ["locus", "trait", "chr", "start", "end"]
    for col in required_columns:
        if col not in loci_file.columns:
            raise ValueError(f"Missing required column: {col}")
    # filter the loci file for the given chromosome and position
    filtered_loci = loci_file.loc[(loci_file["chr"] == chrom) & (loci_file["start"] <= pos) & (loci_file["end"] >= pos)].copy()
    print(filtered_loci)
    if filtered_loci.empty:
        return None
    elif len(filtered_loci) == 1:
        return filtered_loci.loc[:, "locus"].to_numpy()
    else:
        print("Warning: multiple loci found for the given position")
        return filtered_loci.loc[:, "locus"].to_numpy()
        
def get_susiex_cs(chrom, pos, loci_file, pip_file, raf, rbeta):
    locus = get_susiex_locus(chrom, pos, loci_file)
    if locus is None:
        print("No locus found for the given position")
        return None, None, None
    # don't proceed for now if the length of locus is not 1
    if len(locus) != 1:
        print("Multiple loci found for the given position")
        print(locus)
        return None, None, None
    # filter the pip file for the given locus
    filtered_pip = pip_file.loc[pip_file["locus"] == locus[0]].copy()
    if filtered_pip.empty:
        print("No credible set found for the given locus")
        return None, None, None
    
    # replace all non-numeric values in frequency, beta, se, and pip columns with NaN
    for col in ["alt_freq_eur", "beta_eur", "se_eur", "pip_eur", "alt_freq_eas", "beta_eas", "se_eas", "pip_eas"]:
        filtered_pip[col] = pd.to_numeric(filtered_pip[col], errors='coerce')
    
    # get trait-increasing frequencies
    filtered_pip["raf_eur"] = np.where(filtered_pip.beta_eur > 0, filtered_pip["alt_freq_eur"], 1 - filtered_pip["alt_freq_eur"])
    filtered_pip["raf_eas"] = np.where(filtered_pip.beta_eas > 0, filtered_pip["alt_freq_eas"], 1 - filtered_pip["alt_freq_eas"])
    filtered_pip["rbeta_eur"] = np.abs(filtered_pip.beta_eur)
    filtered_pip["rbeta_eas"] = np.abs(filtered_pip.beta_eas)

    def rr(raf, rbeta, raf_b, rbeta_b):
        return rbeta_b / rbeta * np.sqrt(raf_b * (1 - raf_b) / (raf * (1 - raf)))
    
    # compute correlation values for each variant in the table
    filtered_pip["corr_eur"] = np.minimum(rr(raf, rbeta, filtered_pip["raf_eur"], filtered_pip["rbeta_eur"]),
                                            rr(filtered_pip["raf_eur"], filtered_pip["rbeta_eur"], raf, rbeta))   

    # get the locus and CS id with the highest correlation
    max_corr_i = np.argmax(filtered_pip["corr_eur"])
    max_corr_cs = int(filtered_pip.iloc[max_corr_i]["cs_id"])
    max_corr = float(filtered_pip.iloc[max_corr_i]["corr_eur"])
    max_locus = int(filtered_pip.iloc[max_corr_i]["locus"])
    
    return max_corr_cs, max_corr, max_locus

def build_cs_table(chrom_list, pos_list, raf_list, rbeta_list, loci_file, pip_file):
    # replace all non-numeric values in frequency, beta, se, and pip columns with NaN
    for col in ["alt_freq_eur", "beta_eur", "se_eur", "pip_eur", "alt_freq_eas", "beta_eas", "se_eas", "pip_eas"]:
        pip_file[col] = pd.to_numeric(pip_file[col], errors='coerce')
    pip_file["raf_eur"] = np.where(pip_file.beta_eur > 0, pip_file["alt_freq_eur"], 1 - pip_file["alt_freq_eur"])
    pip_file["raf_eas"] = np.where(pip_file.beta_eas > 0, pip_file["alt_freq_eas"], 1 - pip_file["alt_freq_eas"])
    pip_file["rbeta_eur"] = np.abs(pip_file.beta_eur)
    pip_file["rbeta_eas"] = np.abs(pip_file.beta_eas)
    
    # initialize a dict of lists to store the results
    cs_table = {
        "locus":[],
        "chrom": [],
        "pos": [],
        "ref": [],
        "alt": [],
        "raf": [],
        "rbeta": [],
        "se": [],
        "cs_id": [],
        "corr": [],
        "pip": [],
        "orig_locus": [],
        "orig_pos": [],
        "orig_raf": [],
        "orig_rbeta": [],
    }
    
    # iterate over the input lists
    for chrom, pos, raf, rbeta in zip(chrom_list, pos_list, raf_list, rbeta_list):
        locus_id = str(chrom) + ":" + str(pos)
        # get the credible set and correlation
        cs_id, corr, locus = get_susiex_cs(chrom, pos, loci_file, pip_file, raf, rbeta)
        if cs_id is not None:
            # subset the pip file for the given cs_id
            cs_pip = pip_file.loc[(pip_file["cs_id"].to_numpy() == cs_id) & (pip_file["locus"].to_numpy() == locus)].copy()
            # for each row add information to the cs_table
            for _, row in cs_pip.iterrows():
                cs_table["locus"].append(locus)
                cs_table["chrom"].append(chrom)
                cs_table["pos"].append(row["pos"])
                cs_table["ref"].append(row["ref"])
                cs_table["alt"].append(row["alt"])
                cs_table["raf"].append(row["raf_eur"])
                cs_table["rbeta"].append(row["rbeta_eur"])
                cs_table["se"].append(row["se_eur"])
                cs_table["cs_id"].append(cs_id)
                cs_table["corr"].append(corr)
                cs_table["pip"].append(row["pip_eur"])
                cs_table["orig_locus"].append(locus_id)
                cs_table["orig_pos"].append(pos)
                cs_table["orig_raf"].append(raf)
                cs_table["orig_rbeta"].append(rbeta)
        else:
            print(f"No credible set found for locus {locus_id} at position {pos}")
            print(f"chrom: {chrom}, pos: {pos}, raf: {raf}, rbeta: {rbeta}")
            # if no credible set was found, add a row with NaN values
            cs_table["locus"].append(np.nan)
            cs_table["chrom"].append(chrom)
            cs_table["pos"].append(np.nan)
            cs_table["ref"].append(np.nan)
            cs_table["alt"].append(np.nan)
            cs_table["raf"].append(np.nan)
            cs_table["rbeta"].append(np.nan)
            cs_table["se"].append(np.nan)
            cs_table["cs_id"].append(np.nan)
            cs_table["corr"].append(np.nan)
            cs_table["pip"].append(np.nan)
            cs_table["orig_locus"].append(locus_id)
            cs_table["orig_pos"].append(pos)
            cs_table["orig_raf"].append(raf)
            cs_table["orig_rbeta"].append(rbeta)
    # convert the dict to a DataFrame and return
    return pd.DataFrame(cs_table)


def sample_finemap(fm_df):
    # Make a copy of the dataframe
    fm_df = fm_df.copy()
    
    # For rows where locus is NaN, set raf and rbeta to orig_raf and orig_rbeta
    fm_df.loc[fm_df["locus"].isna(), "raf"] = fm_df["orig_raf"]
    fm_df.loc[fm_df["locus"].isna(), "rbeta"] = fm_df["orig_rbeta"]
    
    # For these rows set pip to 1
    fm_df.loc[fm_df["locus"].isna(), "pip"] = 1
    
    # Come up with new, unique locus numbers for these rows and set cs_id to 1
    n_missing = fm_df["locus"].isna().sum()

    if n_missing:
        start_id = int(fm_df["locus"].max(skipna=True)) + 1
        fm_df.loc[fm_df["locus"].isna(), "locus"] = np.arange(start_id,
                                                            start_id + n_missing,
                                                            dtype=int)
        fm_df.loc[fm_df["locus"].isna(), "cs_id"] = 1

    # Reduce to locus, cs_id, raf, rbeta, and pip columns
    fm_df = fm_df[["locus", "cs_id", "raf", "rbeta", "pip"]].copy()
    
    # Remove any duplicate rows
    fm_df = fm_df.drop_duplicates()
    
    # Normalize pip values to sum to 1 within each (locus, cs_id) pair
    def normalize_pip(group):
        total_pip = group["pip"].sum()
        if total_pip > 0:
            group["pip"] = group["pip"] / total_pip
        else:
            group["pip"] = 1.0 / len(group)
        return group
    
    fm_df = fm_df.groupby(["locus", "cs_id"], group_keys=False).apply(normalize_pip)

    # save fm df for debugging
    # fm_df.to_csv("fm_df_debug.csv", index=False)
    
    # Check that pip values sum to 1 within each (locus, cs_id) group with a tolerance
    pip_sums = fm_df.groupby(["locus", "cs_id"])["pip"].sum()
    if not np.allclose(pip_sums, 1, atol=1e-6):
        print("Warning: PIP values do not sum to 1 within some (locus, cs_id) groups")
        print(pip_sums[~np.isclose(pip_sums, 1, atol=1e-6)])
    
    assert np.allclose(pip_sums, 1, atol=1e-6), "PIP values do not sum to 1 within some (locus, cs_id) groups"
    
    # Sample one row from each (locus, cs_id) pair, weighted by pip
    fm_df = fm_df.groupby(["locus", "cs_id"]).apply(lambda x: x.sample(n=1, weights=x["pip"])).reset_index(drop=True)
    
    return fm_df

def read_trait_build_table(trait_fname, trait):
    susiex_loci = pd.read_csv(SUSIEX_LOCI_FILE)
    susiex_pip = pd.read_csv(SUSIEX_PIP_FILE)

    # Verify that the trait exists in the loci and pip files
    if trait not in susiex_loci["trait"].unique():
        raise ValueError(f"Trait {trait} not found in SuSiEx loci file.")
    if trait not in susiex_pip["trait"].unique():
        raise ValueError(f"Trait {trait} not found in SuSiEx PIP file.")

    trait_loci = susiex_loci[susiex_loci["trait"] == trait]
    trait_pip = susiex_pip[susiex_pip["trait"] == trait]

    trait_data = pd.read_csv(trait_fname, sep="\t")

    trait_chroms = trait_data["chr"].to_numpy()
    trait_pos = trait_data["pos"].to_numpy()
    trait_rafs = trait_data["raf"].to_numpy()
    trait_rbetas = trait_data["rbeta"].to_numpy()

    trait_cs_table = build_cs_table(
        trait_chroms, trait_pos, trait_rafs, trait_rbetas,
        trait_loci, trait_pip
    )

    # make the chrom, pos, locus, cs_id, orig_pos, and pip columns integers
    trait_cs_table["locus"] = trait_cs_table["locus"].astype(int)
    trait_cs_table["cs_id"] = trait_cs_table["cs_id"].astype(int)
    trait_cs_table["orig_pos"] = trait_cs_table["orig_pos"].astype(int)
    trait_cs_table["pip"] = trait_cs_table["pip"].astype(int)

    return trait_cs_table

# Do a main where we generate the table for sets of traits
if __name__ == "__main__":
    # Define the traits to process
    matched_trait_names = {"BMI":"bmi", "DBP":"dbp", "HDL-C":"hdl", "HT":"height", "LDL-C":"ldl", "TG":"triglycerides", "WBC":"wbc", "SBP":"sbp"}

    # read the trait files and build the tables
    for trait, original_name in matched_trait_names.items():
        print(trait, original_name)
        trait_fname = os.path.join(ORIGINAL_FINAL_LOC, f"processed.{original_name}.snps_low_r2.tsv")
        if not os.path.exists(trait_fname):
            print(f"Trait file {trait_fname} does not exist. Skipping trait {trait}.")
            continue
        trait_table = read_trait_build_table(trait_fname, trait)
        # Save the table to a file
        trait_table.to_csv(f"susiex_cs_table_{original_name}.csv", index=False)
