import requests
import os
import pandas as pd
import numpy as np
import smilenfer.posterior as post

# Function to get rsID from Ensembl using genomic coordinates
def get_rsid(snp, hg="37"):
    chr, pos, ref, alt = snp.split(':')
    if hg == "37":
        url = f"https://grch37.rest.ensembl.org/overlap/region/human/{chr}:{pos}-{pos}?feature=variation;content-type=application/json"
    else:
        url = f"https://rest.ensembl.org/overlap/region/human/{chr}:{pos}-{pos}?feature=variation;content-type=application/json"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)
    
    if not response.ok:
        print(f"Error: Received status code {response.status_code} for SNP {snp}")
        return None

    try:
        data = response.json()
    except ValueError:
        print(f"Error decoding JSON response for SNP {snp}")
        return None

    for entry in data:
        if 'id' in entry:
            if ref in entry['alleles'] and alt in entry['alleles']:
                return entry['id']

    print(f"No matching rsID found for SNP {snp}")
    return None

def get_ld_r2(rsid1, rsid2, build="grch37"):
    url = f"https://ldlink.nih.gov/LDlinkRest/ldpair?var1={rsid1}&var2={rsid2}&pop=EUR&genome_build={build}&json_out=true&token=a35abb232891"
    headers = {"Content-Type": "application/json"}
    
    response = requests.get(url, headers=headers)
    
    if not response.ok:
        print(f"Error: Received status code {response.status_code} for SNP pair {rsid1}, {rsid2}")
        return np.nan
    
    try:
        data = response.json()
        if len(data) == 0:
            print(f"No data found for SNP pair {rsid1}, {rsid2}")
            return np.nan
    except ValueError:
        print("Error decoding JSON response for SNP pair {rsid1}, {rsid2}")
        return np.nan
    
    try:
        return float(data[0]['statistics']['r2'])
    except KeyError:
        print(data)
        return np.nan
    except ValueError:
        print(data)
        return np.nan
    

main_traits = ["bc", "bmi", "cad", "dbp", "hdl", "height", "ibd", "ldl", "rbc",
               "sbp", "scz", "t2d", "triglycerides", "urate", "wbc"]

hg37_traits = ["bmi", "cad", "dbp", "hdl", "height", "ibd", "ldl", "rbc", "sbp", "scz", "t2d", "triglycerides", "urate", "wbc"]
hg38_traits = ["bc"]


# Do the above for all traits in main_traits
for trait in main_traits:
    print(f"getting rsids for trait {trait}")
    fname = f"clumped.genome_wide_ash.{trait}.tsv"
    data = pd.read_csv(fname, sep="\t")

    hg = "37" if trait in hg37_traits else "38"

    chroms = data.chr.to_numpy()
    positions = data.pos.to_numpy()
    snps = data.snp.to_numpy(dtype="str")

    has_rsid = False
    if "rsit" in data.columns or "rsid" in data.columns:
        has_rsid = True
        try:
            rsids = data.rsid.to_numpy(dtype="str")
        except AttributeError:
            rsids = data.rsit.to_numpy(dtype="str")
        for i in range(len(rsids)):
            if not "rs" in rsids[i]:
                rsids[i] = get_rsid(snps[i], hg)
    else:
        rsids = []
        for i in range(len(chroms)):
            rsids.append(get_rsid(snps[i], hg))

    # save a version of the data with rsids
    data["rsid_new"] = rsids
    data.to_csv(f"clumped.genome_wide_ash.{trait}.rsid.tsv", sep="\t", index=False)

    snp_1 = []
    snp_2 = []
    rsids_1 = []
    rsids_2 = []
    r2_set = []

    # get all pairs of sites on the same chromosome and within 1Mb of each other
    for i in range(len(chroms)):
        rsid1 = rsids[i]
        if rsid1 is None:
            continue
        for j in range(i+1, len(chroms)):
            if chroms[j] > chroms[i]:
                break
            if chroms[i] == chroms[j] and abs(positions[i] - positions[j]) < 1000000:
                rsid2 = rsids[j]
                if rsid2 is None:
                    continue
                if rsid1 is not None and rsid2 is not None:
                    snp_1.append(snps[i])
                    snp_2.append(snps[j])
                    rsids_1.append(rsid1)
                    rsids_2.append(rsid2)
                else:
                    print("Error getting rsIDs for SNPs: ", snps[i], snps[j])

    topld_format_data = pd.DataFrame({"rsid_1": rsids_1, "rsid_2": rsids_2})
    topld_format_data.to_csv(f"rsids_{trait}_topld.csv", index=False, header=False)

    snp_format_data = pd.DataFrame({"snp_1": snp_1, "snp_2": snp_2, "rsid_1": rsids_1, "rsid_2": rsids_2})
    snp_format_data.to_csv(f"snp_pairs_{trait}_1Mb.csv", index=False)

# Get topmed LD data for all traits
topld_path = "/home/evan/Drive/Work/Sunyaev/POLYGENIC_SELECTION/PROGRAMS/topld_api/topld_api"

for trait in main_traits:
    print(f"Running topld for trait {trait}")
    command = f"{topld_path} -thres 0.2 -pop EUR -maf 0.01 -inFile rsids_{trait}_topld.csv -outputLD r2_{trait}_topld.csv -outputInfo info_{trait}_topld.csv"
    os.system(command)

chunk_size = 1000  # Adjust this to control the size of each chunk

def split_file(file_path, chunk_size):
    # Read the input file
    data = pd.read_csv(file_path, header=None)
    
    # Split the data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]
    
    # Save each chunk to a temporary file
    temp_files = []
    for i, chunk in enumerate(chunks):
        temp_file = f"{file_path}_chunk_{i}.csv"
        chunk.to_csv(temp_file, index=False, header=False)
        temp_files.append(temp_file)
    
    return temp_files

def concatenate_files(output_files, output_path):
    # Concatenate all output files
    combined_data = pd.concat([pd.read_csv(f, sep="\t") for f in output_files])
    
    # Save the combined data to the final output file
    combined_data.to_csv(output_path, sep="\t", index=False)

def run_topld_for_trait(trait):
    input_file = f"rsids_{trait}_topld.csv"
    
    # Split the input file into smaller chunks
    temp_input_files = split_file(input_file, chunk_size)
    
    output_files_ld = []
    output_files_info = []

    for i, temp_input_file in enumerate(temp_input_files):
        temp_output_file_ld = f"r2_{trait}_topld_chunk_{i}.csv"
        temp_output_file_info = f"info_{trait}_topld_chunk_{i}.csv"
        
        print(f"Running topld for chunk {i} of trait {trait}")
        command = f"{topld_path} -thres 0.2 -pop EUR -maf 0.01 -inFile {temp_input_file} -outputLD {temp_output_file_ld} -outputInfo {temp_output_file_info}"
        os.system(command)
        
        output_files_ld.append(temp_output_file_ld)
        output_files_info.append(temp_output_file_info)
    
    # Concatenate the chunked output files
    concatenate_files(output_files_ld, f"r2_{trait}_topld.csv")
    concatenate_files(output_files_info, f"info_{trait}_topld.csv")
    
    # Clean up temporary files
    for temp_file in temp_input_files + output_files_ld + output_files_info:
        os.remove(temp_file)

for trait in main_traits:
    run_topld_for_trait(trait)

# Use LDlink for all variant pairs not present in topmed
for trait in main_traits:
    topld_info_file = f"info_{trait}_topld.csv"
    topld_info = pd.read_csv(topld_info_file, sep="\t")
    topld_info_rsids = topld_info.rsID.to_numpy(dtype=str)
    # get the rsids for the pairs that are missing from topmed
    rsid_pairs = pd.read_csv(f"rsids_{trait}_topld.csv", header=None)
    # get all the rsid pairs where one of the two columns is not in topld_info_rsids
    missing_pairs = rsid_pairs[~rsid_pairs[0].isin(topld_info_rsids) | ~rsid_pairs[1].isin(topld_info_rsids)].copy()
    print("trait ", trait, " missing pairs: ", missing_pairs.shape[0])
    # for each missing pair us get_ld_r2 to get the r2 value
    r2_set = []
    for i in range(len(missing_pairs)):
        print(i)
        r2 = get_ld_r2(missing_pairs.iloc[i][0], missing_pairs.iloc[i][1])
        r2_set.append(r2)
        print(f"Pair {i+1}/{len(missing_pairs)}: {missing_pairs.iloc[i][0]}, {missing_pairs.iloc[i][1]}: {r2}")
    missing_pairs["R2"] = np.array(r2_set, dtype=float)
    missing_pairs.to_csv(f"missing_r2_{trait}_topld.csv", index=False, header=False)

for trait in main_traits:
    r2_file = f"r2_{trait}_topld.csv"
    r2_data = pd.read_csv(r2_file, sep="\t")

    missing_r2 = pd.read_csv(f"missing_r2_{trait}_topld.csv", header=None)
    # set missing values in the R2 column to 1
    missing_r2.fillna(2)

    rsid1 = r2_data.rsID1.to_numpy(dtype=str)
    rsid1 = np.append(rsid1, missing_r2[0].to_numpy(dtype=str))
    rsid2 = r2_data.rsID2.to_numpy(dtype=str)
    rsid2 = np.append(rsid2, missing_r2[1].to_numpy(dtype=str))
    r2 = r2_data.R2.to_numpy(dtype=float)
    r2 = np.append(r2, missing_r2[2].to_numpy(dtype=float))
    
    snp_format_data = pd.read_csv(f"snp_pairs_{trait}_1Mb.csv")
    fname = f"clumped.genome_wide_ash.{trait}.tsv"
    trait_data = post.read_and_process_trait_data(fname)

    fname_rsid = f"clumped.genome_wide_ash.{trait}.rsid.tsv"
    rsid_set = pd.read_csv(fname_rsid, sep="\t").rsid_new.to_numpy(dtype=str)
    trait_data["rsid"] = rsid_set
    
    snps = trait_data.snp.to_numpy(dtype="str")
    mlogp = trait_data.mlogp.to_numpy(dtype=float)

    max_r2_set = []

    for i, snp in enumerate(snps):
        if rsid_set[i] is None or rsid_set[i] == "nan":
            max_r2_set.append(2)
            continue
        # check if the SNP is a member of any nearby pairs
        if not snp in snp_format_data.snp_1.to_numpy(dtype=str) and not snp in snp_format_data.snp_2.to_numpy(dtype=str):
            max_r2_set.append(0)
            continue
        # get the rsid for the SNP
        if snp in snp_format_data.snp_1.to_numpy():
            rsid = snp_format_data[snp_format_data.snp_1.to_numpy(dtype=str) == snp].rsid_1.to_numpy(dtype=str)[0]
        else:
            rsid = snp_format_data[snp_format_data.snp_2.to_numpy(dtype=str) == snp].rsid_2.to_numpy(dtype=str)[0]

        
        # get rsids of other SNPs in the r2 file where the SNP is one of the pair
        r2_friends = rsid1==rsid
        r2_friends_rsids = rsid2[r2_friends]
        r2_friends_r2 = r2[r2_friends]

        r2_friends = rsid2==rsid # r2_data.rsID2 == rsid
        r2_friends_rsids = np.append(r2_friends_rsids, rsid1[r2_friends])
        r2_friends_r2 = np.append(r2_friends_r2, r2[r2_friends])
    
        # get the all the pairs that contain the SNP
        pairs = snp_format_data[(snp_format_data.snp_1.to_numpy(dtype=str) == snp) | 
                                (snp_format_data.snp_2.to_numpy(dtype=str) == snp)]

        # get the snp names for r2_friends rsids
        r2_friends_snps = []
        for friend_rsid in r2_friends_rsids:
            if friend_rsid in snp_format_data.rsid_1.to_numpy():
                r2_friends_snps.append(snp_format_data[snp_format_data.rsid_1.to_numpy(dtype=str) == friend_rsid].snp_1.to_numpy(dtype=str)[0])
            else:
                r2_friends_snps.append(snp_format_data[snp_format_data.rsid_2.to_numpy(dtype=str) == friend_rsid].snp_2.to_numpy(dtype=str)[0])
        r2_friends_snps = np.array(r2_friends_snps, dtype=str)

        # get the mlogp values for the r2_friends
        r2_friends_mlogp = []
        for friend_snp in r2_friends_snps:
            if friend_snp in snps:
                r2_friends_mlogp.append(mlogp[np.where(snps == friend_snp)[0][0]])
            else:
                print(f"SNP {friend_snp} not found in {trait} data")
                r2_friends_mlogp.append(np.nan)
        r2_friends_mlogp = np.array(r2_friends_mlogp, dtype=float)

        # get the mlogp value for the focal SNP
        focal_mlogp = mlogp[np.where(snps == snp)[0][0]]

        more_sig_r2 = r2_friends_r2[r2_friends_mlogp > focal_mlogp]

        if len(more_sig_r2) == 0:
            max_r2_set.append(0)
            continue
        max_more_sig_r2 = np.max(more_sig_r2)
        max_r2_set.append(max_more_sig_r2)

    trait_data["max_r2"] = np.array(max_r2_set, dtype=float)
    trait_data.to_csv(f"clumped.genome_wide_ash.{trait}.max_r2.tsv", sep="\t", index=False)
