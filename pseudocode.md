# Pseudocode

## `download_and_reformat_gwas.sh`
- Downloads GWAS summary statistics from public repositories using wget
- Uses awk to select the relevant statistics from these files (allele frequency, effect size estimate, standard error, p-value)
- Creates a slimmed-down copy of each GWAS by taking every fifth line

## `gw_ashr.R`
- Reads in the every-fifth-line copy of GWAS summary statistics for a trait
- Uses these to create a prior distribution for adaptive shrinkage (ash)
- Takes the associated SNPs (genome-wide significant level) and updates them based on the ash prior

## `gw_clumping.R`
- Makes sure that no two SNPs are within 100kb, prioritizing SNPs by P-value

## `Snakemake_trait_fits_simple`
- For each trait's summary statistics, finds the best fit of the models described in Table 1 of the paper
- Creates a matrix of possible paramters (see Table 1)
- Tests the combinations in the matrix to find the parameter set that is most consistent with the summary statistics
- It then calculates the likelihood of the observed summary statistics under these parameters