# To run the code, load the conda environment stored in smilenfer.yml
# The installation of conda is described in https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
# The environment will have to be installed by running the following command (but only the first time)
# Some packages require a linux environment, but generally have equivalents on MacOS
# Installing on a linux cluster with 20Gb of ram to approximately 50 minutes, and slightly longer than 1 hour on an Ubuntu laptop with 16Gb of ram.
conda env create -f smilenfer.yml

# Then the environment will have to be loaded each time
conda activate smilenfer

# We will be using breast cancer as the example trait.

# First it is downloaded and reformated, as in download_and_reformat_gwas.sh
wget http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004988/harmonised/29059683-GCST004988-EFO_0000305.h.tsv.gz
mv 29059683-GCST004988-EFO_0000305.h.tsv.gz michailidou_2017.bc.b37.tsv.gz

# This first lines of this file are
# hm_variant_id  hm_rsid      hm_chrom  hm_pos  hm_other_allele  hm_effect_allele  hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  hm_effect_allele_frequency  hm_code  variant_id              chromosome  base_pair_location  other_allele  effect_allele  effect_allele_frequency  beta     standard_error  p_value  ci_lower  odds_ratio  ci_upper
# 10_14744_A_C   rs569167217  10        14744   A                C                 -0.0273  NA             NA           NA           0.0294                      10       rs569167217             10          14744               A             C              0.0294                   -0.0273  0.0224          0.2244   NA        NA          NA
# 10_15029_A_C   rs61838556   10        15029   A                C                 0.0101   NA             NA           NA           0.3657                      11       rs61838556              10          15029               C             A              0.6343                   -0.0101  0.0085          0.2352   NA        NA          NA
# 10_15391_A_G   rs548639866  10        15391   A                G                 -0.0272  NA             NA           NA           0.0294                      10       rs548639866             10          15391               A             G              0.0294                   -0.0272  0.0224          0.2252   NA        NA          NA
# NA             NA           NA        NA      NA               NA                NA       NA             NA           NA           NA                          15       rs147855157:61372:CA:C  10          15432               CA            C              0.7103                   -0.0047  0.0084          0.5757   NA        NA          NA
# 10_15479_G_A   rs553163044  10        15479   G                A                 -0.0108  NA             NA           NA           0.0624                      10       rs553163044             10          15479               G             A              0.0624                   -0.0108  0.0201          0.5903   NA        NA          NA
# 10_17274_G_C   rs542543788  10        17274   G                C                 -0.0273  NA             NA           NA           0.0294                      5        rs542543788             10          17274               G             C              0.0294                   -0.0273  0.0224          0.2244   NA        NA          NA
# 10_18828_T_C   rs61838558   10        18828   T                C                 0.0145   NA             NA           NA           0.26649999999999996         11       rs61838558              10          18828               C             T              0.7335                   -0.0145  0.0089          0.1024   NA        NA          NA
# 10_18930_C_A   rs556434813  10        18930   C                A                 -0.0093  NA             NA           NA           0.0164                      10       rs556434813             10          18930               C             A              0.0164                   -0.0093  0.0312          0.7664   NA        NA          NA
# 10_19939_G_C   rs28887774   10        19939   G                C                 0.0162   NA             NA           NA           0.26570000000000005         6        rs28887774              10          19939               C             G              0.7343                   -0.0162  0.0087          0.06377  NA        NA          N

# A selection of 10,000 SNPs from chr 1 is contained in toy.michailidou.tsv.gz
# The scripts can be run with this small dataset (to test the environment), but there are far too few SNPs to get meaningful results.

pheno=bc
zcat michailidou_2017.bc.b37.tsv.gz | \
awk 'BEGIN {print "snp\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval\tn"} \
    $14 < 1 || $14 > 22 {next} \
    $14 == 6 && $15 > 27500000 && $15 < 35000000 {next} \
    length($16) != 1 || length($17) != 1 {next} \
    {print $14":"$15":"$16":"$17 "\t" $14 "\t" $15 "\t" $16 "\t" $17 "\t" $18 "\t" $19 "\t" $20 "\t" $21 "\t228951"}' | \
gzip > gwas.bc.b37.tsv.gz

# The first lines of this re-formatted file are
# snp           chr  pos    a1  a2  af      beta     se      pval     n
# 10:14744:A:C  10   14744  A   C   0.0294  -0.0273  0.0224  0.2244   228951
# 10:15029:C:A  10   15029  C   A   0.6343  -0.0101  0.0085  0.2352   228951
# 10:15391:A:G  10   15391  A   G   0.0294  -0.0272  0.0224  0.2252   228951
# 10:15479:G:A  10   15479  G   A   0.0624  -0.0108  0.0201  0.5903   228951
# 10:17274:G:C  10   17274  G   C   0.0294  -0.0273  0.0224  0.2244   228951
# 10:18828:C:T  10   18828  C   T   0.7335  -0.0145  0.0089  0.1024   228951
# 10:18930:C:A  10   18930  C   A   0.0164  -0.0093  0.0312  0.7664   228951
# 10:19939:C:G  10   19939  C   G   0.7343  -0.0162  0.0087  0.06377  228951
# 10:20364:A:G  10   20364  A   G   0.0342  -0.011   0.0217  0.6117   228951

file=gwas.$pheno.b37.tsv.gz # This is part of a loop
out=$(echo $file | sed 's/.b37//g; s/.tsv.gz/.5th.tsv.gz/')
zcat $file | awk 'NR % 5 == 1' | gzip > $out

# Now we run the R script that performs adaptive shrinkage (ash)
# Depending on the number of significant points in the GWAS, this should take from 10 minutes to 1 hour
Rscript test.ashr.R --phen $pheno

# And then perform a clumping based on distance
# This should take under 5 minutes
Rscript gw_clumping.R --phen $pheno

# The output of this is clumped.genome_wide_ash.bc.tsv
# The LD filtering is done in the python scripts run by the snakemake file

# The LD filtering and model fitting can be run with
snakemake \
    --snakefile ../Snakefile_trait_fits_simple \
    --cores 1 \
    --configfile test_run.yml
# Given 2Gb or ram, this process take up to six hours, depending on the number of genome-wide significant associations.