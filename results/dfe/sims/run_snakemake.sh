# If unlock flag is set, unlock and run, otherwise run without unlocking
# Usage: bash run_snakemake.sh [unlock]
if [ "$1" == "unlock" ]; then
    snakemake --snakefile ../../../Snakefile_dfe_sims --cores 4 --configfile config.yml --unlock
    snakemake --snakefile ../../../Snakefile_dfe_sims --cores 4 --configfile config.yml --rerun-incomplete
    else
    snakemake --snakefile ../../../Snakefile_dfe_sims --cores 4 --configfile config.yml --rerun-incomplete
fi
