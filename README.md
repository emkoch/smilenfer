Inference of simple selection models from the distribution of genetic associations.

[Genetic association data are broadly consistent with stabilizing selection shaping human common diseases and traits](https://doi.org/10.1101/2024.06.19.599789)
## Project components
### gwas_processing
Code for downloading and processing GWAS summary statistics.
### results
Processed GWAS summary statistics, model fits, simulation output, and scripts for generating figures.
### smilenfer
Main code used for analyses, simulations, and plotting. This is organized as a python package and can be installed by first creating a conda environment with the necessary dependencies `conda env create -f smilenfer.yml`. After activating the environment with `conda activate smilenfer` run `install.sh`. 
### snakemake
Snakemake pipelines used for model fits and simulations. After installing the package, this is how model fits are actually performed.

#### More information on requirements and runtime can be found in `example_trait/example_trait_analysis.sh`

Tested on Ubuntu 24.04.4 LTS; exact dependency versions are listed in `smilenfer_versions.yml`.
No non-standard hardware is required.

## Demo

From `snakemake/test_run`, run `bash test_run.sh`. The primary expected output is `snakemake/test_run/output/ML_all_flat_5e-08_new.csv`; the breast-cancer results should match those in `results/data/ML/SIMPLE_ALL_TRAITS_NOFILTER_GENOMEWIDE/ML_all_flat_5e-08_new.csv`. Runtime is up to six hours on one core with 2 GB RAM.

## Instructions for use

To analyze another trait, provide a tab-separated input file in the format of `results/data/clumped_ash/clumped.genome_wide_ash.bc.max_r2.tsv`, edit `data_dir`, `out_dir`, `trait_files`, `trait_types`, and `traits` in a copy of `snakemake/test_run/test_run.yml`, and run Snakemake as in `snakemake/test_run/test_run.sh`.

## License and repository

Licensed under the GNU General Public License v3.0. Source code: https://github.com/emkoch/smilenfer.
