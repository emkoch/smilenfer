Inference of simple selection models from the distribution of genetic associations.

[Genetic association data are broadly consistent with stabilizing selection shaping human common diseases and traits](https://doi.org/10.1101/2024.06.19.599789)
## Project components
### gwas_processing
Code for downloading and processing GWAS summary statistics.
### results
Processed GWAS summary statistics, current model-fitting workflows, model fits, simulation output, and scripts for generating figures. Current production model fitting is under `results/all_opt_fits`.
### smilenfer
Main code used for analyses, simulations, and plotting, organized as a Python package.
### snakemake
Legacy workflows used for the previous submission and retained for reproducibility of earlier grid-based fitting, simulation, and SFS-generation steps. These are not the current production model-fitting workflows.

## System requirements

Tested on Ubuntu 24.04.4 LTS; exact dependency versions are listed in `smilenfer_versions.yml`.
No non-standard hardware is required.

## Installation

Create and activate the Conda environment, then install the package:

```bash
conda env create -f smilenfer.yml
conda activate smilenfer
bash install.sh
```

Typical installation takes up to one hour.

## Demo

The breast-cancer example uses the same fitting rule as the current production analysis. From the repository root, run:

```bash
cd results/all_opt_fits/original_traits
snakemake --snakefile orignal_opt_fits.smk --cores 1 bc_standard_fits_post.pkl
```

The expected outputs are `bc_standard_fits_raw.pkl` and `bc_standard_fits_post.pkl`. The posterior-mean fit should match the `bc` row in `opt_results_original_traits_eur_post.csv`. Runtime is approximately two minutes on a normal desktop computer.

## Instructions for use

To analyze another trait, prepare `results/data/final/original_traits/processed.<trait>.snps_low_r2.tsv` in the same format as the included breast-cancer file, add the trait name to `TRAITS` in `results/all_opt_fits/original_traits/orignal_opt_fits.smk`, and run the corresponding `<trait>_standard_fits_post.pkl` target as above.

## License and repository

Licensed under the GNU General Public License v3.0. Source code: https://github.com/emkoch/smilenfer.
